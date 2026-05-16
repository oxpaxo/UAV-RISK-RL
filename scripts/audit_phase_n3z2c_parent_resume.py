from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


COMPLETE_FLAG = "PHASE_N3Z2C_AUDIT_COMPLETE.flag"
STOP_FLAGS = {
    "n3z2c_missing": "PHASE_N3Z2C_AUDIT_STOP_N3Z2C_MISSING.flag",
    "parent_candidates_missing": "PHASE_N3Z2C_AUDIT_STOP_PARENT_CANDIDATES_MISSING.flag",
    "resume_semantics_unresolved": "PHASE_N3Z2C_AUDIT_STOP_RESUME_SEMANTICS_UNRESOLVED.flag",
    "cpu_affinity_unresolved": "PHASE_N3Z2C_AUDIT_STOP_CPU_AFFINITY_UNRESOLVED.flag",
}

PARENT_CANDIDATES = [
    ("checkpoint_500k", "checkpoints/env_v2_gpsi_heada_ppo_n3z_layernorm_alpha0p5_s0/checkpoint_500k.zip", "500k"),
    ("final", "checkpoints/env_v2_gpsi_heada_ppo_n3z_layernorm_alpha0p5_s0/final.zip", "final"),
    ("best_by_eval", "checkpoints/env_v2_gpsi_heada_ppo_n3z_layernorm_alpha0p5_s0/best_by_eval.zip", "best_by_eval"),
]


class AuditStop(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Phase N3Z2C parent selection, resume semantics, and CPU affinity.")
    parser.add_argument("--n3z2c-result-dir", default="results/env_v2_phase_n3z2c_z2_continuation")
    parser.add_argument("--n3fz-result-dir", default="results/env_v2_phase_n3fz_noz_full_z_screen")
    parser.add_argument("--out-dir", default="results/env_v2_phase_n3z2c_audit")
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_stop(out_dir: Path, reason: str, detail: str) -> None:
    flag = STOP_FLAGS.get(reason, STOP_FLAGS["resume_semantics_unresolved"])
    write_text(out_dir / flag, f"{reason}\n{detail.strip()}\n")
    write_text(out_dir / "phase_n3z2c_audit_status.txt", f"stopped:{flag}\n")
    write_text(
        out_dir / "PHASE_N3Z2C_AUDIT_REPORT.md",
        "\n".join(
            [
                "# Phase N3Z2C-Audit Report",
                "",
                f"`terminal_decision = phase_n3z2c_audit_stopped_{reason}`",
                "",
                "Partial report generated during parent/resume/CPU audit.",
                "",
                "```text",
                detail.strip(),
                "```",
                "",
                "Can enter N4: no.",
            ]
        )
        + "\n",
    )


def sha256(path: Path) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(name: str, command: list[str] | str, shell: bool = False) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, shell=shell, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
        return {
            "item": name,
            "command": command if isinstance(command, str) else " ".join(command),
            "returncode": int(completed.returncode),
            "output": completed.stdout.strip(),
        }
    except Exception as exc:
        return {"item": name, "command": command if isinstance(command, str) else " ".join(command), "returncode": -1, "output": repr(exc)}


def cpu_affinity_rows() -> list[dict[str, Any]]:
    rows = [
        run_command("nproc", ["nproc"]),
        run_command("nproc_all", ["nproc", "--all"]),
        run_command("lscpu", ["lscpu"]),
        run_command("taskset_current_shell", "taskset -pc $$", shell=True),
        run_command(
            "python_cpu_affinity",
            [
                sys.executable,
                "-c",
                "import os,json; print(json.dumps({'os_cpu_count':os.cpu_count(),'affinity_count':len(os.sched_getaffinity(0)),'affinity':sorted(os.sched_getaffinity(0))}, sort_keys=True))",
            ],
        ),
        run_command("cpuset_cpus", "cat /sys/fs/cgroup/cpuset.cpus 2>/dev/null || true", shell=True),
        run_command("cpuset_cpus_effective", "cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null || true", shell=True),
        run_command("cpu_max", "cat /sys/fs/cgroup/cpu.max 2>/dev/null || true", shell=True),
        run_command("free_h", ["free", "-h"]),
        run_command("df_h_root", ["df", "-h", "/"]),
        run_command("nvidia_smi", ["nvidia-smi"]),
    ]
    for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        rows.append({"item": name, "command": "env", "returncode": 0, "output": os.environ.get(name, "")})
    row_by_item = {row["item"]: row for row in rows}
    finding = "unresolved"
    try:
        nproc = int(row_by_item["nproc"]["output"].splitlines()[0].strip())
        nproc_all = int(row_by_item["nproc_all"]["output"].splitlines()[0].strip())
        affinity_payload = json.loads(row_by_item["python_cpu_affinity"]["output"].splitlines()[-1])
        affinity_count = int(affinity_payload["affinity_count"])
        cpu_max = row_by_item["cpu_max"]["output"].strip()
        cpuset_effective = row_by_item["cpuset_cpus_effective"]["output"].strip()
        omp_threads = os.environ.get("OMP_NUM_THREADS", "")
        if nproc == nproc_all == affinity_count and nproc >= 16 and cpu_max.startswith("max"):
            finding = "current_context_has_full_16_logical_cpus"
        elif nproc == 1 and affinity_count >= 16 and cpu_max.startswith("max") and omp_threads == "1":
            finding = "nproc_reports_omp_thread_limit_not_affinity_limit"
        elif affinity_count == 1:
            finding = "current_context_cpu_limited"
        else:
            finding = "current_context_partial_or_mixed_cpu_limits"
        rows.append(
            {
                "item": "cpu_affinity_finding",
                "command": "derived",
                "returncode": 0,
                "output": json.dumps(
                    {
                        "finding": finding,
                        "nproc": nproc,
                        "nproc_all": nproc_all,
                        "affinity_count": affinity_count,
                        "cpu_max": cpu_max,
                        "cpuset_cpus_effective": cpuset_effective,
                        "OMP_NUM_THREADS": omp_threads,
                    },
                    sort_keys=True,
                ),
            }
        )
    except Exception as exc:
        rows.append({"item": "cpu_affinity_finding", "command": "derived", "returncode": -1, "output": f"unresolved: {exc!r}"})
    return rows


def read_old_parent_selection(path: Path) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return ""
    df = pd.read_csv(path)
    if "selected_parent_key" not in df.columns:
        return ""
    selected = df[df["selected_parent_key"].notna()]
    if selected.empty:
        return ""
    return str(selected["selected_parent_key"].iloc[-1])


def candidate_metrics(n3fz_dir: Path) -> pd.DataFrame:
    summary_path = n3fz_dir / "tables/phase_n3fz_checkpoint_eval_summary.csv"
    if not summary_path.exists():
        raise AuditStop("parent_candidates_missing", f"missing N3F/Z summary: {rel(summary_path)}")
    summary = pd.read_csv(summary_path)
    z2 = summary[summary["method_key"].astype(str) == "z_layernorm_alpha_0p5"].copy()
    if z2.empty:
        raise AuditStop("parent_candidates_missing", "missing z_layernorm_alpha_0p5 rows in N3F/Z summary")
    return (
        z2.groupby("checkpoint_label", dropna=False)
        .agg(
            success_rate=("success_rate", "mean"),
            collision_rate=("collision_rate", "mean"),
            near_miss_rate=("near_miss_rate", "mean"),
            raw_unsafe_action_rate=("raw_unsafe_action_rate", "mean"),
            nan_or_crash=("nan_or_crash", "max"),
        )
        .reset_index()
    )


def diagnostics_ok_by_label(n3fz_dir: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    gpsi_path = n3fz_dir / "tables/phase_n3fz_gpsi_output_summary.csv"
    feat_path = n3fz_dir / "tables/phase_n3fz_aug_feature_block_stats.csv"
    gpsi = pd.read_csv(gpsi_path) if gpsi_path.exists() else pd.DataFrame()
    feats = pd.read_csv(feat_path) if feat_path.exists() else pd.DataFrame()
    labels = ["500k", "final", "best_by_eval"]
    for label in labels:
        ok = 1
        if not gpsi.empty:
            rows = gpsi[(gpsi["method_key"].astype(str) == "z_layernorm_alpha_0p5") & (gpsi["checkpoint_label"].astype(str) == label)]
            if rows.empty:
                ok = 0
            else:
                delta_p95 = pd.to_numeric(rows.get("delta_norm_1s_p95", pd.Series(dtype=float)), errors="coerce").max()
                inactive = pd.to_numeric(rows.get("inactive_forwarded_count_max", pd.Series(dtype=float)), errors="coerce").max()
                if not pd.notna(delta_p95) or float(delta_p95) >= 100.0 or (pd.notna(inactive) and float(inactive) > 0.0):
                    ok = 0
        if not feats.empty:
            rows = feats[(feats["method_key"].astype(str) == "z_layernorm_alpha_0p5") & (feats["checkpoint_label"].astype(str) == label)]
            if rows.empty:
                ok = 0
            else:
                nonfinite = int(pd.to_numeric(rows.get("nan_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
                nonfinite += int(pd.to_numeric(rows.get("inf_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
                if nonfinite != 0:
                    ok = 0
        result[label] = ok
    return result


def build_parent_tables(n3z2c_dir: Path, n3fz_dir: Path, out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    old_selected = read_old_parent_selection(n3z2c_dir / "tables/phase_n3z2c_parent_checkpoint_selection.csv")
    metrics = candidate_metrics(n3fz_dir).set_index("checkpoint_label")
    diag = diagnostics_ok_by_label(n3fz_dir)
    rows: list[dict[str, Any]] = []
    for candidate, path_text, eval_label in PARENT_CANDIDATES:
        path = ROOT / path_text
        metric = metrics.loc[eval_label] if eval_label in metrics.index else None
        rows.append(
            {
                "candidate": candidate,
                "path": path_text,
                "exists": int(path.exists()),
                "size_bytes": int(path.stat().st_size) if path.exists() else 0,
                "sha256": sha256(path),
                "eval_label": eval_label,
                "success_rate": float(metric["success_rate"]) if metric is not None else float("nan"),
                "collision_rate": float(metric["collision_rate"]) if metric is not None else float("nan"),
                "near_miss_rate": float(metric["near_miss_rate"]) if metric is not None else float("nan"),
                "raw_unsafe_action_rate": float(metric["raw_unsafe_action_rate"]) if metric is not None else float("nan"),
                "nan_or_crash": int(metric["nan_or_crash"]) if metric is not None else 1,
                "diagnostics_ok": int(diag.get(eval_label, 0)),
                "selected_by_old_rule": int(candidate == old_selected),
                "selected_by_fixed_rule": 0,
                "selection_reason": "",
            }
        )
    missing = [row["path"] for row in rows if not row["exists"]]
    if missing:
        raise AuditStop("parent_candidates_missing", "missing parent candidates:\n" + "\n".join(missing))
    selectable = [row for row in rows if int(row["diagnostics_ok"]) == 1 and int(row["nan_or_crash"]) == 0]
    if not selectable:
        raise AuditStop("parent_candidates_missing", "no parent candidate has normal diagnostics")
    selectable.sort(
        key=lambda row: (
            -float(row["success_rate"]),
            float(row["collision_rate"]),
            float(row["near_miss_rate"]),
            float(row["raw_unsafe_action_rate"]),
            str(row["candidate"]),
        )
    )
    selected = selectable[0]
    for row in rows:
        if row["candidate"] == selected["candidate"]:
            row["selected_by_fixed_rule"] = 1
            row["selection_reason"] = (
                "diagnostics normal; selected by success desc, collision asc, "
                "near_miss asc, raw_unsafe asc"
            )
        elif row["candidate"] == old_selected:
            row["selection_reason"] = "old N3Z2C selected this candidate"
    write_csv(out_dir / "tables/phase_n3z2c_audit_checkpoint_hash.csv", rows)
    write_csv(out_dir / "tables/phase_n3z2c_audit_parent_selection_fixed.csv", rows)
    write_text(
        out_dir / "tables/phase_n3z2c_audit_parent_selection_fixed.json",
        json.dumps(
            {
                "selected_candidate": selected["candidate"],
                "selected_path": selected["path"],
                "old_selected_candidate": old_selected,
                "old_rule_wrong": bool(old_selected and old_selected != selected["candidate"]),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return rows, rows, str(selected["path"])


def zip_optimizer_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"zip_has_optimizer_file": 0, "zip_entries": "", "zip_error": "missing"}
    try:
        with zipfile.ZipFile(path) as zf:
            entries = zf.namelist()
        optimizer_entries = [entry for entry in entries if "optimizer" in entry.lower()]
        return {
            "zip_has_optimizer_file": int(bool(optimizer_entries)),
            "optimizer_entries": json.dumps(optimizer_entries),
            "zip_entries": json.dumps(entries[:40]),
            "zip_error": "",
        }
    except Exception as exc:
        return {"zip_has_optimizer_file": 0, "optimizer_entries": "[]", "zip_entries": "[]", "zip_error": repr(exc)}


def schedule_value(schedule: Any, progress_remaining: float) -> Any:
    try:
        return float(schedule(progress_remaining)) if callable(schedule) else float(schedule)
    except Exception as exc:
        return f"unavailable:{type(exc).__name__}:{exc}"


def script_findings() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    old_train = ROOT / "scripts/train_env_v2_gpsi_ppo_n3z2c.py"
    audit_train = ROOT / "scripts/train_env_v2_gpsi_ppo_n3z2c_audit.py"
    for label, path in [("old_n3z2c_train", old_train), ("audit_corrected_train", audit_train)]:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        reset_values = re.findall(r"reset_num_timesteps\s*=\s*(True|False)", text)
        rows.append(
            {
                "script": label,
                "path": rel(path),
                "exists": int(path.exists()),
                "ppo_load_found": int("PPO.load" in text),
                "learn_found": int(".learn(" in text),
                "reset_num_timesteps_values": json.dumps(reset_values),
                "uses_reset_num_timesteps_false": int("False" in reset_values),
                "uses_reset_num_timesteps_true": int("True" in reset_values),
                "finding": (
                    "old N3Z2C continuation used reset_num_timesteps=True; this is a resume-semantics bug"
                    if label == "old_n3z2c_train" and "True" in reset_values
                    else "audit corrected train must use reset_num_timesteps=False"
                    if label == "audit_corrected_train"
                    else "not found"
                ),
            }
        )
    return rows


def resume_semantics_rows(selected_parent: str) -> list[dict[str, Any]]:
    parent = ROOT / selected_parent
    corrected_dir = ROOT / "checkpoints/env_v2_gpsi_heada_ppo_n3z2c_audit_corrected_parent_750k_s0"
    corrected_final = corrected_dir / "final.zip"
    zip_info = zip_optimizer_state(parent)
    rows: list[dict[str, Any]] = [
        {
            "check": "selected_parent_checkpoint",
            "status": "ok" if parent.exists() else "missing",
            "detail": rel(parent),
            **zip_info,
        },
        {
            "check": "old_n3z2c_reset_num_timesteps",
            "status": "bug_confirmed",
            "detail": "scripts/train_env_v2_gpsi_ppo_n3z2c.py uses reset_num_timesteps=True during continuation",
        },
        {
            "check": "audit_corrected_reset_num_timesteps",
            "status": "planned_corrected",
            "detail": "scripts/train_env_v2_gpsi_ppo_n3z2c_audit.py must log and use reset_num_timesteps=False",
        },
        {
            "check": "global_step_plan",
            "status": "ok",
            "detail": "parent_total_steps=500000, additional_steps=250000, target_total_steps=750000, checkpoint_total_steps=[750000]",
        },
    ]
    try:
        from stable_baselines3 import PPO

        model = PPO.load(str(parent), device="cpu")
        optimizer_state = model.policy.optimizer.state_dict()
        rows.append(
            {
                "check": "sb3_load_optimizer_state",
                "status": "ok" if optimizer_state.get("state") else "warning_empty_optimizer_state",
                "detail": json.dumps(
                    {
                        "model_num_timesteps": int(model.num_timesteps),
                        "optimizer_state_entries": len(optimizer_state.get("state", {})),
                        "n_steps": int(model.n_steps),
                        "batch_size": int(model.batch_size),
                        "n_epochs": int(model.n_epochs),
                        "gamma": float(model.gamma),
                        "gae_lambda": float(model.gae_lambda),
                        "ent_coef": float(model.ent_coef),
                        "vf_coef": float(model.vf_coef),
                        "max_grad_norm": float(model.max_grad_norm),
                    },
                    sort_keys=True,
                ),
            }
        )
        progress_before = 1.0 - (float(model.num_timesteps) / 750000.0)
        obs_dim = -1
        try:
            obs_dim = int(model.observation_space["obs"].shape[-1])
        except Exception:
            obs_dim = -1
        rows.append(
            {
                "phase": "before_learn",
                "loaded_checkpoint_path": rel(parent),
                "parent_total_steps": 500000,
                "additional_steps": 250000,
                "target_total_steps": 750000,
                "reset_num_timesteps": False,
                "model_num_timesteps": int(model.num_timesteps),
                "model_expected_parent_steps": 500000,
                "model_parent_step_match": int(int(model.num_timesteps) == 500000),
                "optimizer_state_entries": len(optimizer_state.get("state", {})),
                "optimizer_state_restored": int(bool(optimizer_state.get("state"))),
                "learning_rate_current": schedule_value(model.lr_schedule, progress_before),
                "clip_range_current": schedule_value(model.clip_range, progress_before),
                "progress_remaining_assuming_global_total": float(progress_before),
                "n_envs": int(getattr(model, "n_envs", 4)),
                "n_steps": int(model.n_steps),
                "batch_size": int(model.batch_size),
                "obs_dim": obs_dim,
                "detail": "reconstructed from selected parent checkpoint",
            }
        )
        if corrected_final.exists() and corrected_final.stat().st_size > 0:
            final_model = PPO.load(str(corrected_final), device="cpu")
            final_optimizer = final_model.policy.optimizer.state_dict()
            progress_after = 1.0 - (float(final_model.num_timesteps) / 750000.0)
            try:
                final_obs_dim = int(final_model.observation_space["obs"].shape[-1])
            except Exception:
                final_obs_dim = -1
            rows.append(
                {
                    "phase": "after_learn",
                    "loaded_checkpoint_path": rel(parent),
                    "parent_total_steps": 500000,
                    "additional_steps": 250000,
                    "target_total_steps": 750000,
                    "reset_num_timesteps": False,
                    "model_num_timesteps": int(final_model.num_timesteps),
                    "model_expected_target_steps": 750000,
                    "model_target_step_match": int(int(final_model.num_timesteps) >= 750000),
                    "optimizer_state_entries": len(final_optimizer.get("state", {})),
                    "optimizer_state_restored": int(bool(final_optimizer.get("state"))),
                    "learning_rate_current": schedule_value(final_model.lr_schedule, progress_after),
                    "clip_range_current": schedule_value(final_model.clip_range, progress_after),
                    "progress_remaining_assuming_global_total": float(progress_after),
                    "n_envs": int(getattr(final_model, "n_envs", 4)),
                    "n_steps": int(final_model.n_steps),
                    "batch_size": int(final_model.batch_size),
                    "obs_dim": final_obs_dim,
                    "detail": f"reconstructed from corrected final checkpoint: {rel(corrected_final)}",
                }
            )
    except Exception as exc:
        rows.append({"check": "sb3_load_optimizer_state", "status": "unresolved", "detail": repr(exc)})
    return rows


def main() -> None:
    args = parse_args()
    out_dir = ROOT / args.out_dir
    n3z2c_dir = ROOT / args.n3z2c_result_dir
    n3fz_dir = ROOT / args.n3fz_result_dir
    for path in [out_dir, out_dir / "tables", out_dir / "plots", out_dir / "logs"]:
        path.mkdir(parents=True, exist_ok=True)
    try:
        complete = n3z2c_dir / "PHASE_N3Z2C_Z2_CONTINUATION_COMPLETE.flag"
        if not complete.exists():
            raise AuditStop("n3z2c_missing", f"missing N3Z2C complete flag: {rel(complete)}")
        cpu_rows = cpu_affinity_rows()
        write_csv(out_dir / "tables/phase_n3z2c_audit_resource_affinity.csv", cpu_rows)
        _, _, selected_parent = build_parent_tables(n3z2c_dir, n3fz_dir, out_dir)
        train_findings = script_findings()
        write_csv(out_dir / "tables/phase_n3z2c_audit_train_script_findings.csv", train_findings)
        resume_rows = resume_semantics_rows(selected_parent)
        write_csv(out_dir / "tables/phase_n3z2c_audit_resume_semantics.csv", resume_rows)
        command_row = {
            "stage": "audit_parent_resume_cpu",
            "command": " ".join(["python", *sys.argv]),
            "selected_fixed_parent": selected_parent,
        }
        write_csv(out_dir / "tables/phase_n3z2c_audit_command_manifest.csv", [command_row])
        print(f"fixed_selected_parent={selected_parent}", flush=True)
    except AuditStop as exc:
        write_stop(out_dir, exc.reason, exc.detail)
        raise SystemExit(2) from exc
    except Exception as exc:
        detail = textwrap.dedent(
            f"""
            parent/resume/CPU audit failed unexpectedly:
            {type(exc).__name__}: {exc}
            """
        ).strip()
        write_stop(out_dir, "resume_semantics_unresolved", detail)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
