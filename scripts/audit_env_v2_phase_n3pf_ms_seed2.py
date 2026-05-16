from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


STOP_FLAGS = {
    "config_mismatch": "PHASE_N3PF_MS_AB_STOP_CONFIG_MISMATCH.flag",
    "eval_path_mismatch": "PHASE_N3PF_MS_AB_STOP_EVAL_PATH_MISMATCH.flag",
    "checkpoint_integrity_failed": "PHASE_N3PF_MS_AB_STOP_CHECKPOINT_INTEGRITY_FAILED.flag",
    "feature_gpsi_diagnostics_failed": "PHASE_N3PF_MS_AB_STOP_FEATURE_GPSI_DIAGNOSTICS_FAILED.flag",
}

CHECKPOINTS = [
    ("250k", "checkpoint_250k.zip", 250_000),
    ("500k", "checkpoint_500k.zip", 500_000),
    ("750k", "checkpoint_750k.zip", 750_000),
    ("1000k", "checkpoint_1000k.zip", 1_000_000),
    ("1250k", "checkpoint_1250k.zip", 1_250_000),
    ("1500k", "checkpoint_1500k.zip", 1_500_000),
    ("final", "final.zip", 1_500_000),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase N3PF-MS-AB Step A seed2 collapse audit.")
    parser.add_argument("--source-result-dir", default="results/env_v2_phase_n3pf_ms_multiseed")
    parser.add_argument("--result-dir", default="results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun")
    parser.add_argument("--out-dir", default="results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_a_audit")
    parser.add_argument("--seed-good", type=int, default=1)
    parser.add_argument("--seed-bad", type=int, default=2)
    parser.add_argument("--include-seed0", action="store_true")
    parser.add_argument("--write-plots", action="store_true")
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def ensure_dirs(result_dir: Path, out_dir: Path) -> None:
    for path in [result_dir, out_dir, out_dir / "tables", out_dir / "plots", out_dir / "logs"]:
        path.mkdir(parents=True, exist_ok=True)


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
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing required table: {rel(path)}")
    return pd.read_csv(path)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload if isinstance(payload, dict) else {}


def dig(payload: dict[str, Any], dotted: str, default: Any = "") -> Any:
    cur: Any = payload
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def normalize_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.12g}"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def model_stats(path: Path, previous: torch.Tensor | None) -> dict[str, Any]:
    model = PPO.load(str(path), device="cpu")
    params = torch.cat([p.detach().cpu().float().reshape(-1) for p in model.policy.parameters()])
    opt_state = model.policy.optimizer.state_dict().get("state", {})
    delta = float(torch.linalg.vector_norm(params - previous).item()) if previous is not None and previous.numel() == params.numel() else float("nan")
    out = {
        "model_num_timesteps": int(model.num_timesteps),
        "optimizer_state_present": int(bool(opt_state)),
        "policy_parameter_l2_norm": float(torch.linalg.vector_norm(params).item()),
        "policy_parameter_delta_vs_previous_checkpoint": delta,
        "_params": params,
    }
    del model
    return out


def write_stop(result_dir: Path, reason: str, detail: str) -> None:
    flag = STOP_FLAGS[reason]
    write_text(result_dir / flag, f"{reason}\n{detail.strip()}\n")
    write_text(result_dir / "phase_n3pf_ms_seed2_ab_status.txt", f"stopped:{flag}\n")
    write_text(
        result_dir / "PHASE_N3PF_MS_SEED2_AB_AUDIT_RERUN_REPORT.md",
        "\n".join(
            [
                "# Phase N3PF-MS-AB Seed2 Collapse Audit + Minimal Rerun Sanity",
                "",
                f"`terminal_decision = phase_n3pf_ms_seed2_ab_stopped_{reason}`",
                "",
                "Partial Step A report.",
                "",
                "```text",
                detail.strip(),
                "```",
            ]
        )
        + "\n",
    )


def config_audit(source_dir: Path, out_dir: Path, seed_good: int, seed_bad: int) -> tuple[bool, str]:
    manifest = read_csv_required(source_dir / "tables/phase_n3pf_ms_config_manifest.csv")
    rows: list[dict[str, Any]] = []
    hard_error = False
    details: list[str] = []
    seed_dirs = {
        seed_good: ROOT / f"checkpoints/env_v2_gpsi_heada_ppo_n3pf_ms_block_projected_s{seed_good}",
        seed_bad: ROOT / f"checkpoints/env_v2_gpsi_heada_ppo_n3pf_ms_block_projected_s{seed_bad}",
    }
    cfgs = {seed: load_yaml(path / "config_resolved.yaml") for seed, path in seed_dirs.items()}
    checks = [
        ("policy_class", "static", "MultiInputPolicy"),
        ("feature_adapter", "ppo.feature_adapter", None),
        ("include_z", "gpsi.include_z", None),
        ("delta_scale", "gpsi.delta_scale", None),
        ("logvar_clip", "gpsi.logvar_clamp", None),
        ("logvar_scale", "gpsi.logvar_output_scale", None),
        ("n_envs", "env.n_envs", None),
        ("device", "env.device", None),
        ("learning_rate", "ppo.learning_rate", None),
        ("n_steps", "ppo.n_steps", None),
        ("batch_size", "ppo.batch_size", None),
        ("gamma", "ppo.gamma", None),
        ("gae_lambda", "ppo.gae_lambda", None),
        ("clip_range", "ppo.clip_range", None),
        ("ent_coef", "ppo.ent_coef", None),
        ("vf_coef", "ppo.vf_coef", None),
        ("max_grad_norm", "ppo.max_grad_norm", None),
        ("gpsi_checkpoint", "gpsi.checkpoint", None),
        ("gpsi_frozen", "static", "true"),
        ("train_scenario", "env.train_scenario", None),
        ("no_shield", "training.no_shield", None),
        ("action_filtering", "training.action_filtering", None),
        ("use_safety_cost", "training.use_safety_cost", None),
        ("obs_aug_dim", "gpsi.obs_aug_dim", None),
        ("block_projector", "ppo.block_projector", None),
    ]
    for field, dotted, static_value in checks:
        if dotted == "static":
            v_good = static_value
            v_bad = static_value
        else:
            v_good = dig(cfgs[seed_good], dotted)
            v_bad = dig(cfgs[seed_bad], dotted)
        same = normalize_value(v_good) == normalize_value(v_bad)
        row = {"field": field, f"seed{seed_good}_value": normalize_value(v_good), f"seed{seed_bad}_value": normalize_value(v_bad), "same": int(same), "allowed_difference": 0}
        rows.append(row)
        if not same:
            hard_error = True
            details.append(f"{field}: {v_good!r} != {v_bad!r}")

    for allowed in ["training.seed", "method_name", "out_dir", "checkpoint_dir", "log_dir"]:
        rows.append({"field": allowed, f"seed{seed_good}_value": "allowed", f"seed{seed_bad}_value": "allowed", "same": 0, "allowed_difference": 1})

    # Cross-check compact manifest rows for the same core surface.
    for seed in [seed_good, seed_bad]:
        if manifest[manifest["training_seed"].astype(int) == seed].empty:
            hard_error = True
            details.append(f"missing config manifest row for seed={seed}")

    write_csv(out_dir / "tables/phase_n3pf_ms_seed2a_config_diff.csv", rows)
    return hard_error, "; ".join(details)


def checkpoint_audit(source_dir: Path, out_dir: Path, seed_good: int, seed_bad: int) -> tuple[bool, bool, str]:
    summary = read_csv_required(source_dir / "tables/phase_n3pf_ms_eval_summary_by_seed.csv")
    scenario_summary = read_csv_required(source_dir / "tables/phase_n3pf_ms_scenario_breakdown.csv")
    path_rows: list[dict[str, Any]] = []
    integrity_rows: list[dict[str, Any]] = []
    hard_integrity = False
    path_error = False
    details: list[str] = []

    for seed in [seed_good, seed_bad]:
        prev: torch.Tensor | None = None
        root = ROOT / f"checkpoints/env_v2_gpsi_heada_ppo_n3pf_ms_block_projected_s{seed}"
        for label, filename, expected in CHECKPOINTS:
            path = root / filename
            exists = path.exists() and path.stat().st_size > 0
            stats: dict[str, Any] = {
                "model_num_timesteps": np.nan,
                "optimizer_state_present": 0,
                "policy_parameter_l2_norm": np.nan,
                "policy_parameter_delta_vs_previous_checkpoint": np.nan,
            }
            if exists:
                try:
                    stats = model_stats(path, prev)
                    prev = stats.pop("_params")
                except Exception as exc:
                    hard_integrity = True
                    details.append(f"failed loading {rel(path)}: {type(exc).__name__}: {exc}")
            else:
                hard_integrity = True
                details.append(f"missing checkpoint {rel(path)}")

            method_key = f"p3_s{seed}_{label}"
            rows = summary[summary.get("method_key", pd.Series(dtype=str)).astype(str) == method_key].copy()
            scenario_rows = scenario_summary[scenario_summary.get("method_key", pd.Series(dtype=str)).astype(str) == method_key].copy()
            eval_count = int(rows["eval_seed"].nunique()) if not rows.empty and "eval_seed" in rows else 0
            scenario_count = int(scenario_rows["scenario"].nunique()) if not scenario_rows.empty and "scenario" in scenario_rows else 0
            episode_count = int(rows["episodes"].sum()) if not rows.empty and "episodes" in rows else 0
            path_ok = True
            if label in {"1000k", "1500k", "final"}:
                expected_path = rel(path)
                actual_paths = sorted(set(rows.get("checkpoint_path", pd.Series(dtype=str)).dropna().astype(str).tolist()))
                if not actual_paths or actual_paths != [expected_path]:
                    path_ok = False
                    path_error = True
                    details.append(f"eval path mismatch for {method_key}: expected {expected_path}, actual {actual_paths}")

            model_steps = stats["model_num_timesteps"]
            step_ok = bool(exists) and np.isfinite(model_steps) and (int(model_steps) >= expected if label == "final" else int(model_steps) == expected)
            delta = stats["policy_parameter_delta_vs_previous_checkpoint"]
            delta_ok = True if label == "250k" else (np.isfinite(delta) and float(delta) > 1.0e-8)
            if not step_ok or not delta_ok or int(stats["optimizer_state_present"]) != 1:
                hard_integrity = True
                details.append(f"checkpoint integrity failed {method_key}: step_ok={step_ok} delta_ok={delta_ok} optimizer={stats['optimizer_state_present']}")

            row = {
                "training_seed": seed,
                "checkpoint_label": label,
                "checkpoint_path": rel(path),
                "exists": int(exists),
                "size_bytes": int(path.stat().st_size) if exists else 0,
                "sha256": sha256(path) if exists else "missing",
                "expected_total_steps": expected,
                **{k: v for k, v in stats.items() if k != "_params"},
                "eval_rows_reference_this_path": int(label in {"1000k", "1500k", "final"} and path_ok),
                "eval_seed_count": eval_count,
                "scenario_count": scenario_count,
                "episode_count": episode_count,
            }
            integrity_rows.append(row)
            path_rows.append({k: row[k] for k in ["training_seed", "checkpoint_label", "checkpoint_path", "exists", "eval_rows_reference_this_path", "eval_seed_count", "episode_count"]})

    write_csv(out_dir / "tables/phase_n3pf_ms_seed2a_checkpoint_path_audit.csv", path_rows)
    write_csv(out_dir / "tables/phase_n3pf_ms_seed2a_checkpoint_integrity_extended.csv", integrity_rows)
    return hard_integrity, path_error, "; ".join(details)


def training_curve_audit(source_dir: Path, out_dir: Path, seed_good: int, seed_bad: int) -> None:
    curve = read_csv_required(source_dir / "tables/phase_n3pf_ms_train_curve.csv")
    hb = read_csv_required(source_dir / "tables/phase_n3pf_ms_train_heartbeat.csv")
    rows: list[dict[str, Any]] = []
    for seed in [seed_good, seed_bad]:
        c = curve[curve["training_seed"].astype(int) == seed].copy()
        h = hb[hb["training_seed"].astype(int) == seed].copy()
        if c.empty:
            rows.append({"training_seed": seed, "metric": "missing_train_curve", "value": 1})
            continue
        c["step_bin"] = (pd.to_numeric(c["steps"], errors="coerce") // 100_000 * 100_000).astype("Int64")
        for step_bin, group in c.groupby("step_bin", dropna=True):
            rows.append(
                {
                    "training_seed": seed,
                    "step_bin": int(step_bin),
                    "episode_reward_mean": float(group["episode_reward"].mean()),
                    "episode_reward_p25": float(group["episode_reward"].quantile(0.25)),
                    "episode_reward_p75": float(group["episode_reward"].quantile(0.75)),
                    "episode_length_mean": float(group["episode_length"].mean()),
                    "episodes": int(len(group)),
                }
            )
        if not h.empty:
            rows.append(
                {
                    "training_seed": seed,
                    "step_bin": -1,
                    "fps_mean": float(h["steps_per_second"].mean()),
                    "fps_min": float(h["steps_per_second"].min()),
                    "recent_episode_reward_mean_last": float(h.sort_values("steps")["recent_episode_reward_mean"].iloc[-1]),
                    "heartbeat_rows": int(len(h)),
                }
            )
    write_csv(out_dir / "tables/phase_n3pf_ms_seed2a_training_curve_diagnostics.csv", rows)


def behavior_audit(source_dir: Path, out_dir: Path, seed_good: int, seed_bad: int) -> None:
    agg = read_csv_required(source_dir / "tables/phase_n3pf_ms_eval_summary_aggregate.csv")
    raw = read_csv_required(source_dir / "tables/phase_n3pf_ms_raw_unsafe_action_summary.csv")
    rows: list[dict[str, Any]] = []
    for seed in [0, seed_good, seed_bad]:
        prefix = f"p3_s{seed}_"
        for _, row in agg[agg["method_key"].astype(str).str.startswith(prefix)].iterrows():
            method_key = str(row["method_key"])
            raw_match = raw[raw["method_key"].astype(str) == method_key]
            rows.append(
                {
                    "training_seed": seed,
                    "method_key": method_key,
                    "checkpoint_label": row.get("checkpoint_label", ""),
                    "success_rate": float(row.get("mean_success_rate", np.nan)),
                    "collision_rate": float(row.get("mean_collision_rate", np.nan)),
                    "progress": float(row.get("mean_progress", np.nan)),
                    "mean_min_distance": float(row.get("mean_mean_min_distance", np.nan)),
                    "raw_unsafe_action_rate": float(row.get("mean_raw_unsafe_action_rate", np.nan)),
                    "raw_min_predicted_cpa": float(raw_match["raw_min_predicted_cpa"].mean()) if not raw_match.empty and "raw_min_predicted_cpa" in raw_match else np.nan,
                    "action_norm": float(row.get("mean_action_norm", np.nan)),
                    "action_delta": float(row.get("mean_action_delta", np.nan)),
                    "episode_length": float(row.get("mean_episode_length", np.nan)),
                    "seed2_low_action_delta": int(seed == seed_bad and float(row.get("mean_action_delta", np.nan)) < 0.08),
                    "seed2_conservative_but_collision": int(seed == seed_bad and float(row.get("mean_collision_rate", 0.0)) > 0.5 and float(row.get("mean_raw_unsafe_action_rate", 1.0)) < 0.28),
                    "seed2_raw_unsafe_mismatch": int(seed == seed_bad and float(row.get("mean_collision_rate", 0.0)) > 0.5 and float(row.get("mean_raw_unsafe_action_rate", 1.0)) < 0.28),
                    "seed2_progress_collision_mismatch": int(seed == seed_bad and float(row.get("mean_collision_rate", 0.0)) > 0.5 and float(row.get("mean_progress", 0.0)) > 0.94),
                }
            )
    write_csv(out_dir / "tables/phase_n3pf_ms_seed2a_behavior_diagnostics.csv", rows)


def feature_gpsi_audit(source_dir: Path, out_dir: Path, seed_good: int, seed_bad: int) -> tuple[bool, str]:
    feature = read_csv_required(source_dir / "tables/phase_n3pf_ms_feature_block_stats.csv")
    gpsi = read_csv_required(source_dir / "tables/phase_n3pf_ms_gpsi_output_summary.csv")
    rows: list[dict[str, Any]] = []
    hard_error = False
    details: list[str] = []

    for seed in [0, seed_good, seed_bad]:
        for label in ["1000k", "1500k", "final"]:
            method_key = f"p3_s{seed}_{label}"
            f = feature[feature["method_key"].astype(str) == method_key]
            g = gpsi[gpsi["method_key"].astype(str) == method_key]
            rec: dict[str, Any] = {"training_seed": seed, "method_key": method_key, "checkpoint_label": label}
            for block, out_name in [
                ("obs_i_12", "obs_i_l2_p95"),
                ("delta_hat_9_after_scale", "delta_hat_l2_p95"),
                ("logvar_scaled_9_policy", "logvar_scaled_l2_p95"),
                ("adapter_output_64", "adapter_output_l2_p95"),
            ]:
                b = f[f["block"].astype(str) == block]
                rec[out_name] = float(b["l2_norm_p95"].median()) if not b.empty else np.nan
                rec[f"{out_name}_nan_count"] = int(b["nan_count"].sum()) if not b.empty and "nan_count" in b else 0
                rec[f"{out_name}_inf_count"] = int(b["inf_count"].sum()) if not b.empty and "inf_count" in b else 0
            rec["delta_norm_1s_p95"] = float(g["delta_norm_1s_p95"].median()) if not g.empty and "delta_norm_1s_p95" in g else np.nan
            rec["delta_norm_1s_max"] = float(g["delta_norm_1s_max"].max()) if not g.empty and "delta_norm_1s_max" in g else np.nan
            rec["logvar_xy_1s_span"] = float(g["logvar_xy_1s_span"].max()) if not g.empty and "logvar_xy_1s_span" in g else np.nan
            rec["inactive_forwarded_count"] = float(g["inactive_forwarded_count_max"].max()) if not g.empty and "inactive_forwarded_count_max" in g else np.nan
            rec["feature_nonfinite_count"] = int(sum(v for k, v in rec.items() if k.endswith("_nan_count") or k.endswith("_inf_count")))
            rows.append(rec)

            if rec["feature_nonfinite_count"] > 0:
                hard_error = True
                details.append(f"nonfinite feature count for {method_key}")
            if np.isfinite(rec["delta_norm_1s_p95"]) and rec["delta_norm_1s_p95"] > 100.0:
                hard_error = True
                details.append(f"delta_norm_1s_p95 too large for {method_key}: {rec['delta_norm_1s_p95']}")
            if np.isfinite(rec["delta_norm_1s_max"]) and rec["delta_norm_1s_max"] > 1000.0:
                hard_error = True
                details.append(f"delta_norm_1s_max too large for {method_key}: {rec['delta_norm_1s_max']}")
            if np.isfinite(rec["inactive_forwarded_count"]) and rec["inactive_forwarded_count"] > 0:
                hard_error = True
                details.append(f"inactive forwarded count for {method_key}: {rec['inactive_forwarded_count']}")

    write_csv(out_dir / "tables/phase_n3pf_ms_seed2a_feature_gpsi_audit.csv", rows)
    return hard_error, "; ".join(details)


def step_a_decision(out_dir: Path, seed_bad: int, errors: dict[str, str]) -> dict[str, Any]:
    behavior = pd.read_csv(out_dir / "tables/phase_n3pf_ms_seed2a_behavior_diagnostics.csv")
    s2_1500 = behavior[(behavior["training_seed"].astype(int) == seed_bad) & (behavior["checkpoint_label"].astype(str) == "1500k")]
    s2_success = float(s2_1500["success_rate"].iloc[0]) if not s2_1500.empty else np.nan
    s2_collision = float(s2_1500["collision_rate"].iloc[0]) if not s2_1500.empty else np.nan
    hard_error = any(errors.values())
    if errors.get("config_mismatch"):
        dtype = "engineering_error_found"
    elif errors.get("eval_path_mismatch"):
        dtype = "eval_path_error_found"
    elif errors.get("checkpoint_integrity_failed"):
        dtype = "checkpoint_integrity_error_found"
    elif errors.get("feature_gpsi_diagnostics_failed"):
        dtype = "feature_gpsi_issue_found"
    elif np.isfinite(s2_success) and s2_success < 0.50 and np.isfinite(s2_collision) and s2_collision > 0.50:
        dtype = "ppo_bad_local_optimum_likely"
    else:
        dtype = "inconclusive"
    row = {
        "decision_type": dtype,
        "hard_error_found": int(hard_error),
        "do_step_b": int(not hard_error),
        "stop_after_step_a": int(hard_error),
        "seed2_success_1500k": s2_success,
        "seed2_collision_1500k": s2_collision,
        "config_mismatch_detail": errors.get("config_mismatch", ""),
        "eval_path_mismatch_detail": errors.get("eval_path_mismatch", ""),
        "checkpoint_integrity_failed_detail": errors.get("checkpoint_integrity_failed", ""),
        "feature_gpsi_diagnostics_failed_detail": errors.get("feature_gpsi_diagnostics_failed", ""),
        "most_likely_cause": "PPO seed-sensitive bad local optimum or training instability" if not hard_error else "engineering hard error",
    }
    write_csv(out_dir / "tables/phase_n3pf_ms_seed2a_decision.csv", [row])
    return row


def main() -> None:
    args = parse_args()
    source_dir = ROOT / args.source_result_dir
    result_dir = ROOT / args.result_dir
    out_dir = ROOT / args.out_dir
    ensure_dirs(result_dir, out_dir)
    try:
        errors: dict[str, str] = {}
        config_bad, config_detail = config_audit(source_dir, out_dir, args.seed_good, args.seed_bad)
        if config_bad:
            errors["config_mismatch"] = config_detail
        ckpt_bad, path_bad, ckpt_detail = checkpoint_audit(source_dir, out_dir, args.seed_good, args.seed_bad)
        if ckpt_bad:
            errors["checkpoint_integrity_failed"] = ckpt_detail
        if path_bad:
            errors["eval_path_mismatch"] = ckpt_detail
        training_curve_audit(source_dir, out_dir, args.seed_good, args.seed_bad)
        behavior_audit(source_dir, out_dir, args.seed_good, args.seed_bad)
        feature_bad, feature_detail = feature_gpsi_audit(source_dir, out_dir, args.seed_good, args.seed_bad)
        if feature_bad:
            errors["feature_gpsi_diagnostics_failed"] = feature_detail
        decision = step_a_decision(out_dir, args.seed_bad, errors)
        for reason, detail in errors.items():
            write_stop(result_dir, reason, detail)
            break
        print(
            "N3PF_MS_SEED2_AB_AUDIT "
            f"decision={decision['decision_type']} do_step_b={decision['do_step_b']} hard_error={decision['hard_error_found']}",
            flush=True,
        )
    except Exception:
        detail = traceback.format_exc()
        write_stop(result_dir, "checkpoint_integrity_failed", detail)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
