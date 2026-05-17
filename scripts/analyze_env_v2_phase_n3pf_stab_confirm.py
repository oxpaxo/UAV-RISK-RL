from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RESULT_DIR = ROOT / "results/env_v2_phase_n3pf_stab_confirm"
TABLE_PREFIX = "phase_n3pf_stab_confirm"
COMPLETE_FLAG = "PHASE_N3PF_STAB_CONFIRM_COMPLETE.flag"
REPORT_FILE = "PHASE_N3PF_STAB_CONFIRM_REPORT.md"
STATUS_FILE = "phase_n3pf_stab_confirm_status.txt"
STOP_FLAGS = {
    "preflight_failed": "STOP_PREFLIGHT_FAILED.flag",
    "selector_contaminated": "STOP_SELECTOR_CONTAMINATED.flag",
    "eval_failed": "STOP_EVAL_FAILED.flag",
}
MANDATORY_SEEDS = {0, 1, 2}
CHECKPOINT_ORDER = {"500k": 500_000, "750k": 750_000, "1000k": 1_000_000, "1250k": 1_250_000, "1500k": 1_500_000, "final": 9_999_999}


class ConfirmAnalysisStop(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Phase N3PF-STAB-CONFIRM outputs.")
    parser.add_argument("--result-dir", default="results/env_v2_phase_n3pf_stab_confirm")
    parser.add_argument("--table-prefix", default=TABLE_PREFIX)
    parser.add_argument("--github-sync-commit", default="")
    parser.add_argument("--github-sync-status", default="unknown")
    parser.add_argument("--attention-success", type=float, default=0.6033)
    parser.add_argument("--attention-collision", type=float, default=0.3967)
    parser.add_argument("--noz-success", type=float, default=0.5667)
    parser.add_argument("--noz-collision", type=float, default=0.4333)
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
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        if required:
            raise ConfirmAnalysisStop("eval_failed", f"missing or empty CSV: {rel(path)}")
        return pd.DataFrame()
    return pd.read_csv(path)


def write_stop(result_dir: Path, reason: str, detail: str) -> None:
    flag = STOP_FLAGS.get(reason, STOP_FLAGS["eval_failed"])
    write_text(result_dir / flag, f"{reason}\n{detail.strip()}\n")
    write_text(result_dir / STATUS_FILE, f"stopped:{flag}\n")
    write_text(
        result_dir / REPORT_FILE,
        f"# Phase N3PF-STAB-CONFIRM Report\n\n`terminal_decision = phase_n3pf_stab_confirm_stopped_{reason}`\n\n```text\n{detail.strip()}\n```\n",
    )


def fmt(value: Any) -> str:
    try:
        f = float(value)
    except Exception:
        return str(value)
    if math.isnan(f):
        return "nan"
    return f"{f:.4f}"


def table_md(df: pd.DataFrame, cols: list[str], max_rows: int = 30) -> list[str]:
    if df.empty:
        return ["No rows."]
    cols = [col for col in cols if col in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].head(max_rows).iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) if isinstance(row[col], (float, int, np.floating, np.integer)) else str(row[col]) for col in cols) + " |")
    return lines


def repo_preflight(result_dir: Path) -> pd.DataFrame:
    checks = [
        ("guide", ROOT / "codex_guide/PHASE_N3PF_STAB_CONFIRM_GUIDE.md", "file"),
        ("s2d_config", ROOT / "configs/env_v2_gpsi_heada_ppo_n3pf_stab_confirm_s2d_gated.yaml", "file"),
        ("gpsi_checkpoint", ROOT / "work_dirs/gpsi_heada_v1_nll/best.pth", "file"),
        ("gated_extractor_class", ROOT / "models/gpsi_ppo_policy.py", "GpsiGatedResidualExtractor"),
        ("base_block_projected_class", ROOT / "models/gpsi_ppo_policy.py", "GpsiBlockProjectedNoZExtractor"),
        ("previous_stab_complete", ROOT / "results/env_v2_phase_n3pf_stab/PHASE_N3PF_STAB_COMPLETE.flag", "file"),
    ]
    rows: list[dict[str, Any]] = []
    for item, path, kind in checks:
        if kind == "file":
            ok = path.is_file()
            detail = rel(path)
        else:
            ok = path.is_file() and kind in path.read_text(encoding="utf-8")
            detail = f"{kind} in {rel(path)}"
        rows.append({"item": item, "kind": kind, "ok": int(ok), "detail": detail})
    out = pd.DataFrame(rows)
    out.to_csv(result_dir / f"tables/{TABLE_PREFIX}_preflight_check.csv", index=False)
    if out.empty or int(out["ok"].min()) != 1:
        raise ConfirmAnalysisStop("preflight_failed", out.to_string(index=False))
    return out


def aggregate_eval(result_dir: Path, table_prefix: str, phase: str) -> pd.DataFrame:
    by_seed = read_csv(result_dir / f"tables/{table_prefix}_{phase}_eval_summary_by_seed.csv", required=False)
    if by_seed.empty:
        return pd.DataFrame()
    group_cols = ["eval_phase", "variant", "training_seed", "method_key", "method", "checkpoint", "checkpoint_path", "checkpoint_step", "checkpoint_label"]
    agg = (
        by_seed.groupby(group_cols, dropna=False)
        .agg(
            success_rate=("success_rate", "mean"),
            collision_rate=("collision_rate", "mean"),
            near_miss_rate=("near_miss_rate", "mean"),
            progress=("progress", "mean"),
            raw_unsafe_action_rate=("raw_unsafe_action_rate", "mean"),
            action_delta=("action_delta", "mean"),
            raw_min_predicted_cpa=("raw_min_predicted_cpa", "mean"),
            episodes=("episodes", "sum"),
            eval_seed_count=("eval_seed", "nunique"),
        )
        .reset_index()
    )
    agg.to_csv(result_dir / f"tables/{table_prefix}_{phase}_eval_summary_aggregate.csv", index=False)
    return agg


def enforce_selector_discipline(result_dir: Path, table_prefix: str) -> pd.DataFrame:
    val = read_csv(result_dir / f"tables/{table_prefix}_validation_eval_summary_by_seed.csv")
    if set(val["eval_phase"].astype(str)) != {"validation"}:
        raise ConfirmAnalysisStop("selector_contaminated", "validation summary contains non-validation rows")
    val_seeds = set(pd.to_numeric(val["eval_seed"], errors="coerce").dropna().astype(int))
    if val_seeds != {900, 901}:
        raise ConfirmAnalysisStop("selector_contaminated", f"validation seed mismatch: {sorted(val_seeds)}")
    selector = read_csv(result_dir / f"tables/{table_prefix}_selector_decision.csv")
    if "test_seed_used_for_selection" in selector.columns and int(pd.to_numeric(selector["test_seed_used_for_selection"], errors="coerce").fillna(1).max()) != 0:
        raise ConfirmAnalysisStop("selector_contaminated", "selector indicates test seed usage")
    rows = [
        {
            "validation_seeds": "900,901",
            "test_seeds": "1000,1001,1002",
            "final_heldout_seeds": "1100,1101,1102",
            "selector_used_only_validation": 1,
            "test_used_for_structure_or_checkpoint_selection": 0,
            "selector_rows": int(len(selector)),
        }
    ]
    out = pd.DataFrame(rows)
    out.to_csv(result_dir / f"tables/{table_prefix}_selector_discipline.csv", index=False)
    return out


def validate_test_rows(result_dir: Path, table_prefix: str) -> pd.DataFrame:
    test = read_csv(result_dir / f"tables/{table_prefix}_test_eval_summary_by_seed.csv")
    seeds = set(pd.to_numeric(test["eval_seed"], errors="coerce").dropna().astype(int))
    if seeds != {1000, 1001, 1002}:
        raise ConfirmAnalysisStop("eval_failed", f"test seed mismatch: {sorted(seeds)}")
    training_seeds = set(pd.to_numeric(test["training_seed"], errors="coerce").dropna().astype(int))
    if not MANDATORY_SEEDS.issubset(training_seeds):
        raise ConfirmAnalysisStop("eval_failed", f"missing mandatory training seeds in test eval: got={sorted(training_seeds)}")
    return test


def state_dict(path: Path) -> dict[str, torch.Tensor]:
    model = PPO.load(str(path), device="cpu")
    payload = {key: value.detach().cpu().float().clone() for key, value in model.policy.state_dict().items()}
    del model
    return payload


def l2_for(keys: list[str], lhs: dict[str, torch.Tensor], rhs: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for key in keys:
        if key not in lhs or key not in rhs or lhs[key].shape != rhs[key].shape:
            continue
        diff = lhs[key] - rhs[key]
        total += float(torch.sum(diff * diff).item())
    return float(math.sqrt(total))


def category_keys(keys: list[str]) -> dict[str, list[str]]:
    return {
        "all": keys,
        "feature_extractor": [key for key in keys if "features_extractor" in key],
        "actor_action": [key for key in keys if "action_net" in key or "policy_net" in key],
        "critic_value": [key for key in keys if "value_net" in key],
        "log_std": [key for key in keys if "log_std" in key],
        "gate": [key for key in keys if "gate" in key],
    }


def parameter_drift(result_dir: Path, table_prefix: str) -> pd.DataFrame:
    selector = read_csv(result_dir / f"tables/{table_prefix}_selector_decision.csv")
    rows: list[dict[str, Any]] = []
    cache: dict[str, dict[str, torch.Tensor]] = {}
    for _, row in selector.iterrows():
        ckpt_dir = ROOT / str(Path(str(row["selected_checkpoint_path"])).parent)
        final_path = ckpt_dir / "final.zip"
        ckpt_1000k = ckpt_dir / "checkpoint_1000k.zip"
        selected_path = ROOT / str(row["selected_checkpoint_path"])
        pairs = [
            ("selected_vs_final", selected_path, final_path),
            ("1000k_vs_final", ckpt_1000k, final_path),
        ]
        for label, left, right in pairs:
            if not left.exists() or not right.exists():
                rows.append(
                    {
                        "training_seed": int(row["training_seed"]),
                        "comparison": label,
                        "left_path": rel(left),
                        "right_path": rel(right),
                        "exists": 0,
                    }
                )
                continue
            for path in [left, right]:
                key = str(path)
                if key not in cache:
                    cache[key] = state_dict(path)
            lhs = cache[str(left)]
            rhs = cache[str(right)]
            cats = category_keys(sorted(set(lhs) & set(rhs)))
            base = {
                "training_seed": int(row["training_seed"]),
                "selected_checkpoint_label": str(row["selected_checkpoint_label"]),
                "comparison": label,
                "left_path": rel(left),
                "right_path": rel(right),
                "exists": 1,
            }
            for cat, keys in cats.items():
                base[f"{cat}_l2_delta"] = l2_for(keys, lhs, rhs)
                base[f"{cat}_param_count"] = int(sum(lhs[key].numel() for key in keys if key in lhs))
            rows.append(base)
    out = pd.DataFrame(rows)
    out.to_csv(result_dir / f"tables/{table_prefix}_parameter_drift.csv", index=False)
    return out


def train_diagnostics(result_dir: Path, table_prefix: str) -> pd.DataFrame:
    hb = read_csv(result_dir / f"tables/{table_prefix}_train_heartbeat.csv", required=False)
    if hb.empty:
        return pd.DataFrame()
    wanted = [
        "approx_kl",
        "clip_fraction",
        "entropy_loss",
        "policy_gradient_loss",
        "value_loss",
        "explained_variance",
        "learning_rate",
        "n_updates",
        "std",
        "log_std_mean",
    ]
    agg_spec = {col: (col, "mean") for col in wanted if col in hb.columns}
    agg_spec.update(
        steps_last=("steps", "max"),
        fps_mean=("steps_per_second", "mean"),
        fps_last=("steps_per_second", "last"),
        heartbeat_rows=("steps", "count"),
    )
    out = hb.groupby(["variant", "training_seed"], dropna=False).agg(**agg_spec).reset_index()
    out.to_csv(result_dir / f"tables/{table_prefix}_ppo_training_diagnostics.csv", index=False)
    return out


def attention_audit(result_dir: Path, table_prefix: str) -> pd.DataFrame:
    candidates = [
        (0, ROOT / "checkpoints/env_v2_phase2/attention_full_s0/attention_full_s0_step1500000.zip", "formal_phase2_1500k"),
        (1, ROOT / "checkpoints/attention_full_s1.zip", "legacy_top_level_unknown_protocol"),
        (2, ROOT / "checkpoints/attention_full_s2.zip", "legacy_top_level_unknown_protocol"),
    ]
    rows = []
    for seed, path, source in candidates:
        rows.append(
            {
                "training_seed": seed,
                "path": rel(path),
                "exists": int(path.exists() and path.stat().st_size > 0),
                "size_bytes": int(path.stat().st_size) if path.exists() else 0,
                "source": source,
                "formal_1500k_protocol_confirmed": int(source == "formal_phase2_1500k" and path.exists()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(result_dir / f"tables/{table_prefix}_attention_full_multiseed_audit.csv", index=False)
    return out


def final_decision(result_dir: Path, table_prefix: str, args: argparse.Namespace, test_agg: pd.DataFrame) -> pd.DataFrame:
    mandatory = test_agg[pd.to_numeric(test_agg["training_seed"], errors="coerce").astype(int).isin(MANDATORY_SEEDS)].copy()
    if len(mandatory) < 3:
        raise ConfirmAnalysisStop("eval_failed", "test aggregate missing selected checkpoints for mandatory seeds 0/1/2")
    mean_success = float(mandatory["success_rate"].mean())
    mean_collision = float(mandatory["collision_rate"].mean())
    min_success = float(mandatory["success_rate"].min())
    max_collision = float(mandatory["collision_rate"].max())
    seed_count = int(mandatory["training_seed"].nunique())
    if min_success >= 0.56 and max_collision <= 0.44 and mean_success >= float(args.attention_success) - 0.02 and mean_collision <= float(args.attention_collision) + 0.03:
        terminal = "phase_n3pf_stab_confirm_complete_s2d_confirmed"
        status = "confirmed_stable_candidate"
        n4o_next = "yes_rerun_next_phase"
    elif min_success >= 0.50 and max_collision <= 0.50 and mean_success >= float(args.noz_success) - 0.02:
        terminal = "phase_n3pf_stab_confirm_complete_s2d_promising_not_confirmed"
        status = "promising_not_confirmed"
        n4o_next = "not_until_confirmation_or_user_accepts_risk"
    else:
        terminal = "phase_n3pf_stab_confirm_complete_s2d_failed"
        status = "failed_stability_gate"
        n4o_next = "no"
    rows = [
        {
            "terminal_decision": terminal,
            "s2d_status": status,
            "mandatory_training_seed_count": seed_count,
            "test_mean_success": mean_success,
            "test_mean_collision": mean_collision,
            "test_min_success": min_success,
            "test_max_collision": max_collision,
            "attention_full_reference_success": float(args.attention_success),
            "attention_full_reference_collision": float(args.attention_collision),
            "noz_full_reference_success": float(args.noz_success),
            "noz_full_reference_collision": float(args.noz_collision),
            "comparable_to_attention_seed0_reference": int(mean_success >= float(args.attention_success) - 0.02 and mean_collision <= float(args.attention_collision) + 0.03),
            "decisive_multiseed_attention_claim_allowed": 0,
            "n4o_can_rerun_next": n4o_next,
            "n4u_blocked": "yes",
        }
    ]
    out = pd.DataFrame(rows)
    out.to_csv(result_dir / f"tables/{table_prefix}_final_decision.csv", index=False)
    return out


def plots(result_dir: Path, table_prefix: str, val_agg: pd.DataFrame, test_agg: pd.DataFrame, drift: pd.DataFrame, train_diag: pd.DataFrame) -> None:
    (result_dir / "plots").mkdir(parents=True, exist_ok=True)
    if not val_agg.empty:
        labels = ["500k", "750k", "1000k", "1250k", "1500k", "final"]
        order = {label: idx for idx, label in enumerate(labels)}
        plt.figure(figsize=(10, 5))
        for seed, group in val_agg.groupby("training_seed"):
            group = group.assign(_order=group["checkpoint_label"].map(order)).sort_values("_order")
            plt.plot(group["checkpoint_label"], group["success_rate"], marker="o", label=f"seed{seed} success")
            plt.plot(group["checkpoint_label"], group["collision_rate"], marker="x", linestyle="--", label=f"seed{seed} collision")
        plt.ylim(0, 1)
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(result_dir / f"plots/{table_prefix}_validation_checkpoint_success_collision.png", dpi=140)
        plt.close()
    if not test_agg.empty:
        group = test_agg.sort_values("training_seed")
        x = np.arange(len(group))
        plt.figure(figsize=(7, 4.5))
        plt.bar(x - 0.18, group["success_rate"], width=0.36, label="success")
        plt.bar(x + 0.18, group["collision_rate"], width=0.36, label="collision")
        plt.xticks(x, [f"s{int(seed)} {label}" for seed, label in zip(group["training_seed"], group["checkpoint_label"])], rotation=20, ha="right")
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(result_dir / f"plots/{table_prefix}_test_selected_success_collision.png", dpi=140)
        plt.close()
    if not drift.empty and "all_l2_delta" in drift.columns:
        plt.figure(figsize=(8, 4.5))
        use = drift[drift["exists"].astype(int) == 1].copy()
        labels = [f"s{int(r.training_seed)} {r.comparison}" for _, r in use.iterrows()]
        plt.bar(np.arange(len(use)), use["all_l2_delta"])
        plt.xticks(np.arange(len(use)), labels, rotation=25, ha="right")
        plt.ylabel("L2 delta")
        plt.tight_layout()
        plt.savefig(result_dir / f"plots/{table_prefix}_parameter_drift.png", dpi=140)
        plt.close()
    if not train_diag.empty and "fps_mean" in train_diag.columns:
        plt.figure(figsize=(7, 4))
        plt.bar([f"s{int(seed)}" for seed in train_diag["training_seed"]], train_diag["fps_mean"])
        plt.ylabel("mean steps/s")
        plt.tight_layout()
        plt.savefig(result_dir / f"plots/{table_prefix}_training_fps.png", dpi=140)
        plt.close()


def artifacts(result_dir: Path) -> dict[str, list[str]]:
    return {
        "tables": [rel(path) for path in sorted((result_dir / "tables").glob("*.csv"))],
        "plots": [rel(path) for path in sorted((result_dir / "plots").glob("*.png"))],
        "logs": [rel(path) for path in sorted((result_dir / "logs").glob("*.log"))] + [rel(result_dir / "phase_n3pf_stab_confirm_watcher.log")],
        "flags": [rel(path) for path in sorted(result_dir.glob("*.flag"))],
    }


def report(
    result_dir: Path,
    args: argparse.Namespace,
    preflight: pd.DataFrame,
    selector_discipline: pd.DataFrame,
    selector: pd.DataFrame,
    val_agg: pd.DataFrame,
    test_agg: pd.DataFrame,
    train_diag: pd.DataFrame,
    drift: pd.DataFrame,
    attention: pd.DataFrame,
    decision: pd.DataFrame,
) -> str:
    d = decision.iloc[0].to_dict()
    lines = [
        "# Phase N3PF-STAB-CONFIRM Report",
        "",
        "## Terminal Decision",
        "",
        f"`terminal_decision = {d['terminal_decision']}`",
        "",
        f"GitHub sync status: `{args.github_sync_status}`; commit: `{args.github_sync_commit}`.",
        "",
        "S2-D semantics: `attention_like_gated_gpsi`. This is not strict attention-preserving, because no attention_full warm-start, isomorphic parameter load, or distillation is used.",
        "",
        "N4-U remains blocked. N4-O can only be rerun next phase if the candidate is confirmed.",
        "",
        "## Preflight",
        "",
    ]
    lines.extend(table_md(preflight, ["item", "kind", "ok", "detail"], 20))
    lines.extend(["", "## Parallel Training Strategy", ""])
    lines.append("Mandatory S2-D seeds 0/1/2 were scheduled as separate CPU PPO jobs with n_envs=4, device=cpu, and OMP/MKL/OpenBLAS/NumExpr set to 1. Seed3 is optional and included only if the watcher launched and completed it.")
    lines.extend(["", "## Selector Discipline", ""])
    lines.extend(table_md(selector_discipline, list(selector_discipline.columns), 5))
    lines.extend(["", "## Validation Selector", ""])
    lines.extend(table_md(selector, ["training_seed", "selected_checkpoint_label", "success_rate", "collision_rate", "selection_score", "selector_used_only_validation_seeds", "test_seed_used_for_selection"], 12))
    lines.extend(["", "## Validation Aggregate", ""])
    lines.extend(table_md(val_agg, ["training_seed", "checkpoint_label", "success_rate", "collision_rate", "progress", "raw_unsafe_action_rate", "episodes", "eval_seed_count"], 30))
    lines.extend(["", "## Test Aggregate", ""])
    lines.extend(table_md(test_agg, ["training_seed", "checkpoint_label", "success_rate", "collision_rate", "near_miss_rate", "progress", "raw_unsafe_action_rate", "action_delta", "episodes", "eval_seed_count"], 12))
    lines.extend(["", "## PPO Training Diagnostics", ""])
    lines.extend(table_md(train_diag, list(train_diag.columns), 12))
    lines.extend(["", "## Parameter Drift", ""])
    lines.extend(table_md(drift, ["training_seed", "selected_checkpoint_label", "comparison", "all_l2_delta", "feature_extractor_l2_delta", "actor_action_l2_delta", "critic_value_l2_delta", "log_std_l2_delta", "gate_l2_delta"], 20))
    lines.extend(["", "## Attention Full Audit", ""])
    lines.extend(table_md(attention, ["training_seed", "exists", "source", "formal_1500k_protocol_confirmed", "path"], 5))
    lines.extend(["", "## Final Decision Table", ""])
    lines.extend(table_md(decision, list(decision.columns), 5))
    files = artifacts(result_dir)
    lines.extend(["", "## Artifacts", ""])
    for key in ["tables", "plots", "logs", "flags"]:
        lines.append(f"### {key}")
        values = files.get(key, [])
        lines.extend([f"- `{value}`" for value in values] if values else ["- none"])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    result_dir = ROOT / args.result_dir
    table_prefix = str(args.table_prefix)
    for path in [result_dir, result_dir / "tables", result_dir / "plots", result_dir / "logs"]:
        path.mkdir(parents=True, exist_ok=True)
    try:
        preflight = repo_preflight(result_dir)
        selector_discipline = enforce_selector_discipline(result_dir, table_prefix)
        validate_test_rows(result_dir, table_prefix)
        val_agg = aggregate_eval(result_dir, table_prefix, "validation")
        test_agg = aggregate_eval(result_dir, table_prefix, "test")
        selector = read_csv(result_dir / f"tables/{table_prefix}_selector_decision.csv")
        train_diag = train_diagnostics(result_dir, table_prefix)
        drift = parameter_drift(result_dir, table_prefix)
        attention = attention_audit(result_dir, table_prefix)
        decision = final_decision(result_dir, table_prefix, args, test_agg)
        plots(result_dir, table_prefix, val_agg, test_agg, drift, train_diag)
        write_text(result_dir / REPORT_FILE, report(result_dir, args, preflight, selector_discipline, selector, val_agg, test_agg, train_diag, drift, attention, decision))
        write_text(result_dir / COMPLETE_FLAG, decision.iloc[0]["terminal_decision"] + "\n")
        write_text(result_dir / STATUS_FILE, "complete\n")
        print("STAB_CONFIRM_ANALYSIS_COMPLETE", flush=True)
    except ConfirmAnalysisStop as exc:
        write_stop(result_dir, exc.reason, exc.detail)
        raise SystemExit(2) from exc
    except Exception:
        detail = traceback.format_exc()
        write_stop(result_dir, "eval_failed", detail)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
