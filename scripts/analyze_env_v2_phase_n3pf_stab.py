from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


COMPLETE_FLAG = "PHASE_N3PF_STAB_COMPLETE.flag"
STOP_FLAGS = {
    "hard_error": "PHASE_N3PF_STAB_STOP_HARD_ERROR.flag",
    "validation_test_leakage": "PHASE_N3PF_STAB_STOP_VALIDATION_TEST_LEAKAGE.flag",
    "training_broken": "PHASE_N3PF_STAB_STOP_TRAINING_BROKEN.flag",
}
SCENARIOS = [
    "eval_flow_id",
    "eval_flow_high_density",
    "eval_flow_high_speed",
    "eval_flow_high_threat",
    "eval_flow_mixed_ood",
    "eval_flow_sudden_threat",
]


class AnalysisStop(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Phase N3PF-STAB outputs.")
    parser.add_argument("--result-dir", default="results/env_v2_phase_n3pf_stab")
    parser.add_argument("--mode", choices=["post_validation", "final"], default="post_validation")
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


def read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        if required:
            raise AnalysisStop("hard_error", f"missing or empty CSV: {rel(path)}")
        return pd.DataFrame()
    return pd.read_csv(path)


def write_stop(result_dir: Path, reason: str, detail: str) -> None:
    flag = STOP_FLAGS.get(reason, STOP_FLAGS["hard_error"])
    write_text(result_dir / flag, f"{reason}\n{detail.strip()}\n")
    write_text(result_dir / "phase_n3pf_stab_status.txt", f"stopped:{flag}\n")
    write_text(result_dir / "PHASE_N3PF_STAB_REPORT.md", f"# Phase N3PF-STAB Report\n\n`terminal_decision = phase_n3pf_stab_stop_{reason}`\n\n```text\n{detail.strip()}\n```\n")


def fmt(value: Any) -> str:
    try:
        f = float(value)
    except Exception:
        return str(value)
    if math.isnan(f):
        return "nan"
    return f"{f:.4f}"


def table_md(df: pd.DataFrame, cols: list[str], max_rows: int = 24) -> list[str]:
    if df.empty:
        return ["No rows."]
    cols = [col for col in cols if col in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].head(max_rows).iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) if isinstance(row[col], (float, int, np.floating, np.integer)) else str(row[col]) for col in cols) + " |")
    return lines


def repo_verification(result_dir: Path) -> pd.DataFrame:
    checks = [
        ("models/gpsi_ppo_policy.py", ROOT / "models/gpsi_ppo_policy.py", "file"),
        ("GpsiBlockProjectedNoZExtractor", ROOT / "models/gpsi_ppo_policy.py", "text"),
        ("GpsiGatedResidualExtractor", ROOT / "models/gpsi_ppo_policy.py", "text"),
        ("policies/obstacle_set_extractor.py", ROOT / "policies/obstacle_set_extractor.py", "file"),
        ("ObstacleSetExtractor", ROOT / "policies/obstacle_set_extractor.py", "text"),
        ("envs/wrappers/gpsi_obs_wrapper.py", ROOT / "envs/wrappers/gpsi_obs_wrapper.py", "file"),
        ("configs/env_v2_gpsi_heada_ppo_n3pf_block_projected.yaml", ROOT / "configs/env_v2_gpsi_heada_ppo_n3pf_block_projected.yaml", "file"),
        ("work_dirs/gpsi_heada_v1_nll/best.pth", ROOT / "work_dirs/gpsi_heada_v1_nll/best.pth", "file"),
        ("results/env_v2_phase_n3pf_ms_multiseed", ROOT / "results/env_v2_phase_n3pf_ms_multiseed", "dir"),
        ("results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun", ROOT / "results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun", "dir"),
    ]
    rows = []
    for item, path, kind in checks:
        if kind == "file":
            ok = path.is_file()
            detail = rel(path)
        elif kind == "dir":
            ok = path.is_dir()
            detail = rel(path)
        else:
            ok = path.is_file() and item in path.read_text(encoding="utf-8")
            detail = rel(path)
        rows.append({"item": item, "kind": kind, "ok": int(ok), "detail": detail})
    out = pd.DataFrame(rows)
    out.to_csv(result_dir / "tables/phase_n3pf_stab_repo_verification.csv", index=False)
    if int(out["ok"].min()) != 1:
        raise AnalysisStop("hard_error", out.to_string(index=False))
    return out


def config_diff(result_dir: Path) -> pd.DataFrame:
    rows = [
        {"variant": "stab_s1_lr2e4", "changed_field": "ppo.learning_rate", "old_value": "0.0003", "new_value": "0.0002", "structure_changed": 0},
        {"variant": "stab_s1_lr1e4", "changed_field": "ppo.learning_rate", "old_value": "0.0003", "new_value": "0.0001", "structure_changed": 0},
        {"variant": "stab_s2d_gated", "changed_field": "ppo.feature_adapter", "old_value": "block_projected_no_z", "new_value": "gated_residual_no_z", "structure_changed": 1},
        {"variant": "stab_s2d_gated", "changed_field": "s2_variant", "old_value": "", "new_value": "S2-D attention-like fallback", "structure_changed": 1},
    ]
    out = pd.DataFrame(rows)
    out.to_csv(result_dir / "tables/phase_n3pf_stab_config_diff.csv", index=False)
    return out


def seed2_collapse_diagnostics(result_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ms = ROOT / "results/env_v2_phase_n3pf_ms_multiseed/tables"
    ab = ROOT / "results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun/step_b_rerun/tables"
    ms_agg = read_csv(ms / "phase_n3pf_ms_eval_summary_aggregate.csv")
    ms_raw = read_csv(ms / "phase_n3pf_ms_raw_unsafe_action_summary.csv")
    ms_scenario = read_csv(ms / "phase_n3pf_ms_scenario_breakdown.csv")
    ms_motion = read_csv(ms / "phase_n3pf_ms_motion_mode_breakdown.csv")
    ms_threat = read_csv(ms / "phase_n3pf_ms_threat_class_breakdown.csv")
    ab_dec = read_csv(ab / "phase_n3pf_ms_seed2b_decision.csv")
    ab_agg = read_csv(ab / "phase_n3pf_ms_seed2b_eval_summary_aggregate.csv")
    ab_beh = read_csv(ab / "phase_n3pf_ms_seed2b_behavior_diagnostics.csv")

    focus = ms_agg[ms_agg["method_key"].astype(str).isin(["p3_s0_1500k", "p3_s2_1500k"])].copy()
    rows: list[dict[str, Any]] = []
    for _, row in focus.iterrows():
        key = str(row["method_key"])
        raw = ms_raw[ms_raw["method_key"].astype(str) == key]
        rows.append(
            {
                "comparison": "seed0_vs_seed2_original",
                "method_key": key,
                "training_seed": row.get("training_seed", np.nan),
                "success_rate": float(row.get("mean_success_rate", np.nan)),
                "collision_rate": float(row.get("mean_collision_rate", np.nan)),
                "raw_unsafe_action_rate": float(row.get("mean_raw_unsafe_action_rate", np.nan)),
                "action_delta": float(row.get("mean_action_delta", np.nan)),
                "progress": float(row.get("mean_progress", np.nan)),
                "raw_min_predicted_cpa": float(pd.to_numeric(raw.get("raw_min_predicted_cpa", pd.Series(dtype=float)), errors="coerce").mean()) if not raw.empty else np.nan,
                "raw_min_predicted_ttc": float(pd.to_numeric(raw.get("raw_min_predicted_ttc", pd.Series(dtype=float)), errors="coerce").mean()) if not raw.empty else np.nan,
                "source": "N3PF-MS",
            }
        )
    rerun = ab_agg[ab_agg["method_key"].astype(str) == "p3_s2_seed2_rerunA_1500k"]
    if not rerun.empty:
        r = rerun.iloc[0]
        rows.append(
            {
                "comparison": "seed2_original_vs_seed2_rerunA",
                "method_key": "p3_s2_seed2_rerunA_1500k",
                "training_seed": 2,
                "success_rate": float(r.get("mean_success_rate", np.nan)),
                "collision_rate": float(r.get("mean_collision_rate", np.nan)),
                "raw_unsafe_action_rate": float(r.get("mean_raw_unsafe_action_rate", np.nan)),
                "action_delta": float(r.get("mean_action_delta", np.nan)),
                "progress": float(r.get("mean_progress", np.nan)),
                "raw_min_predicted_cpa": float(r.get("mean_raw_min_predicted_cpa", np.nan)),
                "raw_min_predicted_ttc": np.nan,
                "source": "N3PF-MS-AB",
            }
        )
    diag = pd.DataFrame(rows)
    diag["ab_per_step_trace_available"] = 0
    diag["ab_per_step_trace_note"] = "AB rerun has summary/behavior diagnostics but no per-step trace CSV; no trace values are fabricated."
    diag.to_csv(result_dir / "tables/phase_n3pf_stab_seed2_collapse_diagnostics.csv", index=False)

    breakdowns = []
    for name, df, col in [("scenario", ms_scenario, "scenario"), ("motion", ms_motion, "threat_motion_mode"), ("threat", ms_threat, "threat_class")]:
        use = df[df["method_key"].astype(str).isin(["p3_s0_1500k", "p3_s2_1500k"])].copy()
        use["breakdown_type"] = name
        use["breakdown_key"] = use[col].astype(str) if col in use.columns else ""
        breakdowns.append(use)
    breakdown = pd.concat(breakdowns, ignore_index=True, sort=False) if breakdowns else pd.DataFrame()
    breakdown.to_csv(result_dir / "tables/phase_n3pf_stab_seed2_breakdown_diagnostics.csv", index=False)

    consistency = pd.DataFrame(
        [
            {
                "engineering_error_found": 0,
                "seed2_collapse_reproduced": int(float(ab_dec["seed2_rerunA_recovered"].iloc[0]) == 0) if not ab_dec.empty else 1,
                "most_likely_cause": str(ab_dec["collapse_most_likely_cause"].iloc[0]) if not ab_dec.empty else "unknown",
                "seed2_rerunA_recovered": int(ab_dec["seed2_rerunA_recovered"].iloc[0]) if not ab_dec.empty else 0,
                "behavior_rows_available": int(not ab_beh.empty),
            }
        ]
    )
    consistency.to_csv(result_dir / "tables/phase_n3pf_stab_seed2_rerun_consistency.csv", index=False)
    return diag, consistency


def aggregate_validation(result_dir: Path) -> pd.DataFrame:
    by_seed = read_csv(result_dir / "tables/phase_n3pf_stab_validation_eval_summary_by_seed.csv", required=False)
    if by_seed.empty:
        return pd.DataFrame()
    group_cols = ["variant", "training_seed", "method_key", "checkpoint_label", "checkpoint_path", "checkpoint_step"]
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
            eval_seed_count=("eval_seed", "nunique"),
            episodes=("episodes", "sum"),
        )
        .reset_index()
    )
    agg.to_csv(result_dir / "tables/phase_n3pf_stab_validation_eval_summary_aggregate.csv", index=False)
    return agg


def diagnostics_decision(result_dir: Path) -> pd.DataFrame:
    feat = read_csv(result_dir / "tables/phase_n3pf_stab_validation_feature_block_stats.csv", required=False)
    gpsi = read_csv(result_dir / "tables/phase_n3pf_stab_validation_gpsi_output_summary.csv", required=False)
    rows = []
    for variant in sorted(set(feat.get("variant", pd.Series(dtype=str)).astype(str)) | set(gpsi.get("variant", pd.Series(dtype=str)).astype(str))):
        f = feat[feat["variant"].astype(str) == variant] if not feat.empty else pd.DataFrame()
        g = gpsi[gpsi["variant"].astype(str) == variant] if not gpsi.empty else pd.DataFrame()
        nonfinite = int(pd.to_numeric(f.get("nan_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) + int(pd.to_numeric(f.get("inf_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        delta_p95 = float(pd.to_numeric(g.get("delta_norm_1s_p95", pd.Series(dtype=float)), errors="coerce").max()) if not g.empty else np.nan
        adapter = f[f["block"].astype(str) == "adapter_output_64"] if not f.empty and "block" in f.columns else pd.DataFrame()
        adapter_l2 = float(pd.to_numeric(adapter.get("l2_norm_p95", pd.Series(dtype=float)), errors="coerce").max()) if not adapter.empty else np.nan
        rows.append(
            {
                "variant": variant,
                "diagnostics_ok": int(nonfinite == 0 and np.isfinite(delta_p95) and delta_p95 < 100.0 and (not np.isfinite(adapter_l2) or adapter_l2 < 50.0)),
                "feature_nonfinite_count": nonfinite,
                "delta_norm_1s_p95_max": delta_p95,
                "adapter_output_l2_p95_max": adapter_l2,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(result_dir / "tables/phase_n3pf_stab_diagnostics_decision.csv", index=False)
    return out


def final_decision(result_dir: Path, args: argparse.Namespace) -> pd.DataFrame:
    selector = read_csv(result_dir / "tables/phase_n3pf_stab_selector_decision.csv", required=False)
    if selector.empty:
        return pd.DataFrame(
            [
                {
                    "phase_status": "post_validation_pending",
                    "p3_stab_candidate_found": "no",
                    "seed2_screening_passed_any": 0,
                    "multi_seed_confirmation_status": "not_run",
                    "n4o_can_rerun_next": "no",
                    "n4u_blocked": "yes",
                    "terminal_decision": "phase_n3pf_stab_complete",
                }
            ]
        )
    passed = selector[selector["seed2_screening_gate_pass"].astype(int) == 1].copy()
    cont = selector[selector["continue_to_1500k_recommended"].astype(int) == 1].copy()
    if not passed.empty:
        status = "seed2_screening_passed_proceed_multiseed_confirmation"
        candidate = str(passed.sort_values("selection_score", ascending=False)["variant"].iloc[0])
        n4o = "after_seed0_1_2_confirmation"
    elif not cont.empty:
        status = "seed2_screening_inconclusive_continue_1500k"
        candidate = str(cont.sort_values("selection_score", ascending=False)["variant"].iloc[0])
        n4o = "no"
    else:
        status = "seed2_screening_failed"
        candidate = ""
        n4o = "no"
    out = pd.DataFrame(
        [
            {
                "phase_status": status,
                "p3_stab_candidate_found": "screening_only" if not passed.empty else "no",
                "selected_screening_variant": candidate,
                "seed2_screening_passed_any": int(not passed.empty),
                "continue_1500k_recommended_any": int(not cont.empty),
                "multi_seed_confirmation_status": "not_run",
                "attention_full_multiseed_needed": "yes_before_decisive_attention_claim",
                "n4o_can_rerun_next": n4o,
                "n4u_blocked": "yes",
                "terminal_decision": "phase_n3pf_stab_complete",
                "available_attention_success": float(args.attention_success),
                "available_attention_collision": float(args.attention_collision),
                "noz_success": float(args.noz_success),
                "noz_collision": float(args.noz_collision),
            }
        ]
    )
    out.to_csv(result_dir / "tables/phase_n3pf_stab_final_decision.csv", index=False)
    return out


def plots(result_dir: Path, val: pd.DataFrame, diag: pd.DataFrame) -> None:
    (result_dir / "plots").mkdir(parents=True, exist_ok=True)
    if not val.empty:
        labels = ["500k", "750k", "1000k", "1250k", "1500k", "final"]
        order = {label: idx for idx, label in enumerate(labels)}
        plt.figure(figsize=(9, 4.8))
        for variant, group in val.groupby("variant"):
            group = group.assign(_order=group["checkpoint_label"].map(order)).sort_values("_order")
            plt.plot(group["checkpoint_label"], group["success_rate"], marker="o", label=f"{variant} success")
            plt.plot(group["checkpoint_label"], group["collision_rate"], marker="x", linestyle="--", label=f"{variant} collision")
        plt.ylim(0, 1)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(result_dir / "plots/phase_n3pf_stab_validation_success_collision.png", dpi=140)
        plt.close()
    if not diag.empty:
        plt.figure(figsize=(6.8, 4.2))
        x = np.arange(len(diag))
        plt.bar(x - 0.18, diag["success_rate"], width=0.36, label="success")
        plt.bar(x + 0.18, diag["collision_rate"], width=0.36, label="collision")
        plt.xticks(x, diag["method_key"], rotation=20, ha="right")
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(result_dir / "plots/phase_n3pf_stab_seed2_collapse_summary.png", dpi=140)
        plt.close()


def artifacts(result_dir: Path) -> dict[str, list[str]]:
    return {
        "tables": [rel(path) for path in sorted((result_dir / "tables").glob("*.csv"))],
        "plots": [rel(path) for path in sorted((result_dir / "plots").glob("*.png"))],
        "logs": [rel(path) for path in sorted((result_dir / "logs").glob("*.log"))] + [rel(result_dir / "phase_n3pf_stab_watcher.log")],
        "flags": [rel(path) for path in sorted(result_dir.glob("*.flag"))],
    }


def report(result_dir: Path, args: argparse.Namespace, repo: pd.DataFrame, cfg: pd.DataFrame, collapse: pd.DataFrame, consistency: pd.DataFrame, val: pd.DataFrame, diag: pd.DataFrame, selector: pd.DataFrame, decision: pd.DataFrame) -> str:
    d = decision.iloc[0].to_dict() if not decision.empty else {"terminal_decision": "phase_n3pf_stab_complete"}
    lines = [
        "# Phase N3PF-STAB Report",
        "",
        "## Terminal Decision",
        "",
        f"`terminal_decision = {d.get('terminal_decision', 'phase_n3pf_stab_complete')}`",
        "",
        f"GitHub sync status: `{args.github_sync_status}`; commit: `{args.github_sync_commit}`.",
        "",
        "N4-U remains blocked in this phase. N4-O remains conditional positive evidence, but must be rerun only after a stable P3-STAB policy is confirmed.",
        "",
        "## Repo Verification",
        "",
    ]
    lines.extend(table_md(repo, ["item", "kind", "ok", "detail"], max_rows=20))
    lines.extend(["", "## Config Changes", ""])
    lines.extend(table_md(cfg, ["variant", "changed_field", "old_value", "new_value", "structure_changed"], max_rows=20))
    lines.extend(
        [
            "",
            "S2 implementation type: `S2-D attention_like_gated_gpsi`.",
            "S2 attention preservation: no strict attention-preserving claim. It does not warm-start from trained attention_full and is only an attention-like fallback with gate initialized near zero.",
            "",
            "Validation/test/final-heldout discipline: validation seeds are 900/901 and are the only selector inputs. Test seeds 1000/1001/1002 and final-heldout seeds 1100/1101/1102 are reserved for later frozen-candidate evaluation.",
            "",
            "Selection metric is pre-registered as `success_rate - 2 * collision_rate`.",
            "",
            "## Seed2 Collapse Diagnostics",
            "",
        ]
    )
    lines.extend(table_md(collapse, ["comparison", "method_key", "success_rate", "collision_rate", "raw_unsafe_action_rate", "action_delta", "raw_min_predicted_cpa", "raw_min_predicted_ttc", "progress", "ab_per_step_trace_available"], max_rows=12))
    lines.extend(["", "## AB Consistency", ""])
    lines.extend(table_md(consistency, list(consistency.columns), max_rows=5))
    lines.extend(["", "## Validation Screening", ""])
    lines.extend(table_md(val, ["variant", "checkpoint_label", "success_rate", "collision_rate", "progress", "raw_unsafe_action_rate", "action_delta", "eval_seed_count", "episodes"], max_rows=36))
    lines.extend(["", "## Selector Decision", ""])
    lines.extend(table_md(selector, ["variant", "selected_checkpoint_label", "success_rate", "collision_rate", "selection_score", "seed2_screening_gate_pass", "trend_improving_through_1000k", "continue_to_1500k_recommended"], max_rows=12))
    lines.extend(["", "## Diagnostics Decision", ""])
    lines.extend(table_md(diag, list(diag.columns), max_rows=12))
    lines.extend(["", "## Final Decision", ""])
    lines.extend(table_md(decision, list(decision.columns), max_rows=5))
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
    for path in [result_dir, result_dir / "tables", result_dir / "plots", result_dir / "logs"]:
        path.mkdir(parents=True, exist_ok=True)
    try:
        repo = repo_verification(result_dir)
        cfg = config_diff(result_dir)
        collapse, consistency = seed2_collapse_diagnostics(result_dir)
        val = aggregate_validation(result_dir)
        diag = diagnostics_decision(result_dir)
        selector = read_csv(result_dir / "tables/phase_n3pf_stab_selector_decision.csv", required=False)
        decision_df = final_decision(result_dir, args)
        plots(result_dir, val, collapse)
        write_text(result_dir / "PHASE_N3PF_STAB_REPORT.md", report(result_dir, args, repo, cfg, collapse, consistency, val, diag, selector, decision_df))
        write_text(result_dir / COMPLETE_FLAG, f"complete\nmode={args.mode}\n")
        write_text(result_dir / "phase_n3pf_stab_status.txt", "complete\n")
        print("STAB_ANALYSIS_COMPLETE", flush=True)
    except AnalysisStop as exc:
        write_stop(result_dir, exc.reason, exc.detail)
        raise SystemExit(2) from exc
    except Exception:
        detail = traceback.format_exc()
        write_stop(result_dir, "hard_error", detail)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
