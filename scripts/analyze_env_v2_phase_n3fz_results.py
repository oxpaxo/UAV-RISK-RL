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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


COMPLETE_FLAG = "PHASE_N3FZ_NOZ_FULL_Z_SCREEN_COMPLETE.flag"
STOP_FLAGS = {
    "phase_n3r_missing": "PHASE_N3FZ_STOP_PHASE_N3R_MISSING.flag",
    "phase_n3_5_missing": "PHASE_N3FZ_STOP_PHASE_N3_5_MISSING.flag",
    "gpsi_checkpoint_missing": "PHASE_N3FZ_STOP_GPSI_CHECKPOINT_MISSING.flag",
    "z_stats_failed": "PHASE_N3FZ_STOP_Z_STATS_FAILED.flag",
    "train_failed": "PHASE_N3FZ_STOP_TRAIN_FAILED.flag",
    "eval_failed": "PHASE_N3FZ_STOP_EVAL_FAILED.flag",
    "diagnostics_failed": "PHASE_N3FZ_STOP_DIAGNOSTICS_FAILED.flag",
    "analysis_failed": "PHASE_N3FZ_STOP_ANALYSIS_FAILED.flag",
}
CONFIG_KEYS = ["n3f_no_z_full", "z_l2_scale_4", "z_layernorm_alpha_0p5"]
Z_KEYS = ["z_l2_scale_4", "z_layernorm_alpha_0p5"]
CONFIG_NAMES = {
    "n3f_no_z_full": "gpsi_no_z_full_1500k",
    "z_l2_scale_4": "gpsi_z_l2_scale_4_screen",
    "z_layernorm_alpha_0p5": "gpsi_z_layernorm_alpha_0p5_screen",
}
CHECKPOINT_DIRS = {
    "n3f_no_z_full": ROOT / "checkpoints/env_v2_gpsi_heada_ppo_n3f_no_z_s0",
    "z_l2_scale_4": ROOT / "checkpoints/env_v2_gpsi_heada_ppo_n3z_l2_scale4_s0",
    "z_layernorm_alpha_0p5": ROOT / "checkpoints/env_v2_gpsi_heada_ppo_n3z_layernorm_alpha0p5_s0",
}
N3R_SUCCESS_BASELINE = 0.4233
N3R_COLLISION_BASELINE = 0.5767
SCENARIOS = [
    "eval_flow_id",
    "eval_flow_high_density",
    "eval_flow_high_speed",
    "eval_flow_high_threat",
    "eval_flow_mixed_ood",
    "eval_flow_sudden_threat",
]


class PhaseN3FZAnalysisStop(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Phase N3FZ repaired Gpsi-PPO rerun outputs.")
    parser.add_argument("--result-dir", default="results/env_v2_phase_n3fz_noz_full_z_screen")
    parser.add_argument("--expected-episodes", type=int, default=50)
    parser.add_argument("--expected-noz-steps", type=int, default=1_500_000)
    parser.add_argument("--expected-z-steps", type=int, default=500_000)
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
            raise FileNotFoundError(f"missing or empty CSV: {rel(path)}")
        return pd.DataFrame()
    return pd.read_csv(path)


def write_stop(result_dir: Path, reason: str, detail: str) -> None:
    flag_name = STOP_FLAGS.get(reason, STOP_FLAGS["analysis_failed"])
    write_text(result_dir / flag_name, f"{reason}\n{detail.strip()}\n")
    write_text(result_dir / "phase_n3fz_status.txt", f"stopped:{flag_name}\n")
    write_text(
        result_dir / "PHASE_N3FZ_NOZ_FULL_Z_SCREEN_REPORT.md",
        "\n".join(
            [
                "# Phase N3F/Z No-Z Full + Z Screen Report",
                "",
                f"`terminal_decision = phase_n3fz_stopped_{reason}`",
                "",
                "Partial report generated because analysis reached a stop condition.",
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


def fmt(value: Any) -> str:
    try:
        f = float(value)
    except Exception:
        return str(value)
    if math.isnan(f):
        return "nan"
    return f"{f:.4f}"


def table_md(df: pd.DataFrame, cols: list[str], max_rows: int = 20) -> list[str]:
    if df.empty:
        return ["No rows."]
    view = df[cols].head(max_rows).copy()
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) if isinstance(row[col], (float, int, np.floating, np.integer)) else str(row[col]) for col in cols) + " |")
    return lines


def validate_required_outputs(result_dir: Path, args: argparse.Namespace) -> None:
    required_paths = {
        "n3_5_flag": ROOT / "results/env_v2_phase_n3_5_gpsi_wrapper_audit/PHASE_N3_5_GPSI_WRAPPER_AUDIT_COMPLETE.flag",
        "n3r_flag": ROOT / "results/env_v2_phase_n3r_gpsi_ppo_rerun/PHASE_N3R_GPSI_PPO_RERUN_COMPLETE.flag",
        "n3r_eval_summary": ROOT / "results/env_v2_phase_n3r_gpsi_ppo_rerun/tables/phase_n3r_eval_summary.csv",
        "gpsi_checkpoint": ROOT / "work_dirs/gpsi_heada_v1_nll/best.pth",
        "z_stats_csv": result_dir / "tables/phase_n3fz_z_stats.csv",
        "config_manifest": result_dir / "tables/phase_n3fz_config_manifest.csv",
        "schema_check": result_dir / "tables/phase_n3fz_schema_check.csv",
        "train_curve": result_dir / "tables/phase_n3fz_train_curve.csv",
        "eval_summary": result_dir / "tables/phase_n3fz_eval_summary.csv",
        "checkpoint_eval_summary": result_dir / "tables/phase_n3fz_checkpoint_eval_summary.csv",
        "episode_metrics": result_dir / "tables/phase_n3fz_episode_metrics.csv",
        "scenario_breakdown": result_dir / "tables/phase_n3fz_scenario_breakdown.csv",
        "motion_breakdown": result_dir / "tables/phase_n3fz_motion_mode_breakdown.csv",
        "threat_breakdown": result_dir / "tables/phase_n3fz_threat_class_breakdown.csv",
        "raw_unsafe": result_dir / "tables/phase_n3fz_raw_unsafe_action_summary.csv",
        "gpsi_output": result_dir / "tables/phase_n3fz_gpsi_output_summary.csv",
        "feature_blocks": result_dir / "tables/phase_n3fz_aug_feature_block_stats.csv",
        "command_manifest": result_dir / "tables/phase_n3fz_command_manifest.csv",
    }
    missing = [f"{name}: {rel(path)}" for name, path in required_paths.items() if not path.exists() or path.stat().st_size == 0]
    for key, ckpt_dir in CHECKPOINT_DIRS.items():
        required = [ckpt_dir / "checkpoint_250k.zip", ckpt_dir / "final.zip"]
        if key == "n3f_no_z_full":
            required.extend([ckpt_dir / "checkpoint_500k.zip", ckpt_dir / "checkpoint_1000k.zip", ckpt_dir / "checkpoint_1500k.zip"])
        else:
            required.append(ckpt_dir / "checkpoint_500k.zip")
        for path in required:
            if not path.exists() or path.stat().st_size == 0:
                missing.append(f"{key}_checkpoint: {rel(path)}")
    if missing:
        reason = (
            "phase_n3_5_missing"
            if any("n3_5_flag" in item for item in missing)
            else "phase_n3r_missing"
            if any("n3r_" in item for item in missing)
            else "train_failed"
            if any("checkpoint" in item for item in missing)
            else "eval_failed"
        )
        raise PhaseN3FZAnalysisStop(reason, "missing required artifacts:\n" + "\n".join(missing))


def final_rows(summary: pd.DataFrame) -> pd.DataFrame:
    return summary[summary["checkpoint_label"].astype(str) == "final"].copy()


def final_config_rows(summary: pd.DataFrame) -> pd.DataFrame:
    return final_rows(summary[summary["method_key"].isin(CONFIG_KEYS)].copy())


def build_attention_comparison(summary: pd.DataFrame, result_dir: Path) -> pd.DataFrame:
    final = final_config_rows(summary)
    attention = summary[summary["method_key"] == "attention_full"].copy()
    if final.empty or attention.empty:
        raise PhaseN3FZAnalysisStop("eval_failed", "missing final N3F/Z rows or attention reference rows")
    cols = [
        "method_key",
        "method",
        "scenario",
        "success_rate",
        "collision_rate",
        "near_miss_rate",
        "progress",
        "mean_min_distance",
        "episode_min_distance",
        "raw_unsafe_action_rate",
    ]
    att_cols = [
        "scenario",
        "success_rate",
        "collision_rate",
        "near_miss_rate",
        "progress",
        "mean_min_distance",
        "episode_min_distance",
        "raw_unsafe_action_rate",
    ]
    merged = final[cols].merge(attention[att_cols], on="scenario", how="inner", suffixes=("_config", "_attention"))
    for metric in att_cols[1:]:
        merged[f"delta_{metric}"] = merged[f"{metric}_config"] - merged[f"{metric}_attention"]
    merged.to_csv(result_dir / "tables/phase_n3fz_attention_reference_comparison.csv", index=False)
    return merged


def aggregate_final(summary: pd.DataFrame) -> pd.DataFrame:
    final = final_config_rows(summary)
    if final.empty:
        return pd.DataFrame()
    return (
        final.groupby(["method_key", "method"], dropna=False)
        .agg(
            success_rate=("success_rate", "mean"),
            collision_rate=("collision_rate", "mean"),
            near_miss_rate=("near_miss_rate", "mean"),
            progress=("progress", "mean"),
            mean_min_distance=("mean_min_distance", "mean"),
            raw_unsafe_action_rate=("raw_unsafe_action_rate", "mean"),
            no_response_rate=("no_response_rate", "mean"),
            action_norm=("action_norm", "mean"),
        )
        .reset_index()
    )


def diagnostics_table(gpsi: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    diag_rows: list[dict[str, Any]] = []
    for method_key in CONFIG_KEYS:
        rows = gpsi[(gpsi["method_key"] == method_key) & (gpsi["checkpoint_label"].astype(str) == "final")]
        feature_rows = features[(features["method_key"] == method_key) & (features["checkpoint_label"].astype(str) == "final")]
        delta_p95 = pd.to_numeric(rows.get("delta_norm_1s_p95", pd.Series(dtype=float)), errors="coerce").max()
        delta_max = pd.to_numeric(rows.get("delta_norm_1s_max", pd.Series(dtype=float)), errors="coerce").max()
        inactive = pd.to_numeric(rows.get("inactive_forwarded_count_max", pd.Series(dtype=float)), errors="coerce").max()
        logvar_span = pd.to_numeric(rows.get("logvar_xy_1s_span", pd.Series(dtype=float)), errors="coerce").max()
        z_after = feature_rows[feature_rows["block"].astype(str) == "z_i_64_after_constraint"]
        z_after_p95 = pd.to_numeric(z_after.get("l2_norm_p95", pd.Series(dtype=float)), errors="coerce").max()
        nonfinite = 0
        for col in ["nan_count", "inf_count"]:
            if col in feature_rows:
                nonfinite += int(pd.to_numeric(feature_rows[col], errors="coerce").fillna(0).sum())
        normal = (
            np.isfinite(delta_p95)
            and np.isfinite(delta_max)
            and float(delta_p95) < 100.0
            and float(delta_max) < 1000.0
            and (not np.isfinite(inactive) or float(inactive) <= 0.0)
            and nonfinite == 0
            and (not np.isfinite(logvar_span) or float(logvar_span) > 0.05)
        )
        diag_rows.append(
            {
                "method_key": method_key,
                "diagnostics_ok": int(bool(normal)),
                "delta_norm_1s_p95_max": float(delta_p95) if np.isfinite(delta_p95) else np.nan,
                "delta_norm_1s_max": float(delta_max) if np.isfinite(delta_max) else np.nan,
                "inactive_forwarded_count_max": float(inactive) if np.isfinite(inactive) else np.nan,
                "logvar_xy_1s_span_max": float(logvar_span) if np.isfinite(logvar_span) else np.nan,
                "z_after_constraint_l2_p95_max": float(z_after_p95) if np.isfinite(z_after_p95) else np.nan,
                "feature_nonfinite_count": int(nonfinite),
            }
        )
    return pd.DataFrame(diag_rows)


def build_hard_gate(summary: pd.DataFrame, gpsi: pd.DataFrame, features: pd.DataFrame, result_dir: Path) -> pd.DataFrame:
    agg = aggregate_final(summary)
    if agg.empty:
        raise PhaseN3FZAnalysisStop("eval_failed", "no final config rows for hard-gate decision")
    diag = diagnostics_table(gpsi, features)
    out = agg.merge(diag, on="method_key", how="left")
    out["n3r_no_z_500k_success_baseline"] = N3R_SUCCESS_BASELINE
    out["n3r_no_z_500k_collision_baseline"] = N3R_COLLISION_BASELINE
    out["is_z_variant"] = out["method_key"].isin(Z_KEYS).astype(int)
    out["passes_success_gate"] = (pd.to_numeric(out["success_rate"], errors="coerce") >= N3R_SUCCESS_BASELINE).astype(int)
    out["passes_collision_gate"] = (pd.to_numeric(out["collision_rate"], errors="coerce") <= N3R_COLLISION_BASELINE).astype(int)
    out["passes_diagnostics_gate"] = (pd.to_numeric(out["diagnostics_ok"], errors="coerce").fillna(0) == 1).astype(int)
    out["z_hard_gate_pass"] = (
        (out["is_z_variant"] == 1)
        & (out["passes_success_gate"] == 1)
        & (out["passes_collision_gate"] == 1)
        & (out["passes_diagnostics_gate"] == 1)
    ).astype(int)
    out["eligible_for_1_5m_continuation"] = out["z_hard_gate_pass"].map({1: "yes", 0: "no"})

    passed_z = out[(out["is_z_variant"] == 1) & (out["z_hard_gate_pass"] == 1)].copy()
    if passed_z.empty:
        z_winner_key = "none"
        z_winner_method = "none"
        z_continuation = "no"
        z_decision = "No Z variant passed both success and collision hard gates; do not continue Z to 1.5M."
    else:
        winner = passed_z.sort_values(
            ["collision_rate", "success_rate", "near_miss_rate", "raw_unsafe_action_rate", "progress"],
            ascending=[True, False, True, True, False],
        ).iloc[0]
        z_winner_key = str(winner["method_key"])
        z_winner_method = str(winner["method"])
        z_continuation = "yes"
        z_decision = f"Continue Z winner {z_winner_method} to 1.5M before choosing a final N4 candidate."

    no_z = out[out["method_key"] == "n3f_no_z_full"].copy()
    if no_z.empty or int(no_z["diagnostics_ok"].iloc[0]) != 1:
        no_z_candidate = "no"
        can_enter_n4 = "no"
        no_z_decision = "N3F no_z full is not an eligible N4 policy candidate because diagnostics failed or results are missing."
    elif z_continuation == "yes":
        no_z_candidate = "defer_until_z_continuation"
        can_enter_n4 = "no"
        no_z_decision = "N3F no_z full is clean, but a Z variant passed the continuation gate; run that Z continuation before N4."
    else:
        no_z_candidate = "yes"
        can_enter_n4 = "candidate_ready_but_phase_does_not_enter_n4"
        no_z_decision = "N3F no_z full is the current repaired no-shield policy candidate for the next N4 decision point."

    out["z_winner_key"] = z_winner_key
    out["z_winner_method"] = z_winner_method
    out["z_1_5m_continuation_needed"] = z_continuation
    out["z_hard_gate_decision"] = z_decision
    out["no_z_full_n4_policy_candidate"] = no_z_candidate
    out["can_enter_n4"] = can_enter_n4
    out["recommendation"] = no_z_decision
    out.to_csv(result_dir / "tables/phase_n3fz_z_hard_gate_decision.csv", index=False)
    out.to_csv(result_dir / "tables/phase_n3fz_winner_recommendation.csv", index=False)
    return out


def build_winner(summary: pd.DataFrame, gpsi: pd.DataFrame, features: pd.DataFrame, result_dir: Path) -> pd.DataFrame:
    agg = aggregate_final(summary)
    if agg.empty:
        raise PhaseN3FZAnalysisStop("eval_failed", "no final config rows for winner recommendation")
    agg["score"] = (
        agg["success_rate"].astype(float)
        + 0.25 * agg["progress"].astype(float)
        + 0.02 * agg["mean_min_distance"].astype(float)
        - 1.25 * agg["collision_rate"].astype(float)
        - 0.50 * agg["near_miss_rate"].astype(float)
        - 0.25 * agg["raw_unsafe_action_rate"].astype(float)
    )

    diag_rows: list[dict[str, Any]] = []
    for method_key in CONFIG_KEYS:
        rows = gpsi[(gpsi["method_key"] == method_key) & (gpsi["checkpoint_label"].astype(str) == "final")]
        feature_rows = features[(features["method_key"] == method_key) & (features["checkpoint_label"].astype(str) == "final")]
        delta_p95 = pd.to_numeric(rows.get("delta_norm_1s_p95", pd.Series(dtype=float)), errors="coerce").max()
        delta_max = pd.to_numeric(rows.get("delta_norm_1s_max", pd.Series(dtype=float)), errors="coerce").max()
        inactive = pd.to_numeric(rows.get("inactive_forwarded_count_max", pd.Series(dtype=float)), errors="coerce").max()
        logvar_span = pd.to_numeric(rows.get("logvar_xy_1s_span", pd.Series(dtype=float)), errors="coerce").max()
        nonfinite = 0
        for col in ["nan_count", "inf_count"]:
            if col in feature_rows:
                nonfinite += int(pd.to_numeric(feature_rows[col], errors="coerce").fillna(0).sum())
        normal = (
            np.isfinite(delta_p95)
            and np.isfinite(delta_max)
            and float(delta_p95) < 100.0
            and float(delta_max) < 1000.0
            and (not np.isfinite(inactive) or float(inactive) <= 0.0)
            and nonfinite == 0
            and (not np.isfinite(logvar_span) or float(logvar_span) > 0.05)
        )
        diag_rows.append(
            {
                "method_key": method_key,
                "diagnostics_ok": int(bool(normal)),
                "delta_norm_1s_p95_max": float(delta_p95) if np.isfinite(delta_p95) else np.nan,
                "delta_norm_1s_max": float(delta_max) if np.isfinite(delta_max) else np.nan,
                "inactive_forwarded_count_max": float(inactive) if np.isfinite(inactive) else np.nan,
                "logvar_xy_1s_span_max": float(logvar_span) if np.isfinite(logvar_span) else np.nan,
                "feature_nonfinite_count": int(nonfinite),
            }
        )
    diag = pd.DataFrame(diag_rows)
    out = agg.merge(diag, on="method_key", how="left")
    valid = out[out["diagnostics_ok"] == 1].copy()
    if valid.empty:
        winner_key = "no_winner"
        winner_method = "no_winner"
        need_full = "no"
        can_enter_n4 = "no"
        recommendation = "No N3FZ config is eligible because diagnostics failed."
    else:
        best = valid.sort_values(["score", "success_rate", "progress"], ascending=False).iloc[0]
        winner_key = str(best["method_key"])
        winner_method = str(best["method"])
        promising = float(best["success_rate"]) >= 0.10 or float(best["progress"]) >= 0.35
        need_full = "yes" if promising else "no"
        can_enter_n4 = "no"
        if need_full == "yes":
            recommendation = f"Winner for the next N3-full candidate is {winner_method}; run repaired no-shield PPO to 1.5M before N4."
        else:
            recommendation = f"Best screening config is {winner_method}, but performance is too weak for a full 1.5M rerun without another repair/ablation."
    out["winner_key"] = winner_key
    out["winner_method"] = winner_method
    out["need_full_1_5m_rerun"] = need_full
    out["can_enter_n4"] = can_enter_n4
    out["recommendation"] = recommendation
    out.to_csv(result_dir / "tables/phase_n3fz_winner_recommendation.csv", index=False)
    return out


def validate_tables(tables: dict[str, pd.DataFrame], args: argparse.Namespace) -> None:
    summary = tables["summary"]
    expected_methods = set(CONFIG_KEYS) | {"attention_full"}
    methods = set(summary["method_key"].astype(str))
    missing_methods = expected_methods - methods
    if missing_methods:
        raise PhaseN3FZAnalysisStop("eval_failed", f"missing methods in eval summary: {sorted(missing_methods)}")
    scenarios = set(summary["scenario"].astype(str))
    missing_scenarios = set(SCENARIOS) - scenarios
    if missing_scenarios:
        raise PhaseN3FZAnalysisStop("eval_failed", f"missing scenarios: {sorted(missing_scenarios)}")
    episodes = tables["episodes"]
    counts = episodes.groupby(["method_key", "checkpoint_label", "scenario"]).size().reset_index(name="episodes")
    bad = counts[counts["episodes"] < int(args.expected_episodes)]
    if not bad.empty:
        raise PhaseN3FZAnalysisStop("eval_failed", f"not enough eval episodes:\n{bad.to_string(index=False)}")
    for key in CONFIG_KEYS:
        labels = set(summary[summary["method_key"] == key]["checkpoint_label"].astype(str))
        required_labels = {"250k", "final", "best_by_eval"}
        if key == "n3f_no_z_full":
            required_labels.update({"500k", "1000k", "1500k"})
        else:
            required_labels.add("500k")
        missing_labels = sorted(required_labels - labels)
        if missing_labels:
            raise PhaseN3FZAnalysisStop("eval_failed", f"missing eval rows for {key}: missing={missing_labels} labels={sorted(labels)}")

    gpsi = tables["gpsi"]
    if gpsi.empty:
        raise PhaseN3FZAnalysisStop("diagnostics_failed", "missing Gpsi output diagnostics")
    final_gpsi = gpsi[gpsi["checkpoint_label"].astype(str) == "final"].copy()
    delta_p95 = pd.to_numeric(final_gpsi.get("delta_norm_1s_p95", pd.Series(dtype=float)), errors="coerce")
    delta_max = pd.to_numeric(final_gpsi.get("delta_norm_1s_max", pd.Series(dtype=float)), errors="coerce")
    inactive = pd.to_numeric(final_gpsi.get("inactive_forwarded_count_max", pd.Series(dtype=float)), errors="coerce")
    if delta_p95.dropna().empty or float(delta_p95.max()) >= 100.0 or float(delta_max.max()) >= 1000.0:
        raise PhaseN3FZAnalysisStop("diagnostics_failed", f"Gpsi output scale invalid: delta_p95_max={delta_p95.max()} delta_max={delta_max.max()}")
    if not inactive.dropna().empty and float(inactive.max()) > 0.0:
        raise PhaseN3FZAnalysisStop("diagnostics_failed", f"inactive forwarded count is nonzero: {inactive.max()}")
    features = tables["features"]
    if features.empty:
        raise PhaseN3FZAnalysisStop("diagnostics_failed", "missing augmented feature block stats")
    nonfinite = 0
    for col in ["nan_count", "inf_count"]:
        if col in features:
            nonfinite += int(pd.to_numeric(features[col], errors="coerce").fillna(0).sum())
    if nonfinite != 0:
        raise PhaseN3FZAnalysisStop("diagnostics_failed", f"non-finite augmented feature values found: {nonfinite}")


def plot_success_collision(summary: pd.DataFrame, path: Path) -> None:
    final = final_rows(summary)
    agg = final.groupby("method_key", dropna=False).agg(success=("success_rate", "mean"), collision=("collision_rate", "mean")).reset_index()
    order = ["attention_full", *CONFIG_KEYS]
    agg["order"] = agg["method_key"].map({key: idx for idx, key in enumerate(order)})
    agg = agg.sort_values("order")
    x = np.arange(len(agg))
    width = 0.35
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.5, 4.5))
    if not agg.empty:
        plt.bar(x - width / 2, agg["success"], width, label="success")
        plt.bar(x + width / 2, agg["collision"], width, label="collision")
        plt.xticks(x, agg["method_key"], rotation=20, ha="right")
    plt.ylabel("rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_checkpoint(summary: pd.DataFrame, path: Path) -> None:
    data = (
        summary[summary["method_key"].isin(CONFIG_KEYS)]
        .groupby(["method_key", "checkpoint_label", "checkpoint_step"], dropna=False)
        .agg(success=("success_rate", "mean"), collision=("collision_rate", "mean"))
        .reset_index()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.8))
    for key in CONFIG_KEYS:
        group = data[data["method_key"] == key].copy()
        group["x"] = pd.to_numeric(group["checkpoint_step"], errors="coerce")
        group.loc[group["checkpoint_label"].astype(str) == "best_by_eval", "x"] = group["x"].max()
        group.loc[group["checkpoint_label"].astype(str) == "final", "x"] = group["x"].max()
        group = group.sort_values("x")
        if not group.empty:
            plt.plot(group["x"], group["success"], marker="o", label=f"{key} success")
            plt.plot(group["x"], group["collision"], marker="x", linestyle="--", label=f"{key} collision")
    plt.xlabel("checkpoint step")
    plt.ylabel("rate")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_raw_unsafe(summary: pd.DataFrame, path: Path) -> None:
    final = final_rows(summary)
    agg = final.groupby("method_key", dropna=False).agg(raw_unsafe=("raw_unsafe_action_rate", "mean")).reset_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.2))
    if not agg.empty:
        plt.bar(agg["method_key"], agg["raw_unsafe"])
        plt.xticks(rotation=20, ha="right")
    plt.ylabel("raw unsafe action rate")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_feature_blocks(features: pd.DataFrame, path: Path) -> None:
    final = features[(features["checkpoint_label"].astype(str) == "final") & (features["scenario"].astype(str) == "eval_flow_id")].copy()
    final = (
        final.groupby(["method_key", "block"], dropna=False)
        .agg(l2_norm_p95=("l2_norm_p95", "mean"))
        .reset_index()
    )
    blocks = ["obs_i_12", "z_i_64_raw", "z_i_64_after_constraint", "delta_hat_9_after_scale", "logvar_hat_9_clamped", "full_aug_obs"]
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 4.8))
    width = 0.22
    x = np.arange(len(blocks))
    for idx, key in enumerate(CONFIG_KEYS):
        group = final[final["method_key"] == key].set_index("block")
        values = [float(group.loc[block, "l2_norm_p95"]) if block in group.index else np.nan for block in blocks]
        plt.bar(x + (idx - 1) * width, values, width, label=key)
    plt.xticks(x, blocks, rotation=25, ha="right")
    plt.ylabel("p95 L2 norm")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_gpsi_delta(gpsi: pd.DataFrame, path: Path) -> None:
    final = gpsi[gpsi["checkpoint_label"].astype(str) == "final"].copy()
    agg = final.groupby("method_key", dropna=False).agg(delta=("delta_norm_1s_p95", "max")).reset_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.2))
    if not agg.empty:
        plt.bar(agg["method_key"], agg["delta"])
        plt.xticks(rotation=20, ha="right")
    plt.ylabel("delta norm 1s p95")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_gpsi_logvar(gpsi: pd.DataFrame, path: Path) -> None:
    final = gpsi[gpsi["checkpoint_label"].astype(str) == "final"].copy()
    agg = final.groupby("method_key", dropna=False).agg(logvar=("logvar_xy_1s_mean", "mean"), span=("logvar_xy_1s_span", "max")).reset_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.2))
    if not agg.empty:
        x = np.arange(len(agg))
        plt.bar(x - 0.18, agg["logvar"], 0.36, label="mean")
        plt.bar(x + 0.18, agg["span"], 0.36, label="span")
        plt.xticks(x, agg["method_key"], rotation=20, ha="right")
    plt.ylabel("logvar xy 1s")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_train_curve(train: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    if not train.empty and {"method_key", "steps", "episode_reward"}.issubset(train.columns):
        for key in CONFIG_KEYS:
            group = train[train["method_key"] == key].copy()
            if group.empty:
                continue
            group = group.sort_values("steps")
            smooth = pd.to_numeric(group["episode_reward"], errors="coerce").rolling(25, min_periods=1).mean()
            plt.plot(group["steps"], smooth, label=key)
    plt.xlabel("training steps")
    plt.ylabel("episode reward rolling mean")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_scenario(summary: pd.DataFrame, path: Path) -> None:
    final = final_config_rows(summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5.0))
    scenarios = SCENARIOS
    x = np.arange(len(scenarios))
    width = 0.24
    for idx, key in enumerate(CONFIG_KEYS):
        group = final[final["method_key"] == key].set_index("scenario")
        values = [float(group.loc[s, "success_rate"]) if s in group.index else np.nan for s in scenarios]
        plt.bar(x + (idx - 1) * width, values, width, label=key)
    plt.xticks(x, scenarios, rotation=25, ha="right")
    plt.ylabel("success rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_z_hard_gate(gate: pd.DataFrame, path: Path) -> None:
    z = gate[gate["method_key"].isin(Z_KEYS)].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.0, 4.2))
    if not z.empty:
        x = np.arange(len(z))
        width = 0.35
        plt.bar(x - width / 2, z["success_rate"], width, label="success")
        plt.bar(x + width / 2, z["collision_rate"], width, label="collision")
        plt.axhline(N3R_SUCCESS_BASELINE, color="tab:green", linestyle=":", linewidth=1.5, label="success gate")
        plt.axhline(N3R_COLLISION_BASELINE, color="tab:red", linestyle="--", linewidth=1.5, label="collision gate")
        plt.xticks(x, z["method_key"], rotation=20, ha="right")
    plt.ylabel("rate")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def generate_plots(result_dir: Path, tables: dict[str, pd.DataFrame], gate: pd.DataFrame) -> None:
    plot_success_collision(tables["summary"], result_dir / "plots/success_collision_by_config.png")
    plot_checkpoint(tables["summary"], result_dir / "plots/checkpoint_success_collision_by_config.png")
    plot_raw_unsafe(tables["summary"], result_dir / "plots/raw_unsafe_by_config.png")
    plot_feature_blocks(tables["features"], result_dir / "plots/aug_feature_block_scale_by_config.png")
    plot_gpsi_delta(tables["gpsi"], result_dir / "plots/gpsi_delta_norm_by_config.png")
    plot_gpsi_logvar(tables["gpsi"], result_dir / "plots/gpsi_logvar_by_config.png")
    plot_train_curve(tables["train"], result_dir / "plots/train_reward_by_config.png")
    plot_scenario(tables["summary"], result_dir / "plots/scenario_breakdown_by_config.png")
    plot_z_hard_gate(gate, result_dir / "plots/z_hard_gate_by_variant.png")


def collect_files(result_dir: Path) -> dict[str, list[str]]:
    checkpoints: list[str] = []
    for path in CHECKPOINT_DIRS.values():
        checkpoints.extend(rel(item) for item in sorted(path.glob("*.zip")))
    return {
        "checkpoints": checkpoints,
        "tables": [rel(path) for path in sorted((result_dir / "tables").glob("*.csv"))],
        "plots": [rel(path) for path in sorted((result_dir / "plots").glob("*.png"))],
        "logs": [rel(path) for path in sorted((result_dir / "logs").glob("*.log"))] + [rel(result_dir / "phase_n3fz_watcher.log")],
        "flags": [rel(path) for path in sorted(result_dir.glob("*.flag"))],
    }


def report_lines(
    result_dir: Path,
    tables: dict[str, pd.DataFrame],
    comparison: pd.DataFrame,
    winner: pd.DataFrame,
    files: dict[str, list[str]],
    terminal_decision: str,
    args: argparse.Namespace,
) -> list[str]:
    final = final_config_rows(tables["summary"])
    final_agg = (
        final.groupby(["method_key", "method"], dropna=False)
        .agg(
            success_rate=("success_rate", "mean"),
            collision_rate=("collision_rate", "mean"),
            near_miss_rate=("near_miss_rate", "mean"),
            progress=("progress", "mean"),
            raw_unsafe_action_rate=("raw_unsafe_action_rate", "mean"),
            mean_min_distance=("mean_min_distance", "mean"),
        )
        .reset_index()
    )
    winner_row = winner.iloc[0] if not winner.empty else pd.Series(dtype=object)
    winner_key = str(winner_row.get("winner_key", "no_winner"))
    winner_method = str(winner_row.get("winner_method", "no_winner"))
    need_full = str(winner_row.get("need_full_1_5m_rerun", "no"))
    can_enter = str(winner_row.get("can_enter_n4", "no"))
    recommendation = str(winner_row.get("recommendation", "No recommendation generated."))
    lines = [
        "# Phase N3FZ No-Z Full + Z Screen Report",
        "",
        "## Terminal Decision",
        "",
        f"`terminal_decision = {terminal_decision}`",
        "",
        "Phase N3FZ complete. This is a repaired no-shield PPO screening rerun with z-block ablation; it does not enter N4.",
        "",
        "## Engineering Facts",
        "",
        "- Phase N3.5 complete flag exists and repaired `GpsiObsWrapper` was used.",
        "- Gpsi checkpoint: `work_dirs/gpsi_heada_v1_nll/best.pth`; Gpsi was frozen under `eval()` with no trainable parameters.",
        "- EnvV2 core was not modified for this phase.",
        "- No shield, no action filtering, no dense safety cost, no learned R(s,a), and no Gpsi fine-tuning were used.",
        f"- Training budget used: `{args.expected_train_steps}` PPO steps per A/B/C config, seed 0.",
        f"- Evaluation used `{args.expected_episodes}` episodes per scenario over six scenarios.",
        "",
        "## N3.5 Repair Summary",
        "",
        "The original N3 no-shield result is invalid because online normalization divided nonzero PPO UAV velocities by near-zero N2 hold-position velocity std. N3.5 fixed this by flooring degenerate checkpoint std dimensions and verified offline-online equivalence.",
        "",
        "## Configs",
        "",
        "- A `repaired-full-raw-z`: `[obs_i, z_i_raw, delta_hat_scaled, logvar_hat]`, dim 94.",
        "- B `repaired-full-z-normalized`: `[obs_i, z_i_normalized, delta_hat_scaled, logvar_hat]`, dim 94. z stats are from N1 train split frozen-Gpsi forward only.",
        "- C `repaired-no-z`: `[obs_i, delta_hat_scaled, logvar_hat]`, dim 30.",
        "",
        "## Experiment-Supported Facts",
        "",
    ]
    lines.extend(table_md(final_agg, ["method_key", "success_rate", "collision_rate", "near_miss_rate", "progress", "raw_unsafe_action_rate", "mean_min_distance"]))
    lines.extend(["", "## Attention Reference Comparison", ""])
    comp_cols = [
        "method_key",
        "scenario",
        "success_rate_config",
        "success_rate_attention",
        "delta_success_rate",
        "collision_rate_config",
        "collision_rate_attention",
        "delta_collision_rate",
    ]
    lines.extend(table_md(comparison.sort_values(["method_key", "scenario"]), comp_cols, max_rows=18))
    lines.extend(["", "## Gpsi Output Diagnostics", ""])
    gpsi_final = tables["gpsi"][tables["gpsi"]["checkpoint_label"].astype(str) == "final"].copy()
    gpsi_agg = (
        gpsi_final.groupby("method_key", dropna=False)
        .agg(
            delta_norm_1s_p95=("delta_norm_1s_p95", "max"),
            delta_norm_1s_max=("delta_norm_1s_max", "max"),
            logvar_xy_1s_mean=("logvar_xy_1s_mean", "mean"),
            logvar_xy_1s_span=("logvar_xy_1s_span", "max"),
            projected_std_radial_mean=("projected_std_radial_mean", "mean"),
            projected_std_relvel_mean=("projected_std_relvel_mean", "mean"),
            z_norm_raw_p95=("z_norm_raw_p95", "max"),
            z_norm_after_p95=("z_norm_after_p95", "max"),
            history_valid_ratio_mean=("history_valid_ratio_mean", "mean"),
            inactive_forwarded_count_max=("inactive_forwarded_count_max", "max"),
        )
        .reset_index()
    )
    lines.extend(table_md(gpsi_agg, list(gpsi_agg.columns), max_rows=12))
    lines.extend(["", "## Augmented Feature Block Stats", ""])
    feature_final = tables["features"][(tables["features"]["checkpoint_label"].astype(str) == "final") & (tables["features"]["scenario"].astype(str) == "eval_flow_id")]
    feature_cols = ["method_key", "block", "not_applicable", "l2_norm_p95", "max_abs_p95", "nan_count", "inf_count"]
    lines.extend(table_md(feature_final.sort_values(["method_key", "block"]), feature_cols, max_rows=24))
    lines.extend(["", "## Raw Unsafe Diagnostics", ""])
    raw_final = tables["raw"][tables["raw"]["checkpoint_label"].astype(str) == "final"].copy()
    raw_agg = raw_final.groupby("method_key", dropna=False).agg(raw_unsafe_rate=("raw_unsafe_rate", "mean"), raw_min_predicted_cpa=("raw_min_predicted_cpa", "mean"), no_response_rate=("no_response_rate", "mean")).reset_index()
    lines.extend(table_md(raw_agg, list(raw_agg.columns)))
    lines.extend(["", "## Breakdown Outputs", ""])
    lines.append("Scenario, motion-mode, and threat-class breakdown CSVs were generated under `results/env_v2_phase_n3fz_gpsi_ppo_rerun/tables/`.")
    lines.extend(["", "## Winner Recommendation", ""])
    lines.append(f"- Winner recommendation: `{winner_key}` / `{winner_method}`.")
    lines.append(f"- Recommendation: {recommendation}")
    lines.append(f"- Need full N3 1.5M rerun: {need_full}.")
    lines.append(f"- Can enter N4: {can_enter}.")
    if can_enter != "yes":
        lines.append("- Next step before N4: run the recommended repaired no-shield N3-full 1.5M baseline if the winner is promising; otherwise repair/ablate and rerun N3FZ.")
    lines.extend(["", "## Reasonable Inferences", ""])
    lines.append("- N3FZ is a screening comparison. A promising winner should be used for a full repaired N3 no-shield rerun before any shield comparison.")
    lines.append("- If z-normalized or no-z wins over raw-z, raw latent scale is a likely PPO input conditioning problem rather than a Gpsi HeadA output-scale failure.")
    lines.extend(["", "## Remaining Risks", ""])
    lines.append("- The run uses seed 0 only; the selected winner still needs longer-run and multi-seed confirmation before method-level claims.")
    lines.append("- The attention reference is a 1.5M baseline; N3FZ A/B/C are 500k screening runs and are not a final compute-matched comparison.")
    lines.extend(["", "## Artifacts", ""])
    for key in ["checkpoints", "tables", "plots", "logs", "flags"]:
        lines.append(f"### {key}")
        values = files.get(key, [])
        if values:
            lines.extend([f"- `{value}`" for value in values[:80]])
            if len(values) > 80:
                lines.append(f"- ... {len(values) - 80} more")
        else:
            lines.append("- none")
    return lines


def report_lines(
    result_dir: Path,
    tables: dict[str, pd.DataFrame],
    comparison: pd.DataFrame,
    gate: pd.DataFrame,
    files: dict[str, list[str]],
    terminal_decision: str,
    args: argparse.Namespace,
) -> list[str]:
    final_agg = aggregate_final(tables["summary"])
    gate_row = gate.iloc[0] if not gate.empty else pd.Series(dtype=object)
    z_winner_key = str(gate_row.get("z_winner_key", "none"))
    z_winner_method = str(gate_row.get("z_winner_method", "none"))
    z_continuation = str(gate_row.get("z_1_5m_continuation_needed", "no"))
    can_enter = str(gate_row.get("can_enter_n4", "no"))
    no_z_candidate = str(gate_row.get("no_z_full_n4_policy_candidate", "no"))
    recommendation = str(gate_row.get("recommendation", "No recommendation generated."))
    z_decision = str(gate_row.get("z_hard_gate_decision", "No Z hard-gate decision generated."))

    lines = [
        "# Phase N3F/Z No-Z Full + Z Screen Report",
        "",
        "## Terminal Decision",
        "",
        f"`terminal_decision = {terminal_decision}`",
        "",
        "Phase N3F/Z complete. This is a repaired no-shield PPO no_z full rerun plus constrained z_i screening; it does not enter N4.",
        "",
        "## Engineering Facts",
        "",
        "- Phase N3R complete flag exists and N3R no_z 500k baseline is used for Z hard gates.",
        "- Phase N3.5 complete flag exists and repaired `GpsiObsWrapper` was used.",
        "- Gpsi checkpoint: `work_dirs/gpsi_heada_v1_nll/best.pth`; Gpsi was frozen under `eval()` with no trainable parameters.",
        "- EnvV2 core was not modified.",
        "- No shield, no action filtering, no dense safety cost, no learned R(s,a), and no Gpsi fine-tuning were used.",
        f"- Track 1 budget: N3F no_z `{args.expected_noz_steps}` PPO steps, seed 0.",
        f"- Track 2 budget: Z1/Z2 `{args.expected_z_steps}` PPO steps each, seed 0.",
        f"- Evaluation used `{args.expected_episodes}` episodes per scenario over six scenarios.",
        "- Logvar clip sanity: configs use `[-5, 3]`, already bounded tighter than `|logvar| <= 5`.",
        "",
        "## Configs",
        "",
        "- Track 1 `n3f_no_z_full`: `[obs_i, delta_hat_scaled, logvar_hat]`, dim 30.",
        "- Z1 `z_l2_scale_4`: `[obs_i, z_i / (||z_i||_2 + eps) * 4, delta_hat_scaled, logvar_hat]`, dim 94.",
        "- Z2 `z_layernorm_alpha_0p5`: `[obs_i, 0.5 * LayerNorm(z_i), delta_hat_scaled, logvar_hat]`, dim 94.",
        "- Z3 `z_proj16_layernorm`: not run in this stage; optional and resource-gated.",
        "",
        "## Main Results",
        "",
    ]
    lines.extend(table_md(final_agg, ["method_key", "success_rate", "collision_rate", "near_miss_rate", "progress", "raw_unsafe_action_rate", "mean_min_distance"]))
    lines.extend(["", "## Z Hard Gate", ""])
    gate_cols = [
        "method_key",
        "success_rate",
        "collision_rate",
        "passes_success_gate",
        "passes_collision_gate",
        "diagnostics_ok",
        "z_hard_gate_pass",
        "eligible_for_1_5m_continuation",
    ]
    lines.extend(table_md(gate[gate["method_key"].isin(Z_KEYS)], gate_cols, max_rows=10))
    lines.append(f"- N3R no_z 500k gate: success >= {N3R_SUCCESS_BASELINE:.4f} and collision <= {N3R_COLLISION_BASELINE:.4f}.")
    lines.append(f"- Z winner recommendation: `{z_winner_key}` / `{z_winner_method}`.")
    lines.append(f"- Z 1.5M continuation needed: {z_continuation}.")
    lines.append(f"- Z hard-gate decision: {z_decision}")
    lines.extend(["", "## N4 Candidate", ""])
    lines.append(f"- no_z full N4 policy candidate: {no_z_candidate}.")
    lines.append(f"- Can enter N4 now: {can_enter}.")
    lines.append(f"- Recommendation: {recommendation}")
    if can_enter != "yes":
        next_step = "run the Z winner 1.5M continuation before N4" if z_continuation == "yes" else "use N3F no_z full as the no-shield candidate at the next N4 decision point"
        lines.append(f"- Next step: {next_step}.")
    lines.extend(["", "## Attention Reference Comparison", ""])
    comp_cols = [
        "method_key",
        "scenario",
        "success_rate_config",
        "success_rate_attention",
        "delta_success_rate",
        "collision_rate_config",
        "collision_rate_attention",
        "delta_collision_rate",
    ]
    lines.extend(table_md(comparison.sort_values(["method_key", "scenario"]), comp_cols, max_rows=18))
    lines.extend(["", "## Gpsi Output Diagnostics", ""])
    gpsi_final = tables["gpsi"][tables["gpsi"]["checkpoint_label"].astype(str) == "final"].copy()
    gpsi_agg = (
        gpsi_final.groupby("method_key", dropna=False)
        .agg(
            delta_norm_1s_p95=("delta_norm_1s_p95", "max"),
            delta_norm_1s_max=("delta_norm_1s_max", "max"),
            logvar_xy_1s_mean=("logvar_xy_1s_mean", "mean"),
            logvar_xy_1s_span=("logvar_xy_1s_span", "max"),
            projected_std_radial_mean=("projected_std_radial_mean", "mean"),
            projected_std_relvel_mean=("projected_std_relvel_mean", "mean"),
            z_norm_raw_p95=("z_norm_raw_p95", "max"),
            z_norm_after_p95=("z_norm_after_p95", "max"),
            z_zero_norm_count_max=("z_zero_norm_count_max", "max"),
            history_valid_ratio_mean=("history_valid_ratio_mean", "mean"),
            inactive_forwarded_count_max=("inactive_forwarded_count_max", "max"),
        )
        .reset_index()
    )
    lines.extend(table_md(gpsi_agg, list(gpsi_agg.columns), max_rows=12))
    lines.extend(["", "## Feature Block Stats", ""])
    feature_final = tables["features"][(tables["features"]["checkpoint_label"].astype(str) == "final") & (tables["features"]["scenario"].astype(str) == "eval_flow_id")]
    feature_cols = ["method_key", "block", "z_transform", "not_applicable", "l2_norm_p95", "max_abs_p95", "nan_count", "inf_count"]
    lines.extend(table_md(feature_final.sort_values(["method_key", "block"]), feature_cols, max_rows=24))
    lines.extend(["", "## Raw Unsafe Diagnostics", ""])
    raw_final = tables["raw"][tables["raw"]["checkpoint_label"].astype(str) == "final"].copy()
    raw_agg = raw_final.groupby("method_key", dropna=False).agg(raw_unsafe_rate=("raw_unsafe_rate", "mean"), raw_min_predicted_cpa=("raw_min_predicted_cpa", "mean"), no_response_rate=("no_response_rate", "mean")).reset_index()
    lines.extend(table_md(raw_agg, list(raw_agg.columns)))
    lines.extend(["", "## Breakdown Outputs", ""])
    lines.append("Scenario, motion-mode, and threat-class breakdown CSVs were generated under `results/env_v2_phase_n3fz_noz_full_z_screen/tables/`.")
    lines.extend(["", "## Remaining Risks", ""])
    lines.append("- This stage uses seed 0 only.")
    lines.append("- N3F/Z still evaluates no-shield PPO; shield comparisons remain out of scope for this phase.")
    lines.extend(["", "## Artifacts", ""])
    for key in ["checkpoints", "tables", "plots", "logs", "flags"]:
        lines.append(f"### {key}")
        values = files.get(key, [])
        if values:
            lines.extend([f"- `{value}`" for value in values[:100]])
            if len(values) > 100:
                lines.append(f"- ... {len(values) - 100} more")
        else:
            lines.append("- none")
    return lines


def main() -> None:
    args = parse_args()
    result_dir = ROOT / args.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    try:
        validate_required_outputs(result_dir, args)
        tables = {
            "summary": read_csv(result_dir / "tables/phase_n3fz_eval_summary.csv"),
            "episodes": read_csv(result_dir / "tables/phase_n3fz_episode_metrics.csv"),
            "raw": read_csv(result_dir / "tables/phase_n3fz_raw_unsafe_action_summary.csv"),
            "gpsi": read_csv(result_dir / "tables/phase_n3fz_gpsi_output_summary.csv"),
            "features": read_csv(result_dir / "tables/phase_n3fz_aug_feature_block_stats.csv"),
            "train": read_csv(result_dir / "tables/phase_n3fz_train_curve.csv"),
            "config": read_csv(result_dir / "tables/phase_n3fz_config_manifest.csv"),
            "z_stats": read_csv(result_dir / "tables/phase_n3fz_z_stats.csv"),
        }
        validate_tables(tables, args)
        comparison = build_attention_comparison(tables["summary"], result_dir)
        gate = build_hard_gate(tables["summary"], tables["gpsi"], tables["features"], result_dir)
        generate_plots(result_dir, tables, gate)
        terminal_decision = "phase_n3fz_noz_full_z_screen_complete"
        write_text(result_dir / COMPLETE_FLAG, terminal_decision + "\n")
        write_text(result_dir / "phase_n3fz_status.txt", "complete\n")
        files = collect_files(result_dir)
        lines = report_lines(result_dir, tables, comparison, gate, files, terminal_decision, args)
        write_text(result_dir / "PHASE_N3FZ_NOZ_FULL_Z_SCREEN_REPORT.md", "\n".join(lines) + "\n")
        print(f"terminal_decision = {terminal_decision}", flush=True)
    except PhaseN3FZAnalysisStop as exc:
        write_stop(result_dir, exc.reason, exc.detail)
        raise SystemExit(2) from exc
    except Exception:
        detail = traceback.format_exc()
        write_stop(result_dir, "analysis_failed", detail)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
