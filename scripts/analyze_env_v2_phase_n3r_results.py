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


COMPLETE_FLAG = "PHASE_N3R_GPSI_PPO_RERUN_COMPLETE.flag"
STOP_FLAGS = {
    "phase_n3_5_missing": "PHASE_N3R_STOP_PHASE_N3_5_MISSING.flag",
    "gpsi_checkpoint_missing": "PHASE_N3R_STOP_GPSI_CHECKPOINT_MISSING.flag",
    "z_stats_failed": "PHASE_N3R_STOP_Z_STATS_FAILED.flag",
    "train_failed": "PHASE_N3R_STOP_TRAIN_FAILED.flag",
    "eval_failed": "PHASE_N3R_STOP_EVAL_FAILED.flag",
    "diagnostics_failed": "PHASE_N3R_STOP_DIAGNOSTICS_FAILED.flag",
    "analysis_failed": "PHASE_N3R_STOP_ANALYSIS_FAILED.flag",
}
CONFIG_KEYS = ["raw_z", "z_norm", "no_z"]
CONFIG_NAMES = {
    "raw_z": "gpsi_full_raw_z_repaired",
    "z_norm": "gpsi_full_z_norm_repaired",
    "no_z": "gpsi_no_z_repaired",
}
CHECKPOINT_DIRS = {
    "raw_z": ROOT / "checkpoints/env_v2_gpsi_heada_ppo_n3r_raw_z_s0",
    "z_norm": ROOT / "checkpoints/env_v2_gpsi_heada_ppo_n3r_z_norm_s0",
    "no_z": ROOT / "checkpoints/env_v2_gpsi_heada_ppo_n3r_no_z_s0",
}
SCENARIOS = [
    "eval_flow_id",
    "eval_flow_high_density",
    "eval_flow_high_speed",
    "eval_flow_high_threat",
    "eval_flow_mixed_ood",
    "eval_flow_sudden_threat",
]


class PhaseN3RAnalysisStop(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Phase N3R repaired Gpsi-PPO rerun outputs.")
    parser.add_argument("--result-dir", default="results/env_v2_phase_n3r_gpsi_ppo_rerun")
    parser.add_argument("--expected-episodes", type=int, default=50)
    parser.add_argument("--expected-train-steps", type=int, default=500_000)
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
    write_text(result_dir / "phase_n3r_status.txt", f"stopped:{flag_name}\n")
    write_text(
        result_dir / "PHASE_N3R_GPSI_PPO_RERUN_REPORT.md",
        "\n".join(
            [
                "# Phase N3R Gpsi-PPO Rerun Report",
                "",
                f"`terminal_decision = phase_n3r_stopped_{reason}`",
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
        "gpsi_checkpoint": ROOT / "work_dirs/gpsi_heada_v1_nll/best.pth",
        "z_stats_csv": result_dir / "tables/phase_n3r_z_stats.csv",
        "z_stats_npz": ROOT / "work_dirs/gpsi_heada_v1_nll/z_stats_train_split.npz",
        "config_manifest": result_dir / "tables/phase_n3r_config_manifest.csv",
        "schema_check": result_dir / "tables/phase_n3r_schema_check.csv",
        "train_curve": result_dir / "tables/phase_n3r_train_curve.csv",
        "eval_summary": result_dir / "tables/phase_n3r_eval_summary.csv",
        "checkpoint_eval_summary": result_dir / "tables/phase_n3r_checkpoint_eval_summary.csv",
        "episode_metrics": result_dir / "tables/phase_n3r_episode_metrics.csv",
        "scenario_breakdown": result_dir / "tables/phase_n3r_scenario_breakdown.csv",
        "motion_breakdown": result_dir / "tables/phase_n3r_motion_mode_breakdown.csv",
        "threat_breakdown": result_dir / "tables/phase_n3r_threat_class_breakdown.csv",
        "raw_unsafe": result_dir / "tables/phase_n3r_raw_unsafe_action_summary.csv",
        "gpsi_output": result_dir / "tables/phase_n3r_gpsi_output_summary.csv",
        "feature_blocks": result_dir / "tables/phase_n3r_aug_feature_block_stats.csv",
        "command_manifest": result_dir / "tables/phase_n3r_command_manifest.csv",
    }
    missing = [f"{name}: {rel(path)}" for name, path in required_paths.items() if not path.exists() or path.stat().st_size == 0]
    for key, ckpt_dir in CHECKPOINT_DIRS.items():
        required = [ckpt_dir / "checkpoint_250k.zip", ckpt_dir / "final.zip"]
        for path in required:
            if not path.exists() or path.stat().st_size == 0:
                missing.append(f"{key}_checkpoint: {rel(path)}")
    if missing:
        reason = "phase_n3_5_missing" if any("n3_5_flag" in item for item in missing) else "train_failed" if any("checkpoint" in item for item in missing) else "eval_failed"
        raise PhaseN3RAnalysisStop(reason, "missing required artifacts:\n" + "\n".join(missing))


def final_rows(summary: pd.DataFrame) -> pd.DataFrame:
    return summary[summary["checkpoint_label"].astype(str) == "final"].copy()


def final_config_rows(summary: pd.DataFrame) -> pd.DataFrame:
    return final_rows(summary[summary["method_key"].isin(CONFIG_KEYS)].copy())


def build_attention_comparison(summary: pd.DataFrame, result_dir: Path) -> pd.DataFrame:
    final = final_config_rows(summary)
    attention = summary[summary["method_key"] == "attention_full"].copy()
    if final.empty or attention.empty:
        raise PhaseN3RAnalysisStop("eval_failed", "missing final A/B/C rows or attention reference rows")
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
    merged.to_csv(result_dir / "tables/phase_n3r_attention_reference_comparison.csv", index=False)
    return merged


def build_winner(summary: pd.DataFrame, gpsi: pd.DataFrame, features: pd.DataFrame, result_dir: Path) -> pd.DataFrame:
    final = final_config_rows(summary)
    if final.empty:
        raise PhaseN3RAnalysisStop("eval_failed", "no final config rows for winner recommendation")
    agg = (
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
        recommendation = "No N3R config is eligible because diagnostics failed."
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
    out.to_csv(result_dir / "tables/phase_n3r_winner_recommendation.csv", index=False)
    return out


def validate_tables(tables: dict[str, pd.DataFrame], args: argparse.Namespace) -> None:
    summary = tables["summary"]
    expected_methods = set(CONFIG_KEYS) | {"attention_full"}
    methods = set(summary["method_key"].astype(str))
    missing_methods = expected_methods - methods
    if missing_methods:
        raise PhaseN3RAnalysisStop("eval_failed", f"missing methods in eval summary: {sorted(missing_methods)}")
    scenarios = set(summary["scenario"].astype(str))
    missing_scenarios = set(SCENARIOS) - scenarios
    if missing_scenarios:
        raise PhaseN3RAnalysisStop("eval_failed", f"missing scenarios: {sorted(missing_scenarios)}")
    episodes = tables["episodes"]
    counts = episodes.groupby(["method_key", "checkpoint_label", "scenario"]).size().reset_index(name="episodes")
    bad = counts[counts["episodes"] < int(args.expected_episodes)]
    if not bad.empty:
        raise PhaseN3RAnalysisStop("eval_failed", f"not enough eval episodes:\n{bad.to_string(index=False)}")
    for key in CONFIG_KEYS:
        labels = set(summary[summary["method_key"] == key]["checkpoint_label"].astype(str))
        if "250k" not in labels or "final" not in labels:
            raise PhaseN3RAnalysisStop("eval_failed", f"missing 250k/final eval rows for {key}: labels={sorted(labels)}")

    gpsi = tables["gpsi"]
    if gpsi.empty:
        raise PhaseN3RAnalysisStop("diagnostics_failed", "missing Gpsi output diagnostics")
    final_gpsi = gpsi[gpsi["checkpoint_label"].astype(str) == "final"].copy()
    delta_p95 = pd.to_numeric(final_gpsi.get("delta_norm_1s_p95", pd.Series(dtype=float)), errors="coerce")
    delta_max = pd.to_numeric(final_gpsi.get("delta_norm_1s_max", pd.Series(dtype=float)), errors="coerce")
    inactive = pd.to_numeric(final_gpsi.get("inactive_forwarded_count_max", pd.Series(dtype=float)), errors="coerce")
    if delta_p95.dropna().empty or float(delta_p95.max()) >= 100.0 or float(delta_max.max()) >= 1000.0:
        raise PhaseN3RAnalysisStop("diagnostics_failed", f"Gpsi output scale invalid: delta_p95_max={delta_p95.max()} delta_max={delta_max.max()}")
    if not inactive.dropna().empty and float(inactive.max()) > 0.0:
        raise PhaseN3RAnalysisStop("diagnostics_failed", f"inactive forwarded count is nonzero: {inactive.max()}")
    features = tables["features"]
    if features.empty:
        raise PhaseN3RAnalysisStop("diagnostics_failed", "missing augmented feature block stats")
    nonfinite = 0
    for col in ["nan_count", "inf_count"]:
        if col in features:
            nonfinite += int(pd.to_numeric(features[col], errors="coerce").fillna(0).sum())
    if nonfinite != 0:
        raise PhaseN3RAnalysisStop("diagnostics_failed", f"non-finite augmented feature values found: {nonfinite}")


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
    data = summary[summary["method_key"].isin(CONFIG_KEYS)].groupby(["method_key", "checkpoint_label"], dropna=False).agg(success=("success_rate", "mean"), collision=("collision_rate", "mean")).reset_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.8))
    for key in CONFIG_KEYS:
        group = data[data["method_key"] == key].copy()
        group["x"] = group["checkpoint_label"].map({"250k": 250_000, "final": 500_000}).fillna(group.get("checkpoint_step", 0))
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
    blocks = ["obs_i_12", "z_i_64_raw", "z_i_64_after_norm", "delta_hat_9_after_scale", "logvar_hat_9_clamped", "full_aug_obs"]
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


def generate_plots(result_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    plot_success_collision(tables["summary"], result_dir / "plots/success_collision_by_config.png")
    plot_checkpoint(tables["summary"], result_dir / "plots/checkpoint_success_collision_by_config.png")
    plot_raw_unsafe(tables["summary"], result_dir / "plots/raw_unsafe_by_config.png")
    plot_feature_blocks(tables["features"], result_dir / "plots/aug_feature_block_scale_by_config.png")
    plot_gpsi_delta(tables["gpsi"], result_dir / "plots/gpsi_delta_norm_by_config.png")
    plot_gpsi_logvar(tables["gpsi"], result_dir / "plots/gpsi_logvar_by_config.png")
    plot_train_curve(tables["train"], result_dir / "plots/train_reward_by_config.png")
    plot_scenario(tables["summary"], result_dir / "plots/scenario_breakdown_by_config.png")


def collect_files(result_dir: Path) -> dict[str, list[str]]:
    checkpoints: list[str] = []
    for path in CHECKPOINT_DIRS.values():
        checkpoints.extend(rel(item) for item in sorted(path.glob("*.zip")))
    return {
        "checkpoints": checkpoints,
        "tables": [rel(path) for path in sorted((result_dir / "tables").glob("*.csv"))],
        "plots": [rel(path) for path in sorted((result_dir / "plots").glob("*.png"))],
        "logs": [rel(path) for path in sorted((result_dir / "logs").glob("*.log"))] + [rel(result_dir / "phase_n3r_watcher.log")],
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
        "# Phase N3R Gpsi-PPO Rerun Report",
        "",
        "## Terminal Decision",
        "",
        f"`terminal_decision = {terminal_decision}`",
        "",
        "Phase N3R complete. This is a repaired no-shield PPO screening rerun with z-block ablation; it does not enter N4.",
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
    lines.append("Scenario, motion-mode, and threat-class breakdown CSVs were generated under `results/env_v2_phase_n3r_gpsi_ppo_rerun/tables/`.")
    lines.extend(["", "## Winner Recommendation", ""])
    lines.append(f"- Winner recommendation: `{winner_key}` / `{winner_method}`.")
    lines.append(f"- Recommendation: {recommendation}")
    lines.append(f"- Need full N3 1.5M rerun: {need_full}.")
    lines.append(f"- Can enter N4: {can_enter}.")
    if can_enter != "yes":
        lines.append("- Next step before N4: run the recommended repaired no-shield N3-full 1.5M baseline if the winner is promising; otherwise repair/ablate and rerun N3R.")
    lines.extend(["", "## Reasonable Inferences", ""])
    lines.append("- N3R is a screening comparison. A promising winner should be used for a full repaired N3 no-shield rerun before any shield comparison.")
    lines.append("- If z-normalized or no-z wins over raw-z, raw latent scale is a likely PPO input conditioning problem rather than a Gpsi HeadA output-scale failure.")
    lines.extend(["", "## Remaining Risks", ""])
    lines.append("- The run uses seed 0 only; the selected winner still needs longer-run and multi-seed confirmation before method-level claims.")
    lines.append("- The attention reference is a 1.5M baseline; N3R A/B/C are 500k screening runs and are not a final compute-matched comparison.")
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


def main() -> None:
    args = parse_args()
    result_dir = ROOT / args.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    try:
        validate_required_outputs(result_dir, args)
        tables = {
            "summary": read_csv(result_dir / "tables/phase_n3r_eval_summary.csv"),
            "episodes": read_csv(result_dir / "tables/phase_n3r_episode_metrics.csv"),
            "raw": read_csv(result_dir / "tables/phase_n3r_raw_unsafe_action_summary.csv"),
            "gpsi": read_csv(result_dir / "tables/phase_n3r_gpsi_output_summary.csv"),
            "features": read_csv(result_dir / "tables/phase_n3r_aug_feature_block_stats.csv"),
            "train": read_csv(result_dir / "tables/phase_n3r_train_curve.csv"),
            "config": read_csv(result_dir / "tables/phase_n3r_config_manifest.csv"),
            "z_stats": read_csv(result_dir / "tables/phase_n3r_z_stats.csv"),
        }
        validate_tables(tables, args)
        comparison = build_attention_comparison(tables["summary"], result_dir)
        winner = build_winner(tables["summary"], tables["gpsi"], tables["features"], result_dir)
        generate_plots(result_dir, tables)
        terminal_decision = "phase_n3r_gpsi_ppo_rerun_complete"
        write_text(result_dir / COMPLETE_FLAG, terminal_decision + "\n")
        write_text(result_dir / "phase_n3r_status.txt", "complete\n")
        files = collect_files(result_dir)
        lines = report_lines(result_dir, tables, comparison, winner, files, terminal_decision, args)
        write_text(result_dir / "PHASE_N3R_GPSI_PPO_RERUN_REPORT.md", "\n".join(lines) + "\n")
        print(f"terminal_decision = {terminal_decision}", flush=True)
    except PhaseN3RAnalysisStop as exc:
        write_stop(result_dir, exc.reason, exc.detail)
        raise SystemExit(2) from exc
    except Exception:
        detail = traceback.format_exc()
        write_stop(result_dir, "analysis_failed", detail)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
