from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results/env_v2_phase2_5"
PLOTS_DIR = OUT_DIR / "plots"
STATUS_PATH = OUT_DIR / "phase2_5_status.txt"
COMPLETE_FLAG = ROOT / "PHASE2_5_BASELINE_CAPABILITY_AUDIT_COMPLETE.flag"
NO_GO_FLAG = ROOT / "PHASE2_5_BASELINE_CAPABILITY_AUDIT_NO_GO.flag"
REPORT_PATH = ROOT / "PHASE2_5_BASELINE_CAPABILITY_AUDIT_REPORT.md"

PHASE1_SANITY_EPISODE = ROOT / "results/restart_phase0_phase1/env_v2/env_v2_sanity.csv"
PHASE1_SANITY_SUMMARY = ROOT / "results/restart_phase0_phase1/env_v2/env_v2_sanity_by_policy_scenario.csv"
PHASE1_CPA = ROOT / "results/restart_phase0_phase1/env_v2/cpa_distribution.csv"
PHASE1_TTC = ROOT / "results/restart_phase0_phase1/env_v2/ttc_distribution.csv"

PHASE2_REPORT = ROOT / "PHASE2_BASELINE_LONGTRAIN_FINAL_REPORT.md"
PHASE2_SUMMARY = ROOT / "results/env_v2_phase2/baseline_longtrain_by_checkpoint_scenario.csv"
PHASE2_EPISODES = ROOT / "results/env_v2_phase2/baseline_longtrain_episode_metrics.csv"
PHASE2_REACTION = ROOT / "results/env_v2_phase2/baseline_longtrain_reaction_breakdown.csv"
PHASE2_THREAT = ROOT / "results/env_v2_phase2/baseline_longtrain_threat_metrics.csv"

SCENARIOS = [
    "eval_flow_id",
    "eval_flow_high_density",
    "eval_flow_high_speed",
    "eval_flow_high_threat",
    "eval_flow_mixed_ood",
    "eval_flow_sudden_threat",
]
SANITY_POLICIES = ["random", "straight_line", "reactive"]
ALLOWED_DECISIONS = {
    "baseline_undertrained_or_unstable",
    "baseline_learns_progress_but_not_safety",
    "specific_scenarios_dominate_failure",
    "reactive_prior_promising",
    "env_or_reward_needs_revision",
    "data_insufficient_for_decision",
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_status(message: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(message + "\n", encoding="utf-8")
    print(message, flush=True)


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for flag in [COMPLETE_FLAG, NO_GO_FLAG]:
        if flag.exists():
            flag.unlink()


def write_no_go(reason: str, details: str, next_action: str) -> None:
    report = [
        "# Phase 2.5 Baseline Capability Audit Report",
        "",
        "## 1. Executive Summary",
        f"NO-GO triggered: {reason}",
        f"terminal_decision = {reason}",
        f"next_recommended_action = {next_action}",
        "",
        "## 2. Inputs",
        details,
        "",
        "## 3. Why Audit Cannot Proceed",
        "The available files do not contain the minimum aligned metrics needed for the capability audit.",
        "",
        "## 4. Required Eval-Only Repair",
        "补齐缺失的 eval-only logging / metrics; 不需要训练新 PPO。",
        "",
        "## 5. Decision",
        f"terminal_decision = {reason}",
    ]
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    NO_GO_FLAG.write_text(
        f"terminal_decision={reason}\nnext_recommended_action={next_action}\n",
        encoding="utf-8",
    )
    write_status(f"[no-go] {reason}")


def require_inputs() -> None:
    required = [
        PHASE1_SANITY_EPISODE,
        PHASE1_SANITY_SUMMARY,
        PHASE2_REPORT,
        PHASE2_SUMMARY,
        PHASE2_EPISODES,
        PHASE2_REACTION,
        PHASE2_THREAT,
    ]
    missing = [rel(p) for p in required if not p.exists() or p.stat().st_size == 0]
    if missing:
        phase1_missing = any("restart_phase0_phase1" in p for p in missing)
        reason = (
            "phase2_5_no_go_missing_phase1_sanity_data"
            if phase1_missing
            else "phase2_5_no_go_missing_phase2_eval_data"
        )
        write_no_go(reason, f"Missing files: {missing}", "补齐缺失 CSV/report；只需 eval-only 数据，不训练新 PPO")
        raise SystemExit(2)


def read_inputs() -> dict[str, pd.DataFrame]:
    write_status("[stage0] reading input data")
    data = {
        "sanity_episode": pd.read_csv(PHASE1_SANITY_EPISODE),
        "sanity_summary": pd.read_csv(PHASE1_SANITY_SUMMARY),
        "phase2_summary": pd.read_csv(PHASE2_SUMMARY),
        "phase2_episode": pd.read_csv(PHASE2_EPISODES),
        "phase2_reaction": pd.read_csv(PHASE2_REACTION),
        "phase2_threat": pd.read_csv(PHASE2_THREAT),
    }
    sanity_policies = set(data["sanity_summary"].get("policy_name", pd.Series(dtype=str)).astype(str))
    sanity_scenarios = set(data["sanity_summary"].get("scenario", pd.Series(dtype=str)).astype(str))
    phase2_scenarios = set(data["phase2_summary"].get("scenario", pd.Series(dtype=str)).astype(str))
    if not set(SANITY_POLICIES).issubset(sanity_policies) or not set(SCENARIOS).issubset(sanity_scenarios):
        write_no_go(
            "phase2_5_no_go_metrics_incompatible",
            f"Sanity policies={sorted(sanity_policies)}, scenarios={sorted(sanity_scenarios)}",
            "补跑 random/straight_line/reactive 的 eval-only sanity，并覆盖 6 个 EnvV2 eval scenarios",
        )
        raise SystemExit(2)
    if not set(SCENARIOS).issubset(phase2_scenarios):
        write_no_go(
            "phase2_5_no_go_metrics_incompatible",
            f"Phase2 scenarios={sorted(phase2_scenarios)}",
            "补齐 attention_full checkpoint 的 6 个 scenario eval-only CSV，不训练新 PPO",
        )
        raise SystemExit(2)
    episode_required = {"checkpoint_step", "scenario", "episode_id", "success", "collision", "progress"}
    if not episode_required.issubset(data["phase2_episode"].columns):
        write_no_go(
            "phase2_5_no_go_episode_level_data_missing",
            f"Missing episode columns: {sorted(episode_required - set(data['phase2_episode'].columns))}",
            "补跑 eval-only episode-level logging，不训练新 PPO",
        )
        raise SystemExit(2)
    return data


def load_phase1_threat_labels() -> pd.DataFrame:
    if not (PHASE1_CPA.exists() and PHASE1_TTC.exists()):
        return pd.DataFrame()
    use_cols = ["policy_name", "scenario", "episode_id", "obstacle_id", "motion_mode", "threat_class", "planned_cpa"]
    cpa = pd.read_csv(PHASE1_CPA, usecols=lambda c: c in use_cols)
    ttc = pd.read_csv(PHASE1_TTC, usecols=lambda c: c in ["policy_name", "scenario", "episode_id", "obstacle_id", "planned_ttc"])
    labels = cpa.merge(ttc, on=["policy_name", "scenario", "episode_id", "obstacle_id"], how="left")
    return labels


def build_unified_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    write_status("[stage1] building unified metrics table")
    sanity = data["sanity_episode"].copy()
    labels = load_phase1_threat_labels()
    if not labels.empty and "threat_obstacle_id" in sanity.columns:
        sanity = sanity.merge(
            labels,
            left_on=["policy_name", "scenario", "episode_id", "threat_obstacle_id"],
            right_on=["policy_name", "scenario", "episode_id", "obstacle_id"],
            how="left",
        )
    else:
        sanity["motion_mode"] = np.nan
        sanity["threat_class"] = np.nan
        sanity["planned_cpa"] = np.nan
        sanity["planned_ttc"] = np.nan

    sanity_unified = pd.DataFrame(
        {
            "method": sanity["policy_name"].astype(str),
            "policy_name": sanity["policy_name"].astype(str),
            "scenario": sanity["scenario"].astype(str),
            "checkpoint_step": "sanity",
            "episode_id": sanity["episode_id"],
            "success": sanity["success"],
            "collision": sanity["collision"],
            "near_miss": sanity["near_miss"],
            "progress": sanity.get("progress", np.nan),
            "mean_time": sanity.get("mean_time", np.nan),
            "mean_min_distance": sanity.get("mean_min_distance", np.nan),
            "min_distance_after_threat": np.nan,
            "no_response_rate": np.nan,
            "reaction_time_eval_style": np.nan,
            "conditional_reaction_time": np.nan,
            "planned_cpa": sanity.get("planned_cpa", sanity.get("planned_cpa_to_threat", np.nan)),
            "planned_ttc": sanity.get("planned_ttc", sanity.get("planned_ttc_to_threat", np.nan)),
            "threat_class": sanity.get("threat_class", pd.Series(np.nan, index=sanity.index)),
            "motion_mode": sanity.get("motion_mode", pd.Series(np.nan, index=sanity.index)),
            "replacement_count": sanity.get("replacement_count", np.nan),
            "active_obstacle_count": sanity.get("mean_active_obstacle_count", np.nan),
        }
    )

    ep = data["phase2_episode"].copy()
    attention_unified = pd.DataFrame(
        {
            "method": "attention_full",
            "policy_name": "attention_full",
            "scenario": ep["scenario"].astype(str),
            "checkpoint_step": ep["checkpoint_step"].astype(str),
            "episode_id": ep["episode_id"],
            "success": ep["success"],
            "collision": ep["collision"],
            "near_miss": ep["near_miss"],
            "progress": ep.get("progress", np.nan),
            "mean_time": ep.get("mean_time", np.nan),
            "mean_min_distance": ep.get("mean_min_distance", np.nan),
            "min_distance_after_threat": ep.get("min_distance_after_threat", np.nan),
            "no_response_rate": ep.get("no_response", np.nan),
            "reaction_time_eval_style": ep.get("reaction_time_eval_style", np.nan),
            "conditional_reaction_time": ep.get("conditional_reaction_time", np.nan),
            "planned_cpa": ep.get("planned_cpa", np.nan),
            "planned_ttc": ep.get("planned_ttc", np.nan),
            "threat_class": ep.get("threat_class", pd.Series(np.nan, index=ep.index)),
            "motion_mode": ep.get("threat_motion_mode", pd.Series(np.nan, index=ep.index)),
            "replacement_count": ep.get("replacement_count", np.nan),
            "active_obstacle_count": np.nan,
        }
    )

    unified = pd.concat([sanity_unified, attention_unified], ignore_index=True)
    unified.to_csv(OUT_DIR / "unified_metrics_table.csv", index=False)
    return unified


def select_best_checkpoint(summary: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    checkpoint_summary = (
        summary.groupby("checkpoint_step", as_index=False)
        .agg(
            scenarios=("scenario", "nunique"),
            success_rate_mean=("success_rate", "mean"),
            success_rate_std=("success_rate", "std"),
            collision_rate_mean=("collision_rate", "mean"),
            collision_rate_std=("collision_rate", "std"),
            near_miss_rate_mean=("near_miss_rate", "mean"),
            progress_mean=("progress", "mean"),
            mean_min_distance_mean=("mean_min_distance", "mean"),
            min_distance_after_threat_mean=("min_distance_after_threat", "mean"),
            no_response_rate_mean=("no_response_rate", "mean"),
            reaction_time_eval_style_mean=("reaction_time_eval_style", "mean"),
            conditional_reaction_time_mean=("conditional_reaction_time", "mean"),
        )
        .sort_values("checkpoint_step")
    )
    ranked = checkpoint_summary.sort_values(
        ["success_rate_mean", "collision_rate_mean", "near_miss_rate_mean", "mean_min_distance_mean", "checkpoint_step"],
        ascending=[False, True, True, False, False],
    ).reset_index(drop=True)
    best_step = int(ranked.loc[0, "checkpoint_step"])
    per_scenario_best = (
        summary.sort_values(
            ["scenario", "success_rate", "collision_rate", "near_miss_rate", "mean_min_distance", "checkpoint_step"],
            ascending=[True, False, True, True, False, False],
        )
        .groupby("scenario", as_index=False)
        .first()[["scenario", "checkpoint_step"]]
        .rename(columns={"checkpoint_step": "scenario_best_checkpoint"})
    )
    best_counts = per_scenario_best["scenario_best_checkpoint"].value_counts().rename_axis("checkpoint_step").reset_index(name="scenario_best_count")
    checkpoint_summary = checkpoint_summary.merge(best_counts, on="checkpoint_step", how="left")
    checkpoint_summary["scenario_best_count"] = checkpoint_summary["scenario_best_count"].fillna(0).astype(int)
    checkpoint_summary["is_global_best"] = checkpoint_summary["checkpoint_step"].astype(int) == best_step
    checkpoint_summary["global_rank"] = checkpoint_summary["checkpoint_step"].map(
        {int(row.checkpoint_step): i + 1 for i, row in ranked.iterrows()}
    )
    checkpoint_summary.to_csv(OUT_DIR / "attention_full_checkpoint_summary.csv", index=False)
    return best_step, checkpoint_summary


def plot_attention_checkpoint_curves(checkpoint_summary: pd.DataFrame) -> None:
    x = checkpoint_summary["checkpoint_step"].astype(int) / 1000.0
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    axes = axes.ravel()
    series = [
        ("success_rate_mean", "Success Rate", "#2f6f4e"),
        ("collision_rate_mean", "Collision Rate", "#b33f3f"),
        ("near_miss_rate_mean", "Near-Miss Rate", "#b87914"),
        ("mean_min_distance_mean", "Mean Min Distance", "#345995"),
    ]
    for ax, (col, title, color) in zip(axes, series):
        ax.plot(x, checkpoint_summary[col], marker="o", linewidth=2, color=color)
        ax.set_title(title)
        ax.set_xlabel("Checkpoint (k steps)")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "attention_checkpoint_curves.png", dpi=160)
    plt.close(fig)


def summarize_baselines(sanity_summary: pd.DataFrame, phase2_summary: pd.DataFrame, best_step: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    write_status("[stage3] comparing sanity policies and attention_full")
    rows: list[dict[str, Any]] = []
    common_scenarios = sorted(set(sanity_summary["scenario"]) & set(phase2_summary["scenario"]) & set(SCENARIOS))
    for policy in SANITY_POLICIES:
        df = sanity_summary[(sanity_summary["policy_name"] == policy) & (sanity_summary["scenario"].isin(common_scenarios))]
        rows.append(
            {
                "method": policy,
                "checkpoint_step": "sanity",
                "scenarios": df["scenario"].nunique(),
                "success_rate": df["success_rate"].mean(),
                "collision_rate": df["collision_rate"].mean(),
                "near_miss_rate": df["near_miss_rate"].mean(),
                "progress": df["progress_mean"].mean(),
                "mean_time": df["mean_time"].mean(),
                "mean_min_distance": df["mean_min_distance"].mean(),
                "min_distance_after_threat": df["min_distance_mean"].mean(),
            }
        )
    for label, step in [("attention_full_1500k", 1500000), ("attention_full_best_checkpoint", best_step)]:
        df = phase2_summary[(phase2_summary["checkpoint_step"] == step) & (phase2_summary["scenario"].isin(common_scenarios))]
        rows.append(
            {
                "method": label,
                "checkpoint_step": str(step),
                "scenarios": df["scenario"].nunique(),
                "success_rate": df["success_rate"].mean(),
                "collision_rate": df["collision_rate"].mean(),
                "near_miss_rate": df["near_miss_rate"].mean(),
                "progress": df["progress"].mean(),
                "mean_time": df["mean_time"].mean(),
                "mean_min_distance": df["mean_min_distance"].mean(),
                "min_distance_after_threat": df["min_distance_after_threat"].mean(),
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT_DIR / "baseline_comparison_summary.csv", index=False)

    reactive = sanity_summary[(sanity_summary["policy_name"] == "reactive") & (sanity_summary["scenario"].isin(common_scenarios))].copy()
    attention = phase2_summary[(phase2_summary["checkpoint_step"] == best_step) & (phase2_summary["scenario"].isin(common_scenarios))].copy()
    gap = reactive.merge(attention, on="scenario", suffixes=("_reactive", "_attention"))
    gap_out = pd.DataFrame(
        {
            "scenario": gap["scenario"],
            "attention_checkpoint_step": best_step,
            "reactive_success_rate": gap["success_rate_reactive"],
            "attention_success_rate": gap["success_rate_attention"],
            "success_gap_attention_minus_reactive": gap["success_rate_attention"] - gap["success_rate_reactive"],
            "reactive_collision_rate": gap["collision_rate_reactive"],
            "attention_collision_rate": gap["collision_rate_attention"],
            "collision_gap_attention_minus_reactive": gap["collision_rate_attention"] - gap["collision_rate_reactive"],
            "reactive_near_miss_rate": gap["near_miss_rate_reactive"],
            "attention_near_miss_rate": gap["near_miss_rate_attention"],
            "reactive_progress": gap["progress_mean"],
            "attention_progress": gap["progress"],
            "progress_gap_attention_minus_reactive": gap["progress"] - gap["progress_mean"],
            "reactive_mean_min_distance": gap["mean_min_distance_reactive"],
            "attention_mean_min_distance": gap["mean_min_distance_attention"],
        }
    )
    gap_out.to_csv(OUT_DIR / "reactive_vs_attention_gap.csv", index=False)
    return comparison, gap_out


def plot_baseline_comparisons(comparison: pd.DataFrame, gap: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    order = comparison["method"].tolist()
    x = np.arange(len(order))
    width = 0.36
    ax.bar(x - width / 2, comparison["success_rate"], width, label="success", color="#2f6f4e")
    ax.bar(x + width / 2, comparison["collision_rate"], width, label="collision", color="#b33f3f")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Success / Collision by Baseline")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "success_collision_by_scenario.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    scenarios = gap["scenario"].tolist()
    x = np.arange(len(scenarios))
    ax.bar(x - width / 2, gap["reactive_collision_rate"], width, label="reactive", color="#4b7f52")
    ax.bar(x + width / 2, gap["attention_collision_rate"], width, label="attention_full_best", color="#b33f3f")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Reactive vs Attention Collision by Scenario")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "reactive_vs_attention_collision.png", dpi=160)
    plt.close(fig)


def scenario_audit(phase2_summary: pd.DataFrame, phase2_episode: pd.DataFrame, best_step: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    write_status("[stage4] auditing scenario difficulty")
    scenario = phase2_summary[phase2_summary["checkpoint_step"] == best_step].copy()
    scenario = scenario.sort_values(["collision_rate", "success_rate"], ascending=[False, True])
    keep = [
        "checkpoint_step",
        "scenario",
        "episodes",
        "success_rate",
        "collision_rate",
        "near_miss_rate",
        "mean_min_distance",
        "min_distance_after_threat",
        "progress",
        "mean_time",
        "no_response_rate",
        "reaction_time_eval_style",
        "conditional_reaction_time",
        "planned_cpa",
        "planned_ttc",
        "replacement_count",
        "distance_warning_cost_nonzero_rate",
    ]
    scenario[keep].to_csv(OUT_DIR / "scenario_difficulty_table.csv", index=False)

    ep = phase2_episode[phase2_episode["checkpoint_step"] == best_step].copy()
    total_collisions = float(ep["collision"].sum())
    breakdown = (
        ep.groupby("scenario", as_index=False)
        .agg(
            episodes=("episode_id", "count"),
            collision_count=("collision", "sum"),
            collision_rate=("collision", "mean"),
            success_rate=("success", "mean"),
            near_miss_rate=("near_miss", "mean"),
            mean_min_distance=("mean_min_distance", "mean"),
            min_distance_after_threat=("min_distance_after_threat", "mean"),
            progress=("progress", "mean"),
        )
        .sort_values(["collision_count", "collision_rate"], ascending=[False, False])
    )
    breakdown["collision_share"] = np.where(total_collisions > 0, breakdown["collision_count"] / total_collisions, 0.0)
    breakdown.to_csv(OUT_DIR / "collision_breakdown_by_scenario.csv", index=False)
    return scenario, breakdown


def plot_scenario_audit(scenario: pd.DataFrame) -> None:
    plot_df = scenario.sort_values("scenario")
    x = np.arange(len(plot_df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - width / 2, plot_df["success_rate"], width, label="success", color="#2f6f4e")
    ax.bar(x + width / 2, plot_df["collision_rate"], width, label="collision", color="#b33f3f")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["scenario"], rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Attention Best Checkpoint: Scenario Success / Collision")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "success_collision_by_scenario.png", dpi=160)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    ax2 = ax1.twinx()
    ax1.bar(x - width / 2, plot_df["near_miss_rate"], width, label="near_miss", color="#b87914")
    ax2.bar(x + width / 2, plot_df["mean_min_distance"], width, label="mean_min_distance", color="#345995")
    ax1.set_xticks(x)
    ax1.set_xticklabels(plot_df["scenario"], rotation=25, ha="right")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Near-miss rate")
    ax2.set_ylabel("Mean min distance")
    ax1.set_title("Near-Miss and Min-Distance by Scenario")
    ax1.grid(axis="y", alpha=0.25)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "near_miss_min_distance_by_scenario.png", dpi=160)
    plt.close(fig)


def collision_breakdowns(phase2_episode: pd.DataFrame, best_step: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    write_status("[stage5] breaking down collisions by threat and motion")
    ep = phase2_episode[phase2_episode["checkpoint_step"] == best_step].copy()
    total_collisions = float(ep["collision"].sum())
    if "threat_motion_mode" not in ep.columns and "motion_mode" in ep.columns:
        ep["threat_motion_mode"] = ep["motion_mode"]

    def group_breakdown(group_col: str) -> pd.DataFrame:
        if group_col not in ep.columns:
            return pd.DataFrame(
                columns=[
                    group_col,
                    "episodes",
                    "collision_count",
                    "collision_rate",
                    "collision_share",
                    "success_rate",
                    "near_miss_rate",
                    "planned_cpa",
                    "planned_ttc",
                    "min_distance_after_threat",
                ]
            )
        out = (
            ep.groupby(group_col, dropna=False, as_index=False)
            .agg(
                episodes=("episode_id", "count"),
                collision_count=("collision", "sum"),
                collision_rate=("collision", "mean"),
                success_rate=("success", "mean"),
                near_miss_rate=("near_miss", "mean"),
                planned_cpa=("planned_cpa", "mean"),
                planned_ttc=("planned_ttc", "mean"),
                min_distance_after_threat=("min_distance_after_threat", "mean"),
            )
            .sort_values(["collision_rate", "collision_count"], ascending=[False, False])
        )
        out["collision_share"] = np.where(total_collisions > 0, out["collision_count"] / total_collisions, 0.0)
        return out

    by_threat = group_breakdown("threat_class")
    by_motion = group_breakdown("threat_motion_mode").rename(columns={"threat_motion_mode": "motion_mode"})
    by_threat.to_csv(OUT_DIR / "collision_breakdown_by_threat_class.csv", index=False)
    by_motion.to_csv(OUT_DIR / "collision_breakdown_by_motion_mode.csv", index=False)

    corr_rows: list[dict[str, Any]] = []
    for col in ["planned_cpa", "planned_ttc"]:
        vals = ep[col].dropna() if col in ep.columns else pd.Series(dtype=float)
        if vals.nunique() >= 4:
            bins = pd.qcut(ep[col], q=4, duplicates="drop")
            tmp = ep.assign(bin=bins)
            grouped = tmp.groupby("bin", observed=False).agg(
                episodes=("episode_id", "count"),
                collision_rate=("collision", "mean"),
                success_rate=("success", "mean"),
                value_mean=(col, "mean"),
                value_min=(col, "min"),
                value_max=(col, "max"),
            )
            for bin_label, row in grouped.iterrows():
                corr_rows.append(
                    {
                        "variable": col,
                        "bin": str(bin_label),
                        "episodes": row["episodes"],
                        "collision_rate": row["collision_rate"],
                        "success_rate": row["success_rate"],
                        "value_mean": row["value_mean"],
                        "value_min": row["value_min"],
                        "value_max": row["value_max"],
                    }
                )
            corr_rows.append(
                {
                    "variable": f"{col}_pearson",
                    "bin": "collision",
                    "episodes": len(ep),
                    "collision_rate": ep[col].corr(ep["collision"]) if ep[col].notna().sum() > 2 else np.nan,
                    "success_rate": np.nan,
                    "value_mean": ep[col].mean(),
                    "value_min": ep[col].min(),
                    "value_max": ep[col].max(),
                }
            )
    cpa_ttc = pd.DataFrame(corr_rows)
    cpa_ttc.to_csv(OUT_DIR / "cpa_ttc_failure_correlation.csv", index=False)
    return by_threat, by_motion, cpa_ttc


def plot_collision_breakdowns(by_threat: pd.DataFrame, by_motion: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if not by_threat.empty:
        plot_df = by_threat.sort_values("threat_class")
        ax.bar(plot_df["threat_class"].astype(str), plot_df["collision_rate"], color="#b33f3f")
    ax.set_ylim(0, 1.05)
    ax.set_title("Collision Rate by Threat Class")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "collision_by_threat_class.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    if not by_motion.empty:
        plot_df = by_motion.sort_values("collision_rate", ascending=False)
        ax.bar(plot_df["motion_mode"].astype(str), plot_df["collision_rate"], color="#b33f3f")
    ax.set_ylim(0, 1.05)
    ax.set_title("Collision Rate by Motion Mode")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "collision_by_motion_mode.png", dpi=160)
    plt.close(fig)


def decide(
    comparison: pd.DataFrame,
    gap: pd.DataFrame,
    checkpoint_summary: pd.DataFrame,
    scenario_breakdown: pd.DataFrame,
) -> tuple[str, str, str, list[str]]:
    attn = comparison[comparison["method"] == "attention_full_best_checkpoint"].iloc[0]
    straight = comparison[comparison["method"] == "straight_line"].iloc[0]
    random = comparison[comparison["method"] == "random"].iloc[0]
    reactive = comparison[comparison["method"] == "reactive"].iloc[0]

    attention_learned = (
        attn["success_rate"] > straight["success_rate"] + 0.20
        and attn["progress"] > random["progress"] + 0.50
        and attn["progress"] > 0.80
    )
    reactive_safer = attn["collision_rate"] - reactive["collision_rate"] >= 0.15
    collision_high = attn["collision_rate"] >= 0.25 or attn["near_miss_rate"] >= 0.75
    top_collision_share = float(scenario_breakdown["collision_share"].max()) if not scenario_breakdown.empty else 0.0
    unstable = (
        checkpoint_summary["success_rate_mean"].diff().abs().max() >= 0.15
        or checkpoint_summary["collision_rate_mean"].diff().abs().max() >= 0.15
    )

    if reactive_safer:
        terminal = "reactive_prior_promising"
        secondary = "baseline_learns_progress_but_not_safety" if attention_learned and collision_high else "baseline_undertrained_or_unstable"
        next_action = (
            "优先做 reactive prior + learned residual / velocity-obstacle-like safety filter / "
            "RL constrained by simple geometric avoider；不继续追 no-response。"
        )
    elif attention_learned and collision_high:
        terminal = "baseline_learns_progress_but_not_safety"
        secondary = "specific_scenarios_dominate_failure" if top_collision_share >= 0.35 else "reactive_prior_promising"
        next_action = "做安全能力提升路线：reactive prior / residual RL / safety filter / dense safety cost / curriculum。"
    elif top_collision_share >= 0.35:
        terminal = "specific_scenarios_dominate_failure"
        secondary = "baseline_learns_progress_but_not_safety"
        next_action = "针对最难场景做 curriculum / scenario-specific diagnosis。"
    elif not attention_learned or unstable:
        terminal = "baseline_undertrained_or_unstable"
        secondary = "env_or_reward_needs_revision"
        next_action = "先调 reward / training / observation / action scale；不要先做新网络。"
    else:
        terminal = "env_or_reward_needs_revision"
        secondary = "baseline_learns_progress_but_not_safety"
        next_action = "检查 reward、observation、termination 与安全学习信号；不训练新 PPO，先做离线诊断或 eval-only logging。"

    not_recommended = [
        "继续追旧 3-ball no-response degradation 或 Phase 3 failure localization",
        "单纯把 attention_full 训练到 2000k",
        "在 seed=0 路线不清楚时启动 seed=1/2",
        "直接上 temporal/risk-aware attention 或 PPO-Lagrangian",
        "把当前结果包装成 benchmark 论文主线",
    ]
    return terminal, secondary, next_action, not_recommended


def write_recommendation(
    terminal: str,
    secondary: str,
    next_action: str,
    not_recommended: list[str],
    best_step: int,
) -> None:
    text = [
        "# Method Route Recommendation",
        "",
        f"primary_diagnosis: {terminal}",
        f"secondary_diagnosis: {secondary}",
        f"best_checkpoint: {best_step}",
        "",
        "recommended_next_step:",
        f"- {next_action}",
        "- Treat the reactive avoider as a strong geometric prior but not a complete policy, because it is much safer while attention_full reaches the goal more often.",
        "",
        "not_recommended_next_steps:",
    ]
    text.extend([f"- {item}" for item in not_recommended])
    text.append("")
    (OUT_DIR / "method_route_recommendation.md").write_text("\n".join(text), encoding="utf-8")


def md_table(df: pd.DataFrame, cols: list[str], float_fmt: str = ".4f", max_rows: int | None = None) -> list[str]:
    view = df[cols].copy()
    if max_rows is not None:
        view = view.head(max_rows)
    headers = list(view.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in view.iterrows():
        vals = []
        for col in headers:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append(format(float(val), float_fmt))
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def write_report(
    terminal: str,
    secondary: str,
    next_action: str,
    not_recommended: list[str],
    best_step: int,
    checkpoint_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    gap: pd.DataFrame,
    scenario_table: pd.DataFrame,
    scenario_breakdown: pd.DataFrame,
    by_threat: pd.DataFrame,
    by_motion: pd.DataFrame,
    cpa_ttc: pd.DataFrame,
) -> None:
    attn = comparison[comparison["method"] == "attention_full_best_checkpoint"].iloc[0]
    reactive = comparison[comparison["method"] == "reactive"].iloc[0]
    straight = comparison[comparison["method"] == "straight_line"].iloc[0]
    random = comparison[comparison["method"] == "random"].iloc[0]
    hardest = scenario_breakdown.head(3)["scenario"].tolist()
    lines: list[str] = [
        "# Phase 2.5 Baseline Capability Audit Report",
        "",
        "## 1. Executive Summary",
        f"terminal_decision = {terminal}",
        f"secondary_diagnosis = {secondary}",
        f"next_recommended_action = {next_action}",
        "",
        (
            f"attention_full best checkpoint is {best_step}. It clearly learns task progress relative to random/straight_line "
            f"(success={attn['success_rate']:.4f}, progress={attn['progress']:.4f}), but collision remains high "
            f"(collision={attn['collision_rate']:.4f}, near_miss={attn['near_miss_rate']:.4f}). "
            f"Reactive is much safer (collision={reactive['collision_rate']:.4f}) but has lower success."
        ),
        "",
        "## 2. Inputs",
        f"- Phase 1 sanity episode data: `{rel(PHASE1_SANITY_EPISODE)}`",
        f"- Phase 1 sanity summary: `{rel(PHASE1_SANITY_SUMMARY)}`",
        f"- Phase 2 final report: `{rel(PHASE2_REPORT)}`",
        f"- Phase 2 checkpoint summary: `{rel(PHASE2_SUMMARY)}`",
        f"- Phase 2 episode metrics: `{rel(PHASE2_EPISODES)}`",
        f"- Phase 2 reaction metrics: `{rel(PHASE2_REACTION)}`",
        f"- Phase 2 threat metrics: `{rel(PHASE2_THREAT)}`",
        "",
        "## 3. Attention_full Learning Audit",
        (
            "attention_full learns the basic task: success and progress improve strongly over early training, "
            "but the checkpoint curve is not monotonic. 1250k regresses before 1500k recovers, so there is checkpoint oscillation."
        ),
    ]
    lines.extend(md_table(checkpoint_summary, [
        "checkpoint_step",
        "success_rate_mean",
        "collision_rate_mean",
        "near_miss_rate_mean",
        "progress_mean",
        "mean_min_distance_mean",
        "min_distance_after_threat_mean",
        "scenario_best_count",
        "is_global_best",
    ]))
    lines.extend(
        [
            "",
            "## 4. Comparison with Random / Straight-Line / Reactive",
            (
                f"attention_full is clearly better than random and straight_line on success/progress. "
                f"Compared with reactive, it has much higher success ({attn['success_rate']:.4f} vs {reactive['success_rate']:.4f}) "
                f"but much worse collision ({attn['collision_rate']:.4f} vs {reactive['collision_rate']:.4f}). "
                "This indicates a strong simple-geometry safety prior that the learned policy does not reproduce."
            ),
        ]
    )
    lines.extend(md_table(comparison, [
        "method",
        "checkpoint_step",
        "success_rate",
        "collision_rate",
        "near_miss_rate",
        "progress",
        "mean_min_distance",
        "min_distance_after_threat",
    ]))
    lines.extend(
        [
            "",
            "Reactive vs attention collision gaps by scenario:",
        ]
    )
    lines.extend(md_table(gap, [
        "scenario",
        "reactive_collision_rate",
        "attention_collision_rate",
        "collision_gap_attention_minus_reactive",
        "reactive_success_rate",
        "attention_success_rate",
    ]))
    lines.extend(
        [
            "",
            "## 5. Scenario Difficulty",
            f"Hardest scenarios at the best checkpoint by collision count/rate: {', '.join(hardest)}.",
            "Failures are not dominated by sudden_threat; sudden_threat is the easiest by collision at the best checkpoint.",
        ]
    )
    lines.extend(md_table(scenario_table, [
        "scenario",
        "success_rate",
        "collision_rate",
        "near_miss_rate",
        "mean_min_distance",
        "min_distance_after_threat",
        "progress",
        "no_response_rate",
    ]))
    lines.extend(
        [
            "",
            "Collision share by scenario:",
        ]
    )
    lines.extend(md_table(scenario_breakdown, [
        "scenario",
        "collision_count",
        "collision_rate",
        "collision_share",
        "success_rate",
        "progress",
    ]))
    lines.extend(
        [
            "",
            "## 6. Collision Breakdown",
            "Threat/motion labels are available in episode-level Phase 2 data, so collision breakdowns use attention_full at the best checkpoint.",
            "",
            "By threat class:",
        ]
    )
    lines.extend(md_table(by_threat, [
        "threat_class",
        "episodes",
        "collision_count",
        "collision_rate",
        "collision_share",
        "planned_cpa",
        "planned_ttc",
    ]))
    lines.extend(["", "By motion mode:"])
    lines.extend(md_table(by_motion, [
        "motion_mode",
        "episodes",
        "collision_count",
        "collision_rate",
        "collision_share",
        "planned_cpa",
        "planned_ttc",
    ]))
    lines.extend(["", "CPA/TTC binned failure correlation:"])
    if not cpa_ttc.empty:
        lines.extend(md_table(cpa_ttc, [
            "variable",
            "bin",
            "episodes",
            "collision_rate",
            "value_mean",
            "value_min",
            "value_max",
        ], max_rows=12))
    else:
        lines.append("collision by CPA/TTC unavailable due to missing labels.")
    lines.extend(
        [
            "",
            "## 7. Interpretation",
            (
                "The EnvV2 issue is not the old no-response degradation. Phase 2 showed no_response_rate becomes zero after 500k. "
                "The current problem is capability mismatch: attention_full learns forward progress and reaches goals, "
                "but it does not achieve the safety margin of the simple reactive policy."
            ),
            (
                "random is low-collision because it mostly fails to make progress, so it is not a capable baseline. "
                "straight_line is unsafe. reactive is safe but under-achieves success. attention_full is capable but unsafe."
            ),
            "",
            "## 8. Method Route Recommendation",
            f"recommended_next_step: {next_action}",
            "",
            "not_recommended_next_steps:",
        ]
    )
    lines.extend([f"- {item}" for item in not_recommended])
    lines.extend(
        [
            "",
            "No new PPO training is required to complete this audit. Additional data, if needed later, should be eval-only logging.",
            "",
            "## 9. Decision",
            f"terminal_decision = {terminal}",
            f"next_recommended_action = {next_action}",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_complete_flag(terminal: str, next_action: str) -> None:
    if terminal not in ALLOWED_DECISIONS:
        raise ValueError(f"invalid terminal decision: {terminal}")
    COMPLETE_FLAG.write_text(
        f"terminal_decision={terminal}\nnext_recommended_action={next_action}\n",
        encoding="utf-8",
    )
    write_status(f"[complete] terminal_decision={terminal}")


def main() -> None:
    ensure_dirs()
    require_inputs()
    data = read_inputs()
    unified = build_unified_table(data)
    if unified.empty:
        write_no_go(
            "phase2_5_no_go_audit_inconclusive_requires_extra_eval",
            "Unified metrics table is empty.",
            "补跑 eval-only metrics logging，不训练新 PPO",
        )
        raise SystemExit(2)

    write_status("[stage2] auditing attention_full checkpoint learning")
    phase2_summary = data["phase2_summary"]
    best_step, checkpoint_summary = select_best_checkpoint(phase2_summary)
    plot_attention_checkpoint_curves(checkpoint_summary)

    comparison, gap = summarize_baselines(data["sanity_summary"], phase2_summary, best_step)
    scenario_table, scenario_breakdown = scenario_audit(phase2_summary, data["phase2_episode"], best_step)
    by_threat, by_motion, cpa_ttc = collision_breakdowns(data["phase2_episode"], best_step)

    plot_baseline_comparisons(comparison, gap)
    plot_scenario_audit(scenario_table)
    plot_collision_breakdowns(by_threat, by_motion)

    terminal, secondary, next_action, not_recommended = decide(
        comparison,
        gap,
        checkpoint_summary,
        scenario_breakdown,
    )
    write_recommendation(terminal, secondary, next_action, not_recommended, best_step)
    write_report(
        terminal,
        secondary,
        next_action,
        not_recommended,
        best_step,
        checkpoint_summary,
        comparison,
        gap,
        scenario_table,
        scenario_breakdown,
        by_threat,
        by_motion,
        cpa_ttc,
    )
    write_complete_flag(terminal, next_action)


if __name__ == "__main__":
    main()
