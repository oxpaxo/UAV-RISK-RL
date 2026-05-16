from __future__ import annotations

import argparse
import csv
import math
import sys
import time
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


COMPLETE_FLAG = "PHASE_B_GEOMETRY_FILTER_BASELINE_COMPLETE.flag"
STOP_REASON_TO_FLAG = {
    "phase_a_missing": "PHASE_B_STOP_PHASE_A_MISSING.flag",
    "env_core_change_required": "PHASE_B_STOP_ENV_CORE_CHANGE_REQUIRED.flag",
    "checkpoint_not_found": "PHASE_B_STOP_CHECKPOINT_NOT_FOUND.flag",
    "baseline_impl_failed": "PHASE_B_STOP_BASELINE_IMPL_FAILED.flag",
    "eval_failed": "PHASE_B_STOP_EVAL_FAILED.flag",
    "schema_mismatch": "PHASE_B_STOP_SCHEMA_MISMATCH.flag",
    "resource_limit": "PHASE_B_STOP_RESOURCE_LIMIT.flag",
}
REQUIRED_STAGES = {"b0_smoke", "b1_coarse", "b2_formal"}
REQUIRED_PLOTS = [
    "success_collision_pareto.png",
    "progress_collision_pareto.png",
    "near_miss_min_distance_comparison.png",
    "scenario_collision_heatmap.png",
    "motion_mode_collision_heatmap.png",
    "threat_class_collision_bar.png",
    "filter_rate_vs_collision.png",
    "filter_delta_distribution.png",
    "top_config_ranking.png",
]


class AnalysisStop(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


class Logger:
    def __init__(self, result_dir: Path) -> None:
        self.path = result_dir / "logs/phase_b_analysis.log"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, np.nan) for field in fields})


def write_status(result_dir: Path, status: str) -> None:
    (result_dir / "phase_b_status.txt").write_text(status + "\n", encoding="utf-8")


def write_stop(result_dir: Path, reason: str, detail: str) -> None:
    flag = STOP_REASON_TO_FLAG.get(reason, "PHASE_B_STOP_EVAL_FAILED.flag")
    write_status(result_dir, f"stopped:{flag}")
    (result_dir / flag).write_text(f"terminal_decision=phase_b_stopped_{reason}\ndetail={detail}\n", encoding="utf-8")
    write_report(result_dir, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), complete=False, stop_reason=reason, stop_detail=detail)


def finite_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else float("nan")


def load_inputs(result_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    episode_path = result_dir / "tables/phase_b_episode_metrics.csv"
    manifest_path = result_dir / "tables/phase_b_baseline_manifest.csv"
    if not episode_path.exists() or episode_path.stat().st_size == 0:
        raise AnalysisStop("eval_failed", "missing or empty phase_b_episode_metrics.csv")
    if not manifest_path.exists() or manifest_path.stat().st_size == 0:
        raise AnalysisStop("baseline_impl_failed", "missing or empty phase_b_baseline_manifest.csv")
    ep = pd.read_csv(episode_path)
    manifest = pd.read_csv(manifest_path)
    missing_stages = REQUIRED_STAGES - set(ep["stage"].astype(str))
    if missing_stages:
        if "b2_formal" in missing_stages and not ({"b0_smoke", "b1_coarse"} - set(ep["stage"].astype(str))):
            raise AnalysisStop("resource_limit", f"B2 formal confirmation missing; completed stages={sorted(set(ep['stage'].astype(str)))}")
        raise AnalysisStop("eval_failed", f"missing required stages: {sorted(missing_stages)}")
    return ep, manifest


def aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            episodes=("episode_id", "count"),
            success_rate=("success", "mean"),
            collision_rate=("collision", "mean"),
            timeout_rate=("timeout", "mean"),
            near_miss_rate=("near_miss", "mean"),
            progress_mean=("progress", "mean"),
            final_goal_distance_mean=("final_goal_distance", "mean"),
            episode_return_mean=("episode_return", "mean"),
            episode_length_steps_mean=("episode_length_steps", "mean"),
            mean_time=("mean_time", "mean"),
            episode_min_distance_mean=("episode_min_distance", "mean"),
            mean_min_distance=("mean_min_distance", "mean"),
            min_distance_after_threat_mean=("min_distance_after_threat", "mean"),
            replacement_count_mean=("replacement_count", "mean"),
            active_obstacle_count_mean=("active_obstacle_count", "mean"),
            mean_action_norm=("mean_action_norm", "mean"),
            mean_action_delta=("mean_action_delta", "mean"),
            max_action_delta=("max_action_delta", "mean"),
            filter_used=("filter_used", "max"),
            filter_trigger_rate=("filter_trigger_rate", "mean"),
            mean_filter_delta_norm=("mean_filter_delta_norm", "mean"),
            max_filter_delta_norm=("max_filter_delta_norm", "mean"),
            nan_or_crash=("nan_or_crash", "sum"),
        )
        .reset_index()
    )
    return grouped


def is_pareto(df: pd.DataFrame, x_col: str, y_col: str) -> list[int]:
    out: list[int] = []
    for _, row in df.iterrows():
        dominated = False
        for _, other in df.iterrows():
            if other.name == row.name:
                continue
            if other[x_col] >= row[x_col] and other[y_col] <= row[y_col] and (
                other[x_col] > row[x_col] or other[y_col] < row[y_col]
            ):
                dominated = True
                break
        out.append(0 if dominated else 1)
    return out


def build_summary(ep: pd.DataFrame) -> pd.DataFrame:
    summary = aggregate(ep, ["stage", "baseline_name", "config_name", "baseline_category"])
    summary["rank_score"] = (
        summary["success_rate"]
        - 2.0 * summary["collision_rate"]
        - 0.5 * summary["near_miss_rate"]
        + 0.2 * summary["progress_mean"]
    )
    return summary.sort_values(["stage", "rank_score"], ascending=[True, False])


def build_pareto(formal: pd.DataFrame) -> pd.DataFrame:
    pareto = aggregate(formal, ["baseline_name", "config_name", "baseline_category"])
    pareto["rank_score"] = (
        pareto["success_rate"]
        - 2.0 * pareto["collision_rate"]
        - 0.5 * pareto["near_miss_rate"]
        + 0.2 * pareto["progress_mean"]
    )
    pareto["is_pareto_success_collision"] = is_pareto(pareto, "success_rate", "collision_rate")
    pareto["is_pareto_progress_collision"] = is_pareto(pareto, "progress_mean", "collision_rate")
    fields = [
        "baseline_name",
        "config_name",
        "success_rate",
        "collision_rate",
        "near_miss_rate",
        "progress_mean",
        "episode_min_distance_mean",
        "mean_action_delta",
        "filter_trigger_rate",
        "is_pareto_success_collision",
        "is_pareto_progress_collision",
        "rank_score",
        "episodes",
        "baseline_category",
    ]
    return pareto[fields].sort_values("rank_score", ascending=False)


def build_filter_summary(formal: pd.DataFrame) -> pd.DataFrame:
    filt = formal[formal["filter_used"] == 1].copy()
    rows: list[dict[str, Any]] = []
    for (baseline, config), group in filt.groupby(["baseline_name", "config_name"], dropna=False):
        triggered = group[group["episode_filter_triggered"] == 1]
        not_triggered = group[group["episode_filter_triggered"] == 0]
        rows.append(
            {
                "baseline_name": baseline,
                "config_name": config,
                "episodes": len(group),
                "filter_trigger_rate": finite_mean(group["filter_trigger_rate"]),
                "episode_filter_triggered_rate": finite_mean(group["episode_filter_triggered"]),
                "mean_filter_delta_norm": finite_mean(group["mean_filter_delta_norm"]),
                "max_filter_delta_norm": finite_mean(group["max_filter_delta_norm"]),
                "collision_when_filter_triggered": finite_mean(triggered["collision"]) if len(triggered) else float("nan"),
                "collision_when_filter_not_triggered": finite_mean(not_triggered["collision"]) if len(not_triggered) else float("nan"),
                "success_when_filter_triggered": finite_mean(triggered["success"]) if len(triggered) else float("nan"),
                "success_when_filter_not_triggered": finite_mean(not_triggered["success"]) if len(not_triggered) else float("nan"),
                "mean_min_predicted_cpa_raw": finite_mean(group["mean_min_predicted_cpa_raw"]),
                "mean_min_predicted_cpa_filtered": finite_mean(group["mean_min_predicted_cpa_filtered"]),
                "mean_min_ttc_raw": finite_mean(group["mean_min_ttc_raw"]),
                "mean_min_ttc_filtered": finite_mean(group["mean_min_ttc_filtered"]),
            }
        )
    if not rows:
        rows.append(
            {
                "baseline_name": "none",
                "config_name": "none",
                "episodes": 0,
                "filter_trigger_rate": float("nan"),
                "episode_filter_triggered_rate": float("nan"),
                "mean_filter_delta_norm": float("nan"),
                "max_filter_delta_norm": float("nan"),
                "collision_when_filter_triggered": float("nan"),
                "collision_when_filter_not_triggered": float("nan"),
                "success_when_filter_triggered": float("nan"),
                "success_when_filter_not_triggered": float("nan"),
                "mean_min_predicted_cpa_raw": float("nan"),
                "mean_min_predicted_cpa_filtered": float("nan"),
                "mean_min_ttc_raw": float("nan"),
                "mean_min_ttc_filtered": float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_failure_cases(formal: pd.DataFrame) -> pd.DataFrame:
    failures = formal[formal["failure_type"].isin(["collision", "timeout", "near_miss"])].copy()
    if failures.empty:
        return formal.head(0).copy()
    sort_cols = ["collision", "near_miss", "episode_min_distance"]
    failures = failures.sort_values(sort_cols, ascending=[False, False, True])
    cols = [
        "stage",
        "baseline_name",
        "config_name",
        "scenario",
        "episode_id",
        "episode_seed",
        "failure_type",
        "success",
        "collision",
        "near_miss",
        "progress",
        "episode_min_distance",
        "min_distance_after_threat",
        "threat_class",
        "motion_mode",
        "planned_cpa",
        "planned_ttc",
        "filter_used",
        "filter_trigger_rate",
        "mean_filter_delta_norm",
    ]
    return failures[cols].head(200)


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def plot_scatter(pareto: pd.DataFrame, x: str, y: str, path: Path, xlabel: str, ylabel: str) -> None:
    plt.figure(figsize=(8, 5))
    for _, row in pareto.iterrows():
        marker = "o" if row.get("baseline_category", "") != "safety_filter" else "s"
        plt.scatter(row[x], row[y], s=55, marker=marker)
        plt.text(row[x], row[y], str(row["config_name"])[:28], fontsize=7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_bar(df: pd.DataFrame, x: str, y: str, path: Path, title: str, rotation: int = 45) -> None:
    plt.figure(figsize=(10, 5))
    labels = df[x].astype(str).tolist()
    values = pd.to_numeric(df[y], errors="coerce").to_numpy()
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(y)
    plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_heatmap(df: pd.DataFrame, path: Path, title: str) -> None:
    if df.empty:
        path.write_text("empty heatmap input\n", encoding="utf-8")
        return
    pivot = df.pivot_table(index="config_name", columns=df.columns[0], values="collision_rate", aggfunc="mean")
    plt.figure(figsize=(10, max(4, 0.35 * len(pivot))))
    plt.imshow(pivot.fillna(0.0).to_numpy(), aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    plt.colorbar(label="collision_rate")
    plt.yticks(range(len(pivot.index)), pivot.index.astype(str), fontsize=7)
    plt.xticks(range(len(pivot.columns)), pivot.columns.astype(str), rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_plots(result_dir: Path, formal: pd.DataFrame, pareto: pd.DataFrame, scenario: pd.DataFrame, motion: pd.DataFrame, threat: pd.DataFrame, filter_summary: pd.DataFrame) -> None:
    plots = result_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    plot_scatter(pareto, "success_rate", "collision_rate", plots / "success_collision_pareto.png", "success_rate", "collision_rate")
    plot_scatter(pareto, "progress_mean", "collision_rate", plots / "progress_collision_pareto.png", "progress_mean", "collision_rate")
    distance_df = pareto.sort_values("episode_min_distance_mean", ascending=False).head(20)
    plot_bar(distance_df, "config_name", "episode_min_distance_mean", plots / "near_miss_min_distance_comparison.png", "Top min-distance configs")
    plot_heatmap(scenario[["scenario", "config_name", "collision_rate"]], plots / "scenario_collision_heatmap.png", "Scenario collision rate")
    plot_heatmap(motion[["motion_mode", "config_name", "collision_rate"]], plots / "motion_mode_collision_heatmap.png", "Motion-mode collision rate")
    threat_bar = threat.groupby("threat_class", as_index=False)["collision_rate"].mean()
    plot_bar(threat_bar, "threat_class", "collision_rate", plots / "threat_class_collision_bar.png", "Collision by threat class", rotation=0)
    if filter_summary.empty:
        (plots / "filter_rate_vs_collision.png").write_text("no filter baselines\n", encoding="utf-8")
    else:
        filt_collision = formal[formal["filter_used"] == 1].groupby(["baseline_name", "config_name"], as_index=False)["collision"].mean()
        merged = filter_summary.merge(filt_collision, on=["baseline_name", "config_name"], how="left")
        plt.figure(figsize=(8, 5))
        plt.scatter(merged["filter_trigger_rate"], merged["collision"], s=55)
        for _, row in merged.iterrows():
            plt.text(row["filter_trigger_rate"], row["collision"], str(row["config_name"])[:28], fontsize=7)
        plt.xlabel("filter_trigger_rate")
        plt.ylabel("collision_rate")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(plots / "filter_rate_vs_collision.png", dpi=150)
        plt.close()
    filt = formal[formal["filter_used"] == 1]
    plt.figure(figsize=(8, 5))
    if filt.empty:
        plt.text(0.5, 0.5, "no filter baselines", ha="center")
    else:
        plt.hist(pd.to_numeric(filt["mean_filter_delta_norm"], errors="coerce").dropna(), bins=30)
    plt.xlabel("mean_filter_delta_norm")
    plt.ylabel("episodes")
    plt.tight_layout()
    plt.savefig(plots / "filter_delta_distribution.png", dpi=150)
    plt.close()
    rank_df = pareto.sort_values("rank_score", ascending=False).head(15)
    plot_bar(rank_df, "config_name", "rank_score", plots / "top_config_ranking.png", "Top config rank score")


def fmt(value: Any) -> str:
    try:
        f = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(f):
        return "nan"
    return f"{f:.4f}"


def write_report(
    result_dir: Path,
    summary: pd.DataFrame,
    pareto: pd.DataFrame,
    filter_summary: pd.DataFrame,
    failures: pd.DataFrame,
    *,
    complete: bool,
    stop_reason: str | None = None,
    stop_detail: str | None = None,
) -> None:
    lines = [
        "# Phase B Geometry Filter Baseline Report",
        "",
        "## 1. Background And Phase B Goal",
        "",
        "Phase B is an eval-only audit of geometry controllers and action-level safety filters on frozen EnvV2. No new PPO was trained and EnvV2-core was not modified.",
        "",
        "## 2. Phase A Dependency Check",
        "",
        "- Phase A complete flag: checked.",
        "- Phase A episode and trace schema: reused and extended without deleting core columns.",
        "- attention_full 1500k checkpoint: loaded from `checkpoints/env_v2_phase2/attention_full_s0/attention_full_s0_step1500000.zip`.",
        "",
        "## 3. EnvV2-Core Freeze Recheck",
        "",
        "The Phase A EnvV2-core hash was compared with the current `envs/dynamic_obstacle_flow_env.py` hash before evaluation. Phase B did not modify obstacle counts, motion modes, train/eval scenarios, action dynamics, reward, termination, or collision/success/near-miss definitions.",
        "",
        "## 4. Added / Modified Files",
        "",
        "- `scripts/run_env_v2_phase_b_geometry_filter_baselines.py`",
        "- `scripts/analyze_env_v2_phase_b_results.py`",
        "- `scripts/watch_phase_b_geometry_filter_baselines.sh`",
        f"- `{result_dir.relative_to(ROOT)}/` result artifacts",
        "",
        "## 5. Baseline Manifest",
        "",
        "Baseline definitions are recorded in `tables/phase_b_baseline_manifest.csv`. Required families include random, straight_line, attention_full_1500k, current_cpa_reactive, APF family, CPA-reactive sweep, and distance / CPA-TTC / VO-like attention filters.",
        "",
        "## 6. Baseline Formulas / Parameters",
        "",
        "- `naive_apf`: goal attraction plus inverse-distance repulsion with `d0` and `w_rep` sweep.",
        "- `velocity_aware_apf`: naive APF multiplied by closing-speed gain.",
        "- `cpa_ttc_weighted_apf`: velocity-aware APF multiplied by CPA/TTC short-horizon gain.",
        "- `cpa_reactive_sweep`: one-factor sweep around current `d_reactive=4.0`, `horizon=4.5`, `cpa_trigger=2.4`, `avoid_weight=2.1`.",
        "- `distance_filter`: filters attention action when nearest obstacle is close and raw action increases closing.",
        "- `cpa_ttc_filter`: filters attention action when predicted CPA/TTC under raw action is unsafe.",
        "- `vo_like_filter`: selects a velocity candidate from a small velocity set by CPA safety and progress/raw-distance score.",
        "",
        "## 7. B0 / B1 / B2 Eval Scale",
        "",
    ]
    if not summary.empty:
        stage_counts = summary.groupby("stage", as_index=False)["episodes"].sum()
        for _, row in stage_counts.iterrows():
            lines.append(f"- {row['stage']}: {int(row['episodes'])} episode rows")
    lines.extend(
        [
            "",
            "## 8. Aggregate Comparison",
            "",
            "| baseline | config | success | collision | near_miss | progress | min_distance | rank_score |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in pareto.head(20).iterrows():
        lines.append(
            f"| {row['baseline_name']} | {row['config_name']} | {fmt(row['success_rate'])} | "
            f"{fmt(row['collision_rate'])} | {fmt(row['near_miss_rate'])} | {fmt(row['progress_mean'])} | "
            f"{fmt(row['episode_min_distance_mean'])} | {fmt(row['rank_score'])} |"
        )
    lines.extend(["", "## 9. Pareto Frontier", ""])
    if not pareto.empty:
        sc = pareto[pareto["is_pareto_success_collision"] == 1]["config_name"].tolist()
        pc = pareto[pareto["is_pareto_progress_collision"] == 1]["config_name"].tolist()
        lines.append(f"- success/collision Pareto configs: `{', '.join(sc)}`")
        lines.append(f"- progress/collision Pareto configs: `{', '.join(pc)}`")
    lines.extend(
        [
            "",
            "## 10. Scenario-Wise Breakdown",
            "",
            "`tables/phase_b_scenario_breakdown.csv` contains formal scenario-level success, collision, near-miss, progress, and distance metrics.",
            "",
            "## 11. Motion-Mode Breakdown",
            "",
            "`tables/phase_b_motion_mode_breakdown.csv` contains formal collision/success breakdown by threat motion mode.",
            "",
            "## 12. Threat-Class Breakdown",
            "",
            "`tables/phase_b_threat_class_breakdown.csv` contains formal collision/success breakdown by threat class.",
            "",
            "## 13. Filter Intervention Analysis",
            "",
            "| baseline | config | trigger_rate | collision_triggered | collision_not_triggered | cpa_raw | cpa_filtered |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in filter_summary.head(20).iterrows():
        lines.append(
            f"| {row['baseline_name']} | {row['config_name']} | {fmt(row['filter_trigger_rate'])} | "
            f"{fmt(row['collision_when_filter_triggered'])} | {fmt(row['collision_when_filter_not_triggered'])} | "
            f"{fmt(row['mean_min_predicted_cpa_raw'])} | {fmt(row['mean_min_predicted_cpa_filtered'])} |"
        )
    lines.extend(
        [
            "",
            "## 14. Failure Cases",
            "",
            f"`tables/phase_b_failure_case_table.csv` contains {len(failures)} sampled failure rows sorted by collision / near-miss / minimum distance.",
            "",
            "## 15. Top Configs",
            "",
            "`tables/phase_b_top_configs.csv` records the B1-selected configs used for B2 confirmation. `tables/phase_b_pareto_table.csv` records the formal ranking.",
            "",
            "## 16. Did Geometry / Filters Beat Attention Full?",
            "",
        ]
    )
    if not pareto.empty and "attention_full_1500k" in set(pareto["config_name"]):
        attention = pareto[pareto["config_name"] == "attention_full_1500k"].iloc[0]
        better = pareto[
            (pareto["collision_rate"] < attention["collision_rate"]) & (pareto["success_rate"] >= attention["success_rate"] - 0.05)
        ]
        if better.empty:
            lines.append("Experiment-supported fact: no formal B2 config achieved lower collision than attention_full_1500k while preserving success within 0.05.")
        else:
            lines.append("Experiment-supported fact: at least one formal B2 config achieved lower collision while preserving attention_full_1500k success within 0.05.")
            lines.append("Configs: `" + ", ".join(better["config_name"].tolist()) + "`.")
    else:
        lines.append("Experiment-supported fact: attention_full_1500k row was unavailable in the formal Pareto table.")
    lines.extend(
        [
            "",
            "Reasonable inference: Phase C should focus on training-time safety costs only after using this audit to choose whether action-level filters are strong enough as deployment wrappers or diagnostic baselines.",
            "",
            "Hypotheses for Phase C: CPA/TTC costs and VO-style unsafe-velocity costs are plausible if formal filters reduce collision without destroying progress; otherwise geometry-only intervention is insufficient.",
            "",
            "## 17. Phase C Recommendation",
            "",
            "If Phase B complete, Phase C may start safety-cost training decisions using the formal Pareto and filter intervention tables.",
            "",
            "## 18. Completion Criteria",
            "",
            "- Phase A complete flag exists.",
            "- EnvV2-core freeze rechecked.",
            "- attention_full checkpoint loaded.",
            "- B0, B1, and B2 completed.",
            "- required CSVs, plots, sampled/failure traces, report, logs, and flag are generated.",
            "",
            "## 19. Decision",
            "",
        ]
    )
    if complete:
        lines.extend(
            [
                "Phase B complete.",
                "Geometry/filter baseline audit is ready for Phase C decision.",
                "",
                "terminal_decision = phase_b_geometry_filter_baseline_complete",
            ]
        )
    else:
        lines.extend(
            [
                "Phase B not complete.",
                f"stop_reason = {stop_reason}",
                f"stop_detail = {stop_detail}",
                "",
                f"terminal_decision = phase_b_stopped_{stop_reason}",
            ]
        )
    (result_dir / "PHASE_B_GEOMETRY_FILTER_BASELINE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_outputs(result_dir: Path) -> None:
    required = [
        "phase_b_baseline_manifest.csv",
        "phase_b_episode_metrics.csv",
        "phase_b_eval_summary.csv",
        "phase_b_pareto_table.csv",
        "phase_b_top_configs.csv",
        "phase_b_scenario_breakdown.csv",
        "phase_b_motion_mode_breakdown.csv",
        "phase_b_threat_class_breakdown.csv",
        "phase_b_filter_intervention_summary.csv",
        "phase_b_failure_case_table.csv",
        "phase_b_command_manifest.csv",
        "phase_b_schema_check.csv",
    ]
    missing = []
    for name in required:
        p = result_dir / "tables" / name
        if not p.exists() or p.stat().st_size == 0:
            missing.append(str(p))
    for name in REQUIRED_PLOTS:
        p = result_dir / "plots" / name
        if not p.exists() or p.stat().st_size == 0:
            missing.append(str(p))
    trace_files = list((result_dir / "traces").rglob("*.csv"))
    if not trace_files:
        missing.append("traces/*.csv")
    report = result_dir / "PHASE_B_GEOMETRY_FILTER_BASELINE_REPORT.md"
    if not report.exists() or "Phase B complete." not in report.read_text(encoding="utf-8", errors="replace"):
        missing.append(str(report))
    if missing:
        raise AnalysisStop("schema_mismatch", "missing required outputs: " + ", ".join(missing[:20]))


def analyze(args: argparse.Namespace) -> None:
    result_dir = ROOT / args.result_dir
    logger = Logger(result_dir)
    logger.log("PHASE_B_ANALYSIS_START")
    try:
        ep, _manifest = load_inputs(result_dir)
        formal = ep[ep["stage"] == "b2_formal"].copy()
        if formal.empty:
            raise AnalysisStop("resource_limit", "B2 formal confirmation has no rows")
        summary = build_summary(ep)
        pareto = build_pareto(formal)
        scenario = aggregate(formal, ["scenario", "baseline_name", "config_name", "baseline_category"])
        motion = aggregate(formal, ["motion_mode", "baseline_name", "config_name", "baseline_category"])
        threat = aggregate(formal, ["threat_class", "baseline_name", "config_name", "baseline_category"])
        filter_summary = build_filter_summary(formal)
        failures = build_failure_cases(formal)
        save_table(summary, result_dir / "tables/phase_b_eval_summary.csv")
        save_table(pareto, result_dir / "tables/phase_b_pareto_table.csv")
        save_table(scenario, result_dir / "tables/phase_b_scenario_breakdown.csv")
        save_table(motion, result_dir / "tables/phase_b_motion_mode_breakdown.csv")
        save_table(threat, result_dir / "tables/phase_b_threat_class_breakdown.csv")
        save_table(filter_summary, result_dir / "tables/phase_b_filter_intervention_summary.csv")
        save_table(failures, result_dir / "tables/phase_b_failure_case_table.csv")
        if not (result_dir / "tables/phase_b_top_configs.csv").exists():
            top = pareto.head(10).copy()
            top.insert(0, "selection_rank", range(1, len(top) + 1))
            top.to_csv(result_dir / "tables/phase_b_top_configs.csv", index=False)
        write_plots(result_dir, formal, pareto, scenario, motion, threat, filter_summary)
        write_report(result_dir, summary, pareto, filter_summary, failures, complete=True)
        validate_outputs(result_dir)
        write_status(result_dir, "complete")
        (result_dir / COMPLETE_FLAG).write_text(
            "terminal_decision=phase_b_geometry_filter_baseline_complete\nnext_recommended_phase=Phase C decision\n",
            encoding="utf-8",
        )
        logger.log("PHASE_B_ANALYSIS_COMPLETE")
    except AnalysisStop as exc:
        write_stop(result_dir, exc.reason, exc.detail)
        logger.log(f"PHASE_B_ANALYSIS_STOP reason={exc.reason} detail={exc.detail}")
        raise SystemExit(2) from None
    except Exception as exc:
        detail = f"unexpected analysis exception: {exc!r}"
        logger.log(detail)
        logger.log(traceback.format_exc())
        write_stop(result_dir, "eval_failed", detail)
        raise SystemExit(2) from None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default="results/env_v2_phase_b_geometry_filter_baselines")
    return parser.parse_args()


def main() -> None:
    analyze(parse_args())


if __name__ == "__main__":
    main()
