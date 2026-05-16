from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd


def summarize_file(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    return {
        "success_rate": float(df["success"].mean()),
        "collision_rate": float(df["collision"].mean()),
        "mean_episode_min_distance": float(df["episode_min_distance"].mean()),
        "mean_time_to_goal": float(df["time_to_goal"].mean()),
        "mean_reaction_time": float(df["reaction_time"].mean(skipna=True)) if "reaction_time" in df.columns else np.nan,
        "mean_risk_rise_time": float(df["risk_rise_time"].mean(skipna=True)) if "risk_rise_time" in df.columns else np.nan,
        "mean_min_distance_after_turn": float(df["min_distance_after_turn"].mean(skipna=True)) if "min_distance_after_turn" in df.columns else np.nan,
    }


def main() -> None:
    results_dir = Path("results/ewma_short")
    eval_dir = results_dir / "eval"
    diagnostics_dir = results_dir / "diagnostics"
    rows = []
    random_candidates = sorted(eval_dir.glob("*_random.csv")) + sorted(results_dir.glob("*_random.csv"))
    seen = set()
    for random_csv in random_candidates:
        run_name = random_csv.stem.replace("_random", "")
        run_base = run_name.replace("_s0", "")
        if run_base in seen:
            continue
        seen.add(run_base)
        sudden_csv = eval_dir / f"{run_name}_sudden.csv"
        if not sudden_csv.exists():
            sudden_csv = results_dir / f"{run_base}_sudden.csv"
        random_stats = summarize_file(random_csv)
        sudden_stats = summarize_file(sudden_csv)
        diag_files = list((diagnostics_dir / run_name / "summary").glob("*episode_summary.csv"))
        if not diag_files:
            diag_files = list((diagnostics_dir / run_base).glob("summary/*episode_summary.csv"))
        if not diag_files:
            diag_files = list(diagnostics_dir.glob(f"{run_base}*/summary/*episode_summary.csv"))
        diag_df = pd.read_csv(diag_files[0]) if diag_files else None
        rows.append(
            {
                "run_name": run_base,
                "random_success_rate": random_stats["success_rate"],
                "random_collision_rate": random_stats["collision_rate"],
                "random_mean_min_distance": random_stats["mean_episode_min_distance"],
                "random_mean_time": random_stats["mean_time_to_goal"],
                "sudden_success_rate": sudden_stats["success_rate"],
                "sudden_collision_rate": sudden_stats["collision_rate"],
                "sudden_mean_min_distance": sudden_stats["mean_episode_min_distance"],
                "sudden_mean_time": sudden_stats["mean_time_to_goal"],
                "sudden_mean_reaction_time": sudden_stats["mean_reaction_time"],
                "sudden_nan_rate_reaction_time": float(diag_df["reaction_time"].isna().mean()) if diag_df is not None else np.nan,
                "sudden_mean_risk_rise_time": sudden_stats["mean_risk_rise_time"],
                "sudden_mean_min_distance_after_turn": sudden_stats["mean_min_distance_after_turn"],
                "mean_risk_at_turn": float(diag_df["risk_at_turn"].mean(skipna=True)) if diag_df is not None else np.nan,
                "mean_risk_max_after_turn": float(diag_df["risk_max_after_turn"].mean(skipna=True)) if diag_df is not None else np.nan,
                "mean_risk_rise_time_0p3": float(diag_df["risk_rise_time_0p3"].mean(skipna=True)) if diag_df is not None else np.nan,
                "mean_risk_rise_time_0p5": float(diag_df["risk_rise_time_0p5"].mean(skipna=True)) if diag_df is not None else np.nan,
                "nan_rate_risk_rise_0p5": float(diag_df["risk_rise_time_0p5"].isna().mean()) if diag_df is not None else np.nan,
                "mean_sigma_trace_at_turn": float(diag_df["sigma_trace_at_turn"].mean(skipna=True)) if diag_df is not None else np.nan,
                "mean_sigma_trace_max_after_turn": float(diag_df["sigma_trace_max_after_turn"].mean(skipna=True)) if diag_df is not None else np.nan,
                "mean_sigma_rise_time_2x": float(diag_df["sigma_rise_time_2x"].mean(skipna=True)) if diag_df is not None else np.nan,
                "mean_R_sum_at_turn": float(diag_df["R_sum_at_turn"].mean(skipna=True)) if diag_df is not None else np.nan,
                "mean_R_sum_max_after_turn": float(diag_df["R_sum_max_after_turn"].mean(skipna=True)) if diag_df is not None else np.nan,
                "mean_R_bar_at_turn": float(diag_df["R_bar_at_turn"].mean(skipna=True)) if diag_df is not None else np.nan,
                "mean_R_bar_max_after_turn": float(diag_df["R_bar_max_after_turn"].mean(skipna=True)) if diag_df is not None else np.nan,
                "mean_w_risk_at_turn": float(diag_df["w_risk_at_turn"].mean(skipna=True)) if diag_df is not None else np.nan,
                "mean_w_risk_max_after_turn": float(diag_df["w_risk_max_after_turn"].mean(skipna=True)) if diag_df is not None else np.nan,
                "turning_obstacle_top1_rate_after_turn": float((diag_df["risk_rank_best_after_turn"] == 1).mean()) if diag_df is not None else np.nan,
            }
        )

    out_csv = results_dir / "ewma_short_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
