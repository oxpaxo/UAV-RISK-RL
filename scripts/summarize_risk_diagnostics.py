from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_dir", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--out_md", type=str, required=True)
    return parser.parse_args()


def safe_mean(series: pd.Series) -> float:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan
    return float(np.nanmean(arr))


def safe_nan_rate(series: pd.Series) -> float:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if arr.size == 0:
        return np.nan
    return float(np.mean(np.isnan(arr)))


def fmt(value: float) -> str:
    if np.isnan(value):
        return "NaN"
    return f"{value:.4f}"


def main() -> None:
    args = parse_args()
    summary_dir = Path(args.summary_dir)
    rows = []

    for path in sorted(summary_dir.glob("*_episode_summary.csv")):
        df = pd.read_csv(path)
        method = str(df["method"].iloc[0])
        seed = int(df["seed"].iloc[0])
        rows.append(
            {
                "method": method,
                "seed": seed,
                "episodes": len(df),
                "success_rate": safe_mean(df["success"]),
                "collision_rate": safe_mean(df["collision"]),
                "mean_episode_min_distance": safe_mean(df["episode_min_distance"]),
                "mean_risk_at_turn": safe_mean(df["risk_at_turn"]),
                "mean_risk_max_after_turn": safe_mean(df["risk_max_after_turn"]),
                "mean_risk_rise_time_0p3": safe_mean(df["risk_rise_time_0p3"]),
                "mean_risk_rise_time_0p5": safe_mean(df["risk_rise_time_0p5"]),
                "mean_risk_rise_time_0p7": safe_mean(df["risk_rise_time_0p7"]),
                "nan_rate_risk_rise_0p5": safe_nan_rate(df["risk_rise_time_0p5"]),
                "mean_sigma_trace_at_turn": safe_mean(df["sigma_trace_at_turn"]),
                "mean_sigma_trace_max_after_turn": safe_mean(df["sigma_trace_max_after_turn"]),
                "mean_sigma_rise_time_2x": safe_mean(df["sigma_rise_time_2x"]),
                "nan_rate_sigma_rise_time_2x": safe_nan_rate(df["sigma_rise_time_2x"]),
                "mean_R_sum_at_turn": safe_mean(df["R_sum_at_turn"]),
                "mean_R_sum_max_after_turn": safe_mean(df["R_sum_max_after_turn"]),
                "mean_R_bar_at_turn": safe_mean(df["R_bar_at_turn"]),
                "mean_R_bar_max_after_turn": safe_mean(df["R_bar_max_after_turn"]),
                "mean_w_risk_at_turn": safe_mean(df["w_risk_at_turn"]),
                "mean_w_risk_max_after_turn": safe_mean(df["w_risk_max_after_turn"]),
                "turning_obstacle_top1_rate_at_turn": float(np.mean(df["risk_rank_at_turn"] == 1)),
                "turning_obstacle_top1_rate_after_turn": float(np.mean(df["risk_rank_best_after_turn"] == 1)),
                "mean_reaction_time": safe_mean(df["reaction_time"]),
                "nan_rate_reaction_time": safe_nan_rate(df["reaction_time"]),
            }
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    rules: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        key = f"{row['method']} s{row['seed']}"
        if (not np.isnan(row["mean_risk_rise_time_0p5"]) and row["mean_risk_rise_time_0p5"] > 2.0) or (
            not np.isnan(row["nan_rate_risk_rise_0p5"]) and row["nan_rate_risk_rise_0p5"] > 0.3
        ):
            rules[key].append("风险信号对突变响应偏慢。优先检查 sigma_min、lambda_ewma、R_gate。")
        if (not np.isnan(row["mean_sigma_rise_time_2x"]) and row["mean_sigma_rise_time_2x"] > 2.0) or (
            not np.isnan(row["nan_rate_sigma_rise_time_2x"]) and row["nan_rate_sigma_rise_time_2x"] > 0.3
        ):
            rules[key].append("EWMA 不确定性估计响应偏慢。建议尝试 sigma_min=0.10 或 lambda_ewma=0.15/0.20。")
        if (
            not np.isnan(row["mean_R_bar_at_turn"])
            and not np.isnan(row["mean_R_bar_max_after_turn"])
            and row["mean_R_bar_at_turn"] < 0.3
            and row["mean_R_bar_max_after_turn"] < 0.5
        ):
            rules[key].append("R_bar 对 risk context 可能存在过强压制。建议尝试 risk_full_no_rbar 或 R_bar floor。")
        if not np.isnan(row["turning_obstacle_top1_rate_after_turn"]) and row["turning_obstacle_top1_rate_after_turn"] < 0.5:
            rules[key].append("risk 权重没有稳定关注突变障碍物。需要检查 risk 公式或 beta 参数。")
        if (
            not np.isnan(row["mean_risk_rise_time_0p5"])
            and not np.isnan(row["mean_reaction_time"])
            and row["mean_risk_rise_time_0p5"] < 1.0
            and row["mean_reaction_time"] > 2.0
        ):
            rules[key].append("risk 信号已经升高，但策略没有利用它。问题可能在 R_bar、特征编码、或纯 risk weighting 缺少 ego-conditioned 灵活性。")

    md_lines = ["# Risk Diagnostic Summary", ""]
    md_lines.append("| Method | Seed | Risk0.5 Rise | Sigma2x Rise | R_bar@turn | Top1 After | Reaction |")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        md_lines.append(
            f"| {row['method']} | {row['seed']} | {fmt(row['mean_risk_rise_time_0p5'])} | "
            f"{fmt(row['mean_sigma_rise_time_2x'])} | "
            f"{fmt(row['mean_R_bar_at_turn'])} | "
            f"{fmt(row['turning_obstacle_top1_rate_after_turn'])} | "
            f"{fmt(row['mean_reaction_time'])} |"
        )
    md_lines.append("")
    md_lines.append("## Auto Diagnosis")
    md_lines.append("")
    for key, messages in rules.items():
        md_lines.append(f"### {key}")
        if messages:
            for message in messages:
                md_lines.append(f"- {message}")
        else:
            md_lines.append("- 未触发自动规则。")
        md_lines.append("")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
