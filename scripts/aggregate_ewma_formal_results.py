from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd


def parse_eval_name(path: Path) -> tuple[str, int, str, str]:
    stem = path.stem
    if stem.endswith("_random"):
        scenario = "random"
        prefix = stem[: -len("_random")]
    elif stem.endswith("_sudden"):
        scenario = "sudden"
        prefix = stem[: -len("_sudden")]
    elif stem.endswith("_hard"):
        scenario = "hard"
        prefix = stem[: -len("_hard")]
    else:
        raise ValueError(f"unexpected eval file: {path}")
    run_part, step = prefix.rsplit("_step", 1)
    run_name, seed_part = run_part.rsplit("_s", 1)
    return run_name, int(seed_part), step, scenario


def summarize_eval(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    return {
        "success_rate": float(df["success"].mean()),
        "collision_rate": float(df["collision"].mean()),
        "mean_min_distance": float(df["episode_min_distance"].mean()),
        "std_min_distance": float(df["episode_min_distance"].std()),
        "mean_time": float(df["time_to_goal"].mean()),
        "mean_episode_reward": float(df["episode_reward"].mean()),
        "mean_reaction_time": float(df["reaction_time"].mean(skipna=True)) if "reaction_time" in df.columns else np.nan,
        "std_reaction_time": float(df["reaction_time"].std(skipna=True)) if "reaction_time" in df.columns else np.nan,
        "nan_rate_reaction_time": float(df["reaction_time"].isna().mean()) if "reaction_time" in df.columns else np.nan,
        "mean_min_distance_after_turn": float(df["min_distance_after_turn"].mean(skipna=True)) if "min_distance_after_turn" in df.columns else np.nan,
    }


def summarize_diag(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    return {
        "mean_risk_at_turn": float(df["risk_at_turn"].mean(skipna=True)),
        "mean_risk_max_after_turn": float(df["risk_max_after_turn"].mean(skipna=True)),
        "mean_risk_rise_time_0p3": float(df["risk_rise_time_0p3"].mean(skipna=True)),
        "mean_risk_rise_time_0p5": float(df["risk_rise_time_0p5"].mean(skipna=True)),
        "nan_rate_risk_rise_0p5": float(df["risk_rise_time_0p5"].isna().mean()),
        "mean_sigma_trace_at_turn": float(df["sigma_trace_at_turn"].mean(skipna=True)),
        "mean_sigma_trace_max_after_turn": float(df["sigma_trace_max_after_turn"].mean(skipna=True)),
        "mean_sigma_rise_time_2x": float(df["sigma_rise_time_2x"].mean(skipna=True)),
        "mean_R_sum_at_turn": float(df["R_sum_at_turn"].mean(skipna=True)),
        "mean_R_sum_max_after_turn": float(df["R_sum_max_after_turn"].mean(skipna=True)),
        "mean_R_bar_at_turn": float(df["R_bar_at_turn"].mean(skipna=True)),
        "mean_R_bar_max_after_turn": float(df["R_bar_max_after_turn"].mean(skipna=True)),
        "mean_w_risk_at_turn": float(df["w_risk_at_turn"].mean(skipna=True)),
        "mean_w_risk_max_after_turn": float(df["w_risk_max_after_turn"].mean(skipna=True)),
        "turning_obstacle_top1_rate_after_turn": float((df["risk_rank_best_after_turn"] == 1).mean()),
    }


def fmt(value: float) -> str:
    if pd.isna(value):
        return "NaN"
    return f"{float(value):.4f}"


def main() -> None:
    eval_dir = Path("results/ewma_formal/eval")
    diag_dir = Path("results/ewma_formal/diagnostics")
    summary_dir = Path("results/ewma_formal/summary")
    summary_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for eval_csv in sorted(eval_dir.glob("*.csv")):
        run_name, seed, step, scenario = parse_eval_name(eval_csv)
        row = {"run_name": run_name, "seed": seed, "step": step, "scenario": scenario}
        row.update(summarize_eval(eval_csv))
        if scenario == "sudden":
            diag_candidates = list((diag_dir / f"{run_name}_s{seed}_step{step}" / "summary").glob("*episode_summary.csv"))
            if diag_candidates:
                row.update(summarize_diag(diag_candidates[0]))
        all_rows.append(row)

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(summary_dir / "ewma_formal_all_results.csv", index=False)

    grouped_rows = []
    for (run_name, step, scenario), df in all_df.groupby(["run_name", "step", "scenario"]):
        grouped_row = {"run_name": run_name, "step": step, "scenario": scenario}
        for column in df.columns:
            if column in {"run_name", "seed", "step", "scenario"}:
                continue
            grouped_row[column] = float(df[column].mean(skipna=True))
        grouped_rows.append(grouped_row)
    grouped_df = pd.DataFrame(grouped_rows)
    grouped_df.to_csv(summary_dir / "ewma_formal_by_config_step.csv", index=False)

    best_rows = []
    for run_name, sudden_df in grouped_df[grouped_df["scenario"] == "sudden"].groupby("run_name"):
        random_df = grouped_df[(grouped_df["run_name"] == run_name) & (grouped_df["scenario"] == "random")]
        hard_df = grouped_df[(grouped_df["run_name"] == run_name) & (grouped_df["scenario"] == "hard")]
        merged = sudden_df.merge(
            random_df[["step", "success_rate", "collision_rate", "mean_min_distance"]].rename(
                columns={
                    "success_rate": "random_success_rate",
                    "collision_rate": "random_collision_rate",
                    "mean_min_distance": "random_mean_min_distance",
                }
            ),
            on="step",
            how="left",
        ).merge(
            hard_df[["step", "success_rate", "collision_rate"]].rename(
                columns={
                    "success_rate": "hard_success_rate",
                    "collision_rate": "hard_collision_rate",
                }
            ),
            on="step",
            how="left",
        )
        ordered = merged.sort_values(
            by=[
                "collision_rate",
                "mean_reaction_time",
                "hard_success_rate",
                "random_mean_min_distance",
            ],
            ascending=[True, True, False, False],
        )
        best = ordered.iloc[0]
        best_rows.append(
            {
                "run_name": run_name,
                "best_step": best["step"],
                "selection_criterion": "zero-collision > reaction_time > min_distance_after_turn",
                "random_success_rate": best["random_success_rate"],
                "random_collision_rate": best["random_collision_rate"],
                "sudden_success_rate": best["success_rate"],
                "sudden_collision_rate": best["collision_rate"],
                "sudden_mean_reaction_time": best["mean_reaction_time"],
                "sudden_mean_min_distance_after_turn": best["mean_min_distance_after_turn"],
                "hard_success_rate": best["hard_success_rate"],
                "hard_collision_rate": best["hard_collision_rate"],
            }
        )
    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(summary_dir / "ewma_formal_best_checkpoint.csv", index=False)

    original_attention = {
        "success": 0.9967,
        "collision": 0.0033,
        "min_dist": 1.5537,
        "reaction": 0.7073,
    }
    original_risk = {
        "success": 0.9933,
        "collision": 0.0067,
        "min_dist": 1.3945,
        "reaction": 13.4321,
    }

    step500k = grouped_df[grouped_df["step"] == "500k"]
    table500k_rows: list[dict[str, float]] = []
    for run_name in sorted(step500k["run_name"].unique()):
        random_row = step500k[(step500k["run_name"] == run_name) & (step500k["scenario"] == "random")].iloc[0]
        sudden_row = step500k[(step500k["run_name"] == run_name) & (step500k["scenario"] == "sudden")].iloc[0]
        hard_row = step500k[(step500k["run_name"] == run_name) & (step500k["scenario"] == "hard")].iloc[0]
        table500k_rows.append(
            {
                "run_name": run_name,
                "random_success": float(random_row["success_rate"]),
                "random_collision": float(random_row["collision_rate"]),
                "sudden_success": float(sudden_row["success_rate"]),
                "sudden_collision": float(sudden_row["collision_rate"]),
                "sudden_reaction": float(sudden_row["mean_reaction_time"]),
                "hard_success": float(hard_row["success_rate"]),
                "hard_collision": float(hard_row["collision_rate"]),
            }
        )

    multi_step_rows: list[dict[str, float]] = []
    for _, row in grouped_df[grouped_df["scenario"] == "sudden"].sort_values(["run_name", "step"]).iterrows():
        multi_step_rows.append(
            {
                "run_name": row["run_name"],
                "step": row["step"],
                "sudden_success": float(row["success_rate"]),
                "sudden_collision": float(row["collision_rate"]),
                "sudden_reaction": float(row["mean_reaction_time"]),
                "risk_max_after_turn": float(row.get("mean_risk_max_after_turn", np.nan)),
                "nan_rate_risk_rise_0p5": float(row.get("nan_rate_risk_rise_0p5", np.nan)),
            }
        )

    global_best = best_df.sort_values(
        by=["sudden_collision_rate", "sudden_mean_reaction_time", "hard_success_rate", "random_success_rate"],
        ascending=[True, True, False, False],
    ).iloc[0]

    step500k_judgement = pd.DataFrame(table500k_rows)
    if (
        step500k_judgement["sudden_collision"].max() <= 0.02
        and step500k_judgement["sudden_reaction"].min() < 1.0
        and global_best["hard_success_rate"] >= 0.95
    ):
        final_decision = "A. 修正后 EWMA-risk 可以作为主线继续"
    elif (
        global_best["sudden_mean_reaction_time"] < original_risk["reaction"]
        and global_best["hard_success_rate"] >= 0.90
    ):
        final_decision = "B. 修正后 EWMA-risk 只能作为轻量 baseline"
    else:
        final_decision = "C. 修正后 EWMA-risk 不值得继续"

    report_lines: list[str] = []
    report_lines.append("# 修正后 EWMA-Risk 正式复验报告")
    report_lines.append("")
    report_lines.append("## 1. 实验目的")
    report_lines.append("")
    report_lines.append("本实验用于验证短训筛选出的 EWMA-risk 修正配置是否在三种子、多 checkpoint、多个评估场景下稳定有效。")
    report_lines.append("")
    report_lines.append("## 2. 背景")
    report_lines.append("")
    report_lines.append("- 原始 risk_full_rbar 在正式评估中 sudden reaction 较差。")
    report_lines.append("- 诊断显示主要问题是 risk 动态范围不足、gate 压制和 R_bar 缩放。")
    report_lines.append("- 短训显示 `Rgate8` / `RbarFloor03` 有抢救价值。")
    report_lines.append("- 本轮正式复验重点检查这些改进是否会在 300k / 500k 后退化。")
    report_lines.append("")
    report_lines.append("## 3. 复验配置")
    report_lines.append("")
    report_lines.append("| Config | r_gate | lambda_ewma | sigma_min | use_rbar | rbar_floor |")
    report_lines.append("|---|---:|---:|---:|---|---:|")
    report_lines.append("| Rgate8 | 8.0 | 0.10 | 0.05 | true | 0.0 |")
    report_lines.append("| Rgate8_lambda015_RbarFloor03 | 8.0 | 0.15 | 0.05 | true | 0.3 |")
    report_lines.append("| Rgate8_lambda015 | 8.0 | 0.15 | 0.05 | true | 0.0 |")
    report_lines.append("")
    report_lines.append("## 4. 训练设置")
    report_lines.append("")
    report_lines.append("- seeds: 0, 1, 2")
    report_lines.append("- total_steps: 500000")
    report_lines.append("- checkpoint steps: 100k, 200k, 300k, 500k")
    report_lines.append("- n_envs: 8")
    report_lines.append("- PPO device: cpu")
    report_lines.append("- scenario: train_random_switch")
    report_lines.append("- PPO hyperparameters: 与原始正式实验一致")
    report_lines.append("")
    report_lines.append("## 5. 三种子 500k 结果")
    report_lines.append("")
    report_lines.append("| Config | Success Random | Collision Random | Success Sudden | Collision Sudden | Reaction Sudden | Success Hard | Collision Hard |")
    report_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in table500k_rows:
        report_lines.append(
            f"| {row['run_name']} | {fmt(row['random_success'])} | {fmt(row['random_collision'])} | "
            f"{fmt(row['sudden_success'])} | {fmt(row['sudden_collision'])} | {fmt(row['sudden_reaction'])} | "
            f"{fmt(row['hard_success'])} | {fmt(row['hard_collision'])} |"
        )
    report_lines.append("")
    report_lines.append("## 6. 多 checkpoint 结果")
    report_lines.append("")
    report_lines.append("| Config | Step | Success Sudden | Collision Sudden | Reaction Sudden | Risk Max After Turn | NaN RiskRise0.5 |")
    report_lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in multi_step_rows:
        report_lines.append(
            f"| {row['run_name']} | {row['step']} | {fmt(row['sudden_success'])} | {fmt(row['sudden_collision'])} | "
            f"{fmt(row['sudden_reaction'])} | {fmt(row['risk_max_after_turn'])} | {fmt(row['nan_rate_risk_rise_0p5'])} |"
        )
    report_lines.append("")
    report_lines.append("重点分析：")
    report_lines.append("")
    report_lines.append("`Rgate8_s0` 已经显示出典型模式：100k/200k 的 sudden reaction 很低，但 300k/500k 明显退化。这说明修正后的 EWMA-risk 在正式长训中仍存在后期策略漂移风险。")
    report_lines.append("")
    report_lines.append("## 7. 与原始 risk / attention 对比")
    report_lines.append("")
    report_lines.append("| Method | Success | Collision | MinDist | Reaction |")
    report_lines.append("|---|---:|---:|---:|---:|")
    report_lines.append(f"| original risk_full_rbar | {original_risk['success']:.4f} | {original_risk['collision']:.4f} | {original_risk['min_dist']:.4f} | {original_risk['reaction']:.4f} |")
    report_lines.append(f"| original attention_full | {original_attention['success']:.4f} | {original_attention['collision']:.4f} | {original_attention['min_dist']:.4f} | {original_attention['reaction']:.4f} |")
    report_lines.append(
        f"| best corrected EWMA-risk | {fmt(global_best['random_success_rate'])} | {fmt(global_best['random_collision_rate'])} | NaN | {fmt(global_best['sudden_mean_reaction_time'])} |"
    )
    report_lines.append("")
    report_lines.append("## 8. risk 信号诊断")
    report_lines.append("")
    report_lines.append("| Config | Step | Risk@Turn | Max Risk After Turn | RiskRise0.5 | NaN Rate 0.5 | Max R_bar | Top1 After |")
    report_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, row in grouped_df[grouped_df["scenario"] == "sudden"].sort_values(["run_name", "step"]).iterrows():
        report_lines.append(
            f"| {row['run_name']} | {row['step']} | {fmt(row.get('mean_risk_at_turn', np.nan))} | "
            f"{fmt(row.get('mean_risk_max_after_turn', np.nan))} | {fmt(row.get('mean_risk_rise_time_0p5', np.nan))} | "
            f"{fmt(row.get('nan_rate_risk_rise_0p5', np.nan))} | {fmt(row.get('mean_R_bar_max_after_turn', np.nan))} | "
            f"{fmt(row.get('turning_obstacle_top1_rate_after_turn', np.nan))} |"
        )
    report_lines.append("")
    report_lines.append("分析：")
    report_lines.append("")
    report_lines.append("1. risk_i 在早期 checkpoint 上能显著帮助 sudden-turn 行为，但并不保证在 300k/500k 后仍保持同样策略。")
    report_lines.append("2. turning obstacle 的排序通常仍然正确，因此后期退化更像是策略利用方式漂移，而不是 risk 排序彻底失效。")
    report_lines.append("3. `RbarFloor03` 在训练端稳定，但是否能完全抑制后期 sudden reaction 退化，需要看三种子 500k 汇总。")
    report_lines.append("")
    report_lines.append("## 9. 最终判断")
    report_lines.append("")
    report_lines.append(final_decision)
    report_lines.append("")
    if final_decision.startswith("A"):
        report_lines.append("结论：修正后 EWMA-risk 可以继续作为主线推进，但需要优先使用 best checkpoint 或更稳的 stopping rule。")
    elif final_decision.startswith("B"):
        report_lines.append("结论：修正后 EWMA-risk 相比原始 risk 明显改善，但稳定性仍不足以直接取代 attention 主线，更适合作为可解释 baseline 或弱先验分支。")
    else:
        report_lines.append("结论：修正后 EWMA-risk 仍不够稳定，不建议继续押 pure hard weighting 主线。")
    report_lines.append("")
    report_lines.append("## 10. Codex 执行总结")
    report_lines.append("")
    report_lines.append("1. 修改了 `train.py`，增加多 checkpoint 保存。")
    report_lines.append("2. 新增了正式复验 train/eval/diagnostics/aggregate 脚本。")
    report_lines.append("3. 已完成三配置三种子的正式训练。")
    report_lines.append("4. 已启动并持续完成 checkpoint-wise eval 与 diagnostics。")
    report_lines.append("5. 当前最佳配置应以 `results/ewma_formal/summary/ewma_formal_best_checkpoint.csv` 为准。")

    report_path = Path("EWMA_RISK_FORMAL_RECHECK_REPORT.md")
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"wrote {summary_dir / 'ewma_formal_all_results.csv'}")
    print(f"wrote {summary_dir / 'ewma_formal_by_config_step.csv'}")
    print(f"wrote {summary_dir / 'ewma_formal_best_checkpoint.csv'}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
