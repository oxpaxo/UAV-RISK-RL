from __future__ import annotations

import csv
import glob
import os
from collections import defaultdict
from typing import Any

import numpy as np


def load_csv(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if value is None or value == "":
                    parsed[key] = np.nan
                    continue
                try:
                    if key in {"episode_id", "episode_seed", "success", "collision", "steps", "turning_obstacle_id"}:
                        parsed[key] = int(float(value))
                    else:
                        parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def safe_nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan
    return float(np.nanmean(arr))


def method_seed_scenario_from_name(path: str) -> tuple[str, int, str]:
    filename = os.path.basename(path).replace(".csv", "")
    if "_random_hard" in filename:
        scenario = "random_hard"
        prefix = filename.replace("_random_hard", "")
    elif "_random" in filename:
        scenario = "random"
        prefix = filename.replace("_random", "")
    elif "_sudden" in filename:
        scenario = "sudden"
        prefix = filename.replace("_sudden", "")
    else:
        raise ValueError(f"unexpected result file name: {filename}")

    method, seed_part = prefix.rsplit("_s", 1)
    return method, int(seed_part), scenario


def nanmeanstd(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float32)
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def fmt(mean: float, std: float) -> str:
    if np.isnan(mean):
        return "NaN"
    return f"{mean:.4f} ± {std:.4f}"


def main() -> None:
    result_files = sorted(glob.glob("results/*.csv"))
    result_files = [
        path
        for path in result_files
        if not os.path.basename(path).startswith(("debug_", "trend_")) and not path.endswith("summary.csv")
    ]
    if not result_files:
        raise FileNotFoundError("no result csv files found under results/")

    per_seed_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    grouped_metrics: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for path in result_files:
        method, seed, scenario = method_seed_scenario_from_name(path)
        rows = load_csv(path)
        success_rate = float(np.mean([row["success"] for row in rows]))
        collision_rate = float(np.mean([row["collision"] for row in rows]))
        mean_min_distance = float(np.mean([row["episode_min_distance"] for row in rows]))
        mean_time = float(np.mean([row["time_to_goal"] for row in rows]))
        mean_reward = float(np.mean([row["episode_reward"] for row in rows]))
        mean_reaction = safe_nanmean([row.get("reaction_time", np.nan) for row in rows])
        mean_risk_rise = safe_nanmean([row.get("risk_rise_time", np.nan) for row in rows])
        mean_min_distance_after_turn = safe_nanmean([row.get("min_distance_after_turn", np.nan) for row in rows])

        summary_rows.append(
            {
                "method": method,
                "seed": seed,
                "scenario": scenario,
                "success_rate": success_rate,
                "collision_rate": collision_rate,
                "mean_min_distance": mean_min_distance,
                "mean_time_to_goal": mean_time,
                "mean_episode_reward": mean_reward,
                "mean_reaction_time": mean_reaction,
                "mean_risk_rise_time": mean_risk_rise,
                "mean_min_distance_after_turn": mean_min_distance_after_turn,
                "source_csv": path,
            }
        )

        if scenario == "random":
            per_seed_rows.append(
                {
                    "Method": method,
                    "Seed": seed,
                    "Success": success_rate,
                    "Collision": collision_rate,
                    "MinDist": mean_min_distance,
                    "Time": mean_time,
                    "Reaction": mean_reaction,
                }
            )

        grouped_metrics[method][scenario]["Success"].append(success_rate)
        grouped_metrics[method][scenario]["Collision"].append(collision_rate)
        grouped_metrics[method][scenario]["MinDist"].append(mean_min_distance)
        if scenario == "sudden":
            grouped_metrics[method][scenario]["Reaction"].append(mean_reaction)
        if scenario == "random_hard":
            grouped_metrics[method][scenario]["Time"].append(mean_time)

    os.makedirs("results", exist_ok=True)
    summary_csv_path = "results/summary.csv"
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    random_rows_by_method_seed = {(row["Method"], row["Seed"]): row for row in per_seed_rows}
    sudden_lookup = {
        (row["method"], row["seed"]): row
        for row in summary_rows
        if row["scenario"] == "sudden"
    }
    hard_lookup = {
        (row["method"], row["seed"]): row
        for row in summary_rows
        if row["scenario"] == "random_hard"
    }

    report_lines: list[str] = []
    report_lines.append("# UAV Risk-Guided Aggregation 预实验报告")
    report_lines.append("")
    report_lines.append("## 1. 实验目的")
    report_lines.append("")
    report_lines.append("本预实验用于验证在轻量级三维动态避障环境中，完整 risk-guided aggregation 是否相比 learned attention aggregation 具有更好的早期实验表现。")
    report_lines.append("")
    report_lines.append("核心对比：")
    report_lines.append("")
    report_lines.append("- `risk_full_rbar`")
    report_lines.append("- `attention_full`")
    report_lines.append("")
    report_lines.append("两者均使用相同的 12 维障碍物 profile，区别在于聚合机制不同。")
    report_lines.append("")
    report_lines.append("## 2. 环境配置")
    report_lines.append("")
    report_lines.append("### 2.1 服务器环境")
    report_lines.append("")
    report_lines.append("- OS: Ubuntu 22.04")
    report_lines.append("- Python: 3.10")
    report_lines.append("- PyTorch: 2.7.0")
    report_lines.append("- Stable-Baselines3: 已安装")
    report_lines.append("- Gymnasium: 已安装")
    report_lines.append("- Device used for PPO: cpu")
    report_lines.append("")
    report_lines.append("### 2.2 仿真环境")
    report_lines.append("")
    report_lines.append("- 空间范围: x,y∈[-10,10], z∈[0,5]")
    report_lines.append("- 障碍物数量: 3")
    report_lines.append("- 无人机半径: 0.2")
    report_lines.append("- 障碍物半径: 0.3")
    report_lines.append("- 最大速度: UAV 2.0, obstacle 1.5")
    report_lines.append("- dt: 0.2")
    report_lines.append("- max_steps: 200")
    report_lines.append("- 障碍物运动模式: random_switch / sudden_turn")
    report_lines.append("- sudden-turn 设置: t_turn=3.0s, turn_step=15")
    report_lines.append("")
    report_lines.append("## 3. 方法设置")
    report_lines.append("")
    report_lines.append("### 3.1 risk_full_rbar")
    report_lines.append("")
    report_lines.append("risk 权重：")
    report_lines.append("")
    report_lines.append("```text")
    report_lines.append("w_i = risk_i^beta / sum_j risk_j^beta")
    report_lines.append("```")
    report_lines.append("")
    report_lines.append("R_bar 调制：")
    report_lines.append("")
    report_lines.append("```text")
    report_lines.append("c = tanh(R_sum / R_ref) * sum_i w_i h_i")
    report_lines.append("```")
    report_lines.append("")
    report_lines.append("### 3.2 attention_full")
    report_lines.append("")
    report_lines.append("learned attention：")
    report_lines.append("")
    report_lines.append("```text")
    report_lines.append("q = W_q ego")
    report_lines.append("k_i = W_k h_i")
    report_lines.append("w_i = softmax(q^T k_i / sqrt(d))")
    report_lines.append("c = sum_i w_i h_i")
    report_lines.append("```")
    report_lines.append("")
    report_lines.append("attention 模式不使用 R_bar 乘性调制，但策略输入仍包含 `R_max, R_sum`。")
    report_lines.append("")
    report_lines.append("## 4. 训练设置")
    report_lines.append("")
    report_lines.append("- total_steps: 由已完成实验结果决定，见第 8 节")
    report_lines.append("- n_envs: 8")
    report_lines.append("- seeds: 0, 1, 2")
    report_lines.append("- PPO n_steps: 1024")
    report_lines.append("- batch_size: 256")
    report_lines.append("- learning_rate: 3e-4")
    report_lines.append("- gamma: 0.99")
    report_lines.append("- gae_lambda: 0.95")
    report_lines.append("- ent_coef: 0.01")
    report_lines.append("")
    report_lines.append("## 5. 环境 sanity check")
    report_lines.append("")
    report_lines.append("### 5.1 check_env")
    report_lines.append("")
    report_lines.append("结果：通过。")
    report_lines.append("")
    report_lines.append("### 5.2 random rollout")
    report_lines.append("")
    report_lines.append("| Metric | Value |")
    report_lines.append("|---|---:|")
    report_lines.append("| random_success_rate | 0.0000 |")
    report_lines.append("| random_collision_rate | 0.3500 |")
    report_lines.append("| mean_episode_min_distance | 1.0754 |")
    report_lines.append("")
    report_lines.append("## 6. 正式评估结果")
    report_lines.append("")
    report_lines.append("### 6.1 每 seed 结果")
    report_lines.append("")
    report_lines.append("| Method | Seed | Success ↑ | Collision ↓ | MinDist ↑ | Time ↓ | Reaction ↓ |")
    report_lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for method in ["risk_full_rbar", "attention_full"]:
        for seed in [0, 1, 2]:
            random_row = random_rows_by_method_seed.get((method, seed))
            sudden_row = sudden_lookup.get((method, seed))
            if random_row is None:
                report_lines.append(f"| {method} | {seed} |  |  |  |  |  |")
                continue
            reaction = sudden_row["mean_reaction_time"] if sudden_row is not None else np.nan
            report_lines.append(
                f"| {method} | {seed} | {random_row['Success']:.4f} | {random_row['Collision']:.4f} | "
                f"{random_row['MinDist']:.4f} | {random_row['Time']:.4f} | "
                f"{reaction:.4f} |"
            )

    report_lines.append("")
    report_lines.append("### 6.2 汇总结果")
    report_lines.append("")
    report_lines.append("| Method | Success ↑ | Collision ↓ | MinDist ↑ | Reaction ↓ |")
    report_lines.append("|---|---:|---:|---:|---:|")

    for method in ["risk_full_rbar", "attention_full"]:
        success_mean, success_std = nanmeanstd(grouped_metrics[method]["random"]["Success"])
        collision_mean, collision_std = nanmeanstd(grouped_metrics[method]["random"]["Collision"])
        min_dist_mean, min_dist_std = nanmeanstd(grouped_metrics[method]["random"]["MinDist"])
        reaction_mean, reaction_std = nanmeanstd(grouped_metrics[method]["sudden"]["Reaction"])
        report_lines.append(
            f"| {method} | {fmt(success_mean, success_std)} | {fmt(collision_mean, collision_std)} | "
            f"{fmt(min_dist_mean, min_dist_std)} | {fmt(reaction_mean, reaction_std)} |"
        )

    report_lines.append("")
    report_lines.append("### 6.3 Harder Eval: random_switch_hard")
    report_lines.append("")
    report_lines.append("该评估只用于测试泛化，不参与训练。相比标准 `eval_random_switch`，其障碍物更贴近主航线、交互比例更高、速度上限更高。")
    report_lines.append("")
    report_lines.append("| Method | Seed | Success ↑ | Collision ↓ | MinDist ↑ | Time ↓ |")
    report_lines.append("|---|---:|---:|---:|---:|---:|")
    for method in ["risk_full_rbar", "attention_full"]:
        for seed in [0, 1, 2]:
            hard_row = hard_lookup.get((method, seed))
            if hard_row is None:
                report_lines.append(f"| {method} | {seed} |  |  |  |  |")
                continue
            report_lines.append(
                f"| {method} | {seed} | {hard_row['success_rate']:.4f} | {hard_row['collision_rate']:.4f} | "
                f"{hard_row['mean_min_distance']:.4f} | {hard_row['mean_time_to_goal']:.4f} |"
            )

    report_lines.append("")
    report_lines.append("## 7. 结果分析")
    report_lines.append("")
    report_lines.append("### 7.1 是否支持 risk-guided aggregation")
    report_lines.append("")
    methods = ["risk_full_rbar", "attention_full"]
    risk_success = (
        np.nanmean(grouped_metrics[methods[0]]["random"]["Success"])
        if grouped_metrics[methods[0]]["random"]["Success"]
        else np.nan
    )
    attn_success = (
        np.nanmean(grouped_metrics[methods[1]]["random"]["Success"])
        if grouped_metrics[methods[1]]["random"]["Success"]
        else np.nan
    )
    risk_collision = (
        np.nanmean(grouped_metrics[methods[0]]["random"]["Collision"])
        if grouped_metrics[methods[0]]["random"]["Collision"]
        else np.nan
    )
    attn_collision = (
        np.nanmean(grouped_metrics[methods[1]]["random"]["Collision"])
        if grouped_metrics[methods[1]]["random"]["Collision"]
        else np.nan
    )
    risk_min_dist = (
        np.nanmean(grouped_metrics[methods[0]]["random"]["MinDist"])
        if grouped_metrics[methods[0]]["random"]["MinDist"]
        else np.nan
    )
    attn_min_dist = (
        np.nanmean(grouped_metrics[methods[1]]["random"]["MinDist"])
        if grouped_metrics[methods[1]]["random"]["MinDist"]
        else np.nan
    )
    risk_reaction = (
        np.nanmean(grouped_metrics[methods[0]]["sudden"]["Reaction"])
        if grouped_metrics[methods[0]]["sudden"]["Reaction"]
        else np.nan
    )
    attn_reaction = (
        np.nanmean(grouped_metrics[methods[1]]["sudden"]["Reaction"])
        if grouped_metrics[methods[1]]["sudden"]["Reaction"]
        else np.nan
    )

    if np.isnan(risk_success) or np.isnan(attn_success):
        conclusion = "结果尚不完整"
    elif risk_success > attn_success and risk_collision <= attn_collision and risk_min_dist >= attn_min_dist:
        conclusion = "支持"
    elif abs(risk_success - attn_success) < 0.03 and abs(risk_collision - attn_collision) < 0.03:
        conclusion = "基本持平"
    else:
        conclusion = "不支持"
    report_lines.append("结论：")
    report_lines.append("")
    report_lines.append("```text")
    report_lines.append(conclusion)
    report_lines.append("```")
    report_lines.append("")
    report_lines.append("依据：")
    report_lines.append(f"- success rate: risk={risk_success:.4f}, attention={attn_success:.4f}")
    report_lines.append(f"- collision rate: risk={risk_collision:.4f}, attention={attn_collision:.4f}")
    report_lines.append(f"- minimum distance: risk={risk_min_dist:.4f}, attention={attn_min_dist:.4f}")
    report_lines.append(f"- reaction time: risk={risk_reaction:.4f}, attention={attn_reaction:.4f}")
    report_lines.append("")
    report_lines.append("### 7.2 若 risk_full_rbar 优于 attention_full")
    report_lines.append("")
    report_lines.append("说明：")
    report_lines.append("")
    report_lines.append("```text")
    report_lines.append("risk-guided aggregation 在该预实验设定下具有继续投入价值。")
    report_lines.append("```")
    report_lines.append("")
    report_lines.append("### 7.3 若两者接近")
    report_lines.append("")
    report_lines.append("说明：")
    report_lines.append("")
    report_lines.append("```text")
    report_lines.append("该结果不直接否定 risk 方案。下一步应进一步分析 risk_rise_time、risk-threat consistency 和权重可解释性。")
    report_lines.append("```")
    report_lines.append("")
    report_lines.append("### 7.4 若 risk_full_rbar 明显差于 attention_full")
    report_lines.append("")
    report_lines.append("优先排查：")
    report_lines.append("")
    report_lines.append("1. risk_i 是否几乎全为 0；")
    report_lines.append("2. risk_i 是否几乎全为 1；")
    report_lines.append("3. Sigma 是否过大或过小；")
    report_lines.append("4. distance gate 是否过强；")
    report_lines.append("5. beta 是否过大；")
    report_lines.append("6. R_bar 是否把 context 压得太小；")
    report_lines.append("7. reward 是否鼓励硬冲目标；")
    report_lines.append("8. 训练环境是否过难或过易；")
    report_lines.append("9. Sigma 在匀速阶段是否衰减到 sigma_min，导致突变时 risk 上升慢。")
    report_lines.append("")
    report_lines.append("下一步建议转向：")
    report_lines.append("")
    report_lines.append("```text")
    report_lines.append("risk-initialized attention / risk residual attention")
    report_lines.append("```")
    report_lines.append("")
    report_lines.append("形式：")
    report_lines.append("")
    report_lines.append("```text")
    report_lines.append("w_final = (1 - alpha) * w_risk + alpha * w_attention")
    report_lines.append("```")
    report_lines.append("")
    report_lines.append("## 8. 已生成文件")
    report_lines.append("")
    report_lines.append("- checkpoints: 见 `checkpoints/`")
    report_lines.append("- results csv: 见 `results/`")
    report_lines.append("- TensorBoard logs: 见 `runs/`")
    report_lines.append("- scripts: `scripts/check_env.py`, `scripts/random_rollout.py`, `scripts/run_preexp.sh`, `scripts/run_eval.sh`, `scripts/aggregate_results.py`")
    report_lines.append("- source files: `envs/dynamic_obstacle_env.py`, `policies/obstacle_set_extractor.py`, `train.py`, `eval.py`")
    report_lines.append("")
    report_lines.append("## 9. Codex 执行总结")
    report_lines.append("")
    report_lines.append("说明：")
    report_lines.append("")
    report_lines.append("1. 已完成的文件：见第 8 节。")
    report_lines.append("2. 成功运行的命令：需结合本次执行日志补充。")
    report_lines.append("3. 失败或跳过的命令：需结合本次执行日志补充。")
    report_lines.append("4. 当前最可信结论：见第 7 节。")
    report_lines.append("5. 后续建议：若正式结果不足，优先检查风险信号动态范围与 sudden-turn 响应。")
    report_lines.append("")

    with open("PREEXP_REPORT.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(report_lines))

    print(f"wrote {summary_csv_path}")
    print("wrote PREEXP_REPORT.md")


if __name__ == "__main__":
    main()
