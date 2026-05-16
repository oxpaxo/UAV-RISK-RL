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


COMPLETE_FLAG = "PHASE_N3_GPSI_PPO_NO_SHIELD_COMPLETE.flag"
STOP_FLAGS = {
    "schema_mismatch": "PHASE_N3_STOP_SCHEMA_MISMATCH.flag",
    "train_failed": "PHASE_N3_STOP_TRAIN_FAILED.flag",
    "eval_failed": "PHASE_N3_STOP_EVAL_FAILED.flag",
    "trace_diagnostics_failed": "PHASE_N3_STOP_TRACE_DIAGNOSTICS_FAILED.flag",
}
SCENARIOS = [
    "eval_flow_id",
    "eval_flow_high_density",
    "eval_flow_high_speed",
    "eval_flow_high_threat",
    "eval_flow_mixed_ood",
    "eval_flow_sudden_threat",
]
CHECKPOINT_STEPS = [250_000, 500_000, 1_000_000, 1_500_000]


class PhaseN3AnalysisStop(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Phase N3 Gpsi-PPO no-shield outputs.")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--checkpoint-dir", default="checkpoints/env_v2_gpsi_heada_ppo_s0")
    parser.add_argument("--smoke-dir", default="checkpoints/env_v2_gpsi_heada_ppo_s0_smoke")
    parser.add_argument("--expected-episodes", type=int, default=50)
    parser.add_argument("--expected-train-steps", type=int, default=1_500_000)
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
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_stop(result_dir: Path, reason: str, detail: str) -> None:
    flag_name = STOP_FLAGS.get(reason, STOP_FLAGS["eval_failed"])
    write_text(result_dir / flag_name, f"{reason}\n{detail}\n")
    write_text(result_dir / "phase_n3_status.txt", f"stopped:{flag_name}\n")
    write_report(
        result_dir=result_dir,
        terminal_decision=f"phase_n3_stopped_{reason}",
        complete=False,
        facts=[f"Analysis stopped: {reason}"],
        warnings=[detail.strip()],
        tables={},
        files={},
        n4_ready=False,
    )


def read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        if required:
            raise FileNotFoundError(f"missing or empty CSV: {rel(path)}")
        return pd.DataFrame()
    return pd.read_csv(path)


def fmt(value: Any) -> str:
    try:
        f = float(value)
    except Exception:
        return str(value)
    if math.isnan(f):
        return "nan"
    return f"{f:.4f}"


def check_no_nan(df: pd.DataFrame, path: Path, allow_cols: set[str] | None = None) -> None:
    allow_cols = allow_cols or set()
    numeric = df.select_dtypes(include=[np.number])
    for col in numeric.columns:
        if col in allow_cols:
            continue
        values = numeric[col].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise PhaseN3AnalysisStop("eval_failed", f"inf found in {rel(path)} column {col}")


def validate_required_outputs(result_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_dir = ROOT / args.checkpoint_dir
    smoke_dir = ROOT / args.smoke_dir
    required_paths = {
        "n2_flag": ROOT / "results/env_v2_phase_n2_gpsi_heada_offline/PHASE_N2_HEADA_OFFLINE_COMPLETE.flag",
        "gpsi_checkpoint": ROOT / "work_dirs/gpsi_heada_v1_nll/best.pth",
        "smoke_final": smoke_dir / "final.zip",
        "formal_final": checkpoint_dir / "final.zip",
        "formal_best": checkpoint_dir / "best_by_eval.zip",
        "schema_check": result_dir / "tables/phase_n3_schema_check.csv",
        "train_curve": result_dir / "tables/phase_n3_train_curve.csv",
        "episode_metrics": result_dir / "tables/phase_n3_episode_metrics.csv",
        "eval_summary": result_dir / "tables/phase_n3_eval_summary.csv",
        "scenario_breakdown": result_dir / "tables/phase_n3_scenario_breakdown.csv",
        "motion_breakdown": result_dir / "tables/phase_n3_motion_mode_breakdown.csv",
        "raw_unsafe": result_dir / "tables/phase_n3_raw_unsafe_action_summary.csv",
        "gpsi_output": result_dir / "tables/phase_n3_gpsi_output_summary.csv",
        "gpsi_forward_profile": result_dir / "tables/phase_n3_gpsi_forward_profile.csv",
        "command_manifest": result_dir / "tables/phase_n3_command_manifest.csv",
    }
    for step in CHECKPOINT_STEPS:
        required_paths[f"checkpoint_{step}"] = checkpoint_dir / f"checkpoint_{step // 1000}k.zip"
    missing = [f"{name}: {rel(path)}" for name, path in required_paths.items() if not path.exists() or path.stat().st_size == 0]
    if missing:
        train_missing = any("checkpoint_" in item or "formal_" in item or "smoke_" in item for item in missing)
        raise PhaseN3AnalysisStop("train_failed" if train_missing else "eval_failed", "missing required artifacts:\n" + "\n".join(missing))

    trace_dirs = [
        result_dir / "traces/sampled_success_traces",
        result_dir / "traces/sampled_collision_traces",
        result_dir / "traces/sampled_near_miss_traces",
    ]
    missing_trace_dirs = [rel(path) for path in trace_dirs if not any(path.glob("*.csv"))]
    if missing_trace_dirs:
        raise PhaseN3AnalysisStop("trace_diagnostics_failed", f"missing sampled trace CSVs in: {missing_trace_dirs}")
    return {"checkpoint_dir": checkpoint_dir, "smoke_dir": smoke_dir}


def build_attention_comparison(summary: pd.DataFrame, result_dir: Path) -> pd.DataFrame:
    attention = summary[summary["method"] == "attention_full"].copy()
    gpsi = summary[summary["method"] == "gpsi_heada_ppo_no_shield"].copy()
    final = gpsi[gpsi["checkpoint_step"] == 1_500_000].copy()
    if attention.empty or final.empty:
        raise PhaseN3AnalysisStop("eval_failed", "attention reference or Gpsi final summary missing")
    cols = [
        "scenario",
        "success_rate",
        "collision_rate",
        "near_miss_rate",
        "mean_min_distance",
        "episode_min_distance",
        "mean_time",
        "progress",
        "episode_reward",
        "raw_unsafe_action_rate",
    ]
    merged = final[cols].merge(attention[cols], on="scenario", how="inner", suffixes=("_gpsi", "_attention"))
    for metric in cols[1:]:
        merged[f"delta_{metric}"] = merged[f"{metric}_gpsi"] - merged[f"{metric}_attention"]
    merged.to_csv(result_dir / "tables/phase_n3_attention_reference_comparison.csv", index=False)
    return merged


def plot_train_curve(train: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    if not train.empty and {"steps", "steps_per_second"}.issubset(train.columns):
        plt.plot(train["steps"], train["steps_per_second"], marker="o")
    plt.xlabel("training steps")
    plt.ylabel("steps/second heartbeat")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_success_collision(summary: pd.DataFrame, path: Path) -> None:
    final = summary[summary["method"] == "gpsi_heada_ppo_no_shield"].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.8))
    agg = final.groupby("checkpoint_step").agg(success=("success_rate", "mean"), collision=("collision_rate", "mean")).reset_index()
    if not agg.empty:
        plt.plot(agg["checkpoint_step"], agg["success"], marker="o", label="success")
        plt.plot(agg["checkpoint_step"], agg["collision"], marker="x", label="collision")
    plt.xlabel("checkpoint step")
    plt.ylabel("rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_checkpoint_bars(summary: pd.DataFrame, path: Path) -> None:
    gpsi = summary[summary["method"] == "gpsi_heada_ppo_no_shield"].copy()
    agg = gpsi.groupby("checkpoint_step").agg(success=("success_rate", "mean"), collision=("collision_rate", "mean")).reset_index()
    x = np.arange(len(agg))
    width = 0.35
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    if len(agg):
        plt.bar(x - width / 2, agg["success"], width, label="success")
        plt.bar(x + width / 2, agg["collision"], width, label="collision")
        plt.xticks(x, [str(int(v)) for v in agg["checkpoint_step"]], rotation=20)
    plt.xlabel("checkpoint")
    plt.ylabel("rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_scenario_breakdown(summary: pd.DataFrame, path: Path) -> None:
    final = summary[(summary["method"] == "gpsi_heada_ppo_no_shield") & (summary["checkpoint_step"] == 1_500_000)].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 4.8))
    if not final.empty:
        x = np.arange(len(final))
        plt.bar(x, final["success_rate"], label="success")
        plt.plot(x, final["collision_rate"], marker="x", color="tab:red", label="collision")
        plt.xticks(x, final["scenario"], rotation=30, ha="right")
    plt.ylabel("rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_motion_breakdown(motion: pd.DataFrame, path: Path) -> None:
    final = motion[(motion["method"] == "gpsi_heada_ppo_no_shield") & (motion["checkpoint_step"] == 1_500_000)].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    if not final.empty and "threat_motion_mode" in final.columns:
        x = np.arange(len(final))
        plt.bar(x, final["success_rate"], label="success")
        plt.plot(x, final["near_miss_rate"], marker="x", color="tab:orange", label="near_miss")
        plt.xticks(x, final["threat_motion_mode"], rotation=30, ha="right")
    plt.ylabel("rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_raw_unsafe(raw: pd.DataFrame, path: Path) -> None:
    agg = raw.groupby(["method", "checkpoint_step"], dropna=False).agg(raw_unsafe_rate=("raw_unsafe_rate", "mean")).reset_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    for method, group in agg.groupby("method"):
        group = group.sort_values("checkpoint_step")
        plt.plot(group["checkpoint_step"], group["raw_unsafe_rate"], marker="o", label=method)
    plt.xlabel("checkpoint step")
    plt.ylabel("raw unsafe rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_gpsi_distribution(gpsi_steps: pd.DataFrame, result_dir: Path) -> None:
    final = gpsi_steps[gpsi_steps["checkpoint_step"] == 1_500_000].copy() if "checkpoint_step" in gpsi_steps else gpsi_steps
    for col, filename, xlabel in [
        ("mean_delta_norm_1s", "gpsi_delta_norm_distribution.png", "mean delta norm 1s"),
        ("mean_logvar_xy_1s", "gpsi_logvar_distribution.png", "mean logvar xy 1s"),
    ]:
        path = result_dir / "plots" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6.5, 4.2))
        if col in final.columns and not final.empty:
            values = pd.to_numeric(final[col], errors="coerce").dropna()
            if not values.empty:
                plt.hist(values, bins=40)
        plt.xlabel(xlabel)
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close()


def generate_plots(result_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    plot_train_curve(tables["train"], result_dir / "plots/train_reward_curve.png")
    plot_success_collision(tables["summary"], result_dir / "plots/train_success_collision_curve.png")
    plot_checkpoint_bars(tables["summary"], result_dir / "plots/checkpoint_success_collision.png")
    plot_scenario_breakdown(tables["summary"], result_dir / "plots/scenario_breakdown.png")
    plot_motion_breakdown(tables["motion"], result_dir / "plots/motion_mode_breakdown.png")
    plot_raw_unsafe(tables["raw"], result_dir / "plots/raw_unsafe_rate_by_checkpoint.png")
    plot_gpsi_distribution(tables["gpsi_steps"], result_dir)


def table_head_md(df: pd.DataFrame, cols: list[str], max_rows: int = 12) -> list[str]:
    if df.empty:
        return ["No rows."]
    view = df[cols].head(max_rows).copy()
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) if isinstance(row[col], (float, int, np.floating, np.integer)) else str(row[col]) for col in cols) + " |")
    return lines


def collect_files(result_dir: Path, checkpoint_dir: Path, smoke_dir: Path) -> dict[str, list[str]]:
    return {
        "checkpoints": [rel(path) for path in sorted(checkpoint_dir.glob("*.zip"))] + [rel(path) for path in sorted(smoke_dir.glob("*.zip"))],
        "tables": [rel(path) for path in sorted((result_dir / "tables").glob("*.csv"))],
        "plots": [rel(path) for path in sorted((result_dir / "plots").glob("*.png"))],
        "traces": [rel(path) for path in sorted((result_dir / "traces").glob("**/*.csv"))[:40]],
        "logs": [rel(path) for path in sorted((result_dir / "logs").glob("*.log"))] + [rel(result_dir / "phase_n3_watcher.log")],
        "flags": [rel(path) for path in sorted(result_dir.glob("*.flag"))],
    }


def write_report(
    *,
    result_dir: Path,
    terminal_decision: str,
    complete: bool,
    facts: list[str],
    warnings: list[str],
    tables: dict[str, pd.DataFrame],
    files: dict[str, Any],
    n4_ready: bool,
) -> None:
    lines = [
        "# Phase N3 Gpsi-PPO No-Shield Report",
        "",
        "## Terminal Decision",
        "",
        f"`terminal_decision = {terminal_decision}`",
        "",
        "Phase N3 complete." if complete else "Phase N3 stopped before completion.",
        "Gpsi-PPO no-shield raw policy evaluation is ready for Phase N4 shield fair comparison." if n4_ready else "Phase N4 is blocked until the listed issue is fixed.",
        "",
        "## Background And Goal",
        "",
        "N3 trains a raw PPO velocity policy with frozen Gpsi-HeadA obstacle augmentation. It does not use a safety shield, action filtering, action projection, candidate velocity risk map, learned R(s,a), safety-cost PPO, or Gpsi fine-tuning.",
        "",
        "## Experiment-Supported Facts",
        "",
    ]
    lines.extend([f"- {fact}" for fact in facts])
    lines.extend(
        [
            "",
            "## Gpsi Frozen And Wrapper",
            "",
            "- Gpsi checkpoint: `work_dirs/gpsi_heada_v1_nll/best.pth`.",
            "- Wrapper: `envs/wrappers/gpsi_obs_wrapper.py`.",
            "- Online inputs: `ego_current`, `obs_current`, `history_rel_pos`, `history_rel_vel`, `history_valid_mask`.",
            "- Histories are keyed by `obstacle_id`; replacement creates a new left-padded history.",
            "- Gpsi is set to `eval()`, all parameters use `requires_grad=False`, and forward runs under `torch.no_grad()`.",
            "",
            "## Observation Schema",
            "",
            "`obs_i_aug = [obs_i(12), z_i(64), delta_hat_i(9), logvar_hat_i(9)]`, so obstacle profile dim is `94`.",
            "",
            "## Augmentation Normalization",
            "",
            "- Gpsi input normalization uses train-split-only stats stored in the N2 checkpoint.",
            "- `delta_hat` is divided by `delta_scale=5.0` for PPO input, while raw diagnostics keep unscaled values.",
            "- `logvar_hat` is clamped to `[-5, 3]` before PPO input.",
            "- `z_i` is not additionally normalized in N3 v1.",
            "",
            "## PPO Backbone",
            "",
            "Masked attention over active obstacles is used for both actor and critic through SB3 `MultiInputPolicy`; actor/critic MLP heads are symmetric with `pi=[128,128]`, `vf=[128,128]`.",
            "",
        ]
    )
    if tables:
        summary = tables.get("summary", pd.DataFrame())
        comparison = tables.get("comparison", pd.DataFrame())
        motion = tables.get("motion", pd.DataFrame())
        raw = tables.get("raw", pd.DataFrame())
        gpsi = tables.get("gpsi", pd.DataFrame())
        lines.extend(["## Checkpoint Eval Summary", ""])
        lines.extend(table_head_md(summary.sort_values(["method", "checkpoint_step", "scenario"]) if not summary.empty else summary, ["method", "checkpoint_step", "scenario", "success_rate", "collision_rate", "near_miss_rate", "progress", "raw_unsafe_action_rate"]))
        lines.extend(["", "## Attention Reference Comparison", ""])
        if not comparison.empty:
            cols = ["scenario", "success_rate_gpsi", "success_rate_attention", "delta_success_rate", "collision_rate_gpsi", "collision_rate_attention", "delta_collision_rate", "raw_unsafe_action_rate_gpsi", "raw_unsafe_action_rate_attention"]
            lines.extend(table_head_md(comparison, cols))
        else:
            lines.append("No comparison rows.")
        lines.extend(["", "## Motion-Mode Breakdown", ""])
        motion_cols = [col for col in ["method", "checkpoint_step", "threat_motion_mode", "success_rate", "collision_rate", "near_miss_rate", "progress"] if col in motion.columns]
        lines.extend(table_head_md(motion.sort_values(motion_cols[:3]) if motion_cols and not motion.empty else motion, motion_cols or list(motion.columns[:6])))
        lines.extend(["", "## Raw Action Unsafe Diagnostics", ""])
        raw_cols = [col for col in ["method", "checkpoint_step", "scenario", "motion_mode", "threat_class", "raw_unsafe_rate", "raw_min_predicted_cpa"] if col in raw.columns]
        lines.extend(table_head_md(raw.sort_values(raw_cols[:5]) if raw_cols and not raw.empty else raw, raw_cols or list(raw.columns[:6])))
        lines.extend(["", "## Gpsi Output Diagnostics", ""])
        gpsi_cols = [col for col in ["method", "checkpoint_step", "scenario", "motion_mode", "threat_class", "mean_delta_norm_1s", "mean_logvar_xy_1s", "history_valid_ratio_nearest"] if col in gpsi.columns]
        lines.extend(table_head_md(gpsi.sort_values(gpsi_cols[:5]) if gpsi_cols and not gpsi.empty else gpsi, gpsi_cols or list(gpsi.columns[:6])))
    lines.extend(["", "## Phase B Context", ""])
    lines.append("Phase B geometry/filter baselines remain background upper-bound context only. N3 direct comparison is `attention_full_1500k` versus `Gpsi-HeadA + PPO no shield`.")
    lines.extend(["", "## Reasonable Inferences", ""])
    if complete:
        lines.append("- If no-shield Gpsi-PPO does not outperform attention_full, N4 can still proceed because N2 supports testing Gpsi uncertainty on the shield side.")
    else:
        lines.append("- No N4 inference should be made from this partial run.")
    lines.extend(["", "## Risks / Warnings", ""])
    lines.extend([f"- {warning}" for warning in warnings] if warnings else ["- No blocking warning."])
    lines.extend(["", "## Output Artifacts", ""])
    for key in ["checkpoints", "tables", "plots", "traces", "logs", "flags"]:
        values = files.get(key, []) if files else []
        lines.append(f"### {key}")
        if values:
            lines.extend([f"- `{value}`" for value in values[:80]])
            if len(values) > 80:
                lines.append(f"- ... {len(values) - 80} more")
        else:
            lines.append("- none recorded")
    lines.extend(["", "## N4 Readiness", ""])
    lines.append("Can enter Phase N4: yes." if n4_ready else "Can enter Phase N4: no.")
    if not n4_ready:
        lines.append("Needed before N4: resolve the stop condition and rerun watcher to a complete flag.")
    write_text(result_dir / "PHASE_N3_GPSI_PPO_NO_SHIELD_REPORT.md", "\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    result_dir = ROOT / args.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    try:
        paths = validate_required_outputs(result_dir, args)
        tables = {
            "train": read_csv(result_dir / "tables/phase_n3_train_curve.csv"),
            "summary": read_csv(result_dir / "tables/phase_n3_eval_summary.csv"),
            "episodes": read_csv(result_dir / "tables/phase_n3_episode_metrics.csv"),
            "motion": read_csv(result_dir / "tables/phase_n3_motion_mode_breakdown.csv"),
            "raw": read_csv(result_dir / "tables/phase_n3_raw_unsafe_action_summary.csv"),
            "gpsi": read_csv(result_dir / "tables/phase_n3_gpsi_output_summary.csv"),
            "gpsi_steps": read_csv(result_dir / "tables/phase_n3_gpsi_output_steps.csv"),
            "schema": read_csv(result_dir / "tables/phase_n3_schema_check.csv"),
        }
        check_no_nan(tables["summary"], result_dir / "tables/phase_n3_eval_summary.csv", allow_cols={"reaction_time_nan_style"})
        expected_methods = {"attention_full", "gpsi_heada_ppo_no_shield"}
        methods = set(tables["summary"]["method"].astype(str))
        if not expected_methods.issubset(methods):
            raise PhaseN3AnalysisStop("eval_failed", f"missing methods in summary: expected {expected_methods}, got {methods}")
        gpsi_steps = set(pd.to_numeric(tables["summary"][tables["summary"]["method"] == "gpsi_heada_ppo_no_shield"]["checkpoint_step"], errors="coerce").dropna().astype(int))
        for step in [500_000, 1_000_000, 1_500_000]:
            if step not in gpsi_steps:
                raise PhaseN3AnalysisStop("eval_failed", f"missing Gpsi eval checkpoint step {step}")
        scenarios = set(tables["summary"]["scenario"].astype(str))
        if not set(SCENARIOS).issubset(scenarios):
            raise PhaseN3AnalysisStop("eval_failed", f"missing scenarios: {sorted(set(SCENARIOS) - scenarios)}")
        counts = tables["episodes"].groupby(["method", "checkpoint_step", "scenario"]).size().reset_index(name="episodes")
        bad_counts = counts[counts["episodes"] < args.expected_episodes]
        if not bad_counts.empty:
            raise PhaseN3AnalysisStop("eval_failed", f"not enough eval episodes:\n{bad_counts.to_string(index=False)}")
        comparison = build_attention_comparison(tables["summary"], result_dir)
        tables["comparison"] = comparison
        generate_plots(result_dir, tables)
        facts = [
            "Phase N2 complete flag exists.",
            "Gpsi NLL checkpoint exists and was loaded by wrapper.",
            "Gpsi freeze/schema check CSV was generated.",
            f"Formal PPO checkpoints exist through {args.expected_train_steps} steps.",
            "Evaluation completed for 6 scenarios with required episodes.",
            "Attention reference comparison, scenario/motion/threat breakdowns, raw unsafe diagnostics, Gpsi diagnostics, plots, and sampled traces were generated.",
            "No safety shield, action filtering, action projection, dense safety cost, or Gpsi training was used.",
        ]
        warnings: list[str] = []
        final_gpsi = tables["summary"][(tables["summary"]["method"] == "gpsi_heada_ppo_no_shield") & (tables["summary"]["checkpoint_step"] == 1_500_000)]
        if not final_gpsi.empty:
            mean_success = float(pd.to_numeric(final_gpsi["success_rate"], errors="coerce").mean())
            mean_progress = float(pd.to_numeric(final_gpsi["progress"], errors="coerce").mean())
            if mean_success < 0.05 and mean_progress < 0.20:
                warnings.append(f"Final Gpsi-PPO learned weak task progress: mean_success={mean_success:.4f}, mean_progress={mean_progress:.4f}.")
        files = collect_files(result_dir, paths["checkpoint_dir"], paths["smoke_dir"])
        terminal_decision = "phase_n3_gpsi_ppo_no_shield_complete"
        write_text(result_dir / COMPLETE_FLAG, terminal_decision + "\n")
        files = collect_files(result_dir, paths["checkpoint_dir"], paths["smoke_dir"])
        write_text(result_dir / "phase_n3_status.txt", "complete\n")
        write_report(
            result_dir=result_dir,
            terminal_decision=terminal_decision,
            complete=True,
            facts=facts,
            warnings=warnings,
            tables=tables,
            files=files,
            n4_ready=True,
        )
        print(f"terminal_decision = {terminal_decision}", flush=True)
    except PhaseN3AnalysisStop as exc:
        write_stop(result_dir, exc.reason, exc.detail)
        raise SystemExit(2) from exc
    except Exception:
        detail = traceback.format_exc()
        write_stop(result_dir, "eval_failed", detail)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
