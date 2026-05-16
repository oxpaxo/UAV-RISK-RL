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


COMPLETE_FLAG = "PHASE_N3P_NOZ_REPRESENTATION_ABLATION_COMPLETE.flag"
STOP_FLAGS = {
    "gpsi_checkpoint_missing": "PHASE_N3P_STOP_GPSI_CHECKPOINT_MISSING.flag",
    "baseline_artifacts_missing": "PHASE_N3P_STOP_BASELINE_ARTIFACTS_MISSING.flag",
    "config_invalid": "PHASE_N3P_STOP_CONFIG_INVALID.flag",
    "train_failed": "PHASE_N3P_STOP_TRAIN_FAILED.flag",
    "eval_failed": "PHASE_N3P_STOP_EVAL_FAILED.flag",
    "diagnostics_failed": "PHASE_N3P_STOP_DIAGNOSTICS_FAILED.flag",
    "feature_scale_invalid": "PHASE_N3P_STOP_FEATURE_SCALE_INVALID.flag",
    "watcher_failed": "PHASE_N3P_STOP_WATCHER_FAILED.flag",
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
    parser = argparse.ArgumentParser(description="Analyze Phase N3P no-z representation ablation outputs.")
    parser.add_argument("--result-dir", default="results/env_v2_phase_n3p_noz_representation_ablation")
    parser.add_argument("--expected-episodes", type=int, default=50)
    parser.add_argument("--n3r-noz-success", type=float, default=0.4233)
    parser.add_argument("--n3r-noz-collision", type=float, default=0.5767)
    parser.add_argument("--n3f-noz-success", type=float, default=0.5633)
    parser.add_argument("--n3f-noz-collision", type=float, default=0.4367)
    parser.add_argument("--attention-success", type=float, default=0.6100)
    parser.add_argument("--attention-collision", type=float, default=0.3900)
    parser.add_argument("--z2-success", type=float, default=0.5067)
    parser.add_argument("--z2-collision", type=float, default=0.4933)
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty CSV: {rel(path)}")
    return pd.read_csv(path)


def write_stop(result_dir: Path, reason: str, detail: str) -> None:
    flag = STOP_FLAGS.get(reason, STOP_FLAGS["diagnostics_failed"])
    write_text(result_dir / flag, f"{reason}\n{detail.strip()}\n")
    write_text(result_dir / "phase_n3p_status.txt", f"stopped:{flag}\n")
    write_text(
        result_dir / "PHASE_N3P_NOZ_REPRESENTATION_ABLATION_REPORT.md",
        "\n".join(
            [
                "# Phase N3P No-Z Representation Ablation Report",
                "",
                f"`terminal_decision = phase_n3p_stopped_{reason}`",
                "",
                "Partial report generated because analysis reached a stop condition.",
                "",
                "```text",
                detail.strip(),
                "```",
                "",
                "can_enter_N4: no",
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


def table_md(df: pd.DataFrame, cols: list[str], max_rows: int = 24) -> list[str]:
    if df.empty:
        return ["No rows."]
    view = df[cols].head(max_rows).copy()
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) if isinstance(row[col], (float, int, np.floating, np.integer)) else str(row[col]) for col in cols) + " |")
    return lines


def aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby(["method_key", "method", "checkpoint_label"], dropna=False)
        .agg(
            checkpoint_step=("checkpoint_step", "max"),
            checkpoint_path=("checkpoint_path", "first"),
            episodes=("episodes", "sum"),
            success_rate=("success_rate", "mean"),
            collision_rate=("collision_rate", "mean"),
            near_miss_rate=("near_miss_rate", "mean"),
            progress=("progress", "mean"),
            mean_min_distance=("mean_min_distance", "mean"),
            episode_min_distance=("episode_min_distance", "mean"),
            episode_reward=("episode_reward", "mean"),
            raw_unsafe_action_rate=("raw_unsafe_action_rate", "mean"),
            raw_safe_margin_unsafe_action_rate=("raw_safe_margin_unsafe_action_rate", "mean"),
            action_norm=("action_norm", "mean"),
            action_delta=("action_delta", "mean"),
            no_response_rate=("no_response_rate", "mean"),
            raw_min_predicted_cpa=("raw_min_predicted_cpa", "mean"),
            nan_or_crash=("nan_or_crash", "max"),
        )
        .reset_index()
    )


def validate(result_dir: Path, args: argparse.Namespace) -> None:
    required = [
        result_dir / "tables/phase_n3p_config_manifest.csv",
        result_dir / "tables/phase_n3p_command_manifest.csv",
        result_dir / "tables/phase_n3p_resource_affinity.csv",
        result_dir / "tables/phase_n3p_schema_check.csv",
        result_dir / "tables/phase_n3p_train_curve.csv",
        result_dir / "tables/phase_n3p_train_heartbeat.csv",
        result_dir / "tables/phase_n3p_checkpoint_eval_summary.csv",
        result_dir / "tables/phase_n3p_eval_summary.csv",
        result_dir / "tables/phase_n3p_scenario_breakdown.csv",
        result_dir / "tables/phase_n3p_motion_mode_breakdown.csv",
        result_dir / "tables/phase_n3p_threat_class_breakdown.csv",
        result_dir / "tables/phase_n3p_raw_unsafe_action_summary.csv",
        result_dir / "tables/phase_n3p_gpsi_output_summary.csv",
        result_dir / "tables/phase_n3p_feature_block_stats.csv",
    ]
    ckpt_dirs = [
        ROOT / "checkpoints/env_v2_gpsi_heada_ppo_n3p_obs_delta_only_s0",
        ROOT / "checkpoints/env_v2_gpsi_heada_ppo_n3p_logvar_scaled_s0",
        ROOT / "checkpoints/env_v2_gpsi_heada_ppo_n3p_block_projected_s0",
    ]
    for ckpt_dir in ckpt_dirs:
        for filename in ["checkpoint_250k.zip", "checkpoint_500k.zip", "final.zip", "best_by_eval.zip", "TRAIN_COMPLETE.flag"]:
            required.append(ckpt_dir / filename)
    missing = [rel(path) for path in required if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise AnalysisStop("train_failed", "missing required N3P artifacts:\n" + "\n".join(missing))
    episodes = read_csv(result_dir / "tables/phase_n3p_episode_metrics.csv")
    counts = episodes.groupby(["method_key", "checkpoint_label", "scenario"]).size().reset_index(name="episodes")
    bad = counts[counts["episodes"] < int(args.expected_episodes)]
    if not bad.empty:
        raise AnalysisStop("eval_failed", "not enough eval episodes:\n" + bad.to_string(index=False))
    required_labels = {"250k", "500k", "final", "best_by_eval"}
    methods = {"obs_delta_only", "logvar_scaled", "block_projected"}
    labels = set((str(row.method_key), str(row.checkpoint_label)) for row in episodes[["method_key", "checkpoint_label"]].drop_duplicates().itertuples(index=False))
    missing_pairs = sorted((method, label) for method in methods for label in required_labels if (method, label) not in labels)
    if missing_pairs:
        raise AnalysisStop("eval_failed", f"missing eval method/checkpoint labels: {missing_pairs}")


def diagnostics(result_dir: Path, args: argparse.Namespace) -> pd.DataFrame:
    gpsi = read_csv(result_dir / "tables/phase_n3p_gpsi_output_summary.csv")
    feats = read_csv(result_dir / "tables/phase_n3p_feature_block_stats.csv")
    rows: list[dict[str, Any]] = []
    for method in ["obs_delta_only", "logvar_scaled", "block_projected"]:
        g = gpsi[(gpsi.method_key.astype(str) == method) & (gpsi.checkpoint_label.astype(str) == "500k")]
        f = feats[(feats.method_key.astype(str) == method) & (feats.checkpoint_label.astype(str) == "500k")]
        if g.empty or f.empty:
            raise AnalysisStop("diagnostics_failed", f"missing diagnostics for {method} 500k")
        delta_p95 = pd.to_numeric(g.get("delta_norm_1s_p95", pd.Series(dtype=float)), errors="coerce").max()
        delta_max = pd.to_numeric(g.get("delta_norm_1s_max", pd.Series(dtype=float)), errors="coerce").max()
        inactive = pd.to_numeric(g.get("inactive_forwarded_count_max", pd.Series(dtype=float)), errors="coerce").max()
        logvar_span = pd.to_numeric(g.get("logvar_xy_1s_span", pd.Series(dtype=float)), errors="coerce").max()
        nonfinite = int(pd.to_numeric(f.get("nan_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        nonfinite += int(pd.to_numeric(f.get("inf_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        full_l2 = pd.to_numeric(f[f.block.astype(str) == "full_aug_obs"].get("l2_norm_p95", pd.Series(dtype=float)), errors="coerce").max()
        logvar_raw_l2 = pd.to_numeric(f[f.block.astype(str) == "logvar_raw_9_clamped"].get("l2_norm_p95", pd.Series(dtype=float)), errors="coerce").max()
        logvar_scaled_l2 = pd.to_numeric(f[f.block.astype(str) == "logvar_scaled_9_policy"].get("l2_norm_p95", pd.Series(dtype=float)), errors="coerce").max()
        adapter_l2 = pd.to_numeric(f[f.block.astype(str) == "adapter_output_64"].get("l2_norm_p95", pd.Series(dtype=float)), errors="coerce").max()
        if method == "obs_delta_only":
            scale_ok = not np.isfinite(logvar_scaled_l2)
        elif method == "logvar_scaled":
            scale_ok = np.isfinite(logvar_scaled_l2) and logvar_scaled_l2 <= 4.0 and (not np.isfinite(logvar_raw_l2) or logvar_scaled_l2 < logvar_raw_l2 * 0.35)
        else:
            scale_ok = (
                np.isfinite(logvar_scaled_l2)
                and logvar_scaled_l2 <= 4.0
                and np.isfinite(adapter_l2)
                and 1.0 <= adapter_l2 <= 12.0
                and (not np.isfinite(logvar_raw_l2) or logvar_scaled_l2 < logvar_raw_l2 * 0.35)
            )
        ok = (
            np.isfinite(delta_p95)
            and np.isfinite(delta_max)
            and float(delta_p95) < 100.0
            and float(delta_max) < 1000.0
            and (not np.isfinite(inactive) or float(inactive) <= 0.0)
            and np.isfinite(logvar_span)
            and float(logvar_span) > 0.05
            and np.isfinite(full_l2)
            and nonfinite == 0
            and bool(scale_ok)
        )
        rows.append(
            {
                "method_key": method,
                "checkpoint_label": "500k",
                "diagnostics_ok": int(ok),
                "delta_norm_1s_p95_max": float(delta_p95),
                "delta_norm_1s_max": float(delta_max),
                "inactive_forwarded_count_max": float(inactive) if np.isfinite(inactive) else np.nan,
                "logvar_xy_1s_span_max": float(logvar_span),
                "feature_nonfinite_count": int(nonfinite),
                "full_aug_obs_l2_p95_max": float(full_l2),
                "logvar_raw_l2_p95_max": float(logvar_raw_l2) if np.isfinite(logvar_raw_l2) else np.nan,
                "logvar_scaled_l2_p95_max": float(logvar_scaled_l2) if np.isfinite(logvar_scaled_l2) else np.nan,
                "adapter_output_l2_p95_max": float(adapter_l2) if np.isfinite(adapter_l2) else np.nan,
                "feature_scale_ok": int(bool(scale_ok)),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(result_dir / "tables/phase_n3p_diagnostics_decision.csv", index=False)
    if (out["diagnostics_ok"] != 1).any():
        bad = out[out["diagnostics_ok"] != 1]
        raise AnalysisStop("feature_scale_invalid", bad.to_string(index=False))
    return out


def build_reference_comparison(result_dir: Path, args: argparse.Namespace, diag: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = read_csv(result_dir / "tables/phase_n3p_eval_summary.csv")
    agg = aggregate(summary)
    final500 = agg[agg.checkpoint_label.astype(str) == "500k"].copy()
    final500 = final500.merge(diag[["method_key", "diagnostics_ok", "feature_scale_ok"]], on="method_key", how="left")
    rows: list[dict[str, Any]] = []
    for _, row in final500.iterrows():
        pass_gate = (
            float(row.success_rate) >= float(args.n3r_noz_success)
            and float(row.collision_rate) <= float(args.n3r_noz_collision)
            and int(row.diagnostics_ok) == 1
        )
        rows.append(
            {
                "method_key": row.method_key,
                "checkpoint_label": row.checkpoint_label,
                "success_rate": float(row.success_rate),
                "collision_rate": float(row.collision_rate),
                "near_miss_rate": float(row.near_miss_rate),
                "raw_unsafe_action_rate": float(row.raw_unsafe_action_rate),
                "diagnostics_ok": int(row.diagnostics_ok),
                "feature_scale_ok": int(row.feature_scale_ok),
                "n3r_noz_500k_success": float(args.n3r_noz_success),
                "n3r_noz_500k_collision": float(args.n3r_noz_collision),
                "delta_success_vs_n3r_noz_500k": float(row.success_rate) - float(args.n3r_noz_success),
                "delta_collision_vs_n3r_noz_500k": float(row.collision_rate) - float(args.n3r_noz_collision),
                "n3f_noz_full_success": float(args.n3f_noz_success),
                "n3f_noz_full_collision": float(args.n3f_noz_collision),
                "attention_full_success": float(args.attention_success),
                "attention_full_collision": float(args.attention_collision),
                "corrected_z2_full_success": float(args.z2_success),
                "corrected_z2_full_collision": float(args.z2_collision),
                "hard_gate_pass": int(pass_gate),
            }
        )
    comp = pd.DataFrame(rows).sort_values(["hard_gate_pass", "collision_rate", "success_rate"], ascending=[False, True, False])
    comp.to_csv(result_dir / "tables/phase_n3p_reference_comparison.csv", index=False)

    scenario = read_csv(result_dir / "tables/phase_n3p_scenario_breakdown.csv")
    p500 = scenario[scenario.checkpoint_label.astype(str) == "500k"]
    no_z_ref = read_csv(ROOT / "results/env_v2_phase_n3r_gpsi_ppo_rerun/tables/phase_n3r_scenario_breakdown.csv")
    ref = no_z_ref[(no_z_ref.method_key.astype(str) == "repaired-no-z") | (no_z_ref.method_key.astype(str) == "no_z") | (no_z_ref.method.astype(str).str.contains("no", case=False, na=False))]
    if ref.empty:
        ref = no_z_ref[no_z_ref.checkpoint_label.astype(str).isin(["final", "500k"])].copy()
    ref = ref[ref.checkpoint_label.astype(str).isin(["final", "500k"])].copy()
    ref = ref.groupby("scenario", dropna=False).agg(ref_success=("success_rate", "mean"), ref_collision=("collision_rate", "mean")).reset_index()
    scenario_checks: list[dict[str, Any]] = []
    for method in ["obs_delta_only", "logvar_scaled", "block_projected"]:
        data = p500[p500.method_key.astype(str) == method].merge(ref, on="scenario", how="left")
        high = data[data.scenario.astype(str).isin(["eval_flow_high_speed", "eval_flow_high_threat"])]
        scenario_checks.append(
            {
                "method_key": method,
                "high_speed_success": float(high[high.scenario == "eval_flow_high_speed"].success_rate.mean()) if not high[high.scenario == "eval_flow_high_speed"].empty else np.nan,
                "high_speed_collision": float(high[high.scenario == "eval_flow_high_speed"].collision_rate.mean()) if not high[high.scenario == "eval_flow_high_speed"].empty else np.nan,
                "high_threat_success": float(high[high.scenario == "eval_flow_high_threat"].success_rate.mean()) if not high[high.scenario == "eval_flow_high_threat"].empty else np.nan,
                "high_threat_collision": float(high[high.scenario == "eval_flow_high_threat"].collision_rate.mean()) if not high[high.scenario == "eval_flow_high_threat"].empty else np.nan,
                "not_worse_high_speed_high_threat": int(
                    not high.empty
                    and (pd.to_numeric(high["success_rate"], errors="coerce") >= pd.to_numeric(high["ref_success"], errors="coerce") - 1e-9).all()
                    and (pd.to_numeric(high["collision_rate"], errors="coerce") <= pd.to_numeric(high["ref_collision"], errors="coerce") + 1e-9).all()
                ),
            }
        )
    scenario_gate = pd.DataFrame(scenario_checks)
    scenario_gate.to_csv(result_dir / "tables/phase_n3p_scenario_gate_check.csv", index=False)
    return comp, scenario_gate


def train_still_improving(result_dir: Path, method: str) -> int:
    path = result_dir / "tables/phase_n3p_train_curve.csv"
    if not path.exists() or path.stat().st_size == 0:
        return 0
    train = pd.read_csv(path)
    rows = train[train.method_key.astype(str) == method].copy()
    if rows.empty or len(rows) < 40:
        return 0
    rows["steps_num"] = pd.to_numeric(rows["steps"], errors="coerce")
    rows["reward_num"] = pd.to_numeric(rows["episode_reward"], errors="coerce")
    rows = rows.sort_values("steps_num")
    early = rows[rows.steps_num <= rows.steps_num.quantile(0.40)]["reward_num"].tail(100).mean()
    late = rows[rows.steps_num >= rows.steps_num.quantile(0.70)]["reward_num"].tail(100).mean()
    return int(np.isfinite(early) and np.isfinite(late) and late >= early - 2.0)


def winner_recommendation(result_dir: Path, comp: pd.DataFrame, scenario_gate: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    merged = comp.merge(scenario_gate[["method_key", "not_worse_high_speed_high_threat"]], on="method_key", how="left")
    for _, row in merged.iterrows():
        clearly_better = int(float(row.delta_success_vs_n3r_noz_500k) >= 0.03 or float(row.delta_collision_vs_n3r_noz_500k) <= -0.03)
        improving = train_still_improving(result_dir, str(row.method_key))
        promote = int(int(row.hard_gate_pass) == 1 and clearly_better == 1 and int(row.not_worse_high_speed_high_threat) == 1 and improving == 1)
        rows.append(
            {
                "method_key": row.method_key,
                "success_rate_500k": float(row.success_rate),
                "collision_rate_500k": float(row.collision_rate),
                "near_miss_rate_500k": float(row.near_miss_rate),
                "raw_unsafe_action_rate_500k": float(row.raw_unsafe_action_rate),
                "hard_gate_pass": int(row.hard_gate_pass),
                "clearly_better_than_n3r_noz_500k": clearly_better,
                "not_worse_high_speed_high_threat": int(row.not_worse_high_speed_high_threat),
                "train_curve_still_improving": improving,
                "promote_to_1p5m": promote,
            }
        )
    out = pd.DataFrame(rows).sort_values(
        ["promote_to_1p5m", "hard_gate_pass", "collision_rate_500k", "success_rate_500k", "near_miss_rate_500k", "raw_unsafe_action_rate_500k"],
        ascending=[False, False, True, False, True, True],
    )
    winner = "none"
    if not out[out.promote_to_1p5m == 1].empty:
        winner = str(out[out.promote_to_1p5m == 1].iloc[0].method_key)
    elif not out[out.hard_gate_pass == 1].empty:
        winner = str(out[out.hard_gate_pass == 1].iloc[0].method_key)
    out["winner_if_any"] = winner
    out["overall_promote_to_1p5m"] = "yes" if (out["promote_to_1p5m"] == 1).any() else "no"
    out["can_enter_N4_now"] = "no" if (out["promote_to_1p5m"] == 1).any() else "yes"
    if (out["promote_to_1p5m"] == 1).any():
        rationale = f"{winner} passes the hard gate, improves clearly over N3R no_z 500k, and merits a 1.5M continuation before N4."
    else:
        rationale = "No P variant satisfied all promotion conditions; proceed to N4 with N3F no_z full as the main candidate."
    out["recommendation"] = rationale
    out.to_csv(result_dir / "tables/phase_n3p_winner_recommendation.csv", index=False)
    return out


def plot_success_collision(result_dir: Path, comp: pd.DataFrame) -> None:
    data = comp.sort_values("method_key")
    x = np.arange(len(data))
    plt.figure(figsize=(8, 4.2))
    plt.bar(x - 0.18, data["success_rate"], 0.36, label="success")
    plt.bar(x + 0.18, data["collision_rate"], 0.36, label="collision")
    plt.axhline(0.4233, color="tab:green", linestyle="--", linewidth=1, label="N3R no_z success gate")
    plt.axhline(0.5767, color="tab:red", linestyle="--", linewidth=1, label="N3R no_z collision gate")
    plt.xticks(x, data["method_key"], rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(result_dir / "plots/n3p_success_collision_by_variant.png", dpi=140)
    plt.close()


def plot_checkpoint_curve(result_dir: Path, summary: pd.DataFrame) -> None:
    agg = aggregate(summary)
    data = agg[agg.checkpoint_label.astype(str).isin(["250k", "500k", "final", "best_by_eval"])].copy()
    order = {"250k": 250_000, "500k": 500_000, "final": 500_001, "best_by_eval": 500_002}
    data["order"] = data["checkpoint_label"].map(order)
    plt.figure(figsize=(9, 4.5))
    for method, group in data.groupby("method_key"):
        group = group.sort_values("order")
        x = [order[str(label)] / 1000.0 for label in group["checkpoint_label"]]
        plt.plot(x, group["success_rate"], marker="o", label=f"{method} success")
        plt.plot(x, group["collision_rate"], marker="x", linestyle="--", label=f"{method} collision")
    plt.xlabel("checkpoint k")
    plt.ylabel("rate")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.25)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(result_dir / "plots/n3p_checkpoint_success_collision.png", dpi=140)
    plt.close()


def plot_train(result_dir: Path) -> None:
    train = read_csv(result_dir / "tables/phase_n3p_train_curve.csv")
    plt.figure(figsize=(9, 4.2))
    for method, group in train.groupby("method_key"):
        group = group.sort_values("steps")
        reward = pd.to_numeric(group["episode_reward"], errors="coerce").rolling(50, min_periods=1).mean()
        plt.plot(pd.to_numeric(group["steps"], errors="coerce"), reward, label=str(method))
    plt.xlabel("steps")
    plt.ylabel("episode reward rolling mean")
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "plots/n3p_train_reward.png", dpi=140)
    plt.close()


def plot_feature(result_dir: Path) -> None:
    feats = read_csv(result_dir / "tables/phase_n3p_feature_block_stats.csv")
    data = feats[(feats.checkpoint_label.astype(str) == "500k") & (feats.scenario.astype(str) == "eval_flow_id")]
    group = data.groupby(["method_key", "block"], dropna=False).agg(v=("l2_norm_p95", "mean")).reset_index()
    blocks = ["obs_i_12", "delta_hat_9_after_scale", "logvar_raw_9_clamped", "logvar_scaled_9_policy", "full_aug_obs", "adapter_output_64"]
    methods = ["obs_delta_only", "logvar_scaled", "block_projected"]
    x = np.arange(len(blocks))
    width = 0.24
    plt.figure(figsize=(10, 4.5))
    for idx, method in enumerate(methods):
        m = group[group.method_key == method].set_index("block")
        vals = [float(m.loc[block, "v"]) if block in m.index and np.isfinite(float(m.loc[block, "v"])) else np.nan for block in blocks]
        plt.bar(x + (idx - 1) * width, vals, width, label=method)
    plt.xticks(x, blocks, rotation=25, ha="right")
    plt.ylabel("p95 L2 norm")
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "plots/n3p_feature_block_scale.png", dpi=140)
    plt.close()


def plot_raw(result_dir: Path, comp: pd.DataFrame) -> None:
    data = comp.sort_values("method_key")
    plt.figure(figsize=(7.5, 4.0))
    plt.bar(data["method_key"], data["raw_unsafe_action_rate"])
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("raw unsafe action rate")
    plt.tight_layout()
    plt.savefig(result_dir / "plots/n3p_raw_unsafe_by_variant.png", dpi=140)
    plt.close()


def plot_scenario(result_dir: Path) -> None:
    summary = read_csv(result_dir / "tables/phase_n3p_scenario_breakdown.csv")
    data = summary[summary.checkpoint_label.astype(str) == "500k"].copy()
    methods = ["obs_delta_only", "logvar_scaled", "block_projected"]
    x = np.arange(len(SCENARIOS))
    width = 0.24
    plt.figure(figsize=(10, 4.6))
    for idx, method in enumerate(methods):
        group = data[data.method_key.astype(str) == method].set_index("scenario")
        vals = [float(group.loc[s, "success_rate"]) if s in group.index else np.nan for s in SCENARIOS]
        plt.bar(x + (idx - 1) * width, vals, width, label=method)
    plt.xticks(x, SCENARIOS, rotation=25, ha="right")
    plt.ylabel("success rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "plots/n3p_scenario_breakdown.png", dpi=140)
    plt.close()


def plot_gpsi(result_dir: Path) -> None:
    gpsi = read_csv(result_dir / "tables/phase_n3p_gpsi_output_summary.csv")
    data = gpsi[gpsi.checkpoint_label.astype(str) == "500k"].copy()
    delta = data.groupby("method_key").agg(v=("delta_norm_1s_p95", "mean")).reset_index()
    logvar = data.groupby("method_key").agg(v=("logvar_xy_1s_span", "mean")).reset_index()
    plt.figure(figsize=(7.5, 4.0))
    plt.bar(delta["method_key"], delta["v"])
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("delta_norm_1s_p95")
    plt.tight_layout()
    plt.savefig(result_dir / "plots/n3p_gpsi_delta_norm.png", dpi=140)
    plt.close()
    plt.figure(figsize=(7.5, 4.0))
    plt.bar(logvar["method_key"], logvar["v"])
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("logvar_xy_1s_span")
    plt.tight_layout()
    plt.savefig(result_dir / "plots/n3p_gpsi_logvar.png", dpi=140)
    plt.close()


def generate_plots(result_dir: Path, comp: pd.DataFrame, summary: pd.DataFrame) -> None:
    (result_dir / "plots").mkdir(parents=True, exist_ok=True)
    plot_success_collision(result_dir, comp)
    plot_checkpoint_curve(result_dir, summary)
    plot_train(result_dir)
    plot_feature(result_dir)
    plot_raw(result_dir, comp)
    plot_scenario(result_dir)
    plot_gpsi(result_dir)


def collect_files(result_dir: Path) -> dict[str, list[str]]:
    ckpt_dirs = [
        ROOT / "checkpoints/env_v2_gpsi_heada_ppo_n3p_obs_delta_only_s0",
        ROOT / "checkpoints/env_v2_gpsi_heada_ppo_n3p_logvar_scaled_s0",
        ROOT / "checkpoints/env_v2_gpsi_heada_ppo_n3p_block_projected_s0",
    ]
    checkpoints: list[str] = []
    for ckpt_dir in ckpt_dirs:
        checkpoints.extend(rel(path) for path in sorted(ckpt_dir.glob("*.zip")))
    return {
        "checkpoints": checkpoints,
        "tables": [rel(path) for path in sorted((result_dir / "tables").glob("*")) if path.is_file()],
        "plots": [rel(path) for path in sorted((result_dir / "plots").glob("*.png"))],
        "logs": [rel(path) for path in sorted((result_dir / "logs").glob("*.log"))] + [rel(result_dir / "phase_n3p_watcher.log")],
        "flags": [rel(path) for path in sorted(result_dir.glob("*.flag"))],
    }


def report(
    result_dir: Path,
    comp: pd.DataFrame,
    diag: pd.DataFrame,
    winner: pd.DataFrame,
    summary: pd.DataFrame,
    files: dict[str, list[str]],
) -> list[str]:
    w = winner.iloc[0]
    promote = str(w["overall_promote_to_1p5m"])
    can_n4 = str(w["can_enter_N4_now"])
    lines = [
        "# Phase N3P No-Z Representation Ablation Report",
        "",
        "## Terminal Decision",
        "",
        "`terminal_decision = phase_n3p_noz_representation_ablation_complete`",
        "",
        "P1/P2/P3 500k no-shield representation screening completed with frozen Gpsi and EnvV2 core unchanged.",
        "",
        "## Winner Recommendation",
        "",
    ]
    lines.extend(table_md(winner, ["method_key", "success_rate_500k", "collision_rate_500k", "hard_gate_pass", "clearly_better_than_n3r_noz_500k", "not_worse_high_speed_high_threat", "train_curve_still_improving", "promote_to_1p5m", "winner_if_any", "overall_promote_to_1p5m", "can_enter_N4_now"], max_rows=6))
    lines.append(f"- promote_to_1p5m: {promote}")
    lines.append(f"- can_enter_N4_now: {can_n4}")
    lines.append(f"- recommendation: {w['recommendation']}")
    lines.extend(["", "## Reference Comparison", ""])
    lines.extend(table_md(comp, ["method_key", "success_rate", "collision_rate", "near_miss_rate", "raw_unsafe_action_rate", "hard_gate_pass", "delta_success_vs_n3r_noz_500k", "delta_collision_vs_n3r_noz_500k"], max_rows=8))
    lines.extend(["", "## Diagnostics", ""])
    lines.extend(table_md(diag, list(diag.columns), max_rows=8))
    agg = aggregate(summary)
    lines.extend(["", "## Checkpoint Eval Aggregate", ""])
    lines.extend(table_md(agg, ["method_key", "checkpoint_label", "success_rate", "collision_rate", "near_miss_rate", "progress", "raw_unsafe_action_rate", "action_delta", "mean_min_distance"], max_rows=20))
    lines.extend(["", "## Interpretation", ""])
    p1 = comp[comp.method_key == "obs_delta_only"].iloc[0]
    p2 = comp[comp.method_key == "logvar_scaled"].iloc[0]
    logvar_harmful = "yes" if float(p1.success_rate) > float(p2.success_rate) and float(p1.collision_rate) <= float(p2.collision_rate) else "not_clear"
    lines.append(f"- whether logvar was harmful: {logvar_harmful}.")
    lines.append("- Gpsi output diagnostics remained bounded; no wrapper-scale regression was detected.")
    lines.append("- N4 was not executed in this phase.")
    lines.extend(["", "## Artifacts", ""])
    for key in ["checkpoints", "tables", "plots", "logs", "flags"]:
        lines.append(f"### {key}")
        values = files.get(key, [])
        if values:
            lines.extend([f"- `{value}`" for value in values[:180]])
            if len(values) > 180:
                lines.append(f"- ... {len(values) - 180} more")
        else:
            lines.append("- none")
    return lines


def main() -> None:
    args = parse_args()
    result_dir = ROOT / args.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    try:
        validate(result_dir, args)
        diag = diagnostics(result_dir, args)
        comp, scenario_gate = build_reference_comparison(result_dir, args, diag)
        winner = winner_recommendation(result_dir, comp, scenario_gate)
        summary = read_csv(result_dir / "tables/phase_n3p_eval_summary.csv")
        generate_plots(result_dir, comp, summary)
        write_text(result_dir / COMPLETE_FLAG, "terminal_decision = phase_n3p_noz_representation_ablation_complete\n")
        write_text(result_dir / "phase_n3p_status.txt", "complete\n")
        files = collect_files(result_dir)
        write_text(result_dir / "PHASE_N3P_NOZ_REPRESENTATION_ABLATION_REPORT.md", "\n".join(report(result_dir, comp, diag, winner, summary, files)) + "\n")
        print("terminal_decision = phase_n3p_noz_representation_ablation_complete", flush=True)
    except AnalysisStop as exc:
        write_stop(result_dir, exc.reason, exc.detail)
        raise SystemExit(2) from exc
    except Exception:
        detail = traceback.format_exc()
        write_stop(result_dir, "diagnostics_failed", detail)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
