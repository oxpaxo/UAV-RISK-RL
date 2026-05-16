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


COMPLETE_FLAG = "PHASE_N3Z2C_Z2_CONTINUATION_COMPLETE.flag"
STOP_FLAGS = {
    "phase_n3fz_missing": "PHASE_N3Z2C_STOP_PHASE_N3FZ_MISSING.flag",
    "z2_parent_missing": "PHASE_N3Z2C_STOP_Z2_PARENT_MISSING.flag",
    "train_failed": "PHASE_N3Z2C_STOP_TRAIN_FAILED.flag",
    "eval_failed": "PHASE_N3Z2C_STOP_EVAL_FAILED.flag",
    "diagnostics_failed": "PHASE_N3Z2C_STOP_DIAGNOSTICS_FAILED.flag",
    "checkpoint_ambiguity": "PHASE_N3Z2C_STOP_CHECKPOINT_AMBIGUITY.flag",
    "analysis_failed": "PHASE_N3Z2C_STOP_DIAGNOSTICS_FAILED.flag",
}

Z2_KEY = "z_layernorm_alpha_0p5_cont_1p5m"
NOZ_KEY = "n3f_no_z_full"
ATT_KEY = "attention_full"
SCENARIOS = [
    "eval_flow_id",
    "eval_flow_high_density",
    "eval_flow_high_speed",
    "eval_flow_high_threat",
    "eval_flow_mixed_ood",
    "eval_flow_sudden_threat",
]
Z2_LABELS = ["parent_500k", "750k", "1000k", "1250k", "1500k", "final", "best_by_eval"]


class PhaseN3Z2CAnalysisStop(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Phase N3Z2C outputs.")
    parser.add_argument("--result-dir", default="results/env_v2_phase_n3z2c_z2_continuation")
    parser.add_argument("--expected-episodes", type=int, default=50)
    parser.add_argument("--noz-success", type=float, default=0.5633)
    parser.add_argument("--noz-collision", type=float, default=0.4367)
    parser.add_argument("--attention-success", type=float, default=0.6100)
    parser.add_argument("--attention-collision", type=float, default=0.3900)
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


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty CSV: {rel(path)}")
    return pd.read_csv(path)


def write_stop(result_dir: Path, reason: str, detail: str) -> None:
    flag_name = STOP_FLAGS.get(reason, STOP_FLAGS["analysis_failed"])
    write_text(result_dir / flag_name, f"{reason}\n{detail.strip()}\n")
    write_text(result_dir / "phase_n3z2c_status.txt", f"stopped:{flag_name}\n")
    write_text(
        result_dir / "PHASE_N3Z2C_Z2_CONTINUATION_REPORT.md",
        "\n".join(
            [
                "# Phase N3Z2C Z2 Continuation Report",
                "",
                f"`terminal_decision = phase_n3z2c_stopped_{reason}`",
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


def table_md(df: pd.DataFrame, cols: list[str], max_rows: int = 24) -> list[str]:
    if df.empty:
        return ["No rows."]
    view = df[cols].head(max_rows).copy()
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) if isinstance(row[col], (float, int, np.floating, np.integer)) else str(row[col]) for col in cols) + " |")
    return lines


def validate_required(result_dir: Path, args: argparse.Namespace) -> None:
    required = [
        ROOT / "results/env_v2_phase_n3fz_noz_full_z_screen/PHASE_N3FZ_NOZ_FULL_Z_SCREEN_COMPLETE.flag",
        result_dir / "tables/phase_n3z2c_parent_checkpoint_selection.csv",
        result_dir / "tables/phase_n3z2c_resource_preflight.csv",
        result_dir / "tables/phase_n3z2c_command_manifest.csv",
        result_dir / "tables/phase_n3z2c_config_manifest.csv",
        result_dir / "tables/phase_n3z2c_train_curve.csv",
        result_dir / "tables/phase_n3z2c_train_heartbeat.csv",
        result_dir / "tables/phase_n3z2c_eval_summary.csv",
        result_dir / "tables/phase_n3z2c_checkpoint_eval_summary.csv",
        result_dir / "tables/phase_n3z2c_episode_metrics.csv",
        result_dir / "tables/phase_n3z2c_scenario_breakdown.csv",
        result_dir / "tables/phase_n3z2c_motion_mode_breakdown.csv",
        result_dir / "tables/phase_n3z2c_threat_class_breakdown.csv",
        result_dir / "tables/phase_n3z2c_raw_unsafe_action_summary.csv",
        result_dir / "tables/phase_n3z2c_gpsi_output_summary.csv",
        result_dir / "tables/phase_n3z2c_aug_feature_block_stats.csv",
    ]
    ckpt_dir = ROOT / "checkpoints/env_v2_gpsi_heada_ppo_n3z2c_layernorm_alpha0p5_s0"
    required.extend(
        [
            ckpt_dir / "parent_500k.zip",
            ckpt_dir / "checkpoint_750k.zip",
            ckpt_dir / "checkpoint_1000k.zip",
            ckpt_dir / "checkpoint_1250k.zip",
            ckpt_dir / "checkpoint_1500k.zip",
            ckpt_dir / "final.zip",
            ckpt_dir / "best_by_eval.zip",
        ]
    )
    missing = [rel(path) for path in required if not path.exists() or path.stat().st_size == 0]
    if missing:
        reason = "phase_n3fz_missing" if any("PHASE_N3FZ" in item for item in missing) else "train_failed"
        raise PhaseN3Z2CAnalysisStop(reason, "missing required artifacts:\n" + "\n".join(missing))

    episodes = read_csv(result_dir / "tables/phase_n3z2c_episode_metrics.csv")
    counts = episodes.groupby(["method_key", "checkpoint_label", "scenario"]).size().reset_index(name="episodes")
    bad = counts[counts["episodes"] < int(args.expected_episodes)]
    if not bad.empty:
        raise PhaseN3Z2CAnalysisStop("eval_failed", f"not enough eval episodes:\n{bad.to_string(index=False)}")


def finalish_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary[summary["checkpoint_label"].astype(str).isin(["final", "best_by_eval", "1500k", "parent_500k", "attention_full_1500k"])]
        .groupby(["method_key", "method", "checkpoint_label"], dropna=False)
        .agg(
            success_rate=("success_rate", "mean"),
            collision_rate=("collision_rate", "mean"),
            near_miss_rate=("near_miss_rate", "mean"),
            progress=("progress", "mean"),
            raw_unsafe_action_rate=("raw_unsafe_action_rate", "mean"),
            action_norm=("action_norm", "mean"),
            action_delta=("action_delta", "mean"),
            no_response_rate=("no_response_rate", "mean"),
            mean_min_distance=("mean_min_distance", "mean"),
        )
        .reset_index()
    )


def build_comparisons(summary: pd.DataFrame, result_dir: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    z2 = summary[summary["method_key"] == Z2_KEY].copy()
    noz = summary[(summary["method_key"] == NOZ_KEY) & (summary["checkpoint_label"].astype(str).isin(["final", "best_by_eval"]))].copy()
    attention = summary[summary["method_key"] == ATT_KEY].copy()
    z2_final = z2[z2["checkpoint_label"].astype(str) == "final"].copy()
    z2_best = z2[z2["checkpoint_label"].astype(str) == "best_by_eval"].copy()
    z2_parent = z2[z2["checkpoint_label"].astype(str) == "parent_500k"].copy()
    if z2_final.empty or z2_best.empty or z2_parent.empty or noz.empty or attention.empty:
        raise PhaseN3Z2CAnalysisStop("eval_failed", "missing Z2/no_z/attention comparison rows")

    noz_ref = noz[noz["checkpoint_label"].astype(str) == "final"].copy()
    if noz_ref.empty:
        noz_ref = noz[noz["checkpoint_label"].astype(str) == "best_by_eval"].copy()
    cols = ["scenario", "success_rate", "collision_rate", "near_miss_rate", "progress", "raw_unsafe_action_rate", "action_delta"]

    noz_comp = z2_final[["method_key", "method", "scenario", "success_rate", "collision_rate", "near_miss_rate", "progress", "raw_unsafe_action_rate", "action_delta"]].merge(
        noz_ref[cols], on="scenario", how="inner", suffixes=("_z2", "_noz")
    )
    for metric in cols[1:]:
        noz_comp[f"delta_{metric}"] = noz_comp[f"{metric}_z2"] - noz_comp[f"{metric}_noz"]
    noz_comp.to_csv(result_dir / "tables/phase_n3z2c_noz_reference_comparison.csv", index=False)

    att_comp = z2_final[["method_key", "method", "scenario", "success_rate", "collision_rate", "near_miss_rate", "progress", "raw_unsafe_action_rate", "action_delta"]].merge(
        attention[cols], on="scenario", how="inner", suffixes=("_z2", "_attention")
    )
    for metric in cols[1:]:
        att_comp[f"delta_{metric}"] = att_comp[f"{metric}_z2"] - att_comp[f"{metric}_attention"]
    att_comp.to_csv(result_dir / "tables/phase_n3z2c_attention_reference_comparison.csv", index=False)

    parent_comp = z2_final[["scenario", "success_rate", "collision_rate", "near_miss_rate", "progress", "raw_unsafe_action_rate", "action_delta"]].merge(
        z2_parent[cols], on="scenario", how="inner", suffixes=("_z2_final", "_z2_parent")
    )
    for metric in cols[1:]:
        parent_comp[f"delta_{metric}"] = parent_comp[f"{metric}_z2_final"] - parent_comp[f"{metric}_z2_parent"]
    parent_comp.to_csv(result_dir / "tables/phase_n3z2c_parent_screening_comparison.csv", index=False)
    return noz_comp, att_comp, parent_comp


def diagnostics(summary: pd.DataFrame, gpsi: pd.DataFrame, features: pd.DataFrame, result_dir: Path) -> pd.DataFrame:
    z2_final_gpsi = gpsi[(gpsi["method_key"] == Z2_KEY) & (gpsi["checkpoint_label"].astype(str) == "final")].copy()
    z2_features = features[(features["method_key"] == Z2_KEY) & (features["checkpoint_label"].astype(str) == "final")].copy()
    if z2_final_gpsi.empty or z2_features.empty:
        raise PhaseN3Z2CAnalysisStop("diagnostics_failed", "missing final Z2 diagnostics")
    delta_p95 = pd.to_numeric(z2_final_gpsi.get("delta_norm_1s_p95", pd.Series(dtype=float)), errors="coerce").max()
    delta_max = pd.to_numeric(z2_final_gpsi.get("delta_norm_1s_max", pd.Series(dtype=float)), errors="coerce").max()
    inactive = pd.to_numeric(z2_final_gpsi.get("inactive_forwarded_count_max", pd.Series(dtype=float)), errors="coerce").max()
    logvar_span = pd.to_numeric(z2_final_gpsi.get("logvar_xy_1s_span", pd.Series(dtype=float)), errors="coerce").max()
    z_after = z2_features[z2_features["block"].astype(str) == "z_i_64_after_constraint"]
    z_after_p95 = pd.to_numeric(z_after.get("l2_norm_p95", pd.Series(dtype=float)), errors="coerce").max()
    nonfinite = 0
    for col in ["nan_count", "inf_count"]:
        nonfinite += int(pd.to_numeric(z2_features.get(col, pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    ok = (
        np.isfinite(delta_p95)
        and np.isfinite(delta_max)
        and float(delta_p95) < 100.0
        and float(delta_max) < 1000.0
        and (not np.isfinite(inactive) or float(inactive) <= 0.0)
        and np.isfinite(logvar_span)
        and float(logvar_span) > 0.05
        and nonfinite == 0
        and np.isfinite(z_after_p95)
        and 3.5 <= float(z_after_p95) <= 4.5
    )
    out = pd.DataFrame(
        [
            {
                "method_key": Z2_KEY,
                "checkpoint_label": "final",
                "diagnostics_ok": int(ok),
                "delta_norm_1s_p95_max": float(delta_p95),
                "delta_norm_1s_max": float(delta_max),
                "inactive_forwarded_count_max": float(inactive) if np.isfinite(inactive) else np.nan,
                "logvar_xy_1s_span_max": float(logvar_span),
                "z_after_constraint_l2_p95_max": float(z_after_p95),
                "feature_nonfinite_count": int(nonfinite),
            }
        ]
    )
    out.to_csv(result_dir / "tables/phase_n3z2c_diagnostics_decision.csv", index=False)
    if not ok:
        raise PhaseN3Z2CAnalysisStop("diagnostics_failed", out.to_string(index=False))
    return out


def candidate_decision(summary: pd.DataFrame, diag: pd.DataFrame, result_dir: Path, args: argparse.Namespace) -> pd.DataFrame:
    agg = finalish_summary(summary)
    z2_final = agg[(agg["method_key"] == Z2_KEY) & (agg["checkpoint_label"].astype(str) == "final")]
    z2_best = agg[(agg["method_key"] == Z2_KEY) & (agg["checkpoint_label"].astype(str) == "best_by_eval")]
    noz_final = agg[(agg["method_key"] == NOZ_KEY) & (agg["checkpoint_label"].astype(str) == "final")]
    att = agg[agg["method_key"] == ATT_KEY]
    if z2_final.empty or noz_final.empty or att.empty:
        raise PhaseN3Z2CAnalysisStop("eval_failed", "missing final candidate rows")
    z2 = z2_final.iloc[0]
    noz = noz_final.iloc[0]
    att_row = att.iloc[0]
    z2_success = float(z2["success_rate"])
    z2_collision = float(z2["collision_rate"])
    noz_success = float(noz["success_rate"])
    noz_collision = float(noz["collision_rate"])
    att_success = float(att_row["success_rate"])
    att_collision = float(att_row["collision_rate"])
    diagnostics_ok = int(diag["diagnostics_ok"].iloc[0]) == 1

    if z2_success >= noz_success and z2_collision <= noz_collision:
        selected = "z2_full"
        decision = "Z2 full is primary N4 no-shield candidate."
        include_both = "no"
    elif z2_success < noz_success and z2_collision > noz_collision:
        selected = "no_z_full"
        decision = "Z2 full is worse than no_z on success and collision; no_z remains primary and Z2 is ablation only."
        include_both = "no"
    else:
        selected = "both_ablation"
        decision = "Z2/no_z trade off one metric against the other; preserve both, defaulting to lower collision for shield path."
        include_both = "yes"

    if not diagnostics_ok:
        selected = "undecided"
        decision = "Diagnostics failed; no N4 candidate can be selected."

    beats_attention = z2_success >= att_success and z2_collision <= att_collision
    can_enter = "yes" if diagnostics_ok and selected != "undecided" else "no"
    out = pd.DataFrame(
        [
            {
                "z2_success_rate": z2_success,
                "z2_collision_rate": z2_collision,
                "noz_success_rate": noz_success,
                "noz_collision_rate": noz_collision,
                "attention_success_rate": att_success,
                "attention_collision_rate": att_collision,
                "z2_minus_noz_success": z2_success - noz_success,
                "z2_minus_noz_collision": z2_collision - noz_collision,
                "z2_minus_attention_success": z2_success - att_success,
                "z2_minus_attention_collision": z2_collision - att_collision,
                "diagnostics_ok": int(diagnostics_ok),
                "selected_n4_candidate": selected,
                "include_both_as_ablation": include_both,
                "z2_beats_attention_no_shield": int(beats_attention),
                "can_enter_n4": can_enter,
                "decision": decision,
                "attention_statement": "Gpsi-PPO no-shield beats attention." if beats_attention else "Gpsi-PPO no-shield does not beat attention; comparison is close/weak or scenario-dependent.",
                "checkpoint_policy": "checkpoint_1500k, final, and best_by_eval are evaluated as separate rows; best_by_eval is final unless a later selector overwrote it.",
            }
        ]
    )
    out.to_csv(result_dir / "tables/phase_n3z2c_final_candidate_decision.csv", index=False)
    return out


def plot_success(summary: pd.DataFrame, path: Path) -> None:
    agg = finalish_summary(summary)
    labels = ["attention_full_1500k", "no_z_final", "z2_parent_500k", "z2_1500k", "z2_final", "z2_best"]
    rows = [
        agg[(agg.method_key == ATT_KEY)].head(1),
        agg[(agg.method_key == NOZ_KEY) & (agg.checkpoint_label.astype(str) == "final")],
        agg[(agg.method_key == Z2_KEY) & (agg.checkpoint_label.astype(str) == "parent_500k")],
        agg[(agg.method_key == Z2_KEY) & (agg.checkpoint_label.astype(str) == "1500k")],
        agg[(agg.method_key == Z2_KEY) & (agg.checkpoint_label.astype(str) == "final")],
        agg[(agg.method_key == Z2_KEY) & (agg.checkpoint_label.astype(str) == "best_by_eval")],
    ]
    vals = [(float(r.success_rate.iloc[0]), float(r.collision_rate.iloc[0])) if not r.empty else (np.nan, np.nan) for r in rows]
    x = np.arange(len(labels))
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 4.5))
    plt.bar(x - 0.18, [v[0] for v in vals], 0.36, label="success")
    plt.bar(x + 0.18, [v[1] for v in vals], 0.36, label="collision")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_z2_checkpoint(summary: pd.DataFrame, path: Path) -> None:
    z2 = (
        summary[summary.method_key == Z2_KEY]
        .groupby(["checkpoint_label", "checkpoint_step"], dropna=False)
        .agg(success=("success_rate", "mean"), collision=("collision_rate", "mean"))
        .reset_index()
    )
    order = {"parent_500k": 500000, "750k": 750000, "1000k": 1000000, "1250k": 1250000, "1500k": 1500000, "final": 1510000, "best_by_eval": 1520000}
    z2["x"] = z2["checkpoint_label"].astype(str).map(order).fillna(pd.to_numeric(z2["checkpoint_step"], errors="coerce"))
    z2 = z2.sort_values("x")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.plot(z2["x"], z2["success"], marker="o", label="success")
    plt.plot(z2["x"], z2["collision"], marker="x", linestyle="--", label="collision")
    plt.xticks(z2["x"], z2["checkpoint_label"], rotation=25, ha="right")
    plt.ylabel("rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_metric_bar(df: pd.DataFrame, metric: str, path: Path, ylabel: str) -> None:
    data = finalish_summary(df)
    data = data[((data.method_key == Z2_KEY) & (data.checkpoint_label.astype(str).isin(["parent_500k", "1500k", "final", "best_by_eval"]))) | ((data.method_key == NOZ_KEY) & (data.checkpoint_label.astype(str) == "final"))]
    labels = data["method_key"].astype(str) + "_" + data["checkpoint_label"].astype(str)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, pd.to_numeric(data[metric], errors="coerce"))
    plt.xticks(rotation=25, ha="right")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_scenario(summary: pd.DataFrame, path: Path) -> None:
    data = summary[(summary.method_key == Z2_KEY) & (summary.checkpoint_label.astype(str).isin(["parent_500k", "final"]))].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4.8))
    x = np.arange(len(SCENARIOS))
    width = 0.35
    for idx, label in enumerate(["parent_500k", "final"]):
        group = data[data.checkpoint_label.astype(str) == label].set_index("scenario")
        vals = [float(group.loc[s, "success_rate"]) if s in group.index else np.nan for s in SCENARIOS]
        plt.bar(x + (idx - 0.5) * width, vals, width, label=label)
    plt.xticks(x, SCENARIOS, rotation=25, ha="right")
    plt.ylabel("success rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_feature(features: pd.DataFrame, path: Path) -> None:
    final = features[(features.method_key == Z2_KEY) & (features.checkpoint_label.astype(str) == "final") & (features.scenario.astype(str) == "eval_flow_id")]
    order = ["obs_i_12", "z_i_64_raw", "z_i_64_after_constraint", "delta_hat_9_after_scale", "logvar_hat_9_clamped", "full_aug_obs"]
    group = final.groupby("block", dropna=False).agg(l2=("l2_norm_p95", "mean")).reindex(order)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.bar(group.index.astype(str), group["l2"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("p95 L2 norm")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_gpsi(gpsi: pd.DataFrame, path_delta: Path, path_logvar: Path) -> None:
    final = gpsi[(gpsi.method_key == Z2_KEY) & (gpsi.checkpoint_label.astype(str) == "final")]
    delta = final.groupby("scenario").agg(v=("delta_norm_1s_p95", "max")).reindex(SCENARIOS)
    path_delta.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.3))
    plt.bar(delta.index.astype(str), delta["v"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("delta norm 1s p95")
    plt.tight_layout()
    plt.savefig(path_delta, dpi=140)
    plt.close()

    logvar = final.groupby("scenario").agg(mean=("logvar_xy_1s_mean", "mean"), span=("logvar_xy_1s_span", "max")).reindex(SCENARIOS)
    plt.figure(figsize=(8, 4.3))
    x = np.arange(len(logvar))
    plt.bar(x - 0.18, logvar["mean"], 0.36, label="mean")
    plt.bar(x + 0.18, logvar["span"], 0.36, label="span")
    plt.xticks(x, logvar.index.astype(str), rotation=25, ha="right")
    plt.ylabel("logvar xy 1s")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_logvar, dpi=140)
    plt.close()


def plot_train(train: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.3))
    if not train.empty:
        train = train.sort_values("steps")
        rewards = pd.to_numeric(train["episode_reward"], errors="coerce").rolling(25, min_periods=1).mean()
        plt.plot(pd.to_numeric(train["steps"], errors="coerce"), rewards)
    plt.xlabel("total training step")
    plt.ylabel("episode reward rolling mean")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def generate_plots(result_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    plot_z2_checkpoint(tables["summary"], result_dir / "plots/z2_checkpoint_success_collision.png")
    plot_success(tables["summary"], result_dir / "plots/z2_vs_noz_attention_success_collision.png")
    plot_scenario(tables["summary"], result_dir / "plots/z2_scenario_breakdown.png")
    plot_metric_bar(tables["summary"], "raw_unsafe_action_rate", result_dir / "plots/z2_raw_unsafe_by_checkpoint.png", "raw unsafe action rate")
    plot_metric_bar(tables["summary"], "action_delta", result_dir / "plots/z2_action_dynamics.png", "action delta")
    plot_feature(tables["features"], result_dir / "plots/z2_aug_feature_block_scale.png")
    plot_gpsi(tables["gpsi"], result_dir / "plots/z2_gpsi_delta_norm.png", result_dir / "plots/z2_gpsi_logvar.png")
    plot_train(tables["train"], result_dir / "plots/z2_train_reward.png")


def collect_files(result_dir: Path) -> dict[str, list[str]]:
    ckpt_dir = ROOT / "checkpoints/env_v2_gpsi_heada_ppo_n3z2c_layernorm_alpha0p5_s0"
    return {
        "checkpoints": [rel(path) for path in sorted(ckpt_dir.glob("*.zip"))],
        "tables": [rel(path) for path in sorted((result_dir / "tables").glob("*.csv"))],
        "plots": [rel(path) for path in sorted((result_dir / "plots").glob("*.png"))],
        "logs": [rel(path) for path in sorted((result_dir / "logs").glob("*.log"))] + [rel(result_dir / "phase_n3z2c_watcher.log")],
        "flags": [rel(path) for path in sorted(result_dir.glob("*.flag"))],
    }


def report_lines(result_dir: Path, tables: dict[str, pd.DataFrame], decision: pd.DataFrame, diag: pd.DataFrame, files: dict[str, list[str]]) -> list[str]:
    parent = read_csv(result_dir / "tables/phase_n3z2c_parent_checkpoint_selection.csv")
    resource = read_csv(result_dir / "tables/phase_n3z2c_resource_preflight.csv")
    bench = pd.read_csv(result_dir / "tables/phase_n3z2c_cpu_benchmark.csv") if (result_dir / "tables/phase_n3z2c_cpu_benchmark.csv").exists() else pd.DataFrame()
    final_agg = finalish_summary(tables["summary"])
    d = decision.iloc[0]
    lines = [
        "# Phase N3Z2C Z2 Continuation Report",
        "",
        "## Terminal Decision",
        "",
        "`terminal_decision = phase_n3z2c_z2_continuation_complete`",
        "",
        "Phase N3Z2C complete. This stage continues Z2 `z_layernorm_alpha_0p5` to total 1.5M and selects the final no-shield N4 candidate.",
        "",
        "## Engineering Facts",
        "",
        "- Phase N3F/Z complete flag exists.",
        "- Repaired `GpsiObsWrapper` was used; Gpsi checkpoint `work_dirs/gpsi_heada_v1_nll/best.pth` stayed frozen.",
        "- EnvV2 core was not modified.",
        "- No shield, no action filtering, no dense safety cost, no learned R(s,a), and no Gpsi fine-tuning were used.",
        "- Logvar clip sanity: config uses `[-5, 3]`, already bounded tighter than `|logvar| <= 5`.",
        "",
        "## Parent Selection",
        "",
    ]
    lines.extend(table_md(parent, list(parent.columns), max_rows=10))
    lines.extend(["", "## CPU Strategy", ""])
    lines.extend(table_md(resource, list(resource.columns), max_rows=12))
    if not bench.empty:
        lines.extend(["", "## n_envs Smoke Benchmark", ""])
        lines.extend(table_md(bench, list(bench.columns), max_rows=12))
    lines.extend(["", "## Main Results", ""])
    lines.extend(table_md(final_agg, ["method_key", "checkpoint_label", "success_rate", "collision_rate", "near_miss_rate", "progress", "raw_unsafe_action_rate", "action_delta", "mean_min_distance"], max_rows=20))
    lines.extend(["", "## Candidate Decision", ""])
    lines.extend(table_md(decision, list(decision.columns), max_rows=5))
    lines.append(f"- Selected N4 candidate: `{d['selected_n4_candidate']}`.")
    lines.append(f"- Can enter N4: {d['can_enter_n4']}.")
    lines.append(f"- Decision: {d['decision']}")
    lines.append(f"- Attention comparison: {d['attention_statement']}")
    lines.extend(["", "## Diagnostics", ""])
    lines.extend(table_md(diag, list(diag.columns), max_rows=5))
    gpsi_final = tables["gpsi"][(tables["gpsi"].method_key == Z2_KEY) & (tables["gpsi"].checkpoint_label.astype(str) == "final")]
    gpsi_agg = (
        gpsi_final.groupby("method_key", dropna=False)
        .agg(
            delta_norm_1s_p95=("delta_norm_1s_p95", "max"),
            delta_norm_1s_max=("delta_norm_1s_max", "max"),
            logvar_xy_1s_mean=("logvar_xy_1s_mean", "mean"),
            logvar_xy_1s_span=("logvar_xy_1s_span", "max"),
            projected_std_radial_mean=("projected_std_radial_mean", "mean"),
            projected_std_relvel_mean=("projected_std_relvel_mean", "mean"),
            inactive_forwarded_count_max=("inactive_forwarded_count_max", "max"),
        )
        .reset_index()
    )
    lines.extend(["", "## Gpsi Output Diagnostics", ""])
    lines.extend(table_md(gpsi_agg, list(gpsi_agg.columns), max_rows=5))
    lines.extend(["", "## Feature Block Stats", ""])
    feature_final = tables["features"][(tables["features"].method_key == Z2_KEY) & (tables["features"].checkpoint_label.astype(str) == "final") & (tables["features"].scenario.astype(str) == "eval_flow_id")]
    lines.extend(table_md(feature_final.sort_values("block"), ["block", "z_transform", "l2_norm_p95", "max_abs_p95", "nan_count", "inf_count"], max_rows=12))
    lines.extend(["", "## Checkpoint Semantics", ""])
    lines.append("- `parent_500k.zip` is a copy of the selected N3F/Z Z2 parent checkpoint and is evaluated as the 500k parent.")
    lines.append("- `checkpoint_750k/1000k/1250k/1500k.zip` are saved by global total training step.")
    lines.append("- `final.zip` is saved after continuation completion.")
    lines.append("- `best_by_eval.zip` is copied from `final.zip` in this stage because no train-time eval selector is used; both are still evaluated as distinct labels.")
    lines.extend(["", "## Breakdown Outputs", ""])
    lines.append("Scenario, motion-mode, and threat-class breakdown CSVs were generated under `results/env_v2_phase_n3z2c_z2_continuation/tables/`.")
    lines.extend(["", "## Artifacts", ""])
    for key in ["checkpoints", "tables", "plots", "logs", "flags"]:
        lines.append(f"### {key}")
        values = files.get(key, [])
        if values:
            lines.extend([f"- `{value}`" for value in values[:120]])
            if len(values) > 120:
                lines.append(f"- ... {len(values) - 120} more")
        else:
            lines.append("- none")
    return lines


def main() -> None:
    args = parse_args()
    result_dir = ROOT / args.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    try:
        validate_required(result_dir, args)
        tables = {
            "summary": read_csv(result_dir / "tables/phase_n3z2c_eval_summary.csv"),
            "episodes": read_csv(result_dir / "tables/phase_n3z2c_episode_metrics.csv"),
            "raw": read_csv(result_dir / "tables/phase_n3z2c_raw_unsafe_action_summary.csv"),
            "gpsi": read_csv(result_dir / "tables/phase_n3z2c_gpsi_output_summary.csv"),
            "features": read_csv(result_dir / "tables/phase_n3z2c_aug_feature_block_stats.csv"),
            "train": read_csv(result_dir / "tables/phase_n3z2c_train_curve.csv"),
        }
        labels = set(tables["summary"][tables["summary"].method_key == Z2_KEY]["checkpoint_label"].astype(str))
        missing_labels = sorted(set(Z2_LABELS) - labels)
        if missing_labels:
            raise PhaseN3Z2CAnalysisStop("checkpoint_ambiguity", f"missing Z2 eval labels: {missing_labels}")
        scenarios = set(tables["summary"]["scenario"].astype(str))
        missing_scenarios = sorted(set(SCENARIOS) - scenarios)
        if missing_scenarios:
            raise PhaseN3Z2CAnalysisStop("eval_failed", f"missing scenarios: {missing_scenarios}")
        build_comparisons(tables["summary"], result_dir, args)
        diag = diagnostics(tables["summary"], tables["gpsi"], tables["features"], result_dir)
        decision = candidate_decision(tables["summary"], diag, result_dir, args)
        generate_plots(result_dir, tables)
        terminal_decision = "phase_n3z2c_z2_continuation_complete"
        write_text(result_dir / COMPLETE_FLAG, terminal_decision + "\n")
        write_text(result_dir / "phase_n3z2c_status.txt", "complete\n")
        files = collect_files(result_dir)
        lines = report_lines(result_dir, tables, decision, diag, files)
        write_text(result_dir / "PHASE_N3Z2C_Z2_CONTINUATION_REPORT.md", "\n".join(lines) + "\n")
        print(f"terminal_decision = {terminal_decision}", flush=True)
    except PhaseN3Z2CAnalysisStop as exc:
        write_stop(result_dir, exc.reason, exc.detail)
        raise SystemExit(2) from exc
    except Exception:
        detail = traceback.format_exc()
        write_stop(result_dir, "analysis_failed", detail)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
