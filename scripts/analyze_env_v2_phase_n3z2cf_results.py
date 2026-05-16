from __future__ import annotations

import argparse
import csv
import hashlib
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
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


COMPLETE_FLAG = "PHASE_N3Z2CF_CORRECTED_Z2_FULL_COMPLETE.flag"
STOP_FLAGS = {
    "parent_missing": "PHASE_N3Z2CF_STOP_PARENT_MISSING.flag",
    "resume_semantics_invalid": "PHASE_N3Z2CF_STOP_RESUME_SEMANTICS_INVALID.flag",
    "train_failed": "PHASE_N3Z2CF_STOP_TRAIN_FAILED.flag",
    "eval_failed": "PHASE_N3Z2CF_STOP_EVAL_FAILED.flag",
    "diagnostics_failed": "PHASE_N3Z2CF_STOP_DIAGNOSTICS_FAILED.flag",
    "checkpoint_integrity_failed": "PHASE_N3Z2CF_STOP_CHECKPOINT_INTEGRITY_FAILED.flag",
    "watcher_failed": "PHASE_N3Z2CF_STOP_WATCHER_FAILED.flag",
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
    parser = argparse.ArgumentParser(description="Analyze Phase N3Z2CF corrected Z2 full outputs.")
    parser.add_argument("--result-dir", default="results/env_v2_phase_n3z2cf_corrected_z2_full")
    parser.add_argument("--checkpoint-dir", default="checkpoints/env_v2_gpsi_heada_ppo_n3z2cf_layernorm_alpha0p5_s0")
    parser.add_argument("--expected-episodes", type=int, default=50)
    parser.add_argument("--noz-success", type=float, default=0.5633)
    parser.add_argument("--noz-collision", type=float, default=0.4367)
    parser.add_argument("--attention-success", type=float, default=0.6100)
    parser.add_argument("--attention-collision", type=float, default=0.3900)
    parser.add_argument("--old-z2c-success", type=float, default=0.4500)
    parser.add_argument("--old-z2c-collision", type=float, default=0.5500)
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_stop(result_dir: Path, reason: str, detail: str) -> None:
    flag = STOP_FLAGS.get(reason, STOP_FLAGS["diagnostics_failed"])
    write_text(result_dir / flag, f"{reason}\n{detail.strip()}\n")
    write_text(result_dir / "phase_n3z2cf_status.txt", f"stopped:{flag}\n")
    write_text(
        result_dir / "PHASE_N3Z2CF_CORRECTED_Z2_FULL_REPORT.md",
        "\n".join(
            [
                "# Phase N3Z2CF Corrected Z2 Full Report",
                "",
                f"`terminal_decision = phase_n3z2cf_stopped_{reason}`",
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
        result_dir / "tables/phase_n3z2cf_resource_affinity.csv",
        result_dir / "tables/phase_n3z2cf_resume_semantics.csv",
        result_dir / "tables/phase_n3z2cf_train_curve.csv",
        result_dir / "tables/phase_n3z2cf_train_heartbeat.csv",
        result_dir / "tables/phase_n3z2cf_checkpoint_eval_summary.csv",
        result_dir / "tables/phase_n3z2cf_eval_summary.csv",
        result_dir / "tables/phase_n3z2cf_scenario_breakdown.csv",
        result_dir / "tables/phase_n3z2cf_motion_mode_breakdown.csv",
        result_dir / "tables/phase_n3z2cf_threat_class_breakdown.csv",
        result_dir / "tables/phase_n3z2cf_raw_unsafe_action_summary.csv",
        result_dir / "tables/phase_n3z2cf_gpsi_output_summary.csv",
        result_dir / "tables/phase_n3z2cf_feature_block_stats.csv",
    ]
    ckpt_dir = ROOT / args.checkpoint_dir
    for filename in ["parent_500k.zip", "checkpoint_750k.zip", "checkpoint_1000k.zip", "checkpoint_1250k.zip", "checkpoint_1500k.zip", "final.zip", "best_by_eval.zip"]:
        required.append(ckpt_dir / filename)
    missing = [rel(path) for path in required if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise AnalysisStop("train_failed", "missing required N3Z2CF artifacts:\n" + "\n".join(missing))
    episodes = read_csv(result_dir / "tables/phase_n3z2cf_episode_metrics.csv")
    counts = episodes.groupby(["method_key", "checkpoint_label", "scenario"]).size().reset_index(name="episodes")
    bad = counts[counts["episodes"] < int(args.expected_episodes)]
    if not bad.empty:
        raise AnalysisStop("eval_failed", "not enough eval episodes:\n" + bad.to_string(index=False))
    required_labels = {"parent_500k", "750k", "1000k", "1250k", "1500k", "final", "best_by_eval", "old_final", "attention_full_1500k"}
    labels = set(episodes["checkpoint_label"].astype(str))
    missing_labels = sorted(required_labels - labels)
    if missing_labels:
        raise AnalysisStop("eval_failed", f"missing eval labels: {missing_labels}")


def flat_policy_vector(model: PPO) -> torch.Tensor:
    tensors: list[torch.Tensor] = []
    for name, param in model.policy.state_dict().items():
        if torch.is_floating_point(param):
            tensors.append(param.detach().cpu().reshape(-1).to(torch.float64))
    if not tensors:
        return torch.zeros(0, dtype=torch.float64)
    return torch.cat(tensors)


def integrity(result_dir: Path, args: argparse.Namespace) -> pd.DataFrame:
    ckpt_dir = ROOT / args.checkpoint_dir
    eval_summary = read_csv(result_dir / "tables/phase_n3z2cf_checkpoint_eval_summary.csv")
    specs = [
        ("parent_500k", "parent_500k.zip", 500_000),
        ("750k", "checkpoint_750k.zip", 750_000),
        ("1000k", "checkpoint_1000k.zip", 1_000_000),
        ("1250k", "checkpoint_1250k.zip", 1_250_000),
        ("1500k", "checkpoint_1500k.zip", 1_500_000),
        ("final", "final.zip", 1_500_000),
        ("best_by_eval", "best_by_eval.zip", 1_500_000),
    ]
    parent_path = ckpt_dir / "parent_500k.zip"
    parent_model = PPO.load(str(parent_path), device="cpu")
    parent_vec = flat_policy_vector(parent_model)
    rows: list[dict[str, Any]] = []
    for label, filename, step in specs:
        path = ckpt_dir / filename
        exists = path.exists() and path.stat().st_size > 0
        model_num_timesteps = np.nan
        optimizer_present = 0
        l2 = np.nan
        max_abs = np.nan
        policy_l2 = np.nan
        if exists:
            model = PPO.load(str(path), device="cpu")
            model_num_timesteps = int(model.num_timesteps)
            optimizer_present = int(bool(model.policy.optimizer.state_dict().get("state", {})))
            vec = flat_policy_vector(model)
            if vec.numel() == parent_vec.numel():
                diff = vec - parent_vec
                l2 = float(torch.linalg.vector_norm(diff).item())
                max_abs = float(torch.max(torch.abs(diff)).item()) if diff.numel() else 0.0
                policy_l2 = l2
        eval_rows = eval_summary[(eval_summary["method_key"].astype(str) == "z2_corrected_full") & (eval_summary["checkpoint_label"].astype(str) == label)]
        eval_path = ""
        if not eval_rows.empty and "checkpoint_path" in eval_rows.columns:
            eval_path = str(eval_rows["checkpoint_path"].iloc[0])
        expected_eval_path = rel(path)
        rows.append(
            {
                "checkpoint_label": label,
                "checkpoint_path": rel(path),
                "exists": int(exists),
                "size_bytes": int(path.stat().st_size) if exists else 0,
                "sha256": sha256(path) if exists else "missing",
                "global_total_step": int(step),
                "model_num_timesteps": model_num_timesteps,
                "parameter_l2_delta_vs_parent": l2,
                "parameter_max_abs_delta_vs_parent": max_abs,
                "policy_parameter_l2_delta_vs_parent": policy_l2,
                "optimizer_state_present": optimizer_present,
                "eval_row_checkpoint_path": eval_path,
                "eval_path_matches": int(eval_path == expected_eval_path),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(result_dir / "tables/phase_n3z2cf_checkpoint_integrity.csv", index=False)
    bad = out[(out["exists"] != 1) | (out["optimizer_state_present"] != 1) | (out["eval_path_matches"] != 1)]
    zero_delta = out[(out["checkpoint_label"].astype(str) != "parent_500k") & (pd.to_numeric(out["parameter_l2_delta_vs_parent"], errors="coerce") <= 0.0)]
    parent_bad = out[(out["checkpoint_label"].astype(str) == "parent_500k") & (pd.to_numeric(out["model_num_timesteps"], errors="coerce") != 500_000)]
    final_bad = out[(out["checkpoint_label"].astype(str).isin(["1500k", "final", "best_by_eval"])) & (pd.to_numeric(out["model_num_timesteps"], errors="coerce") < 1_500_000)]
    if not bad.empty or not zero_delta.empty or not parent_bad.empty or not final_bad.empty:
        detail = "\n".join(
            [
                "bad rows:",
                bad.to_string(index=False),
                "zero-delta rows:",
                zero_delta.to_string(index=False),
                "parent bad:",
                parent_bad.to_string(index=False),
                "final bad:",
                final_bad.to_string(index=False),
            ]
        )
        raise AnalysisStop("checkpoint_integrity_failed", detail)
    return out


def diagnostics(result_dir: Path) -> pd.DataFrame:
    gpsi = read_csv(result_dir / "tables/phase_n3z2cf_gpsi_output_summary.csv")
    feats = read_csv(result_dir / "tables/phase_n3z2cf_feature_block_stats.csv")
    rows_gpsi = gpsi[gpsi["method_key"].astype(str) == "z2_corrected_full"]
    rows_feat = feats[feats["method_key"].astype(str) == "z2_corrected_full"]
    if rows_gpsi.empty or rows_feat.empty:
        raise AnalysisStop("diagnostics_failed", "missing corrected Z2 full diagnostics rows")
    delta_p95 = pd.to_numeric(rows_gpsi.get("delta_norm_1s_p95", pd.Series(dtype=float)), errors="coerce").max()
    delta_max = pd.to_numeric(rows_gpsi.get("delta_norm_1s_max", pd.Series(dtype=float)), errors="coerce").max()
    inactive = pd.to_numeric(rows_gpsi.get("inactive_forwarded_count_max", pd.Series(dtype=float)), errors="coerce").max()
    logvar_span = pd.to_numeric(rows_gpsi.get("logvar_xy_1s_span", pd.Series(dtype=float)), errors="coerce").max()
    z_after = rows_feat[rows_feat["block"].astype(str) == "z_i_64_after_constraint"]
    z_l2 = pd.to_numeric(z_after.get("l2_norm_p95", pd.Series(dtype=float)), errors="coerce").max()
    nonfinite = int(pd.to_numeric(rows_feat.get("nan_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    nonfinite += int(pd.to_numeric(rows_feat.get("inf_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    ok = (
        np.isfinite(delta_p95)
        and np.isfinite(delta_max)
        and float(delta_p95) < 100.0
        and float(delta_max) < 1000.0
        and (not np.isfinite(inactive) or float(inactive) <= 0.0)
        and np.isfinite(logvar_span)
        and float(logvar_span) > 0.05
        and np.isfinite(z_l2)
        and 3.0 <= float(z_l2) <= 5.0
        and nonfinite == 0
    )
    out = pd.DataFrame(
        [
            {
                "method_key": "z2_corrected_full",
                "diagnostics_ok": int(ok),
                "delta_norm_1s_p95_max": float(delta_p95),
                "delta_norm_1s_max": float(delta_max),
                "inactive_forwarded_count_max": float(inactive) if np.isfinite(inactive) else np.nan,
                "logvar_xy_1s_span_max": float(logvar_span),
                "z_after_constraint_l2_p95_max": float(z_l2),
                "feature_nonfinite_count": int(nonfinite),
            }
        ]
    )
    out.to_csv(result_dir / "tables/phase_n3z2cf_diagnostics_decision.csv", index=False)
    if not ok:
        raise AnalysisStop("diagnostics_failed", out.to_string(index=False))
    return out


def build_comparisons(result_dir: Path, args: argparse.Namespace, diag: pd.DataFrame) -> pd.DataFrame:
    summary = read_csv(result_dir / "tables/phase_n3z2cf_eval_summary.csv")
    agg = aggregate(summary)
    z2 = agg[(agg.method_key == "z2_corrected_full") & (agg.checkpoint_label.astype(str) == "final")]
    if z2.empty:
        z2 = agg[(agg.method_key == "z2_corrected_full") & (agg.checkpoint_label.astype(str) == "1500k")]
    noz = agg[(agg.method_key == "n3f_no_z_full") & (agg.checkpoint_label.astype(str) == "final")]
    attention = agg[(agg.method_key == "attention_full") & (agg.checkpoint_label.astype(str) == "attention_full_1500k")]
    old = agg[(agg.method_key == "z2_old_n3z2c") & (agg.checkpoint_label.astype(str) == "old_final")]
    if z2.empty or noz.empty or attention.empty or old.empty:
        raise AnalysisStop("eval_failed", "missing comparison aggregate rows")
    z, n, a, o = z2.iloc[0], noz.iloc[0], attention.iloc[0], old.iloc[0]
    noz_rows = [
        {
            "metric": "success_rate",
            "corrected_z2": float(z.success_rate),
            "no_z_full": float(n.success_rate),
            "delta_z2_minus_noz": float(z.success_rate) - float(n.success_rate),
            "baseline_value_from_guide": float(args.noz_success),
        },
        {
            "metric": "collision_rate",
            "corrected_z2": float(z.collision_rate),
            "no_z_full": float(n.collision_rate),
            "delta_z2_minus_noz": float(z.collision_rate) - float(n.collision_rate),
            "baseline_value_from_guide": float(args.noz_collision),
        },
    ]
    attention_rows = [
        {
            "metric": "success_rate",
            "corrected_z2": float(z.success_rate),
            "attention_full": float(a.success_rate),
            "delta_z2_minus_attention": float(z.success_rate) - float(a.success_rate),
            "baseline_value_from_guide": float(args.attention_success),
        },
        {
            "metric": "collision_rate",
            "corrected_z2": float(z.collision_rate),
            "attention_full": float(a.collision_rate),
            "delta_z2_minus_attention": float(z.collision_rate) - float(a.collision_rate),
            "baseline_value_from_guide": float(args.attention_collision),
        },
    ]
    old_rows = [
        {
            "metric": "success_rate",
            "corrected_z2": float(z.success_rate),
            "old_z2c": float(o.success_rate),
            "delta_z2_minus_old": float(z.success_rate) - float(o.success_rate),
            "baseline_value_from_guide": float(args.old_z2c_success),
        },
        {
            "metric": "collision_rate",
            "corrected_z2": float(z.collision_rate),
            "old_z2c": float(o.collision_rate),
            "delta_z2_minus_old": float(z.collision_rate) - float(o.collision_rate),
            "baseline_value_from_guide": float(args.old_z2c_collision),
        },
    ]
    pd.DataFrame(noz_rows).to_csv(result_dir / "tables/phase_n3z2cf_noz_reference_comparison.csv", index=False)
    pd.DataFrame(attention_rows).to_csv(result_dir / "tables/phase_n3z2cf_attention_reference_comparison.csv", index=False)
    pd.DataFrame(old_rows).to_csv(result_dir / "tables/phase_n3z2cf_old_z2c_comparison.csv", index=False)

    z2_beats_noz = float(z.success_rate) >= float(args.noz_success) and float(z.collision_rate) <= float(args.noz_collision)
    z2_lower_collision = float(z.collision_rate) <= float(args.noz_collision)
    z2_higher_success = float(z.success_rate) >= float(args.noz_success)
    z2_beats_attention = float(z.success_rate) >= float(args.attention_success) and float(z.collision_rate) <= float(args.attention_collision)
    if z2_beats_noz:
        selected = "corrected_Z2"
        rationale = "corrected Z2 meets or beats no_z on both success and collision."
    elif (not z2_higher_success) and z2_lower_collision:
        selected = "both"
        rationale = "corrected Z2 has lower success but lower collision; keep both candidates as a tradeoff."
    elif z2_higher_success and (not z2_lower_collision) and float(z.collision_rate) - float(args.noz_collision) <= 0.03:
        selected = "both"
        rationale = "corrected Z2 has higher success with a small collision increase; keep both candidates."
    else:
        selected = "no_z"
        rationale = "corrected Z2 is not better than no_z under the decision rule; keep Z2 as ablation."
    can_enter_n4 = "yes"
    decision = pd.DataFrame(
        [
            {
                "corrected_z2_checkpoint_label": str(z.checkpoint_label),
                "corrected_z2_success": float(z.success_rate),
                "corrected_z2_collision": float(z.collision_rate),
                "noz_success": float(n.success_rate),
                "noz_collision": float(n.collision_rate),
                "attention_success": float(a.success_rate),
                "attention_collision": float(a.collision_rate),
                "old_z2c_success": float(o.success_rate),
                "old_z2c_collision": float(o.collision_rate),
                "corrected_minus_noz_success": float(z.success_rate) - float(n.success_rate),
                "corrected_minus_noz_collision": float(z.collision_rate) - float(n.collision_rate),
                "corrected_minus_attention_success": float(z.success_rate) - float(a.success_rate),
                "corrected_minus_attention_collision": float(z.collision_rate) - float(a.collision_rate),
                "corrected_minus_old_success": float(z.success_rate) - float(o.success_rate),
                "corrected_minus_old_collision": float(z.collision_rate) - float(o.collision_rate),
                "z2_beats_noz_gate": int(z2_beats_noz),
                "z2_beats_attention_gate": int(z2_beats_attention),
                "do_not_claim_beats_attention": int(not z2_beats_attention),
                "diagnostics_ok": int(diag["diagnostics_ok"].iloc[0]),
                "selected_n4_candidate": selected,
                "can_enter_n4": can_enter_n4,
                "decision": rationale,
            }
        ]
    )
    decision.to_csv(result_dir / "tables/phase_n3z2cf_final_candidate_decision.csv", index=False)
    return decision


def plot_checkpoint_curve(result_dir: Path, summary: pd.DataFrame) -> None:
    agg = aggregate(summary)
    data = agg[(agg.method_key == "z2_corrected_full") & (agg.checkpoint_label.astype(str).isin(["parent_500k", "750k", "1000k", "1250k", "1500k", "final", "best_by_eval"]))].copy()
    order = {"parent_500k": 500_000, "750k": 750_000, "1000k": 1_000_000, "1250k": 1_250_000, "1500k": 1_500_000, "final": 1_500_001, "best_by_eval": 1_500_002}
    data["order"] = data["checkpoint_label"].map(order)
    data = data.sort_values("order")
    plt.figure(figsize=(9, 4.5))
    x = np.arange(len(data))
    plt.plot(x, data["success_rate"], marker="o", label="success")
    plt.plot(x, data["collision_rate"], marker="o", label="collision")
    plt.xticks(x, data["checkpoint_label"], rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("rate")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "plots/z2cf_checkpoint_success_collision.png", dpi=140)
    plt.close()


def plot_vs_refs(result_dir: Path, summary: pd.DataFrame) -> None:
    agg = aggregate(summary)
    specs = [
        ("z2_corrected_full", "final", "corrected_Z2"),
        ("n3f_no_z_full", "final", "no_z"),
        ("attention_full", "attention_full_1500k", "attention"),
        ("z2_old_n3z2c", "old_final", "old_Z2C"),
    ]
    rows = []
    for method, label, name in specs:
        row = agg[(agg.method_key == method) & (agg.checkpoint_label.astype(str) == label)]
        if not row.empty:
            rows.append((name, float(row.success_rate.iloc[0]), float(row.collision_rate.iloc[0])))
    x = np.arange(len(rows))
    plt.figure(figsize=(8.5, 4.2))
    plt.bar(x - 0.18, [r[1] for r in rows], 0.36, label="success")
    plt.bar(x + 0.18, [r[2] for r in rows], 0.36, label="collision")
    plt.xticks(x, [r[0] for r in rows], rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "plots/z2cf_vs_noz_attention_success_collision.png", dpi=140)
    plt.close()


def plot_scenario(result_dir: Path, summary: pd.DataFrame) -> None:
    data = summary[((summary.method_key == "z2_corrected_full") & (summary.checkpoint_label.astype(str) == "final")) | ((summary.method_key == "n3f_no_z_full") & (summary.checkpoint_label.astype(str) == "final"))]
    plt.figure(figsize=(10, 4.6))
    x = np.arange(len(SCENARIOS))
    width = 0.35
    for idx, (method, label) in enumerate([("z2_corrected_full", "corrected_Z2"), ("n3f_no_z_full", "no_z")]):
        group = data[data.method_key == method].set_index("scenario")
        vals = [float(group.loc[s, "success_rate"]) if s in group.index else np.nan for s in SCENARIOS]
        plt.bar(x + (idx - 0.5) * width, vals, width, label=label)
    plt.xticks(x, SCENARIOS, rotation=25, ha="right")
    plt.ylabel("success rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "plots/z2cf_scenario_breakdown.png", dpi=140)
    plt.close()


def plot_raw(result_dir: Path, summary: pd.DataFrame) -> None:
    agg = aggregate(summary)
    data = agg[(agg.method_key == "z2_corrected_full") & (agg.checkpoint_label.astype(str).isin(["parent_500k", "750k", "1000k", "1250k", "1500k", "final"]))]
    plt.figure(figsize=(9, 4.2))
    plt.bar(data["checkpoint_label"].astype(str), data["raw_unsafe_action_rate"])
    plt.ylabel("raw unsafe action rate")
    plt.tight_layout()
    plt.savefig(result_dir / "plots/z2cf_raw_unsafe_by_checkpoint.png", dpi=140)
    plt.close()


def plot_action(result_dir: Path, summary: pd.DataFrame) -> None:
    agg = aggregate(summary)
    data = agg[(agg.method_key == "z2_corrected_full") & (agg.checkpoint_label.astype(str).isin(["parent_500k", "750k", "1000k", "1250k", "1500k", "final"]))]
    x = np.arange(len(data))
    plt.figure(figsize=(9, 4.2))
    plt.plot(x, data["action_norm"], marker="o", label="action_norm")
    plt.plot(x, data["action_delta"], marker="o", label="action_delta")
    plt.xticks(x, data["checkpoint_label"], rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "plots/z2cf_action_dynamics.png", dpi=140)
    plt.close()


def plot_feature(result_dir: Path) -> None:
    feats = read_csv(result_dir / "tables/phase_n3z2cf_feature_block_stats.csv")
    data = feats[(feats.method_key == "z2_corrected_full") & (feats.checkpoint_label.astype(str) == "final") & (feats.scenario.astype(str) == "eval_flow_id")]
    group = data.groupby("block", dropna=False).agg(v=("l2_norm_p95", "mean")).reset_index()
    plt.figure(figsize=(8.5, 4.2))
    plt.bar(group["block"], group["v"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("p95 L2 norm")
    plt.tight_layout()
    plt.savefig(result_dir / "plots/z2cf_feature_block_scale.png", dpi=140)
    plt.close()


def plot_gpsi(result_dir: Path) -> None:
    gpsi = read_csv(result_dir / "tables/phase_n3z2cf_gpsi_output_summary.csv")
    data = gpsi[gpsi.method_key == "z2_corrected_full"].copy()
    group = data.groupby("checkpoint_label", dropna=False).agg(delta=("delta_norm_1s_p95", "mean"), logvar_span=("logvar_xy_1s_span", "mean")).reset_index()
    order = {"parent_500k": 0, "750k": 1, "1000k": 2, "1250k": 3, "1500k": 4, "final": 5, "best_by_eval": 6}
    group["order"] = group["checkpoint_label"].map(order)
    group = group.sort_values("order")
    x = np.arange(len(group))
    plt.figure(figsize=(9, 4.2))
    plt.plot(x, group["delta"], marker="o")
    plt.xticks(x, group["checkpoint_label"], rotation=20, ha="right")
    plt.ylabel("delta_norm_1s_p95")
    plt.tight_layout()
    plt.savefig(result_dir / "plots/z2cf_gpsi_delta_norm.png", dpi=140)
    plt.close()
    plt.figure(figsize=(9, 4.2))
    plt.plot(x, group["logvar_span"], marker="o")
    plt.xticks(x, group["checkpoint_label"], rotation=20, ha="right")
    plt.ylabel("logvar_xy_1s_span")
    plt.tight_layout()
    plt.savefig(result_dir / "plots/z2cf_gpsi_logvar.png", dpi=140)
    plt.close()


def plot_train(result_dir: Path) -> None:
    train = read_csv(result_dir / "tables/phase_n3z2cf_train_curve.csv")
    plt.figure(figsize=(9, 4.2))
    if not train.empty:
        train = train.sort_values("steps")
        reward = pd.to_numeric(train["episode_reward"], errors="coerce").rolling(50, min_periods=1).mean()
        plt.plot(pd.to_numeric(train["steps"], errors="coerce"), reward)
    plt.xlabel("global training step")
    plt.ylabel("episode reward rolling mean")
    plt.tight_layout()
    plt.savefig(result_dir / "plots/z2cf_train_reward.png", dpi=140)
    plt.close()


def plot_integrity(result_dir: Path, integ: pd.DataFrame) -> None:
    data = integ.copy()
    plt.figure(figsize=(9, 4.2))
    plt.bar(data["checkpoint_label"], data["parameter_l2_delta_vs_parent"])
    plt.yscale("symlog")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("parameter L2 delta vs parent")
    plt.tight_layout()
    plt.savefig(result_dir / "plots/z2cf_checkpoint_integrity.png", dpi=140)
    plt.close()


def generate_plots(result_dir: Path, summary: pd.DataFrame, integ: pd.DataFrame) -> None:
    (result_dir / "plots").mkdir(parents=True, exist_ok=True)
    plot_checkpoint_curve(result_dir, summary)
    plot_vs_refs(result_dir, summary)
    plot_scenario(result_dir, summary)
    plot_raw(result_dir, summary)
    plot_action(result_dir, summary)
    plot_feature(result_dir)
    plot_gpsi(result_dir)
    plot_train(result_dir)
    plot_integrity(result_dir, integ)


def collect_files(result_dir: Path, args: argparse.Namespace) -> dict[str, list[str]]:
    ckpt_dir = ROOT / args.checkpoint_dir
    return {
        "checkpoints": [rel(path) for path in sorted(ckpt_dir.glob("*.zip"))],
        "tables": [rel(path) for path in sorted((result_dir / "tables").glob("*")) if path.is_file()],
        "plots": [rel(path) for path in sorted((result_dir / "plots").glob("*.png"))],
        "logs": [rel(path) for path in sorted((result_dir / "logs").glob("*.log"))] + [rel(result_dir / "phase_n3z2cf_watcher.log")],
        "flags": [rel(path) for path in sorted(result_dir.glob("*.flag"))],
    }


def report(result_dir: Path, decision: pd.DataFrame, diag: pd.DataFrame, integ: pd.DataFrame, summary: pd.DataFrame, files: dict[str, list[str]]) -> list[str]:
    resume = read_csv(result_dir / "tables/phase_n3z2cf_resume_semantics.csv")
    agg = aggregate(summary)
    d = decision.iloc[0]
    lines = [
        "# Phase N3Z2CF Corrected Z2 Full Report",
        "",
        "## Terminal Decision",
        "",
        "`terminal_decision = phase_n3z2cf_corrected_z2_full_complete`",
        "",
        "Corrected Z2 full continuation completed from the fixed checkpoint_500k parent with reset_num_timesteps=False.",
        "",
        "## Final Candidate Decision",
        "",
    ]
    lines.extend(table_md(decision, list(decision.columns), max_rows=3))
    lines.append(f"- Can enter N4: {d['can_enter_n4']}.")
    lines.append(f"- Selected N4 candidate: `{d['selected_n4_candidate']}`.")
    lines.append(f"- Attention claim guard: do_not_claim_beats_attention={d['do_not_claim_beats_attention']}.")
    lines.extend(["", "## Resume Semantics", ""])
    lines.extend(table_md(resume, [col for col in ["phase", "loaded_checkpoint_path", "parent_sha256", "reset_num_timesteps", "model_num_timesteps", "model_parent_step_match", "model_target_step_match", "optimizer_state_restored", "n_envs", "n_steps", "batch_size"] if col in resume.columns], max_rows=10))
    lines.extend(["", "## Checkpoint Integrity", ""])
    lines.extend(table_md(integ, ["checkpoint_label", "checkpoint_path", "sha256", "global_total_step", "model_num_timesteps", "parameter_l2_delta_vs_parent", "parameter_max_abs_delta_vs_parent", "optimizer_state_present", "eval_path_matches"], max_rows=10))
    lines.extend(["", "## Aggregate Eval", ""])
    lines.extend(table_md(agg, ["method_key", "checkpoint_label", "success_rate", "collision_rate", "near_miss_rate", "progress", "raw_unsafe_action_rate", "action_delta", "mean_min_distance"], max_rows=30))
    lines.extend(["", "## Diagnostics", ""])
    lines.extend(table_md(diag, list(diag.columns), max_rows=5))
    lines.extend(["", "## Artifacts", ""])
    for key in ["checkpoints", "tables", "plots", "logs", "flags"]:
        lines.append(f"### {key}")
        values = files.get(key, [])
        if values:
            lines.extend([f"- `{value}`" for value in values[:160]])
            if len(values) > 160:
                lines.append(f"- ... {len(values) - 160} more")
        else:
            lines.append("- none")
    return lines


def main() -> None:
    args = parse_args()
    result_dir = ROOT / args.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    try:
        validate(result_dir, args)
        summary = read_csv(result_dir / "tables/phase_n3z2cf_eval_summary.csv")
        integ = integrity(result_dir, args)
        diag = diagnostics(result_dir)
        decision = build_comparisons(result_dir, args, diag)
        generate_plots(result_dir, summary, integ)
        write_text(result_dir / COMPLETE_FLAG, "terminal_decision = phase_n3z2cf_corrected_z2_full_complete\n")
        write_text(result_dir / "phase_n3z2cf_status.txt", "complete\n")
        files = collect_files(result_dir, args)
        write_text(result_dir / "PHASE_N3Z2CF_CORRECTED_Z2_FULL_REPORT.md", "\n".join(report(result_dir, decision, diag, integ, summary, files)) + "\n")
        print("terminal_decision = phase_n3z2cf_corrected_z2_full_complete", flush=True)
    except AnalysisStop as exc:
        write_stop(result_dir, exc.reason, exc.detail)
        raise SystemExit(2) from exc
    except Exception:
        detail = traceback.format_exc()
        write_stop(result_dir, "diagnostics_failed", detail)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
