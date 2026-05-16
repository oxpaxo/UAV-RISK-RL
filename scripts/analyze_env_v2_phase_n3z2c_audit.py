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


COMPLETE_FLAG = "PHASE_N3Z2C_AUDIT_COMPLETE.flag"
STOP_FLAGS = {
    "n3z2c_missing": "PHASE_N3Z2C_AUDIT_STOP_N3Z2C_MISSING.flag",
    "parent_candidates_missing": "PHASE_N3Z2C_AUDIT_STOP_PARENT_CANDIDATES_MISSING.flag",
    "resume_semantics_unresolved": "PHASE_N3Z2C_AUDIT_STOP_RESUME_SEMANTICS_UNRESOLVED.flag",
    "corrected_parent_missing": "PHASE_N3Z2C_AUDIT_STOP_CORRECTED_PARENT_MISSING.flag",
    "train_failed": "PHASE_N3Z2C_AUDIT_STOP_TRAIN_FAILED.flag",
    "eval_failed": "PHASE_N3Z2C_AUDIT_STOP_EVAL_FAILED.flag",
    "diagnostics_failed": "PHASE_N3Z2C_AUDIT_STOP_DIAGNOSTICS_FAILED.flag",
    "cpu_affinity_unresolved": "PHASE_N3Z2C_AUDIT_STOP_CPU_AFFINITY_UNRESOLVED.flag",
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
    parser = argparse.ArgumentParser(description="Analyze Phase N3Z2C-Audit outputs.")
    parser.add_argument("--result-dir", default="results/env_v2_phase_n3z2c_audit")
    parser.add_argument("--old-n3z2c-result-dir", default="results/env_v2_phase_n3z2c_z2_continuation")
    parser.add_argument("--n3fz-result-dir", default="results/env_v2_phase_n3fz_noz_full_z_screen")
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
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_stop(result_dir: Path, reason: str, detail: str) -> None:
    flag = STOP_FLAGS.get(reason, STOP_FLAGS["diagnostics_failed"])
    write_text(result_dir / flag, f"{reason}\n{detail.strip()}\n")
    write_text(result_dir / "phase_n3z2c_audit_status.txt", f"stopped:{flag}\n")
    write_text(
        result_dir / "PHASE_N3Z2C_AUDIT_REPORT.md",
        "\n".join(
            [
                "# Phase N3Z2C-Audit Report",
                "",
                f"`terminal_decision = phase_n3z2c_audit_stopped_{reason}`",
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


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty CSV: {rel(path)}")
    return pd.read_csv(path)


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
            success_rate=("success_rate", "mean"),
            collision_rate=("collision_rate", "mean"),
            near_miss_rate=("near_miss_rate", "mean"),
            progress=("progress", "mean"),
            raw_unsafe_action_rate=("raw_unsafe_action_rate", "mean"),
            action_delta=("action_delta", "mean"),
            mean_min_distance=("mean_min_distance", "mean"),
            nan_or_crash=("nan_or_crash", "max"),
        )
        .reset_index()
    )


def validate(result_dir: Path, args: argparse.Namespace) -> None:
    required = [
        ROOT / "results/env_v2_phase_n3z2c_z2_continuation/PHASE_N3Z2C_Z2_CONTINUATION_COMPLETE.flag",
        result_dir / "tables/phase_n3z2c_audit_resource_affinity.csv",
        result_dir / "tables/phase_n3z2c_audit_checkpoint_hash.csv",
        result_dir / "tables/phase_n3z2c_audit_parent_selection_fixed.csv",
        result_dir / "tables/phase_n3z2c_audit_resume_semantics.csv",
        result_dir / "tables/phase_n3z2c_audit_train_script_findings.csv",
        result_dir / "tables/phase_n3z2c_audit_command_manifest.csv",
        result_dir / "tables/phase_n3z2c_audit_short_continuation_train_curve.csv",
        result_dir / "tables/phase_n3z2c_audit_short_continuation_heartbeat.csv",
        result_dir / "tables/phase_n3z2c_audit_checkpoint_eval_summary.csv",
        result_dir / "tables/phase_n3z2c_audit_eval_summary.csv",
        result_dir / "tables/phase_n3z2c_audit_scenario_breakdown.csv",
        result_dir / "tables/phase_n3z2c_audit_raw_unsafe_summary.csv",
        result_dir / "tables/phase_n3z2c_audit_gpsi_output_summary.csv",
        result_dir / "tables/phase_n3z2c_audit_feature_block_stats.csv",
        ROOT / "checkpoints/env_v2_gpsi_heada_ppo_n3z2c_audit_corrected_parent_750k_s0/parent_500k.zip",
        ROOT / "checkpoints/env_v2_gpsi_heada_ppo_n3z2c_audit_corrected_parent_750k_s0/checkpoint_750k.zip",
        ROOT / "checkpoints/env_v2_gpsi_heada_ppo_n3z2c_audit_corrected_parent_750k_s0/final.zip",
    ]
    missing = [rel(path) for path in required if not path.exists() or path.stat().st_size == 0]
    if missing:
        reason = "n3z2c_missing" if any("PHASE_N3Z2C_Z2_CONTINUATION_COMPLETE" in item for item in missing) else "train_failed"
        raise AnalysisStop(reason, "missing required Audit artifacts:\n" + "\n".join(missing))
    episodes = read_csv(result_dir / "tables/phase_n3z2c_audit_episode_metrics.csv")
    counts = episodes.groupby(["method_key", "checkpoint_label", "scenario"]).size().reset_index(name="episodes")
    bad = counts[counts["episodes"] < int(args.expected_episodes)]
    if not bad.empty:
        raise AnalysisStop("eval_failed", "not enough eval episodes:\n" + bad.to_string(index=False))
    labels = set(episodes["checkpoint_label"].astype(str))
    required_labels = {"attention_full_1500k", "final", "old_parent_500k", "old_750k", "corrected_parent_500k", "corrected_750k"}
    missing_labels = sorted(required_labels - labels)
    if missing_labels:
        raise AnalysisStop("eval_failed", f"missing eval labels: {missing_labels}")


def diagnostics(result_dir: Path) -> pd.DataFrame:
    gpsi = read_csv(result_dir / "tables/phase_n3z2c_audit_gpsi_output_summary.csv")
    feats = read_csv(result_dir / "tables/phase_n3z2c_audit_feature_block_stats.csv")
    rows_gpsi = gpsi[(gpsi["method_key"].astype(str) == "z2_corrected_parent") & (gpsi["checkpoint_label"].astype(str) == "corrected_750k")]
    rows_feat = feats[(feats["method_key"].astype(str) == "z2_corrected_parent") & (feats["checkpoint_label"].astype(str) == "corrected_750k")]
    if rows_gpsi.empty or rows_feat.empty:
        raise AnalysisStop("diagnostics_failed", "missing corrected_750k diagnostics rows")
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
        and 3.5 <= float(z_l2) <= 4.5
        and nonfinite == 0
    )
    out = pd.DataFrame(
        [
            {
                "method_key": "z2_corrected_parent",
                "checkpoint_label": "corrected_750k",
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
    out.to_csv(result_dir / "tables/phase_n3z2c_audit_diagnostics_decision.csv", index=False)
    if not ok:
        raise AnalysisStop("diagnostics_failed", out.to_string(index=False))
    return out


def build_decision(result_dir: Path, args: argparse.Namespace, diag: pd.DataFrame) -> pd.DataFrame:
    summary = read_csv(result_dir / "tables/phase_n3z2c_audit_eval_summary.csv")
    agg = aggregate(summary)
    parent = read_csv(result_dir / "tables/phase_n3z2c_audit_parent_selection_fixed.csv")
    resume = read_csv(result_dir / "tables/phase_n3z2c_audit_resume_semantics.csv")
    cpu = read_csv(result_dir / "tables/phase_n3z2c_audit_resource_affinity.csv")
    fixed = parent[parent["selected_by_fixed_rule"] == 1]
    if fixed.empty:
        raise AnalysisStop("parent_candidates_missing", "fixed parent selection row missing")
    corrected = agg[(agg.method_key == "z2_corrected_parent") & (agg.checkpoint_label.astype(str) == "corrected_750k")]
    corrected_parent = agg[(agg.method_key == "z2_corrected_parent") & (agg.checkpoint_label.astype(str) == "corrected_parent_500k")]
    old = agg[(agg.method_key == "z2_old_n3z2c") & (agg.checkpoint_label.astype(str) == "old_750k")]
    noz = agg[(agg.method_key == "n3f_no_z_full") & (agg.checkpoint_label.astype(str) == "final")]
    attention = agg[(agg.method_key == "attention_full") & (agg.checkpoint_label.astype(str) == "attention_full_1500k")]
    if corrected.empty or corrected_parent.empty or old.empty or noz.empty or attention.empty:
        raise AnalysisStop("eval_failed", "missing comparison aggregate rows")
    c, cp, o, n, a = corrected.iloc[0], corrected_parent.iloc[0], old.iloc[0], noz.iloc[0], attention.iloc[0]
    resume_reset_ok = bool((resume.get("reset_num_timesteps", pd.Series(dtype=object)).astype(str).str.lower() == "false").any())
    optimizer_ok = bool(pd.to_numeric(resume.get("optimizer_state_restored", pd.Series(dtype=float)), errors="coerce").fillna(0).max() >= 1)
    cpu_finding_rows = cpu[cpu["item"].astype(str) == "cpu_affinity_finding"]
    cpu_finding = cpu_finding_rows["output"].iloc[-1] if not cpu_finding_rows.empty else "missing"
    old_parent_wrong = bool(int(fixed["selected_by_old_rule"].iloc[0]) == 0)
    corrected_beats_old_750 = float(c.success_rate) >= float(o.success_rate) and float(c.collision_rate) <= float(o.collision_rate)
    corrected_still_degraded = float(c.success_rate) < float(cp.success_rate) and float(c.collision_rate) > float(cp.collision_rate)
    corrected_beats_noz = float(c.success_rate) >= float(n.success_rate) and float(c.collision_rate) <= float(n.collision_rate)
    if not resume_reset_ok or not optimizer_ok:
        final_decision = "unresolved"
        can_enter_n4 = "no"
        need_corrected_full = "no"
        decision_text = "Resume semantics are not fully confirmed; do not advance."
    elif corrected_beats_noz:
        final_decision = "rerun_corrected_z2_full"
        can_enter_n4 = "no"
        need_corrected_full = "yes"
        decision_text = "Corrected short continuation beats no_z; run corrected Z2 full before N4."
    elif corrected_beats_old_750 and not corrected_still_degraded and float(c.success_rate) >= float(cp.success_rate) - 0.02:
        final_decision = "rerun_corrected_z2_full"
        can_enter_n4 = "no"
        need_corrected_full = "yes"
        decision_text = "Corrected parent materially changes short-continuation behavior; rerun corrected Z2 full before final N4 selection."
    else:
        final_decision = "accept_no_z_n4_candidate"
        can_enter_n4 = "yes"
        need_corrected_full = "no"
        decision_text = "Corrected short continuation does not rescue Z2; accept no_z_full as N4 candidate and keep Z2 as ablation."
    rows = [
        {
            "fixed_parent_candidate": fixed["candidate"].iloc[0],
            "fixed_parent_path": fixed["path"].iloc[0],
            "old_parent_selection_wrong": int(old_parent_wrong),
            "cpu_affinity_finding": cpu_finding,
            "resume_reset_num_timesteps_false": int(resume_reset_ok),
            "optimizer_state_restored": int(optimizer_ok),
            "diagnostics_ok": int(diag["diagnostics_ok"].iloc[0]),
            "corrected_parent_success": float(cp.success_rate),
            "corrected_parent_collision": float(cp.collision_rate),
            "corrected_750k_success": float(c.success_rate),
            "corrected_750k_collision": float(c.collision_rate),
            "old_750k_success": float(o.success_rate),
            "old_750k_collision": float(o.collision_rate),
            "noz_success": float(n.success_rate),
            "noz_collision": float(n.collision_rate),
            "attention_success": float(a.success_rate),
            "attention_collision": float(a.collision_rate),
            "corrected_minus_old_success": float(c.success_rate) - float(o.success_rate),
            "corrected_minus_old_collision": float(c.collision_rate) - float(o.collision_rate),
            "corrected_minus_noz_success": float(c.success_rate) - float(n.success_rate),
            "corrected_minus_noz_collision": float(c.collision_rate) - float(n.collision_rate),
            "corrected_beats_old_750_gate": int(corrected_beats_old_750),
            "corrected_still_degraded_vs_parent": int(corrected_still_degraded),
            "need_corrected_z2_full_rerun": need_corrected_full,
            "final_decision": final_decision,
            "can_enter_n4": can_enter_n4,
            "selected_n4_candidate_if_yes": "no_z_full" if can_enter_n4 == "yes" else "",
            "decision": decision_text,
        }
    ]
    out = pd.DataFrame(rows)
    out.to_csv(result_dir / "tables/phase_n3z2c_audit_decision.csv", index=False)
    return out


def plot_parent(result_dir: Path) -> None:
    parent = read_csv(result_dir / "tables/phase_n3z2c_audit_parent_selection_fixed.csv")
    x = np.arange(len(parent))
    plt.figure(figsize=(8, 4.2))
    plt.bar(x - 0.18, parent["success_rate"], 0.36, label="success")
    plt.bar(x + 0.18, parent["collision_rate"], 0.36, label="collision")
    plt.xticks(x, parent["candidate"], rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "plots/audit_parent_selection_comparison.png", dpi=140)
    plt.close()


def plot_success(result_dir: Path, summary: pd.DataFrame) -> None:
    agg = aggregate(summary)
    labels = ["old_750k", "corrected_750k", "no_z", "attention"]
    rows = [
        agg[(agg.method_key == "z2_old_n3z2c") & (agg.checkpoint_label.astype(str) == "old_750k")],
        agg[(agg.method_key == "z2_corrected_parent") & (agg.checkpoint_label.astype(str) == "corrected_750k")],
        agg[(agg.method_key == "n3f_no_z_full") & (agg.checkpoint_label.astype(str) == "final")],
        agg[(agg.method_key == "attention_full")],
    ]
    vals = [(float(row.success_rate.iloc[0]), float(row.collision_rate.iloc[0])) if not row.empty else (np.nan, np.nan) for row in rows]
    x = np.arange(len(labels))
    plt.figure(figsize=(8.5, 4.3))
    plt.bar(x - 0.18, [v[0] for v in vals], 0.36, label="success")
    plt.bar(x + 0.18, [v[1] for v in vals], 0.36, label="collision")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "plots/audit_corrected_vs_old_750k_success_collision.png", dpi=140)
    plt.close()


def plot_train(result_dir: Path) -> None:
    train = read_csv(result_dir / "tables/phase_n3z2c_audit_short_continuation_train_curve.csv")
    plt.figure(figsize=(8, 4.2))
    if not train.empty:
        train = train.sort_values("steps")
        reward = pd.to_numeric(train["episode_reward"], errors="coerce").rolling(25, min_periods=1).mean()
        plt.plot(pd.to_numeric(train["steps"], errors="coerce"), reward)
    plt.xlabel("global training step")
    plt.ylabel("episode reward rolling mean")
    plt.tight_layout()
    plt.savefig(result_dir / "plots/audit_short_continuation_curve.png", dpi=140)
    plt.close()


def plot_raw(result_dir: Path, summary: pd.DataFrame) -> None:
    agg = aggregate(summary)
    data = agg[agg["checkpoint_label"].astype(str).isin(["old_750k", "corrected_750k", "final", "attention_full_1500k"])]
    labels = data["method_key"].astype(str) + "_" + data["checkpoint_label"].astype(str)
    plt.figure(figsize=(9, 4.2))
    plt.bar(labels, data["raw_unsafe_action_rate"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("raw unsafe action rate")
    plt.tight_layout()
    plt.savefig(result_dir / "plots/audit_raw_unsafe_comparison.png", dpi=140)
    plt.close()


def plot_scenario(result_dir: Path, summary: pd.DataFrame) -> None:
    data = summary[((summary.method_key == "z2_old_n3z2c") & (summary.checkpoint_label.astype(str) == "old_750k")) | ((summary.method_key == "z2_corrected_parent") & (summary.checkpoint_label.astype(str) == "corrected_750k"))]
    plt.figure(figsize=(10, 4.5))
    x = np.arange(len(SCENARIOS))
    width = 0.35
    for idx, (method, label) in enumerate([("z2_old_n3z2c", "old_750k"), ("z2_corrected_parent", "corrected_750k")]):
        group = data[(data.method_key == method) & (data.checkpoint_label.astype(str) == label)].set_index("scenario")
        vals = [float(group.loc[s, "success_rate"]) if s in group.index else np.nan for s in SCENARIOS]
        plt.bar(x + (idx - 0.5) * width, vals, width, label=label)
    plt.xticks(x, SCENARIOS, rotation=25, ha="right")
    plt.ylabel("success rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "plots/audit_scenario_breakdown.png", dpi=140)
    plt.close()


def plot_feature(result_dir: Path) -> None:
    feats = read_csv(result_dir / "tables/phase_n3z2c_audit_feature_block_stats.csv")
    data = feats[(feats.method_key == "z2_corrected_parent") & (feats.checkpoint_label.astype(str) == "corrected_750k") & (feats.scenario.astype(str) == "eval_flow_id")]
    group = data.groupby("block", dropna=False).agg(v=("l2_norm_p95", "mean")).reset_index()
    plt.figure(figsize=(8.5, 4.2))
    plt.bar(group["block"], group["v"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("p95 L2 norm")
    plt.tight_layout()
    plt.savefig(result_dir / "plots/audit_feature_block_scale.png", dpi=140)
    plt.close()


def build_checkpoint_curve(result_dir: Path, args: argparse.Namespace, summary: pd.DataFrame) -> pd.DataFrame:
    n3fz_path = ROOT / args.n3fz_result_dir / "tables/phase_n3fz_checkpoint_eval_summary.csv"
    n3fz = read_csv(n3fz_path)
    rows: list[dict[str, Any]] = []

    for step, label in [(250000, "250k"), (500000, "500k")]:
        data = n3fz[
            (n3fz["method_key"].astype(str) == "z_layernorm_alpha_0p5")
            & (pd.to_numeric(n3fz["checkpoint_step"], errors="coerce") == step)
            & (n3fz["checkpoint_label"].astype(str) == label)
        ]
        if data.empty:
            raise AnalysisStop("diagnostics_failed", f"missing N3F/Z Z2 checkpoint eval rows for {label}: {rel(n3fz_path)}")
        rows.append(
            {
                "checkpoint_step": step,
                "checkpoint_label": label,
                "success_rate": float(pd.to_numeric(data["success_rate"], errors="coerce").mean()),
                "collision_rate": float(pd.to_numeric(data["collision_rate"], errors="coerce").mean()),
                "episodes": int(pd.to_numeric(data["episodes"], errors="coerce").sum()),
                "source": rel(n3fz_path),
                "source_method_key": "z_layernorm_alpha_0p5",
            }
        )

    data_750 = summary[
        (summary["method_key"].astype(str) == "z2_corrected_parent")
        & (pd.to_numeric(summary["checkpoint_step"], errors="coerce") == 750000)
        & (summary["checkpoint_label"].astype(str) == "corrected_750k")
    ]
    if data_750.empty:
        raise AnalysisStop("diagnostics_failed", "missing Audit corrected_750k checkpoint eval rows")
    rows.append(
        {
            "checkpoint_step": 750000,
            "checkpoint_label": "750k",
            "success_rate": float(pd.to_numeric(data_750["success_rate"], errors="coerce").mean()),
            "collision_rate": float(pd.to_numeric(data_750["collision_rate"], errors="coerce").mean()),
            "episodes": int(pd.to_numeric(data_750["episodes"], errors="coerce").sum()),
            "source": rel(result_dir / "tables/phase_n3z2c_audit_checkpoint_eval_summary.csv"),
            "source_method_key": "z2_corrected_parent",
        }
    )

    out = pd.DataFrame(rows).sort_values("checkpoint_step")
    out.to_csv(result_dir / "tables/phase_n3z2c_audit_checkpoint_success_collision_curve.csv", index=False)
    return out


def plot_checkpoint_success_collision(result_dir: Path, args: argparse.Namespace, summary: pd.DataFrame) -> None:
    data = build_checkpoint_curve(result_dir, args, summary)
    x = data["checkpoint_step"].astype(float) / 1000.0
    plt.figure(figsize=(7.2, 4.2))
    plt.plot(x, data["success_rate"], marker="o", linewidth=2.0, label="success")
    plt.plot(x, data["collision_rate"], marker="o", linewidth=2.0, label="collision")
    for _, row in data.iterrows():
        xpos = float(row["checkpoint_step"]) / 1000.0
        plt.annotate(fmt(row["success_rate"]), (xpos, float(row["success_rate"])), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
        plt.annotate(fmt(row["collision_rate"]), (xpos, float(row["collision_rate"])), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=8)
    plt.xticks(x, data["checkpoint_label"].astype(str))
    plt.ylim(0.0, 1.0)
    plt.xlabel("checkpoint")
    plt.ylabel("rate")
    plt.title("Z2 checkpoint success / collision")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(result_dir / "plots/checkpoint_success_collision.png", dpi=140)
    plt.close()


def generate_plots(result_dir: Path, args: argparse.Namespace, summary: pd.DataFrame) -> None:
    (result_dir / "plots").mkdir(parents=True, exist_ok=True)
    plot_parent(result_dir)
    plot_success(result_dir, summary)
    plot_checkpoint_success_collision(result_dir, args, summary)
    plot_train(result_dir)
    plot_raw(result_dir, summary)
    plot_scenario(result_dir, summary)
    plot_feature(result_dir)


def collect_files(result_dir: Path) -> dict[str, list[str]]:
    ckpt_dir = ROOT / "checkpoints/env_v2_gpsi_heada_ppo_n3z2c_audit_corrected_parent_750k_s0"
    return {
        "checkpoints": [rel(path) for path in sorted(ckpt_dir.glob("*.zip"))],
        "tables": [rel(path) for path in sorted((result_dir / "tables").glob("*")) if path.is_file()],
        "plots": [rel(path) for path in sorted((result_dir / "plots").glob("*.png"))],
        "logs": [rel(path) for path in sorted((result_dir / "logs").glob("*.log"))] + [rel(result_dir / "phase_n3z2c_audit_watcher.log")],
        "flags": [rel(path) for path in sorted(result_dir.glob("*.flag"))],
    }


def report(result_dir: Path, decision: pd.DataFrame, diag: pd.DataFrame, summary: pd.DataFrame, files: dict[str, list[str]]) -> list[str]:
    parent = read_csv(result_dir / "tables/phase_n3z2c_audit_parent_selection_fixed.csv")
    resume = read_csv(result_dir / "tables/phase_n3z2c_audit_resume_semantics.csv")
    cpu = read_csv(result_dir / "tables/phase_n3z2c_audit_resource_affinity.csv")
    findings = read_csv(result_dir / "tables/phase_n3z2c_audit_train_script_findings.csv")
    agg = aggregate(summary)
    d = decision.iloc[0]
    lines = [
        "# Phase N3Z2C-Audit Report",
        "",
        "## Terminal Decision",
        "",
        "`terminal_decision = phase_n3z2c_audit_complete`",
        "",
        "Phase N3Z2C-Audit complete. This stage audited parent selection, resume semantics, CPU affinity, and ran a corrected short continuation sanity from the fixed 500k parent.",
        "",
        "## Final Decision",
        "",
    ]
    lines.extend(table_md(decision, list(decision.columns), max_rows=3))
    lines.append(f"- Can enter N4: {d['can_enter_n4']}.")
    lines.append(f"- Selected N4 candidate if yes: `{d['selected_n4_candidate_if_yes']}`.")
    lines.append(f"- Need corrected Z2 full rerun: {d['need_corrected_z2_full_rerun']}.")
    lines.append(f"- Decision: {d['decision']}")
    lines.extend(["", "## Parent Selection Audit", ""])
    lines.extend(table_md(parent, ["candidate", "exists", "size_bytes", "eval_label", "success_rate", "collision_rate", "near_miss_rate", "raw_unsafe_action_rate", "diagnostics_ok", "selected_by_old_rule", "selected_by_fixed_rule", "selection_reason"], max_rows=10))
    lines.extend(["", "## Resume Semantics Audit", ""])
    lines.extend(table_md(findings, ["script", "uses_reset_num_timesteps_false", "uses_reset_num_timesteps_true", "finding"], max_rows=10))
    lines.extend(table_md(resume, [col for col in ["phase", "check", "status", "reset_num_timesteps", "model_num_timesteps", "optimizer_state_restored", "n_envs", "n_steps", "batch_size", "detail"] if col in resume.columns], max_rows=20))
    lines.extend(["", "## CPU Affinity Finding", ""])
    lines.extend(table_md(cpu[cpu["item"].astype(str).isin(["nproc", "nproc_all", "taskset_current_shell", "python_cpu_affinity", "cpuset_cpus_effective", "cpu_max", "cpu_affinity_finding"])], ["item", "returncode", "output"], max_rows=12))
    lines.extend(["", "## Corrected Short Continuation Results", ""])
    lines.extend(table_md(agg, ["method_key", "checkpoint_label", "success_rate", "collision_rate", "near_miss_rate", "progress", "raw_unsafe_action_rate", "action_delta", "mean_min_distance"], max_rows=20))
    lines.extend(["", "## Diagnostics", ""])
    lines.extend(table_md(diag, list(diag.columns), max_rows=5))
    lines.extend(["", "## Artifacts", ""])
    for key in ["checkpoints", "tables", "plots", "logs", "flags"]:
        lines.append(f"### {key}")
        values = files.get(key, [])
        if values:
            lines.extend([f"- `{value}`" for value in values[:140]])
            if len(values) > 140:
                lines.append(f"- ... {len(values) - 140} more")
        else:
            lines.append("- none")
    return lines


def main() -> None:
    args = parse_args()
    result_dir = ROOT / args.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    try:
        validate(result_dir, args)
        summary = read_csv(result_dir / "tables/phase_n3z2c_audit_eval_summary.csv")
        diag = diagnostics(result_dir)
        decision = build_decision(result_dir, args, diag)
        generate_plots(result_dir, args, summary)
        write_text(result_dir / COMPLETE_FLAG, "terminal_decision = phase_n3z2c_audit_complete\n")
        write_text(result_dir / "phase_n3z2c_audit_status.txt", "complete\n")
        files = collect_files(result_dir)
        write_text(result_dir / "PHASE_N3Z2C_AUDIT_REPORT.md", "\n".join(report(result_dir, decision, diag, summary, files)) + "\n")
        print("terminal_decision = phase_n3z2c_audit_complete", flush=True)
    except AnalysisStop as exc:
        write_stop(result_dir, exc.reason, exc.detail)
        raise SystemExit(2) from exc
    except Exception:
        detail = traceback.format_exc()
        write_stop(result_dir, "diagnostics_failed", detail)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
