from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from phase_n3_5_audit_common import (
    COMPLETE_FLAG,
    N3_REPORT,
    ROOT,
    PhaseN35Stop,
    check_prerequisites,
    command_manifest_row,
    ensure_dirs,
    finite_max,
    safe_mean,
    write_csv,
    write_stop,
    write_text,
)


REQUIRED_TABLES = [
    "phase_n3_5_offline_online_equivalence.csv",
    "phase_n3_5_input_distribution_compare.csv",
    "phase_n3_5_output_scale_summary.csv",
    "phase_n3_5_aug_feature_block_stats.csv",
    "phase_n3_5_field_order_check.csv",
    "phase_n3_5_history_buffer_check.csv",
    "phase_n3_5_active_mask_check.csv",
    "phase_n3_5_normalization_check.csv",
    "phase_n3_5_slicing_check.csv",
    "phase_n3_5_short_rollout_output_summary.csv",
    "phase_n3_5_repair_actions.csv",
    "phase_n3_5_rerun_recommendation.csv",
    "phase_n3_5_command_manifest.csv",
    "phase_n3_5_online_audit_status.csv",
    "phase_n3_5_feature_scale_status.csv",
]

REQUIRED_PLOTS = [
    "offline_vs_online_delta_scatter.png",
    "offline_vs_online_logvar_scatter.png",
    "online_delta_norm_distribution_before_after.png",
    "online_logvar_distribution_before_after.png",
    "aug_feature_block_scale.png",
    "input_distribution_shift.png",
    "z_norm_distribution.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize Phase N3.5 Gpsi wrapper audit report and flags.")
    parser.add_argument("--checkpoint", default="work_dirs/gpsi_heada_v1_nll/best.pth")
    parser.add_argument("--out-dir", default="results/env_v2_phase_n3_5_gpsi_wrapper_audit")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def table_status(rows: list[dict[str, Any]]) -> str:
    statuses = [str(row.get("status", "")).lower() for row in rows]
    if any(status == "fail" for status in statuses):
        return "fail"
    if any(status == "warn" for status in statuses):
        return "warn"
    return "pass"


def require_artifacts(out_dir: Path) -> None:
    missing = [str(out_dir / "tables" / name) for name in REQUIRED_TABLES if not (out_dir / "tables" / name).exists()]
    missing.extend(str(out_dir / "plots" / name) for name in REQUIRED_PLOTS if not (out_dir / "plots" / name).exists())
    missing.extend(str(out_dir / "logs" / name) for name in ["phase_n3_5_offline_online_compare.log", "phase_n3_5_online_audit.log", "phase_n3_5_feature_scale.log"] if not (out_dir / "logs" / name).exists())
    if missing:
        raise PhaseN35Stop("watcher_failed", "N3.5 required artifacts missing:\n" + "\n".join(missing))


def summarize(out_dir: Path) -> dict[str, Any]:
    eq = read_rows(out_dir / "tables/phase_n3_5_offline_online_equivalence.csv")
    online_status = read_rows(out_dir / "tables/phase_n3_5_online_audit_status.csv")
    feature_status = read_rows(out_dir / "tables/phase_n3_5_feature_scale_status.csv")
    output_steps = read_rows(out_dir / "tables/phase_n3_5_output_scale_steps.csv")
    aug = read_rows(out_dir / "tables/phase_n3_5_aug_feature_block_stats.csv")
    norm = read_rows(out_dir / "tables/phase_n3_5_normalization_check.csv")
    active = read_rows(out_dir / "tables/phase_n3_5_active_mask_check.csv")
    history = read_rows(out_dir / "tables/phase_n3_5_history_buffer_check.csv")
    repair = read_rows(out_dir / "tables/phase_n3_5_repair_actions.csv")
    rerun = read_rows(out_dir / "tables/phase_n3_5_rerun_recommendation.csv")

    if not eq or not output_steps or not aug:
        raise PhaseN35Stop("watcher_failed", "core N3.5 tables are empty")
    failed_eq = [
        row
        for row in eq
        if row.get("repair_stage") == "after_fix"
        and row.get("component") in {"z", "delta_hat", "logvar_hat", "history_valid_mask"}
        and str(row.get("allclose_pass")) not in {"1", "1.0", "True", "true"}
    ]
    if failed_eq:
        raise PhaseN35Stop("offline_online_mismatch", f"after-fix offline-online equivalence failed: {failed_eq}")
    if table_status(online_status) == "fail":
        raise PhaseN35Stop("output_scale_invalid", f"after-fix online audit failed: {online_status}")
    if table_status(feature_status) == "fail":
        raise PhaseN35Stop("feature_scale_invalid", f"after-fix feature-scale audit failed: {feature_status}")

    before = [row for row in output_steps if row.get("repair_stage") == "before_fix"]
    after = [row for row in output_steps if row.get("repair_stage") == "after_fix"]
    before_delta = [to_float(row.get("delta_norm_1s")) for row in before]
    after_delta = [to_float(row.get("delta_norm_1s")) for row in after]
    before_logvar = [to_float(row.get("logvar_xy_1s")) for row in before]
    after_logvar = [to_float(row.get("logvar_xy_1s")) for row in after]
    repaired_dims = [row for row in norm if str(row.get("std_repaired")) in {"1", "1.0", "True", "true"}]
    inactive_forwarded = sum(int(to_float(row.get("inactive_forwarded_count"), 0.0)) for row in active)
    duplicate_ids = sum(int(to_float(row.get("duplicate_active_ids"), 0.0)) for row in active)
    left_pad_violations = sum(int(to_float(row.get("left_padding_violations"), 0.0)) for row in history)
    z_rows = [row for row in aug if row.get("repair_stage") == "after_fix" and row.get("block") == "z_i_64"]
    obs_rows = [row for row in aug if row.get("repair_stage") == "after_fix" and row.get("block") == "obs_i_12"]

    return {
        "eq_rows": eq,
        "online_status": online_status,
        "feature_status": feature_status,
        "before_delta_mean": safe_mean(before_delta),
        "before_delta_max": finite_max(before_delta),
        "after_delta_mean": safe_mean(after_delta),
        "after_delta_max": finite_max(after_delta),
        "before_logvar_mean": safe_mean(before_logvar),
        "after_logvar_mean": safe_mean(after_logvar),
        "after_logvar_min": float(np.nanmin(after_logvar)) if after_logvar else float("nan"),
        "after_logvar_max": float(np.nanmax(after_logvar)) if after_logvar else float("nan"),
        "repaired_dims": repaired_dims,
        "inactive_forwarded": inactive_forwarded,
        "duplicate_ids": duplicate_ids,
        "left_pad_violations": left_pad_violations,
        "repair": repair,
        "rerun": rerun,
        "z_l2_p95": to_float(z_rows[0].get("l2_norm_p95")) if z_rows else float("nan"),
        "obs_l2_p95": to_float(obs_rows[0].get("l2_norm_p95")) if obs_rows else float("nan"),
        "samples_after": len(after),
    }


def markdown_table(rows: list[dict[str, Any]], cols: list[str], max_rows: int = 10) -> list[str]:
    if not rows:
        return ["No rows."]
    selected = rows[:max_rows]
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in selected:
        values = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                text = str(value)
                values.append(text[:80])
        out.append("| " + " | ".join(values) + " |")
    return out


def write_report(out_dir: Path, summary: dict[str, Any]) -> None:
    n3_text = N3_REPORT.read_text(encoding="utf-8") if N3_REPORT.exists() else ""
    n3_has_anomaly = "mean_delta_norm_1s" in n3_text and "-5.0000" in n3_text
    z_ratio = summary["z_l2_p95"] / max(summary["obs_l2_p95"], 1.0e-9) if np.isfinite(summary["z_l2_p95"]) and np.isfinite(summary["obs_l2_p95"]) else float("nan")
    lines = [
        "# Phase N3.5 Gpsi Wrapper Audit Report",
        "",
        "## Terminal Decision",
        "",
        "`terminal_decision = phase_n3_5_gpsi_wrapper_audit_complete`",
        "",
        "Phase N3.5 complete. The audit found and repaired an engineering bug in the online Gpsi normalization path.",
        "",
        "## Engineering Facts",
        "",
        "- Phase N2 complete flag, Phase N3 artifacts, and `work_dirs/gpsi_heada_v1_nll/best.pth` were present.",
        "- Offline-online equivalence passes after using the same normalization semantics in the N2 eval path and N3 wrapper path.",
        f"- Before repair, online 1s delta norm mean/max were `{summary['before_delta_mean']:.6g}` / `{summary['before_delta_max']:.6g}`.",
        f"- After repair, online 1s delta norm mean/max are `{summary['after_delta_mean']:.6g}` / `{summary['after_delta_max']:.6g}`.",
        f"- After repair, logvar_xy_1s mean/min/max are `{summary['after_logvar_mean']:.6g}` / `{summary['after_logvar_min']:.6g}` / `{summary['after_logvar_max']:.6g}`.",
        f"- Degenerate checkpoint std dimensions repaired: `{len(summary['repaired_dims'])}`.",
        f"- Active-mask audit inactive-forwarded count: `{summary['inactive_forwarded']}`; duplicate active id count: `{summary['duplicate_ids']}`.",
        f"- History left-padding violations: `{summary['left_pad_violations']}`.",
        "",
        "## Confirmed Bug",
        "",
        "N2 data was collected with a hold-position policy, so the first three `ego_current` dimensions, normalized UAV velocity, have checkpoint std near `1e-6`. N3 online PPO supplies nonzero UAV velocity, so the legacy wrapper divided by near-zero std and pushed normalized Gpsi inputs to extreme scale. This reproduced the N3 symptom: huge `delta_hat` and logvar clamped to `-5`.",
        "",
        "Repair action: `GpsiObsWrapper` now floors degenerate checkpoint std dimensions before normalization and exposes raw/normalized inputs plus unclamped logvar diagnostics.",
        "",
        "## Offline-Online Equivalence",
        "",
        *markdown_table(
            [row for row in summary["eq_rows"] if row.get("repair_stage") == "after_fix"],
            ["component", "max_abs_diff", "mean_abs_diff", "rmse_diff", "corr", "allclose_pass"],
            max_rows=8,
        ),
        "",
        "## Online And Feature Scale",
        "",
        *markdown_table(summary["online_status"], ["metric", "value", "status"], max_rows=10),
        "",
        *markdown_table(summary["feature_status"], ["metric", "value", "status"], max_rows=12),
        "",
        "## Feature-Scale Finding",
        "",
        f"`z_i` after-fix p95 L2 norm is `{summary['z_l2_p95']:.6g}` versus base `obs_i(12)` p95 L2 norm `{summary['obs_l2_p95']:.6g}`; ratio `{z_ratio:.6g}`. This is not a stop condition here, but it remains a PPO input-scale risk because N3 v1 did not normalize `z_i`.",
        "",
        "## Artifacts",
        "",
        "- Tables: `results/env_v2_phase_n3_5_gpsi_wrapper_audit/tables/`",
        "- Plots: `results/env_v2_phase_n3_5_gpsi_wrapper_audit/plots/`",
        "- Logs: `results/env_v2_phase_n3_5_gpsi_wrapper_audit/logs/` and `phase_n3_5_watcher.log`",
        "",
        "## Decision",
        "",
        "- N3 original result validity: invalid as a method conclusion. It used the legacy online normalization path that produced severe feature-scale corruption.",
        "- Must rerun N3: yes. At minimum rerun N3-lite or full N3 with repaired wrapper and consider z-block normalization or z ablation.",
        "- Can enter N4: no. N4 shield comparison is blocked until repaired N3/N3-lite establishes a valid no-shield baseline.",
        "- Recommended rerun: repaired Gpsi wrapper, no shield, same PPO backbone, plus an ablation of `z_i` normalization or removing `z_i` while keeping delta/logvar.",
    ]
    if not n3_has_anomaly:
        lines.extend(["", "## Remaining Risk", "", "The N3 report text did not expose the exact expected anomaly string, so the invalidity conclusion is based on N3.5 reproduced before-fix scale and repaired after-fix scale."])
    write_text(out_dir / "PHASE_N3_5_GPSI_WRAPPER_AUDIT_REPORT.md", "\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    out_dir = ROOT / args.out_dir
    checkpoint = ROOT / args.checkpoint
    ensure_dirs(out_dir)
    try:
        check_prerequisites(out_dir, checkpoint)
        require_artifacts(out_dir)
        summary = summarize(out_dir)
        manifest_path = out_dir / "tables/phase_n3_5_command_manifest.csv"
        manifest = read_rows(manifest_path)
        manifest.append(command_manifest_row("finalize_n3_5_audit", " ".join(sys.argv), "completed"))
        write_csv(manifest_path, manifest, ["command_name", "command", "status"])
        write_report(out_dir, summary)
        write_text(out_dir / COMPLETE_FLAG, "phase_n3_5_gpsi_wrapper_audit_complete\n")
        write_text(out_dir / "phase_n3_5_status.txt", "complete\n")
        return 0
    except PhaseN35Stop as exc:
        write_stop(out_dir, exc.reason, exc.detail)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
