from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_p2_rich_motion import (
    CHECKPOINT_STEPS,
    OUT_DIR,
    PARETO_SCENARIOS,
    PLOTS_DIR,
    STAGE2_METHODS,
    STAGE2_SCENARIOS,
    STAGE4_METHODS,
    build_risk_adaptation_summary,
    build_seed0_main,
    csv_row_count,
    ensure_dirs,
    fmt,
    plot_seed0_pareto,
    run_stage2_seed0,
    run_stage4,
    stage3_gate,
    write_lines,
    write_stage3_report,
)

COMPLETE_FLAG = OUT_DIR / "P2_STAGE2_TO_FINAL_COMPLETE.flag"
NO_GO_FLAG = OUT_DIR / "P2_STAGE2_TO_FINAL_NO_GO.flag"
PATCHED_STAGE1_FLAG = OUT_DIR / "P2_PATCHED_STAGE1_COMPLETE.flag"
KEY_SCENARIOS = [
    "eval_mixed_v2",
    "eval_sinusoidal",
    "eval_accel_decel",
    "eval_ar1",
    "eval_threat_validated_sudden",
    "eval_random_switch_hard",
]


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{ts()}] {message}", flush=True)


def mean_numeric(group: pd.DataFrame, column: str) -> float:
    if group.empty or column not in group:
        return float("nan")
    values = pd.to_numeric(group[column], errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.mean())


def method_mean(df: pd.DataFrame, scenario: str, method: str) -> pd.Series:
    rows = df[
        (df["scenario"] == scenario)
        & (df["method"] == method)
        & (df["step"].isin([750000, 1000000]))
    ]
    if rows.empty:
        return pd.Series(dtype=float)
    return rows.mean(numeric_only=True)


def safety_issue(series: pd.Series) -> bool:
    if series.empty:
        return False
    reaction = float(series.get("reaction_time_eval_style", 0.0))
    if math.isnan(reaction):
        reaction = 0.0
    return bool(
        float(series.get("near_miss_rate", 0.0)) >= 0.10
        or float(series.get("collision_rate", 0.0)) >= 0.05
        or reaction > 1.0
    )


def risk_safety_close(risk: pd.Series, wide: pd.Series, scenario: str) -> bool:
    if risk.empty or wide.empty:
        return False
    close = (
        float(risk["collision_rate"]) <= float(wide["collision_rate"]) + 0.05
        and float(risk["near_miss_rate"]) <= float(wide["near_miss_rate"]) + 0.10
        and float(risk["mean_min_distance"]) >= float(wide["mean_min_distance"]) - 0.25
    )
    if scenario in {"eval_sudden_turn", "eval_mixed_v2", "eval_threat_validated_sudden"}:
        r_react = float(risk.get("reaction_time_eval_style", np.nan))
        w_react = float(wide.get("reaction_time_eval_style", np.nan))
        if not math.isnan(r_react) and not math.isnan(w_react):
            close = close and r_react <= w_react + 0.50
    return bool(close)


def risk_faster_or_safer(risk: pd.Series, wide: pd.Series) -> tuple[bool, bool]:
    if risk.empty or wide.empty:
        return False, False
    faster = float(risk["mean_time"]) <= float(wide["mean_time"]) - 0.25
    safer = (
        float(risk["collision_rate"]) < float(wide["collision_rate"]) - 0.02
        or float(risk["near_miss_rate"]) < float(wide["near_miss_rate"]) - 0.05
        or float(risk["mean_min_distance"]) > float(wide["mean_min_distance"]) + 0.25
    )
    return bool(faster), bool(safer)


def build_stage3_answers(seed0_df: pd.DataFrame, adaptation: pd.DataFrame, gate_df: pd.DataFrame) -> dict[str, Any]:
    base_issues: list[str] = []
    close_safety: list[str] = []
    efficient_scenarios: list[str] = []
    pareto_scenarios: list[str] = []
    single_mode: dict[str, dict[str, Any]] = {}
    mixed_v2_stability = "insufficient data"

    for scenario in PARETO_SCENARIOS:
        base = method_mean(seed0_df, scenario, "attention_full")
        risk = method_mean(seed0_df, scenario, "attention_full_risk_penalty")
        wide = method_mean(seed0_df, scenario, "attention_full_distance_penalty_wide_d2")
        if safety_issue(base):
            base_issues.append(scenario)
        close = risk_safety_close(risk, wide, scenario)
        faster, safer = risk_faster_or_safer(risk, wide)
        if close:
            close_safety.append(scenario)
        if close and faster:
            efficient_scenarios.append(scenario)
        if close and (faster or safer):
            pareto_scenarios.append(scenario)
        if scenario in {"eval_sinusoidal", "eval_accel_decel", "eval_ar1"}:
            single_mode[scenario] = {
                "baseline_safety_issue": safety_issue(base),
                "risk_safety_close_to_wide": close,
                "risk_faster_than_wide": faster,
                "risk_pareto_positive": close and (faster or safer),
                "risk_mean_time": float(risk.get("mean_time", np.nan)) if not risk.empty else np.nan,
                "wide_mean_time": float(wide.get("mean_time", np.nan)) if not wide.empty else np.nan,
                "risk_near_miss": float(risk.get("near_miss_rate", np.nan)) if not risk.empty else np.nan,
                "wide_near_miss": float(wide.get("near_miss_rate", np.nan)) if not wide.empty else np.nan,
                "risk_collision": float(risk.get("collision_rate", np.nan)) if not risk.empty else np.nan,
                "wide_collision": float(wide.get("collision_rate", np.nan)) if not wide.empty else np.nan,
            }

    risk_mixed = method_mean(seed0_df, "eval_mixed_v2", "attention_full_risk_penalty")
    wide_mixed = method_mean(seed0_df, "eval_mixed_v2", "attention_full_distance_penalty_wide_d2")
    if not risk_mixed.empty and not wide_mixed.empty:
        risk_bad = float(risk_mixed["collision_rate"]) + float(risk_mixed["near_miss_rate"])
        wide_bad = float(wide_mixed["collision_rate"]) + float(wide_mixed["near_miss_rate"])
        if risk_bad < wide_bad - 0.05:
            mixed_v2_stability = "risk_penalty more stable"
        elif wide_bad < risk_bad - 0.05:
            mixed_v2_stability = "wide_d2 more stable"
        else:
            mixed_v2_stability = "similar safety; compare mean_time"

    risk_adapt = adaptation[adaptation["method"] == "attention_full_risk_penalty"].copy()
    adaptation_signal = False
    if not risk_adapt.empty:
        risk_range = float(risk_adapt["risk_sum_mean"].max() - risk_adapt["risk_sum_mean"].min())
        min_dist_range = float(risk_adapt["mean_min_distance"].max() - risk_adapt["mean_min_distance"].min())
        time_range = float(risk_adapt["mean_time"].max() - risk_adapt["mean_time"].min())
        adaptation_signal = bool(risk_range > 0.01 and (min_dist_range > 0.10 or time_range > 0.25))

    return {
        "baseline_safety_reaction_drift_scenarios": base_issues,
        "risk_safety_close_scenarios": close_safety,
        "risk_more_efficient_scenarios": efficient_scenarios,
        "risk_pareto_scenarios": pareto_scenarios,
        "risk_on_pareto_front": bool(pareto_scenarios),
        "single_mode": single_mode,
        "mixed_v2_stability": mixed_v2_stability,
        "risk_motion_mode_adaptation_signal": adaptation_signal,
        "stage4_worth_running": len([s for s in pareto_scenarios if s in KEY_SCENARIOS]) >= 2,
        "gate_table_rows": int(len(gate_df)),
    }


def decide_no_go(seed0_df: pd.DataFrame, gate_df: pd.DataFrame, stage3_go: bool) -> tuple[str, str]:
    if stage3_go:
        return "", ""

    use = seed0_df[seed0_df["step"].isin([750000, 1000000]) & seed0_df["scenario"].isin(KEY_SCENARIOS)]
    if use.empty:
        return "stage3_no_go_environment_no_discrimination", "missing key-scenario rows for Stage 3 gate"

    if use["success_rate"].max() < 0.50 and use["collision_rate"].min() > 0.50:
        return "stage3_no_go_all_methods_fail", "all methods fail in key scenarios"
    if use["success_rate"].min() > 0.95 and use["near_miss_rate"].max() < 0.05 and use["collision_rate"].max() < 0.02:
        return "stage3_no_go_environment_no_discrimination", "all methods are near-perfect in key scenarios"

    base_issues = []
    risk_positive = []
    wide_dominates = []
    near_identical = []
    risk_not_close = []
    for scenario in KEY_SCENARIOS:
        base = method_mean(seed0_df, scenario, "attention_full")
        risk = method_mean(seed0_df, scenario, "attention_full_risk_penalty")
        wide = method_mean(seed0_df, scenario, "attention_full_distance_penalty_wide_d2")
        if not base.empty and safety_issue(base):
            base_issues.append(scenario)
        if risk.empty or wide.empty:
            continue
        close = risk_safety_close(risk, wide, scenario)
        faster, safer = risk_faster_or_safer(risk, wide)
        if close and (faster or safer):
            risk_positive.append(scenario)
        if (
            float(wide["collision_rate"]) <= float(risk["collision_rate"]) + 0.02
            and float(wide["near_miss_rate"]) <= float(risk["near_miss_rate"]) + 0.05
            and float(wide["mean_time"]) <= float(risk["mean_time"]) + 0.25
            and float(wide["mean_min_distance"]) >= float(risk["mean_min_distance"]) - 0.10
        ):
            wide_dominates.append(scenario)
        if (
            abs(float(wide["collision_rate"]) - float(risk["collision_rate"])) <= 0.02
            and abs(float(wide["near_miss_rate"]) - float(risk["near_miss_rate"])) <= 0.05
            and abs(float(wide["mean_time"]) - float(risk["mean_time"])) <= 0.25
            and abs(float(wide["mean_min_distance"]) - float(risk["mean_min_distance"])) <= 0.10
        ):
            near_identical.append(scenario)
        if not close:
            risk_not_close.append(scenario)

    if not base_issues:
        return "stage3_no_go_baseline_no_safety_issue", "baseline no longer shows safety/reaction drift in key scenarios"
    if len(wide_dominates) >= max(2, len(KEY_SCENARIOS) // 2):
        return "stage3_no_go_wide_d2_dominates", f"wide_d2 dominates or matches risk in {len(wide_dominates)} key scenarios"
    if len(near_identical) >= max(2, len(KEY_SCENARIOS) // 2):
        return "stage3_no_go_risk_wide_d2_no_difference", f"risk and wide_d2 are nearly identical in {len(near_identical)} key scenarios"
    if len(risk_positive) < 2:
        if len(risk_not_close) >= 2:
            return "stage3_no_go_risk_not_pareto", f"risk is not safety-close to wide_d2 in {len(risk_not_close)} key scenarios"
        return "stage3_no_go_risk_not_pareto", f"risk Pareto-positive key scenarios={len(risk_positive)} < 2"

    return "stage3_no_go_risk_not_pareto", "Stage 3 gate threshold not met"


def no_go_recommendation(decision: str) -> str:
    if decision in {"stage3_no_go_wide_d2_dominates", "stage3_no_go_risk_wide_d2_no_difference"}:
        return "downgrade risk as the primary empirical claim and pivot to safety-margin cost design principles."
    if decision in {"stage3_no_go_all_methods_fail", "stage3_no_go_environment_no_discrimination", "stage3_no_go_baseline_no_safety_issue"}:
        return "do not make a method claim; revise the environment/discrimination setup before more seeds."
    return "continue risk only as a secondary hypothesis; primary next step should be safety-margin cost design principles."


def write_final_report(
    terminal_decision: str,
    completed_stage: str,
    stage4_triggered: bool,
    seed0_df: pd.DataFrame,
    adaptation: pd.DataFrame,
    gate_df: pd.DataFrame,
    answers: dict[str, Any],
    no_go_reason: str = "",
) -> None:
    lines = [
        "# P2 Rich Motion Generalization Report",
        "",
        "## 1. Motivation",
        "P2 tests whether risk_penalty retains a safety-efficiency Pareto advantage over distance_penalty_wide_d2 under richer obstacle motion.",
        "",
        "## 2. New Motion Modes",
        "- Stage 0/1 were already completed before this run; this report continues from patched Stage 1.",
        "- Rich-motion evidence uses train_mixed_modes_v2 and eval_sinusoidal / eval_accel_decel / eval_ar1 / eval_mixed_v2 / eval_threat_validated_sudden.",
        "",
        "## 3. Environment Sanity Check",
        "- Patched Stage 0 sanity passed before Stage 2. See P2_ENVIRONMENT_SANITY_REPORT_PATCHED.md.",
        "",
        "## 4. Existing Checkpoint OOD Evaluation",
        "- Patched Stage 1 Existing Checkpoint OOD Evaluation passed before Stage 2.",
        "- See P2_STAGE1_OOD_EVAL_REPORT.md and results/p2_rich_motion/p2_stage1_ood_eval.csv.",
        "",
        "## 5. Rich-Motion Training Seed0",
        f"- Completed stage: {completed_stage}.",
        "- Trained attention_full, attention_full_distance_penalty_wide_d2, and attention_full_risk_penalty on train_mixed_modes_v2 with seed=0.",
        "- Evaluated checkpoints 250k / 500k / 750k / 1000k over all Stage 2 scenarios.",
        "",
        "## 6. New Single-Mode Scenario Analysis",
        "| scenario | baseline_issue | risk_close_to_wide | risk_faster | risk_pareto | risk_time | wide_time | risk_near | wide_near |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario in ["eval_sinusoidal", "eval_accel_decel", "eval_ar1"]:
        row = answers["single_mode"].get(scenario, {})
        lines.append(
            f"| {scenario} | {int(bool(row.get('baseline_safety_issue', False)))} | "
            f"{int(bool(row.get('risk_safety_close_to_wide', False)))} | {int(bool(row.get('risk_faster_than_wide', False)))} | "
            f"{int(bool(row.get('risk_pareto_positive', False)))} | {fmt(row.get('risk_mean_time', np.nan))} | "
            f"{fmt(row.get('wide_mean_time', np.nan))} | {fmt(row.get('risk_near_miss', np.nan))} | {fmt(row.get('wide_near_miss', np.nan))} |"
        )
    lines += [
        "",
        "## 7. Risk vs Wide Distance Pareto",
        f"- Risk safety-close scenarios: {', '.join(answers['risk_safety_close_scenarios']) if answers['risk_safety_close_scenarios'] else 'none'}.",
        f"- Risk more-efficient scenarios: {', '.join(answers['risk_more_efficient_scenarios']) if answers['risk_more_efficient_scenarios'] else 'none'}.",
        f"- Risk Pareto-positive scenarios: {', '.join(answers['risk_pareto_scenarios']) if answers['risk_pareto_scenarios'] else 'none'}.",
        "",
        "## 8. Risk Adaptation Analysis",
        f"- Motion-mode adaptation signal: {answers['risk_motion_mode_adaptation_signal']}.",
        "- See results/p2_rich_motion/p2_risk_adaptation_summary.csv for per-scenario risk_sum / risk_max / distance cost / min distance / mean_time.",
        "",
        "## 9. Failure Cases",
        f"- mixed_v2 stability: {answers['mixed_v2_stability']}.",
        "",
        "## 10. Go/No-Go for Three Seeds",
        f"- Stage 4 triggered: {stage4_triggered}.",
        f"- Terminal decision: {terminal_decision}.",
        f"- No-go reason: {no_go_reason if no_go_reason else 'none'}.",
        "",
        "## 11. Final P2 Decision",
        f"- risk_penalty mainline value retained: {stage4_triggered or terminal_decision == 'stage4_complete'}.",
        f"- safety-margin cost design pivot: {terminal_decision != 'stage4_complete'}.",
        f"- Recommendation: {no_go_recommendation(terminal_decision) if terminal_decision != 'stage4_complete' else 'keep risk_penalty as a live Pareto-efficiency candidate and use Stage 4 confirmation for robustness.'}",
        "",
        "## Required Stage 3 Answers",
        f"1. train_mixed_modes_v2 baseline safety/reaction drift: {', '.join(answers['baseline_safety_reaction_drift_scenarios']) if answers['baseline_safety_reaction_drift_scenarios'] else 'not evident in key scenarios'}.",
        f"2. risk_penalty safety close to wide_d2: {bool(answers['risk_safety_close_scenarios'])}; scenarios={', '.join(answers['risk_safety_close_scenarios']) if answers['risk_safety_close_scenarios'] else 'none'}.",
        f"3. risk_penalty more efficient than wide_d2: {bool(answers['risk_more_efficient_scenarios'])}; scenarios={', '.join(answers['risk_more_efficient_scenarios']) if answers['risk_more_efficient_scenarios'] else 'none'}.",
        f"4. risk on Pareto front: {answers['risk_on_pareto_front']}.",
        "5. sinusoidal / accel_decel / ar1: see Section 6 table.",
        f"6. mixed_v2 stability: {answers['mixed_v2_stability']}.",
        f"7. motion-mode adaptation: {answers['risk_motion_mode_adaptation_signal']}.",
        f"8. worth Stage 4 three-seed confirmation: {answers['stage4_worth_running']}.",
        "",
        "## Key Artifacts",
        "- P2_STAGE2_RICH_TRAINING_SEED0_REPORT.md",
        "- P2_STAGE3_SEED0_PARETO_REPORT.md",
        "- P2_RICH_MOTION_FINAL_REPORT.md",
        "- results/p2_rich_motion/p2_seed0_by_step_scenario.csv",
        "- results/p2_rich_motion/p2_seed0_main_750k_1000k_table.csv",
        "- results/p2_rich_motion/p2_risk_adaptation_summary.csv",
        "- results/p2_rich_motion/plots/seed0_pareto_*.png",
    ]
    if stage4_triggered or terminal_decision == "stage4_complete":
        lines += [
            "- P2_THREE_SEED_CONFIRMATION_REPORT.md",
            "- results/p2_rich_motion/p2_three_seed_summary.csv",
        ]
    write_lines(ROOT / "P2_RICH_MOTION_FINAL_REPORT.md", lines)


def required_artifacts(stage4: bool) -> tuple[bool, list[str]]:
    missing: list[str] = []
    files = [
        ROOT / "P2_STAGE2_RICH_TRAINING_SEED0_REPORT.md",
        ROOT / "P2_RICH_MOTION_FINAL_REPORT.md",
        OUT_DIR / "p2_seed0_by_step_scenario.csv",
        OUT_DIR / "p2_seed0_main_750k_1000k_table.csv",
        OUT_DIR / "p2_risk_adaptation_summary.csv",
    ]
    for path in files:
        if not path.exists() or path.stat().st_size == 0:
            missing.append(str(path.relative_to(ROOT)))
    if csv_row_count(OUT_DIR / "p2_seed0_by_step_scenario.csv") != len(STAGE2_METHODS) * len(CHECKPOINT_STEPS) * len(STAGE2_SCENARIOS):
        missing.append("results/p2_rich_motion/p2_seed0_by_step_scenario.csv:unexpected_row_count")
    if csv_row_count(OUT_DIR / "p2_seed0_main_750k_1000k_table.csv") != len(STAGE2_METHODS) * 2 * len(STAGE2_SCENARIOS):
        missing.append("results/p2_rich_motion/p2_seed0_main_750k_1000k_table.csv:unexpected_row_count")
    if csv_row_count(OUT_DIR / "p2_risk_adaptation_summary.csv") != len(STAGE2_METHODS) * len(STAGE2_SCENARIOS):
        missing.append("results/p2_rich_motion/p2_risk_adaptation_summary.csv:unexpected_row_count")
    if len(list(PLOTS_DIR.glob("seed0_pareto_*.png"))) < len(PARETO_SCENARIOS) * 5:
        missing.append("results/p2_rich_motion/plots/seed0_pareto_*.png:too_few")
    if stage4:
        for path in [ROOT / "P2_THREE_SEED_CONFIRMATION_REPORT.md", OUT_DIR / "p2_three_seed_summary.csv"]:
            if not path.exists() or path.stat().st_size == 0:
                missing.append(str(path.relative_to(ROOT)))
        expected_rows = len(STAGE4_METHODS) * 3 * len(CHECKPOINT_STEPS) * len(STAGE2_SCENARIOS)
        if csv_row_count(OUT_DIR / "p2_three_seed_summary.csv") != expected_rows:
            missing.append("results/p2_rich_motion/p2_three_seed_summary.csv:unexpected_row_count")
    return not missing, missing


def write_terminal_flag(path: Path, terminal_decision: str, stage4: bool, extra: dict[str, Any]) -> None:
    ok, missing = required_artifacts(stage4=stage4)
    if not ok:
        raise RuntimeError(f"terminal artifacts incomplete: {missing}")
    payload = {
        "completed_at": ts(),
        "terminal_decision": terminal_decision,
        "completed_stage": "Stage 4" if stage4 else "Stage 3",
        "stage4_triggered": bool(stage4),
        "report": "P2_RICH_MOTION_FINAL_REPORT.md",
        **extra,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"terminal flag written path={path.relative_to(ROOT)} terminal_decision={terminal_decision}")


def verify_prereqs() -> None:
    if not PATCHED_STAGE1_FLAG.exists():
        raise FileNotFoundError(f"missing patched Stage 1 flag: {PATCHED_STAGE1_FLAG}")
    flag = json.loads(PATCHED_STAGE1_FLAG.read_text(encoding="utf-8"))
    if flag.get("terminal_decision") != "patched_stage1_complete":
        raise RuntimeError(f"patched Stage 1 not complete: {flag}")
    stage1_csv = OUT_DIR / "p2_stage1_ood_eval.csv"
    if csv_row_count(stage1_csv) != 40:
        raise RuntimeError(f"patched Stage 1 CSV incomplete: {stage1_csv}")


def main() -> None:
    ensure_dirs()
    verify_prereqs()
    n_envs = 16

    log("[Stage 2] richer-motion training seed=0 entering")
    seed0_df = run_stage2_seed0(n_envs=n_envs)
    log("[Stage 2 Eval] all checkpoints/scenarios evaluated")

    build_seed0_main(seed0_df)
    adaptation = build_risk_adaptation_summary(seed0_df)
    plot_seed0_pareto(seed0_df)
    raw_stage3_go, stage3_reasons, gate_df = stage3_gate(seed0_df)
    answers = build_stage3_answers(seed0_df, adaptation, gate_df)
    stage3_go = bool(raw_stage3_go and answers["stage4_worth_running"])
    if raw_stage3_go and not stage3_go:
        stage3_reasons = [*stage3_reasons, "key_scenario_stage4_threshold_not_met"]
    write_stage3_report(seed0_df, adaptation, gate_df, stage3_go, stage3_reasons)
    log(f"[Stage 3] Pareto analysis complete go={stage3_go} reasons={stage3_reasons}")

    if not stage3_go:
        terminal_decision, reason = decide_no_go(seed0_df, gate_df, stage3_go)
        write_final_report(
            terminal_decision=terminal_decision,
            completed_stage="Stage 3 no-go",
            stage4_triggered=False,
            seed0_df=seed0_df,
            adaptation=adaptation,
            gate_df=gate_df,
            answers=answers,
            no_go_reason=reason,
        )
        write_terminal_flag(
            NO_GO_FLAG,
            terminal_decision,
            stage4=False,
            extra={
                "no_go_reason": reason,
                "seed0_rows": len(seed0_df),
                "stage3_gate_rows": len(gate_df),
                "risk_pareto_scenarios": answers["risk_pareto_scenarios"],
            },
        )
        log(f"NO-GO triggered: {reason}")
        log(f"terminal_decision = {terminal_decision}")
        return

    log("[Stage 4] go decision reached; training wide_d2/risk seed=1,2")
    three_seed_df = run_stage4(n_envs=n_envs, seed0_df=seed0_df)
    write_final_report(
        terminal_decision="stage4_complete",
        completed_stage="Stage 4 complete",
        stage4_triggered=True,
        seed0_df=seed0_df,
        adaptation=adaptation,
        gate_df=gate_df,
        answers=answers,
    )
    write_terminal_flag(
        COMPLETE_FLAG,
        "stage4_complete",
        stage4=True,
        extra={
            "seed0_rows": len(seed0_df),
            "stage3_gate_rows": len(gate_df),
            "three_seed_rows": len(three_seed_df),
            "risk_pareto_scenarios": answers["risk_pareto_scenarios"],
        },
    )
    log("COMPLETE P2 Stage2-to-final artifacts verified")
    log("terminal_decision = stage4_complete")


if __name__ == "__main__":
    main()
