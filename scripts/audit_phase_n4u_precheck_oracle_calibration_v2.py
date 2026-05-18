from __future__ import annotations

import argparse
import json
import subprocess
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results/env_v2_phase_n4u_precheck_oracle_calibration_v2"
PREFIX = "phase_n4u_precheck_v2"
REPORT_NAME = "PHASE_N4U_PRECHECK_ORACLE_CALIBRATION_V2_REPORT.md"
STATUS_NAME = "phase_n4u_precheck_oracle_calibration_v2_status.txt"
STOP_FLAG = "STOP_PREMISE_INVALIDATED.flag"
TERMINAL = "phase_n4u_precheck_v2_stopped_premise_invalidated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase N4U precheck v2 CPA/TTC premise audit.")
    parser.add_argument("--result-dir", default=str(RESULT_DIR.relative_to(ROOT)))
    parser.add_argument("--github-sync-commit", default="")
    parser.add_argument("--github-sync-status", default="unknown")
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def line_no(path: Path, needle: str) -> int:
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if needle in line:
            return idx
    return -1


def git_short_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return ""


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def phase_b_anchor() -> dict[str, Any]:
    manifest_path = ROOT / "results/env_v2_phase_b_geometry_filter_baselines/tables/phase_b_baseline_manifest.csv"
    summary_path = ROOT / "results/env_v2_phase_b_geometry_filter_baselines/tables/phase_b_eval_summary.csv"
    manifest = read_csv(manifest_path)
    summary = read_csv(summary_path)
    row = manifest[manifest.get("config_name", pd.Series(dtype=str)).astype(str) == "vo_like_filter_h45_cpa1p2_h16"]
    metrics = summary[
        (summary.get("stage", pd.Series(dtype=str)).astype(str) == "b2_formal")
        & (summary.get("config_name", pd.Series(dtype=str)).astype(str) == "vo_like_filter_h45_cpa1p2_h16")
    ]
    params: dict[str, Any] = {}
    if not row.empty:
        params = json.loads(str(row.iloc[0].get("config_params", "{}")))
    out = {
        "strong_geom_anchor": "Phase B vo_like_filter_h45_cpa1p2_h16",
        "manifest_path": rel(manifest_path),
        "summary_path": rel(summary_path),
        "manifest_exists": int(not row.empty),
        "formal_summary_exists": int(not metrics.empty),
        "baseline_name": str(row.iloc[0].get("baseline_name", "")) if not row.empty else "",
        "kind": str(row.iloc[0].get("kind", "")) if not row.empty else "",
        "filter_used": int(row.iloc[0].get("filter_used", 0)) if not row.empty else 0,
        "horizon": float(params.get("horizon", float("nan"))),
        "cpa_safe": float(params.get("cpa_safe", float("nan"))),
        "num_headings": int(params.get("num_headings", -1)) if params else -1,
        "candidate_velocity_set": "raw_action; goal_action; current_action; away_lateral_action; headings at speeds 0.4/0.7/1.0 over num_headings=16",
        "candidate_count_nominal": 4 + 3 * int(params.get("num_headings", 16)),
        "scoring_logic": "-1.2*||cand-raw|| + 0.8*progress_alignment + 0.35*min_cpa - 0.2*||cand-current_action||",
        "unsafe_action_handling": "trigger if runtime analytic candidate CPA has 0<=tcpa<=horizon and min_cpa<cpa_safe; select best safe candidate if any, else best_any",
    }
    if not metrics.empty:
        m = metrics.iloc[0]
        out.update(
            {
                "formal_success": float(m.get("success_rate", float("nan"))),
                "formal_collision": float(m.get("collision_rate", float("nan"))),
                "formal_near_miss": float(m.get("near_miss_rate", float("nan"))),
                "formal_progress": float(m.get("progress_mean", float("nan"))),
                "formal_filter_trigger_rate": float(m.get("filter_trigger_rate", float("nan"))),
            }
        )
    return out


def build_tables(result_dir: Path, args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    tables = result_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    env_path = ROOT / "envs/dynamic_obstacle_flow_env.py"
    phase_b_path = ROOT / "scripts/run_env_v2_phase_b_geometry_filter_baselines.py"
    n4o_path = ROOT / "scripts/eval_env_v2_phase_n4o_ordinary_shield.py"

    github = pd.DataFrame(
        [
            {
                "github_sync_status": args.github_sync_status,
                "github_sync_commit": args.github_sync_commit or git_short_head(),
                "current_head": git_short_head(),
            }
        ]
    )
    github.to_csv(tables / f"{PREFIX}_github_sync.csv", index=False)

    source_rows = [
        {
            "component": "EnvV2 observation obs_i planned_cpa",
            "file": rel(env_path),
            "line": line_no(env_path, "obstacle.planned_cpa / 5.0"),
            "formula_or_source": "obs_i[6] = obstacle.planned_cpa / 5.0",
            "uses_constant_velocity_analytic_runtime_cpa_ttc": 0,
            "uses_ground_truth_future_trajectory": 0,
            "depends_on_current_uav_velocity": 0,
            "depends_on_candidate_velocity": 0,
            "units_and_normalization": "meters, normalized by /5.0 in obs_i",
            "invalid_handling": "stored per active obstacle; inactive obs rows zero-padded by mask",
            "audit_conclusion": "spawn-time planned design quantity, not runtime analytic CPA",
        },
        {
            "component": "EnvV2 observation obs_i planned_ttc",
            "file": rel(env_path),
            "line": line_no(env_path, "obstacle.planned_ttc / 20.0"),
            "formula_or_source": "obs_i[7] = obstacle.planned_ttc / 20.0",
            "uses_constant_velocity_analytic_runtime_cpa_ttc": 0,
            "uses_ground_truth_future_trajectory": 0,
            "depends_on_current_uav_velocity": 0,
            "depends_on_candidate_velocity": 0,
            "units_and_normalization": "seconds, normalized by /20.0 in obs_i",
            "invalid_handling": "info also reports remaining planned_ttc clipped at >=0",
            "audit_conclusion": "spawn-time planned design quantity, not runtime analytic TTC",
        },
        {
            "component": "EnvV2 obstacle spawn planned_cpa",
            "file": rel(env_path),
            "line": line_no(env_path, "planned_cpa = float(self.np_random.uniform(cpa_low, cpa_high))"),
            "formula_or_source": "planned_cpa sampled from CPA_RANGES[threat_class]",
            "uses_constant_velocity_analytic_runtime_cpa_ttc": 0,
            "uses_ground_truth_future_trajectory": 0,
            "depends_on_current_uav_velocity": 0,
            "depends_on_candidate_velocity": 0,
            "units_and_normalization": "meters before obs normalization",
            "invalid_handling": "_planned_threat_valid checks class range and finite value",
            "audit_conclusion": "not computed from current relative position/velocity",
        },
        {
            "component": "EnvV2 obstacle spawn planned_ttc",
            "file": rel(env_path),
            "line": line_no(env_path, "planned_ttc = float(self.np_random.uniform(ttc_low, ttc_high))"),
            "formula_or_source": "planned_ttc sampled from scenario ttc_range, then may be adjusted by target_s/path progress",
            "uses_constant_velocity_analytic_runtime_cpa_ttc": 0,
            "uses_ground_truth_future_trajectory": 0,
            "depends_on_current_uav_velocity": 0,
            "depends_on_candidate_velocity": 0,
            "units_and_normalization": "seconds before obs normalization",
            "invalid_handling": "_planned_threat_valid requires finite and 0.5<=planned_ttc<=20",
            "audit_conclusion": "not computed as tcpa=-dot(rel,rel_vel)/||rel_vel||^2",
        },
        {
            "component": "Phase B runtime CPA/TTC",
            "file": rel(phase_b_path),
            "line": line_no(phase_b_path, "tcpa = float(np.clip(-np.dot(rel, rel_vel) / rel_speed_sq, 0.0, horizon))"),
            "formula_or_source": "tcpa=clip(-dot(rel,rel_vel)/||rel_vel||^2,0,horizon); cpa=||rel+rel_vel*tcpa||",
            "uses_constant_velocity_analytic_runtime_cpa_ttc": 1,
            "uses_ground_truth_future_trajectory": 0,
            "depends_on_current_uav_velocity": 1,
            "depends_on_candidate_velocity": 1,
            "units_and_normalization": "meters and seconds, unnormalized",
            "invalid_handling": "if no obstacles -> NaN; if rel_speed_sq<=1e-8 -> tcpa=0, cpa=current distance",
            "audit_conclusion": "independent runtime analytic CPA/TTC recomputation for candidate actions",
        },
        {
            "component": "N4-O ordinary VO shield",
            "file": rel(n4o_path),
            "line": line_no(n4o_path, "return phase_b.BaselineConfig("),
            "formula_or_source": "loads Phase B vo_like_filter_h45_cpa1p2_h16 manifest and calls Phase B act/filter path",
            "uses_constant_velocity_analytic_runtime_cpa_ttc": 1,
            "uses_ground_truth_future_trajectory": 0,
            "depends_on_current_uav_velocity": 1,
            "depends_on_candidate_velocity": 1,
            "units_and_normalization": "same Phase B unnormalized meters/seconds",
            "invalid_handling": "inherits Phase B logic",
            "audit_conclusion": "parameter-equivalent ordinary VO wrapper around Phase B implementation",
        },
    ]
    source = pd.DataFrame(source_rows)
    source.to_csv(tables / f"{PREFIX}_cpa_ttc_source_audit.csv", index=False)

    anchor = pd.DataFrame([phase_b_anchor()])
    anchor.to_csv(tables / f"{PREFIX}_strong_geom_anchor_manifest.csv", index=False)

    n4o_manifest_path = ROOT / "results/env_v2_phase_n4o_ordinary_shield_fair_comparison/tables/phase_n4o_shield_config_manifest.csv"
    n4o_manifest = read_csv(n4o_manifest_path)
    n4o_ok = False
    if not n4o_manifest.empty:
        params = {str(v) for v in n4o_manifest.get("shield_params", pd.Series(dtype=str)).astype(str).unique()}
        n4o_ok = (
            len(params) == 1
            and '"horizon":4.5' in next(iter(params))
            and '"cpa_safe":1.2' in next(iter(params))
            and '"num_headings":16' in next(iter(params))
            and int(pd.to_numeric(n4o_manifest["ordinary_shield_uses_sigma2"], errors="coerce").max()) == 0
            and int(pd.to_numeric(n4o_manifest["ordinary_shield_uses_future_truth"], errors="coerce").max()) == 0
        )
    equivalence = pd.DataFrame(
        [
            {
                "phase_b_config": "vo_like_filter_h45_cpa1p2_h16",
                "n4o_manifest_path": rel(n4o_manifest_path),
                "n4o_manifest_exists": int(n4o_manifest_path.exists()),
                "n4o_same_params_for_attention_noz_p3": int(n4o_manifest.get("same_params_for_attention_noz_p3", pd.Series([0])).astype(int).min()) if not n4o_manifest.empty else 0,
                "n4o_ordinary_shield_uses_sigma2": int(pd.to_numeric(n4o_manifest.get("ordinary_shield_uses_sigma2", pd.Series([1])), errors="coerce").max()) if not n4o_manifest.empty else 1,
                "n4o_ordinary_shield_uses_future_truth": int(pd.to_numeric(n4o_manifest.get("ordinary_shield_uses_future_truth", pd.Series([1])), errors="coerce").max()) if not n4o_manifest.empty else 1,
                "parameter_equivalent_to_phase_b_anchor": int(n4o_ok),
                "formula_equivalent_to_phase_b_anchor": 1,
                "equivalence_conclusion": "N4-O ordinary VO loads Phase B manifest params and uses Phase B vo_like_filter logic; EnvV2 obs planned CPA/TTC are not the same quantity.",
            }
        ]
    )
    equivalence.to_csv(tables / f"{PREFIX}_phase_b_n4o_equivalence.csv", index=False)

    premise = pd.DataFrame(
        [
            {
                "gate": "clean_env_constant_velocity_correction_route",
                "envv2_obs_planned_cpa_ttc_is_constant_velocity_runtime_analytic": 0,
                "envv2_obs_planned_cpa_ttc_is_ground_truth_future_oracle": 0,
                "shield_cpa_ttc_is_constant_velocity_runtime_analytic": 1,
                "phase_b_n4o_formula_compatible": int(n4o_ok),
                "premise_valid": 0,
                "stop_flag": STOP_FLAG,
                "terminal_decision": TERMINAL,
                "reason": "EnvV2 obs planned_cpa/planned_ttc are spawn-time planned design quantities, while Phase B/N4-O shields independently recompute runtime constant-velocity CPA/TTC for candidate actions. Therefore the framing 'Gpsi corrects EnvV2 planned CPA/TTC constant-velocity analytic quantity' is invalid for clean EnvV2.",
            }
        ]
    )
    premise.to_csv(tables / f"{PREFIX}_premise_gate.csv", index=False)
    return {
        "github": github,
        "source": source,
        "anchor": anchor,
        "equivalence": equivalence,
        "premise": premise,
    }


def md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "No rows.\n"
    use = df.head(max_rows).copy()
    cols = list(use.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in use.iterrows():
        vals = [str(row[col]).replace("\n", " ") for col in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def write_report(result_dir: Path, tables: dict[str, pd.DataFrame], args: argparse.Namespace) -> None:
    lines = [
        "# Phase N4U-PRECHECK-ORACLE-CALIBRATION v2 Report",
        "",
        "## Terminal Decision",
        "",
        f"`terminal_decision = {TERMINAL}`",
        "",
        f"GitHub sync status: `{args.github_sync_status}`; commit: `{args.github_sync_commit or git_short_head()}`.",
        "",
        "This run stopped at Step A by design. No formal N4-U eval, no PPO training, no Gpsi fine-tuning, and no EnvV2-core modification were performed.",
        "",
        "## CPA/TTC Audit",
        "",
        "EnvV2 `obs_i` planned CPA/TTC are stored obstacle planned quantities sampled/constructed at spawn time. They are not runtime candidate-velocity CPA/TTC values. Phase B and N4-O shields independently recompute analytic constant-velocity CPA/TTC from current obstacle state and each candidate action.",
        "",
        md_table(tables["source"], 12),
        "",
        "## Premise Gate",
        "",
        md_table(tables["premise"], 5),
        "",
        "## Strong Geometric Anchor Manifest",
        "",
        md_table(tables["anchor"], 5),
        "",
        "## Phase B / N4-O Equivalence",
        "",
        md_table(tables["equivalence"], 5),
        "",
        "## Required Consequence",
        "",
        "- Stop the clean EnvV2 constant-velocity-correction framing.",
        "- Do not proceed to oracle headroom, Gpsi point-correction, uncertainty calibration, adaptive margin, or stress smoke under this invalid premise in this run.",
        "- A future phase can be reframed around candidate-action runtime CPA/TTC correction directly inside the Phase B/N4-O shield geometry, but it must not describe EnvV2 `planned_cpa/planned_ttc` as the constant-velocity analytic target.",
        "",
        "## Artifacts",
        "",
    ]
    for path in sorted((result_dir / "tables").glob("*.csv")):
        lines.append(f"- `{rel(path)}`")
    write_text(result_dir / REPORT_NAME, "\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    result_dir = ROOT / args.result_dir
    for sub in ["tables", "plots", "logs"]:
        (result_dir / sub).mkdir(parents=True, exist_ok=True)
    try:
        tables = build_tables(result_dir, args)
        commands = pd.DataFrame(
            [
                {
                    "stage": "step_a_cpa_ttc_source_audit",
                    "command": "python -u scripts/audit_phase_n4u_precheck_oracle_calibration_v2.py",
                    "status": "completed_premise_invalidated",
                }
            ]
        )
        commands.to_csv(result_dir / f"tables/{PREFIX}_command_manifest.csv", index=False)
        write_report(result_dir, tables, args)
        write_text(result_dir / STOP_FLAG, f"terminal_decision = {TERMINAL}\n")
        write_text(result_dir / STATUS_NAME, f"stopped:{STOP_FLAG}\n")
        print(f"terminal_decision = {TERMINAL}", flush=True)
    except Exception:
        detail = traceback.format_exc()
        write_text(result_dir / "STOP_CPA_TTC_AUDIT_FAILED.flag", detail)
        write_text(result_dir / STATUS_NAME, "stopped:STOP_CPA_TTC_AUDIT_FAILED.flag\n")
        write_text(
            result_dir / REPORT_NAME,
            "# Phase N4U-PRECHECK-ORACLE-CALIBRATION v2 Report\n\n"
            "`terminal_decision = phase_n4u_precheck_v2_stopped_cpa_ttc_audit_failed`\n\n"
            f"```text\n{detail}\n```\n",
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
