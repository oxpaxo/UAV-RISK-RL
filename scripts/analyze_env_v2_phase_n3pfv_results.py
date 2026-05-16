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


COMPLETE_FLAG = "PHASE_N3PFV_CHECKPOINT_VERIFICATION_COMPLETE.flag"
STOP_FLAGS = {
    "required_checkpoint_missing": "PHASE_N3PFV_STOP_REQUIRED_CHECKPOINT_MISSING.flag",
    "reference_checkpoint_missing": "PHASE_N3PFV_STOP_REFERENCE_CHECKPOINT_MISSING.flag",
    "eval_failed": "PHASE_N3PFV_STOP_EVAL_FAILED.flag",
    "schema_invalid": "PHASE_N3PFV_STOP_SCHEMA_INVALID.flag",
    "diagnostics_failed": "PHASE_N3PFV_STOP_DIAGNOSTICS_FAILED.flag",
    "watcher_failed": "PHASE_N3PFV_STOP_WATCHER_FAILED.flag",
}

REQUIRED_POLICIES = ["p3_1000k", "p3_1500k", "p3_final", "attention_full", "no_z_full", "z2_corrected_full"]
P3_POLICIES = ["p3_1000k", "p3_1500k", "p3_final"]
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
    parser = argparse.ArgumentParser(description="Analyze Phase N3PF-V checkpoint verification eval.")
    parser.add_argument("--result-dir", default="results/env_v2_phase_n3pfv_checkpoint_verification")
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=[1000, 1001, 1002])
    parser.add_argument("--expected-episodes", type=int, default=50)
    parser.add_argument("--attention-success", type=float, default=0.6100)
    parser.add_argument("--attention-collision", type=float, default=0.3900)
    parser.add_argument("--noz-success", type=float, default=0.5633)
    parser.add_argument("--noz-collision", type=float, default=0.4367)
    parser.add_argument("--z2-success", type=float, default=0.5067)
    parser.add_argument("--z2-collision", type=float, default=0.4933)
    parser.add_argument("--p3-parent-success", type=float, default=0.5333)
    parser.add_argument("--p3-parent-collision", type=float, default=0.4667)
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
    write_text(result_dir / "phase_n3pfv_status.txt", f"stopped:{flag}\n")
    write_text(
        result_dir / "PHASE_N3PFV_CHECKPOINT_VERIFICATION_REPORT.md",
        "\n".join(
            [
                "# Phase N3PF-V Checkpoint Verification Report",
                "",
                f"`terminal_decision = phase_n3pfv_stopped_{reason}`",
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


def validate(result_dir: Path, args: argparse.Namespace) -> None:
    required_tables = [
        "phase_n3pfv_checkpoint_manifest.csv",
        "phase_n3pfv_command_manifest.csv",
        "phase_n3pfv_eval_summary_by_seed.csv",
        "phase_n3pfv_scenario_breakdown.csv",
        "phase_n3pfv_motion_mode_breakdown.csv",
        "phase_n3pfv_threat_class_breakdown.csv",
        "phase_n3pfv_raw_unsafe_action_summary.csv",
        "phase_n3pfv_action_dynamics_summary.csv",
        "phase_n3pfv_schema_check.csv",
    ]
    missing = [name for name in required_tables if not (result_dir / "tables" / name).exists() or (result_dir / "tables" / name).stat().st_size == 0]
    if missing:
        raise AnalysisStop("eval_failed", f"missing required tables: {missing}")
    manifest = read_csv(result_dir / "tables/phase_n3pfv_checkpoint_manifest.csv")
    required_missing = manifest[(manifest["selected_for_required_eval"].astype(int) == 1) & (manifest["exists"].astype(int) != 1)]
    if not required_missing.empty:
        raise AnalysisStop("required_checkpoint_missing", required_missing.to_string(index=False))
    schema = read_csv(result_dir / "tables/phase_n3pfv_schema_check.csv")
    if "schema_ok" not in schema.columns or (pd.to_numeric(schema["schema_ok"], errors="coerce") != 1).any():
        raise AnalysisStop("schema_invalid", schema.to_string(index=False))
    episodes = read_csv(result_dir / "tables/phase_n3pfv_episode_metrics.csv")
    required = episodes[episodes["method_key"].astype(str).isin(REQUIRED_POLICIES)]
    counts = required.groupby(["method_key", "eval_seed", "scenario"], dropna=False).size().reset_index(name="episodes")
    bad = counts[counts["episodes"] < int(args.expected_episodes)]
    if not bad.empty:
        raise AnalysisStop("eval_failed", "not enough required eval episodes:\n" + bad.to_string(index=False))
    missing_seed_rows = []
    for policy in REQUIRED_POLICIES:
        for seed in args.expected_seeds:
            for scenario in SCENARIOS:
                if not ((counts.method_key.astype(str) == policy) & (pd.to_numeric(counts.eval_seed, errors="coerce") == seed) & (counts.scenario.astype(str) == scenario)).any():
                    missing_seed_rows.append({"method_key": policy, "eval_seed": seed, "scenario": scenario})
    if missing_seed_rows:
        raise AnalysisStop("eval_failed", "missing required policy/seed/scenario rows:\n" + pd.DataFrame(missing_seed_rows).to_string(index=False))


def aggregate_from_by_seed(by_seed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = ["success_rate", "collision_rate", "near_miss_rate", "progress", "mean_min_distance", "episode_reward", "episode_length", "raw_unsafe_action_rate", "raw_safe_margin_unsafe_action_rate", "action_norm", "action_delta", "no_response_rate", "raw_min_predicted_cpa"]
    for keys, group in by_seed.groupby(["method_key", "method", "checkpoint_label", "checkpoint_path"], dropna=False):
        method_key, method, label, path = keys
        row: dict[str, Any] = {
            "method_key": method_key,
            "method": method,
            "checkpoint_label": label,
            "checkpoint_path": path,
            "num_eval_seeds": int(group["eval_seed"].nunique()),
            "num_episodes_total": int(pd.to_numeric(group["episodes"], errors="coerce").sum()),
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"mean_{metric}"] = float(values.mean())
            row[f"std_{metric}"] = float(values.std(ddof=1)) if len(values.dropna()) > 1 else 0.0
        success = row["mean_success_rate"]
        episodes = max(row["num_episodes_total"], 1)
        se = math.sqrt(max(success * (1.0 - success), 0.0) / episodes)
        row["success_normal_ci95_low"] = float(max(0.0, success - 1.96 * se))
        row["success_normal_ci95_high"] = float(min(1.0, success + 1.96 * se))
        collision = row["mean_collision_rate"]
        cse = math.sqrt(max(collision * (1.0 - collision), 0.0) / episodes)
        row["collision_normal_ci95_low"] = float(max(0.0, collision - 1.96 * cse))
        row["collision_normal_ci95_high"] = float(min(1.0, collision + 1.96 * cse))
        rows.append(row)
    out = pd.DataFrame(rows)
    order = {"p3_parent_500k": 0, "p3_1000k": 1, "p3_1250k": 2, "p3_1500k": 3, "p3_final": 4, "attention_full": 5, "no_z_full": 6, "z2_corrected_full": 7}
    out["order"] = out["method_key"].map(order).fillna(99)
    return out.sort_values(["order", "method_key"]).drop(columns=["order"])


def pairwise(by_seed: pd.DataFrame, scenario: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    agg = aggregate_from_by_seed(by_seed).set_index("method_key")
    pairs = []
    for p3 in P3_POLICIES:
        pairs.append((p3, "attention_full"))
        pairs.append((p3, "no_z_full"))
    pairs += [("p3_1000k", "p3_1500k"), ("p3_1500k", "p3_final")]
    for a, b in pairs:
        if a not in agg.index or b not in agg.index:
            continue
        ar = agg.loc[a]
        br = agg.loc[b]
        seed_rows = []
        for seed in sorted(set(by_seed.eval_seed)):
            ad = by_seed[(by_seed.method_key == a) & (by_seed.eval_seed == seed)]
            bd = by_seed[(by_seed.method_key == b) & (by_seed.eval_seed == seed)]
            if not ad.empty and not bd.empty:
                seed_rows.append(
                    {
                        "success_diff": float(ad.success_rate.iloc[0]) - float(bd.success_rate.iloc[0]),
                        "collision_diff": float(ad.collision_rate.iloc[0]) - float(bd.collision_rate.iloc[0]),
                    }
                )
        seed_df = pd.DataFrame(seed_rows)
        scenario_rows = []
        for sc in sorted(set(scenario.scenario)):
            ad = scenario[(scenario.method_key == a) & (scenario.scenario == sc)].groupby("eval_seed", dropna=False).agg(success=("success_rate", "mean"), collision=("collision_rate", "mean")).reset_index()
            bd = scenario[(scenario.method_key == b) & (scenario.scenario == sc)].groupby("eval_seed", dropna=False).agg(success=("success_rate", "mean"), collision=("collision_rate", "mean")).reset_index()
            merged = ad.merge(bd, on="eval_seed", suffixes=("_a", "_b"))
            if not merged.empty:
                scenario_rows.append(
                    {
                        "scenario": sc,
                        "success_diff": float((merged.success_a - merged.success_b).mean()),
                        "collision_diff": float((merged.collision_a - merged.collision_b).mean()),
                    }
                )
        sc_df = pd.DataFrame(scenario_rows)
        rows.append(
            {
                "policy_a": a,
                "policy_b": b,
                "success_diff_mean": float(ar.mean_success_rate - br.mean_success_rate),
                "collision_diff_mean": float(ar.mean_collision_rate - br.mean_collision_rate),
                "seed_success_diff_mean": float(seed_df.success_diff.mean()) if not seed_df.empty else np.nan,
                "seed_success_diff_std": float(seed_df.success_diff.std(ddof=1)) if len(seed_df) > 1 else 0.0,
                "seed_collision_diff_mean": float(seed_df.collision_diff.mean()) if not seed_df.empty else np.nan,
                "seed_collision_diff_std": float(seed_df.collision_diff.std(ddof=1)) if len(seed_df) > 1 else 0.0,
                "worst_scenario_success_diff": float(sc_df.success_diff.min()) if not sc_df.empty else np.nan,
                "worst_scenario_collision_diff": float(sc_df.collision_diff.max()) if not sc_df.empty else np.nan,
                "better_success": int(ar.mean_success_rate >= br.mean_success_rate),
                "better_collision": int(ar.mean_collision_rate <= br.mean_collision_rate),
                "better_both": int(ar.mean_success_rate >= br.mean_success_rate and ar.mean_collision_rate <= br.mean_collision_rate),
            }
        )
    return pd.DataFrame(rows)


def diagnostics(result_dir: Path) -> pd.DataFrame:
    feats = read_csv(result_dir / "tables/phase_n3pfv_feature_block_stats.csv")
    gpsi = read_csv(result_dir / "tables/phase_n3pfv_gpsi_output_summary.csv")
    p3_feats = feats[feats["method_key"].astype(str).str.startswith("p3_")]
    p3_gpsi = gpsi[gpsi["method_key"].astype(str).str.startswith("p3_")]
    if p3_feats.empty or p3_gpsi.empty:
        raise AnalysisStop("diagnostics_failed", "missing P3 diagnostics rows")
    nonfinite = int(pd.to_numeric(p3_feats.get("nan_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    nonfinite += int(pd.to_numeric(p3_feats.get("inf_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    delta_p95 = pd.to_numeric(p3_gpsi.get("delta_norm_1s_p95", pd.Series(dtype=float)), errors="coerce").max()
    delta_max = pd.to_numeric(p3_gpsi.get("delta_norm_1s_max", pd.Series(dtype=float)), errors="coerce").max()
    inactive = pd.to_numeric(p3_gpsi.get("inactive_forwarded_count_max", pd.Series(dtype=float)), errors="coerce").max()
    adapter = p3_feats[p3_feats["block"].astype(str) == "adapter_output_64"]
    adapter_l2 = pd.to_numeric(adapter.get("l2_norm_p95", pd.Series(dtype=float)), errors="coerce").max()
    ok = np.isfinite(delta_p95) and float(delta_p95) < 100.0 and np.isfinite(delta_max) and float(delta_max) < 1000.0 and (not np.isfinite(inactive) or float(inactive) <= 0.0) and np.isfinite(adapter_l2) and 1.0 <= float(adapter_l2) <= 12.0 and nonfinite == 0
    out = pd.DataFrame(
        [
            {
                "diagnostics_ok": int(ok),
                "delta_norm_1s_p95_max": float(delta_p95),
                "delta_norm_1s_max": float(delta_max),
                "inactive_forwarded_count_max": float(inactive) if np.isfinite(inactive) else np.nan,
                "adapter_output_l2_p95_max": float(adapter_l2),
                "feature_nonfinite_count": int(nonfinite),
            }
        ]
    )
    out.to_csv(result_dir / "tables/phase_n3pfv_diagnostics_decision.csv", index=False)
    if not ok:
        raise AnalysisStop("diagnostics_failed", out.to_string(index=False))
    return out


def candidate_decision(agg: pd.DataFrame, pairs: pd.DataFrame, diag: pd.DataFrame) -> pd.DataFrame:
    table = agg.set_index("method_key")
    p1000 = table.loc["p3_1000k"]
    p1500 = table.loc["p3_1500k"]
    pfinal = table.loc["p3_final"]
    att = table.loc["attention_full"]
    noz = table.loc["no_z_full"]
    def beats(row, ref) -> bool:
        return float(row.mean_success_rate) >= float(ref.mean_success_rate) and float(row.mean_collision_rate) <= float(ref.mean_collision_rate)
    def comparable(row, ref) -> bool:
        succ_gap = float(row.mean_success_rate) - float(ref.mean_success_rate)
        coll_gap = float(row.mean_collision_rate) - float(ref.mean_collision_rate)
        return succ_gap >= -0.02 and coll_gap <= 0.02
    id_scenario = read_csv(Path(args_global.result_dir) / "tables/phase_n3pfv_scenario_breakdown.csv") if args_global is not None else pd.DataFrame()
    weak_1000_id = False
    if not id_scenario.empty:
        p1000_id = id_scenario[(id_scenario.method_key == "p3_1000k") & (id_scenario.scenario == "eval_flow_id")]
        p1500_id = id_scenario[(id_scenario.method_key == "p3_1500k") & (id_scenario.scenario == "eval_flow_id")]
        if not p1000_id.empty and not p1500_id.empty:
            weak_1000_id = float(p1000_id.success_rate.mean()) + 0.05 < float(p1500_id.success_rate.mean())
    if beats(p1500, att) or comparable(p1500, att):
        selected = "P3_checkpoint_1500k"
        decision = "P3 1500k is comparable to or better than attention_full; prefer exact 1.5M snapshot."
    elif beats(p1000, p1500) and not weak_1000_id and beats(p1000, noz):
        selected = "P3_checkpoint_1000k"
        decision = "P3 1000k is selected as an offline-eval-selected checkpoint because it is stronger than 1500k without an ID weakness."
    elif beats(pfinal, p1500) and beats(pfinal, noz):
        selected = "P3_final"
        decision = "P3 final matches/exceeds 1500k and remains above no_z."
    elif beats(noz, p1000) and beats(noz, p1500) and beats(noz, pfinal):
        selected = "no_z_full"
        decision = "All P3 checkpoints are worse than no_z; fallback to no_z."
    else:
        selected = "P3_checkpoint_1500k"
        decision = "P3 1500k remains the cleanest candidate but should be described as not decisively above attention."
    p3_best_attention = any(beats(table.loc[key], att) for key in P3_POLICIES)
    row = {
        "selected_n4_candidate": selected,
        "can_enter_n4": "yes",
        "decision": decision,
        "diagnostics_ok": int(diag.diagnostics_ok.iloc[0]),
        "p3_1000k_success": float(p1000.mean_success_rate),
        "p3_1000k_collision": float(p1000.mean_collision_rate),
        "p3_1500k_success": float(p1500.mean_success_rate),
        "p3_1500k_collision": float(p1500.mean_collision_rate),
        "p3_final_success": float(pfinal.mean_success_rate),
        "p3_final_collision": float(pfinal.mean_collision_rate),
        "attention_success": float(att.mean_success_rate),
        "attention_collision": float(att.mean_collision_rate),
        "noz_success": float(noz.mean_success_rate),
        "noz_collision": float(noz.mean_collision_rate),
        "p3_1500k_matches_or_exceeds_attention": int(beats(p1500, att) or comparable(p1500, att)),
        "p3_1000k_offline_eval_selected": int(selected == "P3_checkpoint_1000k"),
        "do_not_claim_decisive_attention_win": int(not p3_best_attention or (abs(float(p1500.mean_success_rate - att.mean_success_rate)) < 0.02)),
    }
    return pd.DataFrame([row])


args_global: argparse.Namespace | None = None


def plot_success_collision(result_dir: Path, agg: pd.DataFrame) -> None:
    data = agg.copy()
    labels = data["method_key"].astype(str).tolist()
    x = np.arange(len(data))
    plt.figure(figsize=(10, 4.8))
    plt.bar(x - 0.18, data["mean_success_rate"], 0.36, yerr=data["std_success_rate"], capsize=3, label="success")
    plt.bar(x + 0.18, data["mean_collision_rate"], 0.36, yerr=data["std_collision_rate"], capsize=3, label="collision")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "plots/n3pfv_success_collision_mean_ci.png", dpi=140)
    plt.close()


def plot_checkpoint_comparison(result_dir: Path, agg: pd.DataFrame) -> None:
    order = ["p3_parent_500k", "p3_1000k", "p3_1250k", "p3_1500k", "p3_final"]
    data = agg[agg.method_key.isin(order)].copy()
    data["order"] = data["method_key"].map({key: idx for idx, key in enumerate(order)})
    data = data.sort_values("order")
    x = np.arange(len(data))
    plt.figure(figsize=(9, 4.5))
    plt.plot(x, data["mean_success_rate"], marker="o", label="success")
    plt.plot(x, data["mean_collision_rate"], marker="o", label="collision")
    plt.xticks(x, data["method_key"], rotation=25, ha="right")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "plots/n3pfv_checkpoint_comparison.png", dpi=140)
    plt.close()


def plot_scenario(result_dir: Path, scenario: pd.DataFrame) -> None:
    data = scenario[scenario.method_key.isin(["p3_1000k", "p3_1500k", "p3_final", "attention_full", "no_z_full"])]
    pivot = data.groupby(["method_key", "scenario"], dropna=False).agg(success=("success_rate", "mean")).reset_index()
    methods = ["p3_1000k", "p3_1500k", "p3_final", "attention_full", "no_z_full"]
    x = np.arange(len(SCENARIOS))
    width = 0.15
    plt.figure(figsize=(12, 5.0))
    for idx, method in enumerate(methods):
        vals = []
        subset = pivot[pivot.method_key == method].set_index("scenario")
        for scenario_name in SCENARIOS:
            vals.append(float(subset.loc[scenario_name, "success"]) if scenario_name in subset.index else np.nan)
        plt.bar(x + (idx - 2) * width, vals, width, label=method)
    plt.xticks(x, SCENARIOS, rotation=25, ha="right")
    plt.ylabel("success rate")
    plt.ylim(0, 1)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(result_dir / "plots/n3pfv_scenario_breakdown.png", dpi=140)
    plt.close()


def plot_seed_stability(result_dir: Path, by_seed: pd.DataFrame) -> None:
    data = by_seed[by_seed.method_key.isin(["p3_1000k", "p3_1500k", "p3_final", "attention_full"])]
    plt.figure(figsize=(9, 4.5))
    for method, group in data.groupby("method_key"):
        group = group.sort_values("eval_seed")
        plt.plot(group["eval_seed"], group["success_rate"], marker="o", label=method)
    plt.xlabel("eval_seed")
    plt.ylabel("success rate")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "plots/n3pfv_seed_stability.png", dpi=140)
    plt.close()


def plot_raw_action_pairwise(result_dir: Path, agg: pd.DataFrame, pairs: pd.DataFrame) -> None:
    data = agg.copy()
    plt.figure(figsize=(9, 4.2))
    plt.bar(data["method_key"], data["mean_raw_unsafe_action_rate"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("mean raw unsafe action rate")
    plt.tight_layout()
    plt.savefig(result_dir / "plots/n3pfv_raw_unsafe_comparison.png", dpi=140)
    plt.close()
    plt.figure(figsize=(9, 4.2))
    plt.plot(np.arange(len(data)), data["mean_action_norm"], marker="o", label="action_norm")
    plt.plot(np.arange(len(data)), data["mean_action_delta"], marker="o", label="action_delta")
    plt.xticks(np.arange(len(data)), data["method_key"], rotation=25, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "plots/n3pfv_action_dynamics.png", dpi=140)
    plt.close()
    vs_att = pairs[pairs.policy_b == "attention_full"]
    plt.figure(figsize=(8, 4.2))
    plt.bar(vs_att["policy_a"], vs_att["success_diff_mean"], label="success_diff")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("success diff vs attention")
    plt.tight_layout()
    plt.savefig(result_dir / "plots/n3pfv_pairwise_vs_attention.png", dpi=140)
    plt.close()


def generate_plots(result_dir: Path, agg: pd.DataFrame, by_seed: pd.DataFrame, scenario: pd.DataFrame, pairs: pd.DataFrame) -> None:
    (result_dir / "plots").mkdir(parents=True, exist_ok=True)
    plot_success_collision(result_dir, agg)
    plot_checkpoint_comparison(result_dir, agg)
    plot_scenario(result_dir, scenario)
    plot_seed_stability(result_dir, by_seed)
    plot_raw_action_pairwise(result_dir, agg, pairs)


def collect_files(result_dir: Path) -> dict[str, list[str]]:
    return {
        "tables": [rel(path) for path in sorted((result_dir / "tables").glob("*.csv"))],
        "plots": [rel(path) for path in sorted((result_dir / "plots").glob("*.png"))],
        "logs": [rel(path) for path in sorted((result_dir / "logs").glob("*.log"))] + [rel(result_dir / "phase_n3pfv_watcher.log")],
        "flags": [rel(path) for path in sorted(result_dir.glob("*.flag"))],
    }


def report(result_dir: Path, agg: pd.DataFrame, by_seed: pd.DataFrame, pairs: pd.DataFrame, decision: pd.DataFrame, diag: pd.DataFrame, files: dict[str, list[str]]) -> list[str]:
    manifest = read_csv(result_dir / "tables/phase_n3pfv_checkpoint_manifest.csv")
    scenario = read_csv(result_dir / "tables/phase_n3pfv_scenario_breakdown.csv")
    motion = read_csv(result_dir / "tables/phase_n3pfv_motion_mode_breakdown.csv")
    threat = read_csv(result_dir / "tables/phase_n3pfv_threat_class_breakdown.csv")
    d = decision.iloc[0]
    lines = [
        "# Phase N3PF-V Checkpoint Verification Report",
        "",
        "## Terminal Decision",
        "",
        "`terminal_decision = phase_n3pfv_checkpoint_verification_complete`",
        "",
        "N3PF-V completed eval-only multi-seed verification. No PPO or Gpsi training was run.",
        "",
        "## Candidate Decision",
        "",
    ]
    lines.extend(table_md(decision, list(decision.columns), max_rows=3))
    lines.extend(["", "## Aggregate Mean/Std", ""])
    lines.extend(
        table_md(
            agg,
            [
                "method_key",
                "checkpoint_label",
                "num_eval_seeds",
                "num_episodes_total",
                "mean_success_rate",
                "std_success_rate",
                "mean_collision_rate",
                "std_collision_rate",
                "mean_near_miss_rate",
                "mean_raw_unsafe_action_rate",
            ],
            max_rows=12,
        )
    )
    lines.extend(["", "## Pairwise", ""])
    lines.extend(table_md(pairs, ["policy_a", "policy_b", "success_diff_mean", "collision_diff_mean", "seed_success_diff_std", "worst_scenario_success_diff", "better_both"], max_rows=24))
    lines.extend(["", "## Per-Seed Summary", ""])
    lines.extend(table_md(by_seed, ["method_key", "checkpoint_label", "eval_seed", "episodes", "success_rate", "collision_rate", "near_miss_rate", "raw_unsafe_action_rate"], max_rows=30))
    lines.extend(["", "## Scenario Focus", ""])
    focus = scenario[scenario["method_key"].isin(["p3_1000k", "p3_1500k", "p3_final", "attention_full", "no_z_full"])]
    lines.extend(table_md(focus, ["method_key", "checkpoint_label", "eval_seed", "scenario", "success_rate", "collision_rate", "raw_unsafe_action_rate"], max_rows=36))
    lines.extend(["", "## Motion/Threat Notes", ""])
    motion_focus = motion[(motion["method_key"].isin(["p3_1000k", "p3_1500k", "p3_final"])) & (motion["threat_motion_mode"].astype(str) == "linear")]
    lines.extend(table_md(motion_focus, ["method_key", "eval_seed", "threat_motion_mode", "episodes", "success_rate", "collision_rate"], max_rows=12))
    threat_focus = threat[threat["method_key"].isin(["p3_1000k", "p3_1500k", "p3_final"])]
    lines.extend(table_md(threat_focus, ["method_key", "eval_seed", "threat_class", "episodes", "success_rate", "collision_rate"], max_rows=12))
    lines.extend(["", "## Diagnostics", ""])
    lines.extend(table_md(diag, list(diag.columns), max_rows=5))
    lines.extend(["", "## Checkpoint Manifest", ""])
    lines.extend(table_md(manifest, ["policy_key", "checkpoint_label", "checkpoint_path", "sha256", "source_phase", "selected_for_required_eval", "selected_for_diagnostic_eval"], max_rows=12))
    lines.extend(["", "## Artifacts", ""])
    for key in ["tables", "plots", "logs", "flags"]:
        lines.append(f"### {key}")
        values = files.get(key, [])
        if values:
            lines.extend([f"- `{value}`" for value in values[:180]])
        else:
            lines.append("- none")
    lines.extend(["", f"Can enter N4: {d['can_enter_n4']}.", f"Selected N4 candidate: `{d['selected_n4_candidate']}`."])
    return lines


def main() -> None:
    global args_global
    args = parse_args()
    args_global = args
    result_dir = ROOT / args.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    try:
        validate(result_dir, args)
        by_seed = read_csv(result_dir / "tables/phase_n3pfv_eval_summary_by_seed.csv")
        scenario = read_csv(result_dir / "tables/phase_n3pfv_scenario_breakdown.csv")
        agg = aggregate_from_by_seed(by_seed)
        agg.to_csv(result_dir / "tables/phase_n3pfv_eval_summary_aggregate.csv", index=False)
        pairs = pairwise(by_seed, scenario)
        pairs.to_csv(result_dir / "tables/phase_n3pfv_pairwise_comparison.csv", index=False)
        diag = diagnostics(result_dir)
        decision = candidate_decision(agg, pairs, diag)
        decision.to_csv(result_dir / "tables/phase_n3pfv_candidate_decision.csv", index=False)
        generate_plots(result_dir, agg, by_seed, scenario, pairs)
        write_text(result_dir / COMPLETE_FLAG, "terminal_decision = phase_n3pfv_checkpoint_verification_complete\n")
        write_text(result_dir / "phase_n3pfv_status.txt", "complete\n")
        files = collect_files(result_dir)
        write_text(result_dir / "PHASE_N3PFV_CHECKPOINT_VERIFICATION_REPORT.md", "\n".join(report(result_dir, agg, by_seed, pairs, decision, diag, files)) + "\n")
        print("terminal_decision = phase_n3pfv_checkpoint_verification_complete", flush=True)
    except AnalysisStop as exc:
        write_stop(result_dir, exc.reason, exc.detail)
        raise SystemExit(2) from exc
    except Exception:
        write_stop(result_dir, "diagnostics_failed", traceback.format_exc())
        raise SystemExit(2)


if __name__ == "__main__":
    main()
