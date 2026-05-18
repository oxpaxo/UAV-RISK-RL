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
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COMPLETE_FLAG = "PHASE_N3PF_DECOUPLE_COMPLETE.flag"
REPORT_FILE = "PHASE_N3PF_DECOUPLE_REPORT.md"
STATUS_FILE = "phase_n3pf_decouple_status.txt"
TABLE_PREFIX = "phase_n3pf_decouple"


class DecoupleAnalysisStop(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Phase N3PF-DECOUPLE outputs.")
    parser.add_argument("--result-dir", default="results/env_v2_phase_n3pf_decouple")
    parser.add_argument("--table-prefix", default=TABLE_PREFIX)
    parser.add_argument("--github-sync-commit", default="")
    parser.add_argument("--github-sync-status", default="unknown")
    parser.add_argument("--attention-success", type=float, default=0.6033)
    parser.add_argument("--attention-collision", type=float, default=0.3967)
    parser.add_argument("--noz-success", type=float, default=0.5667)
    parser.add_argument("--noz-collision", type=float, default=0.4333)
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        if required:
            raise DecoupleAnalysisStop("analysis_failed", f"missing or empty CSV: {rel(path)}")
        return pd.DataFrame()
    return pd.read_csv(path)


def write_stop(result_dir: Path, reason: str, detail: str) -> None:
    flag = {
        "analysis_failed": "STOP_ANALYSIS_FAILED.flag",
        "selector_contaminated": "STOP_SELECTOR_CONTAMINATED.flag",
        "eval_failed": "STOP_EVAL_FAILED.flag",
        "no_valid_variant": "STOP_NO_VALID_VARIANT.flag",
    }.get(reason, "STOP_ANALYSIS_FAILED.flag")
    write_text(result_dir / flag, f"{reason}\n{detail.strip()}\n")
    write_text(result_dir / STATUS_FILE, f"stopped:{flag}\n")
    write_text(result_dir / REPORT_FILE, f"# Phase N3PF-DECOUPLE Report\n\n`terminal_decision = phase_n3pf_decouple_stopped_{reason}`\n\n```text\n{detail.strip()}\n```\n")


def fmt(value: Any) -> str:
    try:
        f = float(value)
    except Exception:
        return str(value)
    if math.isnan(f):
        return "nan"
    return f"{f:.4f}"


def table_md(df: pd.DataFrame, cols: list[str], max_rows: int = 30) -> list[str]:
    if df.empty:
        return ["No rows."]
    cols = [col for col in cols if col in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].head(max_rows).iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) if isinstance(row[col], (float, int, np.floating, np.integer)) else str(row[col]) for col in cols) + " |")
    return lines


def preflight_table(result_dir: Path) -> pd.DataFrame:
    checks = [
        ("guide", ROOT / "codex_guide/PHASE_N3PF_DECOUPLE_GUIDE.md", "file"),
        ("gpsi_checkpoint", ROOT / "work_dirs/gpsi_heada_v1_nll/best.pth", "file"),
        ("nearest_k_extractor", ROOT / "models/gpsi_ppo_policy.py", "GpsiNearestKExtractor"),
        ("deepsets_extractor", ROOT / "models/gpsi_ppo_policy.py", "GpsiDeepSetsExtractor"),
        ("train_entry", ROOT / "scripts/train_env_v2_gpsi_ppo_n3pf_stab.py", "nearest_k_no_attention"),
        ("eval_entry", ROOT / "scripts/eval_env_v2_gpsi_ppo_n3pf_stab.py", "nearestk_selected_features"),
    ]
    rows = []
    for item, path, kind in checks:
        if kind == "file":
            ok = path.is_file()
        else:
            ok = path.is_file() and kind in path.read_text(encoding="utf-8")
        rows.append({"item": item, "ok": int(ok), "detail": rel(path), "kind": kind})
    out = pd.DataFrame(rows)
    out.to_csv(result_dir / f"tables/{TABLE_PREFIX}_preflight_check.csv", index=False)
    if int(out["ok"].min()) != 1:
        raise DecoupleAnalysisStop("analysis_failed", out.to_string(index=False))
    return out


def aggregate_eval(result_dir: Path, table_prefix: str, phase: str) -> pd.DataFrame:
    by_seed = read_csv(result_dir / f"tables/{table_prefix}_{phase}_eval_summary_by_seed.csv", required=False)
    if by_seed.empty:
        return pd.DataFrame()
    group_cols = ["eval_phase", "variant", "training_seed", "method_key", "method", "checkpoint", "checkpoint_path", "checkpoint_step", "checkpoint_label"]
    agg = (
        by_seed.groupby(group_cols, dropna=False)
        .agg(
            success_rate=("success_rate", "mean"),
            collision_rate=("collision_rate", "mean"),
            near_miss_rate=("near_miss_rate", "mean"),
            progress=("progress", "mean"),
            raw_unsafe_action_rate=("raw_unsafe_action_rate", "mean"),
            action_delta=("action_delta", "mean"),
            raw_min_predicted_cpa=("raw_min_predicted_cpa", "mean"),
            episodes=("episodes", "sum"),
            eval_seed_count=("eval_seed", "nunique"),
        )
        .reset_index()
    )
    agg.to_csv(result_dir / f"tables/{table_prefix}_{phase}_eval_summary_aggregate.csv", index=False)
    return agg


def pairwise_from_selected(selected: pd.DataFrame, agg: pd.DataFrame, phase: str) -> pd.DataFrame:
    if selected.empty or agg.empty:
        return pd.DataFrame()
    if phase == "validation":
        selected_keys = selected[["variant", "training_seed", "selected_checkpoint_label"]].rename(columns={"selected_checkpoint_label": "checkpoint_label"})
        use = agg.merge(selected_keys, on=["variant", "training_seed", "checkpoint_label"], how="inner")
    else:
        selected_keys = selected[["variant", "training_seed"]].drop_duplicates()
        use = agg.merge(selected_keys, on=["variant", "training_seed"], how="inner")
    mean = use.groupby("variant", dropna=False).agg(success_rate=("success_rate", "mean"), collision_rate=("collision_rate", "mean"), progress=("progress", "mean"), raw_unsafe_action_rate=("raw_unsafe_action_rate", "mean"), seed_count=("training_seed", "nunique")).reset_index()
    rows = []
    for family, obs_name, gpsi_name in [("nk", "decouple_nk_obs", "decouple_nk_gpsi"), ("deepsets", "decouple_deepsets_obs", "decouple_deepsets_gpsi")]:
        obs = mean[mean["variant"].astype(str) == obs_name]
        gpsi = mean[mean["variant"].astype(str) == gpsi_name]
        if obs.empty or gpsi.empty:
            continue
        o, g = obs.iloc[0], gpsi.iloc[0]
        rows.append(
            {
                "eval_phase": phase,
                "family": family,
                "obs_variant": obs_name,
                "gpsi_variant": gpsi_name,
                "obs_success": float(o.success_rate),
                "obs_collision": float(o.collision_rate),
                "gpsi_success": float(g.success_rate),
                "gpsi_collision": float(g.collision_rate),
                "gpsi_minus_obs_success": float(g.success_rate) - float(o.success_rate),
                "gpsi_minus_obs_collision": float(g.collision_rate) - float(o.collision_rate),
                "obs_seed_count": int(o.seed_count),
                "gpsi_seed_count": int(g.seed_count),
            }
        )
    return pd.DataFrame(rows)


def diagnostics_decision(result_dir: Path, table_prefix: str) -> pd.DataFrame:
    feature = read_csv(result_dir / f"tables/{table_prefix}_validation_feature_block_stats.csv", required=False)
    gpsi = read_csv(result_dir / f"tables/{table_prefix}_validation_gpsi_output_summary.csv", required=False)
    rows = []
    variants = sorted(set(feature.get("variant", pd.Series(dtype=str)).astype(str)) | set(gpsi.get("variant", pd.Series(dtype=str)).astype(str)))
    for variant in variants:
        f = feature[feature["variant"].astype(str) == variant] if not feature.empty else pd.DataFrame()
        nonfinite = int(pd.to_numeric(f.get("nan_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum() + pd.to_numeric(f.get("inf_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not f.empty else 0
        g = gpsi[gpsi["variant"].astype(str) == variant] if not gpsi.empty else pd.DataFrame()
        delta_p95 = float(pd.to_numeric(g.get("delta_norm_1s_p95", pd.Series(dtype=float)), errors="coerce").max()) if not g.empty else np.nan
        rows.append({"variant": variant, "diagnostics_ok": int(nonfinite == 0 and (not np.isfinite(delta_p95) or delta_p95 < 100.0)), "feature_nonfinite_count": nonfinite, "delta_norm_1s_p95_max": delta_p95})
    out = pd.DataFrame(rows)
    out.to_csv(result_dir / f"tables/{table_prefix}_diagnostics_decision.csv", index=False)
    return out


def state_dict(path: Path) -> dict[str, torch.Tensor]:
    model = PPO.load(str(path), device="cpu")
    payload = {key: value.detach().cpu().float().clone() for key, value in model.policy.state_dict().items()}
    del model
    return payload


def l2_for(keys: list[str], lhs: dict[str, torch.Tensor], rhs: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for key in keys:
        if key not in lhs or key not in rhs or lhs[key].shape != rhs[key].shape:
            continue
        diff = lhs[key] - rhs[key]
        total += float(torch.sum(diff * diff).item())
    return float(math.sqrt(total))


def category_keys(keys: list[str]) -> dict[str, list[str]]:
    return {
        "all": keys,
        "feature_extractor": [key for key in keys if "features_extractor" in key],
        "actor_action": [key for key in keys if "action_net" in key or "policy_net" in key],
        "critic_value": [key for key in keys if "value_net" in key],
        "log_std": [key for key in keys if "log_std" in key],
        "nearestk": [key for key in keys if "nearest" in key.lower() or "slot" in key.lower()],
        "deepsets": [key for key in keys if "phi" in key.lower() or "rho" in key.lower() or "deepset" in key.lower()],
    }


def parameter_drift(result_dir: Path, table_prefix: str, selector: pd.DataFrame, stage_plan: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cache: dict[str, dict[str, torch.Tensor]] = {}
    target_by_variant = {
        str(row["variant"]): int(pd.to_numeric(row.get("stage_b_target_steps", 0), errors="coerce") or 0)
        for _, row in stage_plan.iterrows()
    }
    for _, row in selector.iterrows():
        variant = str(row["variant"])
        seed = int(row["training_seed"])
        selected_path = ROOT / str(row["selected_checkpoint_path"])
        ckpt_dir = selected_path.parent
        final_path = ckpt_dir / "final.zip"
        pairs: list[tuple[str, Path, Path]] = [("selected_vs_final", selected_path, final_path)]
        target = int(target_by_variant.get(variant, 0))
        if target > 0:
            target_path = ckpt_dir / f"checkpoint_{target // 1000}k.zip"
            pairs.append((f"selected_vs_{target // 1000}k", selected_path, target_path))
            pairs.append((f"{target // 1000}k_vs_final", target_path, final_path))
        for label, left, right in pairs:
            base: dict[str, Any] = {
                "variant": variant,
                "training_seed": seed,
                "selected_checkpoint_label": str(row["selected_checkpoint_label"]),
                "comparison": label,
                "left_path": rel(left),
                "right_path": rel(right),
                "exists": int(left.exists() and right.exists()),
            }
            if not left.exists() or not right.exists():
                rows.append(base)
                continue
            for path in [left, right]:
                key = str(path)
                if key not in cache:
                    cache[key] = state_dict(path)
            lhs = cache[str(left)]
            rhs = cache[str(right)]
            cats = category_keys(sorted(set(lhs) & set(rhs)))
            for cat, keys in cats.items():
                base[f"{cat}_l2_delta"] = l2_for(keys, lhs, rhs)
                base[f"{cat}_param_count"] = int(sum(lhs[key].numel() for key in keys if key in lhs))
            rows.append(base)
    out = pd.DataFrame(rows)
    out.to_csv(result_dir / f"tables/{table_prefix}_parameter_drift.csv", index=False)
    return out


def final_decision(args: argparse.Namespace, selected_pair: pd.DataFrame, test_pair: pd.DataFrame, selected_agg: pd.DataFrame) -> pd.DataFrame:
    if not test_pair.empty:
        evidence = test_pair
        phase = "test"
    else:
        evidence = selected_pair
        phase = "validation_only"
    gpsi_positive = bool((evidence["gpsi_minus_obs_success"] >= 0.03).any() and (evidence["gpsi_minus_obs_collision"] <= -0.03).any()) if not evidence.empty else False
    best_success = float(selected_agg["success_rate"].max()) if not selected_agg.empty else np.nan
    best_collision = float(selected_agg.sort_values(["success_rate", "collision_rate"], ascending=[False, True])["collision_rate"].iloc[0]) if not selected_agg.empty else np.nan
    all_weak = bool(np.isfinite(best_success) and best_success < 0.50)
    if all_weak:
        terminal = "phase_n3pf_decouple_complete_backbones_too_weak"
        recommendation = "do_not_conclude_gpsi_useless_backbones_are_weak"
    elif gpsi_positive and best_success >= float(args.noz_success) - 0.03:
        terminal = "phase_n3pf_decouple_complete_gpsi_nonattention_positive"
        recommendation = "nonattention_gpsi_positive_consider_followup_confirmation"
    elif gpsi_positive:
        terminal = "phase_n3pf_decouple_complete_gpsi_nonattention_promising_not_competitive"
        recommendation = "gpsi_has_relative_gain_but_policy_not_competitive"
    else:
        terminal = "phase_n3pf_decouple_complete_gpsi_nonattention_failed"
        recommendation = "pivot_to_stable_policy_plus_gpsi_uncertainty_shield"
    rows = [
        {
            "terminal_decision": terminal,
            "decision_basis_phase": phase,
            "best_selected_success": best_success,
            "best_selected_collision": best_collision,
            "attention_reference_success": float(args.attention_success),
            "attention_reference_collision": float(args.attention_collision),
            "noz_reference_success": float(args.noz_success),
            "noz_reference_collision": float(args.noz_collision),
            "gpsi_nonattention_positive": int(gpsi_positive),
            "nonattention_backbones_too_weak": int(all_weak),
            "current_evidence_supports_attention_gpsi_incompatibility": "weak_or_inconclusive",
            "current_evidence_supports_drop_attention_keep_gpsi_ppo": "no" if not gpsi_positive else "not_yet_without_multiseed_competitiveness",
            "recommended_next": recommendation,
            "n4o_paused": "yes",
            "n4u_blocked": "yes",
        }
    ]
    return pd.DataFrame(rows)


def plots(result_dir: Path, table_prefix: str, val_agg: pd.DataFrame, test_agg: pd.DataFrame, pair: pd.DataFrame) -> None:
    (result_dir / "plots").mkdir(parents=True, exist_ok=True)
    labels = ["250k", "500k", "750k", "1000k", "1250k", "1500k", "final"]
    order = {label: idx for idx, label in enumerate(labels)}
    if not val_agg.empty:
        plt.figure(figsize=(11, 5.2))
        mean = val_agg.groupby(["variant", "checkpoint_label"], dropna=False).agg(success_rate=("success_rate", "mean"), collision_rate=("collision_rate", "mean")).reset_index()
        for variant, group in mean.groupby("variant"):
            group = group.assign(_order=group["checkpoint_label"].map(order)).sort_values("_order")
            plt.plot(group["checkpoint_label"], group["success_rate"], marker="o", label=f"{variant} success")
            plt.plot(group["checkpoint_label"], group["collision_rate"], marker="x", linestyle="--", label=f"{variant} collision")
        plt.ylim(0, 1)
        plt.legend(fontsize=6, ncol=2)
        plt.tight_layout()
        plt.savefig(result_dir / f"plots/{table_prefix}_validation_checkpoint_success_collision.png", dpi=140)
        plt.close()
    if not test_agg.empty:
        mean = test_agg.groupby("variant", dropna=False).agg(success_rate=("success_rate", "mean"), collision_rate=("collision_rate", "mean")).reset_index()
        x = np.arange(len(mean))
        plt.figure(figsize=(9, 4.8))
        plt.bar(x - 0.18, mean["success_rate"], width=0.36, label="success")
        plt.bar(x + 0.18, mean["collision_rate"], width=0.36, label="collision")
        plt.xticks(x, mean["variant"], rotation=25, ha="right")
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(result_dir / f"plots/{table_prefix}_test_selected_success_collision.png", dpi=140)
        plt.close()
    if not pair.empty:
        plt.figure(figsize=(7.2, 4.2))
        x = np.arange(len(pair))
        plt.bar(x - 0.18, pair["gpsi_minus_obs_success"], width=0.36, label="success delta")
        plt.bar(x + 0.18, pair["gpsi_minus_obs_collision"], width=0.36, label="collision delta")
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.xticks(x, pair["family"])
        plt.legend()
        plt.tight_layout()
        plt.savefig(result_dir / f"plots/{table_prefix}_gpsi_vs_obs_delta.png", dpi=140)
        plt.close()


def artifacts(result_dir: Path) -> dict[str, list[str]]:
    return {
        "tables": [rel(path) for path in sorted((result_dir / "tables").glob("*.csv"))],
        "plots": [rel(path) for path in sorted((result_dir / "plots").glob("*.png"))],
        "logs": [rel(path) for path in sorted((result_dir / "logs").glob("*.log"))] + [rel(result_dir / "phase_n3pf_decouple_watcher.log")],
        "flags": [rel(path) for path in sorted(result_dir.glob("*.flag"))],
    }


def report(result_dir: Path, args: argparse.Namespace, preflight: pd.DataFrame, selector: pd.DataFrame, stage_plan: pd.DataFrame, val_agg: pd.DataFrame, test_agg: pd.DataFrame, pair_val: pd.DataFrame, pair_test: pd.DataFrame, diagnostics: pd.DataFrame, drift: pd.DataFrame, decision: pd.DataFrame) -> str:
    d = decision.iloc[0].to_dict()
    lines = [
        "# Phase N3PF-DECOUPLE Report",
        "",
        "## Terminal Decision",
        "",
        f"`terminal_decision = {d['terminal_decision']}`",
        "",
        f"GitHub sync status: `{args.github_sync_status}`; commit: `{args.github_sync_commit}`.",
        "",
        "This phase tests Gpsi features under non-attention aggregators only. No shield, reward rewrite, recurrent policy, attention, or Gpsi fine-tuning is used.",
        "",
        "## Preflight",
    ]
    lines.extend(table_md(preflight, ["item", "ok", "kind", "detail"], 20))
    lines.extend(["", "## Stage B Plan"])
    lines.extend(table_md(stage_plan, list(stage_plan.columns), 20))
    lines.extend(["", "## Validation Selector"])
    lines.extend(table_md(selector, ["variant", "training_seed", "selected_checkpoint_label", "success_rate", "collision_rate", "selection_score", "selector_used_only_validation_seeds", "test_seed_used_for_selection"], 40))
    lines.extend(["", "## Validation Aggregate"])
    lines.extend(table_md(val_agg, ["variant", "training_seed", "checkpoint_label", "success_rate", "collision_rate", "progress", "raw_unsafe_action_rate", "episodes"], 50))
    lines.extend(["", "## Validation Gpsi Vs Obs"])
    lines.extend(table_md(pair_val, list(pair_val.columns), 10))
    lines.extend(["", "## Test Aggregate"])
    lines.extend(table_md(test_agg, ["variant", "training_seed", "checkpoint_label", "success_rate", "collision_rate", "progress", "raw_unsafe_action_rate", "episodes"], 50))
    lines.extend(["", "## Test Gpsi Vs Obs"])
    lines.extend(table_md(pair_test, list(pair_test.columns), 10))
    lines.extend(["", "## Diagnostics"])
    lines.extend(table_md(diagnostics, list(diagnostics.columns), 20))
    lines.extend(["", "## Parameter Drift"])
    lines.extend(table_md(drift, ["variant", "training_seed", "selected_checkpoint_label", "comparison", "exists", "all_l2_delta", "feature_extractor_l2_delta", "actor_action_l2_delta", "critic_value_l2_delta", "log_std_l2_delta"], 80))
    lines.extend(
        [
            "",
            "## Direct Answers",
            "",
            f"- Gpsi under NK non-attention aggregator: `{('positive' if not pair_test.empty and (pair_test[(pair_test['family']=='nk')]['gpsi_minus_obs_success'] >= 0.03).any() and (pair_test[(pair_test['family']=='nk')]['gpsi_minus_obs_collision'] <= -0.03).any() else 'not_positive_or_not_tested')}`.",
            f"- Gpsi under DeepSets non-attention aggregator: `{('positive' if not pair_test.empty and (pair_test[(pair_test['family']=='deepsets')]['gpsi_minus_obs_success'] >= 0.03).any() and (pair_test[(pair_test['family']=='deepsets')]['gpsi_minus_obs_collision'] <= -0.03).any() else 'not_positive_or_not_tested')}`.",
            "- Multi-seed stability is judged from seed0/1/2 selected-checkpoint test rows when present; validation-only evidence is not a final method conclusion.",
            "- If all non-attention backbones are weak, this does not by itself prove Gpsi features are useless.",
            f"- Current evidence supports attention/Gpsi incompatibility: `{d['current_evidence_supports_attention_gpsi_incompatibility']}`.",
            f"- Current evidence supports dropping attention and keeping Gpsi-PPO mainline: `{d['current_evidence_supports_drop_attention_keep_gpsi_ppo']}`.",
            "- N4-O remains paused. N4-U remains blocked.",
            "",
            "## Final Decision Table",
        ]
    )
    lines.extend(table_md(decision, list(decision.columns), 5))
    files = artifacts(result_dir)
    lines.extend(["", "## Artifacts"])
    for key in ["tables", "plots", "logs", "flags"]:
        lines.append(f"### {key}")
        values = files.get(key, [])
        lines.extend([f"- `{value}`" for value in values] if values else ["- none"])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    result_dir = ROOT / args.result_dir
    table_prefix = str(args.table_prefix)
    for path in [result_dir, result_dir / "tables", result_dir / "plots", result_dir / "logs"]:
        path.mkdir(parents=True, exist_ok=True)
    try:
        preflight = preflight_table(result_dir)
        val_agg = aggregate_eval(result_dir, table_prefix, "validation")
        test_agg = aggregate_eval(result_dir, table_prefix, "test")
        selector = read_csv(result_dir / f"tables/{table_prefix}_selector_decision.csv")
        stage_plan = read_csv(result_dir / f"tables/{table_prefix}_stage_b_plan.csv")
        pair_val = pairwise_from_selected(selector, val_agg, "validation")
        pair_test = pairwise_from_selected(selector, test_agg, "test")
        pair_val.to_csv(result_dir / f"tables/{table_prefix}_validation_pairwise_gpsi_vs_obs.csv", index=False)
        if not pair_test.empty:
            pair_test.to_csv(result_dir / f"tables/{table_prefix}_test_pairwise_gpsi_vs_obs.csv", index=False)
        diagnostics = diagnostics_decision(result_dir, table_prefix)
        drift = parameter_drift(result_dir, table_prefix, selector, stage_plan)
        selected_keys = selector[["variant", "training_seed", "selected_checkpoint_label"]].rename(columns={"selected_checkpoint_label": "checkpoint_label"})
        selected_agg = val_agg.merge(selected_keys, on=["variant", "training_seed", "checkpoint_label"], how="inner") if not val_agg.empty else pd.DataFrame()
        decision = final_decision(args, pair_val, pair_test, selected_agg)
        decision.to_csv(result_dir / f"tables/{table_prefix}_final_decision.csv", index=False)
        plots(result_dir, table_prefix, val_agg, test_agg, pair_test if not pair_test.empty else pair_val)
        write_text(result_dir / REPORT_FILE, report(result_dir, args, preflight, selector, stage_plan, val_agg, test_agg, pair_val, pair_test, diagnostics, drift, decision))
        write_text(result_dir / COMPLETE_FLAG, decision.iloc[0]["terminal_decision"] + "\n")
        write_text(result_dir / STATUS_FILE, "complete\n")
        print("DECOUPLE_ANALYSIS_COMPLETE", flush=True)
    except DecoupleAnalysisStop as exc:
        write_stop(result_dir, exc.reason, exc.detail)
        raise SystemExit(2) from exc
    except Exception:
        detail = traceback.format_exc()
        write_stop(result_dir, "analysis_failed", detail)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
