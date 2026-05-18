from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CHECKPOINT_ORDER = {"250k": 250_000, "500k": 500_000, "750k": 750_000, "1000k": 1_000_000, "1250k": 1_250_000, "1500k": 1_500_000, "final": 9_999_999}


class DecoupleSelectorStop(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select Phase N3PF-DECOUPLE checkpoints and Stage B variants from validation rows.")
    parser.add_argument("--result-dir", default="results/env_v2_phase_n3pf_decouple")
    parser.add_argument("--table-prefix", default="phase_n3pf_decouple")
    parser.add_argument("--validation-seeds", nargs="+", type=int, default=[900, 901])
    parser.add_argument("--stage-a", action="store_true")
    parser.add_argument("--gpsi-success-margin", type=float, default=0.03)
    parser.add_argument("--gpsi-collision-margin", type=float, default=0.03)
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


def write_stop(result_dir: Path, reason: str, detail: str) -> None:
    flag = "STOP_SELECTOR_CONTAMINATED.flag" if reason == "selector_contaminated" else "STOP_PREFLIGHT_FAILED.flag"
    write_text(result_dir / flag, f"{reason}\n{detail.strip()}\n")
    write_text(result_dir / "phase_n3pf_decouple_status.txt", f"stopped:{flag}\n")


def metric(row: pd.Series) -> float:
    return float(row["success_rate"]) - 2.0 * float(row["collision_rate"])


def checkpoint_agg(path: Path, expected_seeds: set[int]) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise DecoupleSelectorStop("hard_error", f"missing validation summary: {rel(path)}")
    by_seed = pd.read_csv(path)
    if by_seed.empty:
        raise DecoupleSelectorStop("hard_error", "validation summary is empty")
    phases = set(by_seed["eval_phase"].astype(str))
    if phases != {"validation"}:
        raise DecoupleSelectorStop("selector_contaminated", f"selector may only read validation rows, got phases={sorted(phases)}")
    seeds = set(pd.to_numeric(by_seed["eval_seed"], errors="coerce").dropna().astype(int))
    if seeds != expected_seeds:
        raise DecoupleSelectorStop("selector_contaminated", f"validation seeds mismatch: got={sorted(seeds)} expected={sorted(expected_seeds)}")
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
    agg["selection_score"] = agg.apply(metric, axis=1)
    agg["_order"] = agg["checkpoint_label"].astype(str).map(CHECKPOINT_ORDER).fillna(0)
    return agg


def select_rows(agg: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (variant, training_seed), group in agg.groupby(["variant", "training_seed"], dropna=False):
        choice = group.sort_values(
            ["selection_score", "collision_rate", "success_rate", "progress", "raw_unsafe_action_rate", "_order"],
            ascending=[False, True, False, False, True, True],
        ).iloc[0]
        rows.append(
            {
                "variant": variant,
                "training_seed": int(training_seed),
                "selected_checkpoint_label": str(choice["checkpoint_label"]),
                "selected_checkpoint_path": str(choice["checkpoint_path"]),
                "selected_checkpoint_step": int(choice["checkpoint_step"]),
                "selection_score": float(choice["selection_score"]),
                "success_rate": float(choice["success_rate"]),
                "collision_rate": float(choice["collision_rate"]),
                "progress": float(choice["progress"]),
                "raw_unsafe_action_rate": float(choice["raw_unsafe_action_rate"]),
                "selector_used_only_validation_seeds": 1,
                "test_seed_used_for_selection": 0,
            }
        )
    return pd.DataFrame(rows)


def stage_b_plan(selected: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    variant_mean = (
        selected.groupby("variant", dropna=False)
        .agg(
            success_rate=("success_rate", "mean"),
            collision_rate=("collision_rate", "mean"),
            progress=("progress", "mean"),
            raw_unsafe_action_rate=("raw_unsafe_action_rate", "mean"),
            selected_seed_count=("training_seed", "nunique"),
        )
        .reset_index()
    )
    pairs = {
        "nk": ("decouple_nk_obs", "decouple_nk_gpsi"),
        "deepsets": ("decouple_deepsets_obs", "decouple_deepsets_gpsi"),
    }
    rows: list[dict[str, Any]] = []
    continue_variants: set[str] = set()
    for family, (obs_name, gpsi_name) in pairs.items():
        obs = variant_mean[variant_mean["variant"].astype(str) == obs_name]
        gpsi = variant_mean[variant_mean["variant"].astype(str) == gpsi_name]
        if obs.empty or gpsi.empty:
            continue
        o = obs.iloc[0]
        g = gpsi.iloc[0]
        success_delta = float(g.success_rate) - float(o.success_rate)
        collision_delta = float(g.collision_rate) - float(o.collision_rate)
        gpsi_gain_gate = success_delta >= float(args.gpsi_success_margin) and collision_delta <= -float(args.gpsi_collision_margin)
        obs_competitive = float(o.success_rate) >= 0.52 and float(o.collision_rate) <= 0.48
        gpsi_competitive = float(g.success_rate) >= 0.52 and float(g.collision_rate) <= 0.48
        continue_pair = bool(gpsi_gain_gate or obs_competitive or gpsi_competitive)
        if continue_pair:
            continue_variants.update([obs_name, gpsi_name] if gpsi_gain_gate else [gpsi_name if gpsi_competitive else obs_name])
        rows.append(
            {
                "family": family,
                "obs_variant": obs_name,
                "gpsi_variant": gpsi_name,
                "obs_success": float(o.success_rate),
                "obs_collision": float(o.collision_rate),
                "gpsi_success": float(g.success_rate),
                "gpsi_collision": float(g.collision_rate),
                "gpsi_minus_obs_success": success_delta,
                "gpsi_minus_obs_collision": collision_delta,
                "gpsi_strict_gain_gate": int(gpsi_gain_gate),
                "obs_backbone_competitive": int(obs_competitive),
                "gpsi_backbone_competitive": int(gpsi_competitive),
                "continue_stage_b": int(continue_pair),
            }
        )
    if not continue_variants:
        best_gpsi = variant_mean[variant_mean["variant"].astype(str).str.endswith("_gpsi")].sort_values(["success_rate", "collision_rate"], ascending=[False, True]).head(1)
        best_obs = variant_mean[variant_mean["variant"].astype(str).str.endswith("_obs")].sort_values(["success_rate", "collision_rate"], ascending=[False, True]).head(1)
        continue_variants.update(best_gpsi["variant"].astype(str).tolist())
        continue_variants.update(best_obs["variant"].astype(str).tolist())
        fallback = "all_variants_weak_continue_best_gpsi_and_best_obs_to_1000k"
        target_steps = 1_000_000
    else:
        fallback = "continue_validation_useful_variants_to_1500k"
        target_steps = 1_500_000
    plan_rows = []
    for _, row in variant_mean.iterrows():
        variant = str(row["variant"])
        plan_rows.append(
            {
                "variant": variant,
                "stage_a_success": float(row.success_rate),
                "stage_a_collision": float(row.collision_rate),
                "continue_stage_b": int(variant in continue_variants),
                "stage_b_target_steps": int(target_steps if variant in continue_variants else 0),
                "stage_b_policy": fallback,
                "stage_b_checkpoint_steps": "1000000" if target_steps == 1_000_000 else "1000000,1250000,1500000",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(plan_rows)


def main() -> None:
    args = parse_args()
    result_dir = ROOT / args.result_dir
    table_prefix = str(args.table_prefix)
    try:
        agg = checkpoint_agg(result_dir / f"tables/{table_prefix}_validation_eval_summary_by_seed.csv", set(args.validation_seeds))
        agg.drop(columns=["_order"]).to_csv(result_dir / f"tables/{table_prefix}_validation_checkpoint_scores.csv", index=False)
        selected = select_rows(agg)
        selected.to_csv(result_dir / f"tables/{table_prefix}_selector_decision.csv", index=False)
        pair, plan = stage_b_plan(selected, args)
        pair.to_csv(result_dir / f"tables/{table_prefix}_stage_a_pairwise_gpsi_vs_obs.csv", index=False)
        plan.to_csv(result_dir / f"tables/{table_prefix}_stage_b_plan.csv", index=False)
        print(f"DECOUPLE_SELECTOR_COMPLETE variants={selected['variant'].nunique()} stage_b={int(plan['continue_stage_b'].sum())}", flush=True)
    except DecoupleSelectorStop as exc:
        write_stop(result_dir, exc.reason, exc.detail)
        raise SystemExit(2) from exc
    except Exception:
        detail = traceback.format_exc()
        write_stop(result_dir, "hard_error", detail)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
