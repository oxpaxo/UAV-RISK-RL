from __future__ import annotations

import argparse
import csv
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


STOP_FLAGS = {
    "validation_test_leakage": "PHASE_N3PF_STAB_STOP_VALIDATION_TEST_LEAKAGE.flag",
    "hard_error": "PHASE_N3PF_STAB_STOP_HARD_ERROR.flag",
}
CHECKPOINT_ORDER = {"500k": 500_000, "750k": 750_000, "1000k": 1_000_000, "1250k": 1_250_000, "1500k": 1_500_000, "final": 9_999_999}


class SelectorStop(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select Phase N3PF-STAB checkpoint from validation seeds only.")
    parser.add_argument("--result-dir", default="results/env_v2_phase_n3pf_stab")
    parser.add_argument("--selection-metric", choices=["success_minus_2collision"], default="success_minus_2collision")
    parser.add_argument("--validation-seeds", nargs="+", type=int, default=[900, 901])
    parser.add_argument("--gate-success", type=float, default=0.58)
    parser.add_argument("--gate-collision", type=float, default=0.42)
    parser.add_argument("--hard-collision", type=float, default=0.45)
    parser.add_argument("--hard-progress", type=float, default=0.93)
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
    flag = STOP_FLAGS.get(reason, STOP_FLAGS["hard_error"])
    write_text(result_dir / flag, f"{reason}\n{detail.strip()}\n")
    write_text(result_dir / "phase_n3pf_stab_status.txt", f"stopped:{flag}\n")
    write_text(result_dir / "PHASE_N3PF_STAB_REPORT.md", f"# Phase N3PF-STAB Report\n\n`terminal_decision = phase_n3pf_stab_stop_{reason}`\n\n```text\n{detail.strip()}\n```\n")


def metric(row: pd.Series) -> float:
    return float(row["success_rate"]) - 2.0 * float(row["collision_rate"])


def trend_for(group: pd.DataFrame) -> tuple[bool, str]:
    available = group[group["checkpoint_label"].astype(str).isin(["500k", "750k", "1000k"])].copy()
    if available.empty:
        return False, "no_500_750_1000_validation_rows"
    available["_order"] = available["checkpoint_label"].map(CHECKPOINT_ORDER)
    available = available.sort_values("_order")
    scores = [metric(row) for _, row in available.iterrows()]
    improving = len(scores) >= 3 and scores[-1] > scores[-2] > scores[-3]
    return improving, "score_sequence=" + ",".join(f"{s:.4f}" for s in scores)


def main() -> None:
    args = parse_args()
    result_dir = ROOT / args.result_dir
    try:
        by_seed_path = result_dir / "tables/phase_n3pf_stab_validation_eval_summary_by_seed.csv"
        if not by_seed_path.exists():
            raise SelectorStop("hard_error", f"missing validation summary: {rel(by_seed_path)}")
        by_seed = pd.read_csv(by_seed_path)
        if by_seed.empty:
            raise SelectorStop("hard_error", "validation summary is empty")
        phases = set(by_seed.get("eval_phase", pd.Series(dtype=str)).astype(str))
        if phases != {"validation"}:
            raise SelectorStop("validation_test_leakage", f"selector may only read validation rows, got phases={sorted(phases)}")
        seeds = set(pd.to_numeric(by_seed.get("eval_seed"), errors="coerce").dropna().astype(int))
        if seeds != set(args.validation_seeds):
            raise SelectorStop("validation_test_leakage", f"validation seed set mismatch: got={sorted(seeds)} expected={sorted(args.validation_seeds)}")
        group_cols = ["variant", "training_seed", "method_key", "checkpoint_label", "checkpoint_path", "checkpoint_step"]
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
        rows: list[dict[str, Any]] = []
        for _, row in agg.iterrows():
            hard_pass = float(row.collision_rate) <= float(args.hard_collision) and float(row.progress) >= float(args.hard_progress)
            gate_pass = float(row.success_rate) >= float(args.gate_success) and float(row.collision_rate) <= float(args.gate_collision) and hard_pass
            out = row.to_dict()
            out["selection_metric"] = args.selection_metric
            out["selection_score"] = metric(row)
            out["hard_filter_pass"] = int(hard_pass)
            out["seed2_screening_gate_pass"] = int(gate_pass)
            rows.append(out)
        scored = pd.DataFrame(rows)
        scored["_order"] = scored["checkpoint_label"].astype(str).map(CHECKPOINT_ORDER).fillna(0)
        valid = scored[scored["hard_filter_pass"].astype(int) == 1].copy()
        selected_rows: list[dict[str, Any]] = []
        for variant, group in scored.groupby("variant", dropna=False):
            improving, trend_detail = trend_for(group)
            choices = valid[valid["variant"].astype(str) == str(variant)].copy()
            if choices.empty:
                choices = group.copy()
            choices = choices.sort_values(["selection_score", "_order"], ascending=[False, False])
            best = choices.iloc[0].to_dict()
            checkpoint_label = str(best["checkpoint_label"])
            continue_to_1500k = int(checkpoint_label == "1000k" and improving and float(best["collision_rate"]) <= 0.50 and float(best["progress"]) >= float(args.hard_progress))
            selected_rows.append(
                {
                    "variant": variant,
                    "selected_checkpoint_label": checkpoint_label,
                    "selected_checkpoint_path": best["checkpoint_path"],
                    "selected_checkpoint_step": int(best["checkpoint_step"]),
                    "selection_metric": args.selection_metric,
                    "selection_score": float(best["selection_score"]),
                    "success_rate": float(best["success_rate"]),
                    "collision_rate": float(best["collision_rate"]),
                    "progress": float(best["progress"]),
                    "raw_unsafe_action_rate": float(best["raw_unsafe_action_rate"]),
                    "hard_filter_pass": int(best["hard_filter_pass"]),
                    "seed2_screening_gate_pass": int(best["seed2_screening_gate_pass"]),
                    "trend_improving_through_1000k": int(improving),
                    "trend_detail": trend_detail,
                    "continue_to_1500k_recommended": continue_to_1500k,
                    "selector_used_only_validation_seeds": 1,
                    "test_seed_used_for_selection": 0,
                }
            )
        scored = scored.drop(columns=["_order"])
        write_csv(result_dir / "tables/phase_n3pf_stab_validation_checkpoint_scores.csv", scored.to_dict("records"))
        write_csv(result_dir / "tables/phase_n3pf_stab_selector_decision.csv", selected_rows)
        print(f"STAB_SELECTOR_COMPLETE variants={len(selected_rows)} metric={args.selection_metric}", flush=True)
    except SelectorStop as exc:
        write_stop(result_dir, exc.reason, exc.detail)
        raise SystemExit(2) from exc
    except Exception:
        detail = traceback.format_exc()
        write_stop(result_dir, "hard_error", detail)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
