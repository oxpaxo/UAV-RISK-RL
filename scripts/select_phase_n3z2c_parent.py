from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results/env_v2_phase_n3z2c_z2_continuation"
Z2_DIR = ROOT / "checkpoints/env_v2_gpsi_heada_ppo_n3z_layernorm_alpha0p5_s0"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select Phase N3Z2C Z2 parent checkpoint.")
    parser.add_argument("--result-dir", default=str(RESULT_DIR))
    parser.add_argument("--z2-dir", default=str(Z2_DIR))
    parser.add_argument("--eval-summary", default="results/env_v2_phase_n3fz_noz_full_z_screen/tables/phase_n3fz_eval_summary.csv")
    parser.add_argument("--out", default="results/env_v2_phase_n3z2c_z2_continuation/tables/phase_n3z2c_parent_checkpoint_selection.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    z2_dir = ROOT / args.z2_dir
    eval_summary = ROOT / args.eval_summary
    out_arg = Path(args.out)
    out = out_arg if out_arg.is_absolute() else ROOT / out_arg
    result_arg = Path(args.result_dir)
    result_dir = result_arg if result_arg.is_absolute() else ROOT / result_arg
    candidates = {
        "checkpoint_500k": z2_dir / "checkpoint_500k.zip",
        "final": z2_dir / "final.zip",
        "best_by_eval": z2_dir / "best_by_eval.zip",
    }
    missing = [name for name, path in candidates.items() if not path.exists() or path.stat().st_size == 0]
    if "checkpoint_500k" in missing and "final" in missing and "best_by_eval" in missing:
        raise SystemExit("no Z2 parent checkpoint candidates found")
    if not eval_summary.exists() or eval_summary.stat().st_size == 0:
        raise SystemExit(f"missing eval summary: {rel(eval_summary)}")
    df = pd.read_csv(eval_summary)
    rows: list[dict[str, object]] = []
    metrics: dict[str, tuple[float, float]] = {}
    for label in ["checkpoint_500k", "final", "best_by_eval"]:
        eval_label = "500k" if label == "checkpoint_500k" else label
        sub = df[(df["method_key"].astype(str) == "z_layernorm_alpha_0p5") & (df["checkpoint_label"].astype(str) == eval_label)]
        success = float(sub["success_rate"].mean()) if not sub.empty else float("nan")
        collision = float(sub["collision_rate"].mean()) if not sub.empty else float("nan")
        metrics[label] = (success, collision)
        path = candidates[label]
        rows.append(
            {
                "candidate": label,
                "path": rel(path),
                "exists": int(path.exists()),
                "size": int(path.stat().st_size) if path.exists() else 0,
                "eval_label": eval_label,
                "success_rate": success,
                "collision_rate": collision,
            }
        )

    final_exists = candidates["final"].exists()
    best_exists = candidates["best_by_eval"].exists()
    selected_key = ""
    reason = ""
    if best_exists and final_exists:
        best_success, best_collision = metrics["best_by_eval"]
        final_success, final_collision = metrics["final"]
        if best_success >= final_success and best_collision <= final_collision:
            selected_key = "best_by_eval"
            reason = "best_by_eval exists and is not worse than final in both success and collision"
        else:
            selected_key = "final"
            reason = "best_by_eval is worse than final in success or collision"
    elif final_exists:
        selected_key = "final"
        reason = "final exists and best_by_eval is missing"
    elif candidates["checkpoint_500k"].exists():
        selected_key = "checkpoint_500k"
        reason = "final missing; using checkpoint_500k"
    else:
        raise SystemExit("no usable Z2 parent checkpoint after selection")

    selected_path = candidates[selected_key]
    rows.append(
        {
            "candidate": "selected",
            "path": rel(selected_path),
            "exists": 1,
            "size": int(selected_path.stat().st_size),
            "eval_label": "500k_parent",
            "success_rate": metrics.get(selected_key, (float("nan"), float("nan")))[0],
            "collision_rate": metrics.get(selected_key, (float("nan"), float("nan")))[1],
            "selected_parent_key": selected_key,
            "selected_parent_path": rel(selected_path),
            "selection_reason": reason,
            "parent_total_steps": 500000,
            "additional_steps": 1000000,
            "target_total_steps": 1500000,
        }
    )
    write_csv(out, rows)
    json_path = result_dir / "tables/phase_n3z2c_parent_checkpoint_selection.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(rows[-1], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(rel(selected_path))


if __name__ == "__main__":
    main()
