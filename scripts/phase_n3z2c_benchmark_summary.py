from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
    parser = argparse.ArgumentParser(description="Summarize Phase N3Z2C n_envs smoke benchmarks.")
    parser.add_argument("--logs", nargs="+", required=True)
    parser.add_argument("--out", default="results/env_v2_phase_n3z2c_z2_continuation/tables/phase_n3z2c_cpu_benchmark.csv")
    parser.add_argument("--selected-n-envs", type=int, default=4)
    parser.add_argument("--selection-reason", default="Kept n_envs=4 because increasing n_envs changes PPO rollout batch semantics; benchmark is audit-only.")
    parser.add_argument("--skipped-n-envs", nargs="*", type=int, default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for log_name in args.logs:
        path = ROOT / log_name
        text = path.read_text(errors="ignore") if path.exists() else ""
        match = re.search(r"benchmark_nenv(\d+)", path.name)
        n_envs = int(match.group(1)) if match else -1
        rates = [float(x) for x in re.findall(r"rate=\s*([0-9.]+)\s*step/s", text)]
        timesteps = [int(x) for x in re.findall(r"total_timesteps\s+\|\s+(\d+)", text)]
        bad = any(token in text for token in ["Traceback", "RuntimeError", "Exception", "nan_or_crash"])
        rows.append(
            {
                "n_envs": n_envs,
                "status": "completed" if not bad and path.exists() else "failed_or_missing",
                "log": log_name,
                "fps_last": rates[-1] if rates else "",
                "fps_max": max(rates) if rates else "",
                "last_total_timesteps": timesteps[-1] if timesteps else "",
                "bad_signal": int(bad),
                "selected_n_envs": int(args.selected_n_envs),
                "selection_reason": args.selection_reason,
                "ppo_semantics_changed_if_selected": int(n_envs != int(args.selected_n_envs)),
            }
        )
    for n_envs in args.skipped_n_envs:
        rows.append(
            {
                "n_envs": int(n_envs),
                "status": "skipped",
                "log": "",
                "fps_last": "",
                "fps_max": "",
                "last_total_timesteps": "",
                "bad_signal": 0,
                "selected_n_envs": int(args.selected_n_envs),
                "selection_reason": args.selection_reason,
                "ppo_semantics_changed_if_selected": 1,
            }
        )
    write_csv(ROOT / args.out, rows)


if __name__ == "__main__":
    main()
