from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


TABLE_SUFFIXES = [
    "episode_metrics",
    "eval_summary_by_seed",
    "scenario_breakdown",
    "motion_mode_breakdown",
    "threat_class_breakdown",
    "raw_unsafe_action_summary",
    "gpsi_output_summary",
    "feature_block_stats",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge Phase N3PF-DECOUPLE parallel eval shard CSVs.")
    parser.add_argument("--result-dir", default="results/env_v2_phase_n3pf_decouple")
    parser.add_argument("--table-prefix", default="phase_n3pf_decouple")
    parser.add_argument("--eval-phase", choices=["validation", "test", "final_heldout"], required=True)
    parser.add_argument("--shard-root", required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def main() -> None:
    args = parse_args()
    result_dir = ROOT / args.result_dir
    shard_root = ROOT / args.shard_root
    table_prefix = str(args.table_prefix)
    phase = str(args.eval_phase)
    if not shard_root.exists():
        raise SystemExit(f"missing shard root: {shard_root}")
    for suffix in TABLE_SUFFIXES:
        rows: list[dict[str, Any]] = []
        for path in sorted(shard_root.glob(f"*/tables/{table_prefix}_{phase}_{suffix}.csv")):
            rows.extend(read_rows(path))
        write_csv(result_dir / f"tables/{table_prefix}_{phase}_{suffix}.csv", rows)
    manifest_rows: list[dict[str, Any]] = []
    for path in sorted(shard_root.glob(f"*/tables/{table_prefix}_checkpoint_manifest.csv")):
        manifest_rows.extend(read_rows(path))
    write_csv(result_dir / f"tables/{table_prefix}_{phase}_checkpoint_manifest.csv", manifest_rows)
    command_rows: list[dict[str, Any]] = []
    for path in sorted(shard_root.glob(f"*/tables/{table_prefix}_eval_command_manifest.csv")):
        command_rows.extend(read_rows(path))
    write_csv(result_dir / f"tables/{table_prefix}_{phase}_eval_command_manifest.csv", command_rows)
    readme = result_dir / f"tables/{table_prefix}_{phase}_raw_unsafe_action_steps.README.txt"
    readme.write_text("Per-step raw unsafe tables were intentionally skipped in parallel eval shards; summaries are merged here.\n", encoding="utf-8")
    print(f"DECOUPLE_MERGE_COMPLETE phase={phase} shards={len(list(shard_root.glob('*')))}", flush=True)


if __name__ == "__main__":
    main()
