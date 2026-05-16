from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.returncode, proc.stdout.strip()


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
    parser = argparse.ArgumentParser(description="Write Phase N3Z2C resource preflight CSV.")
    parser.add_argument("--out", default="results/env_v2_phase_n3z2c_z2_continuation/tables/phase_n3z2c_resource_preflight.csv")
    parser.add_argument("--log", default="results/env_v2_phase_n3z2c_z2_continuation/logs/phase_n3z2c_resource_preflight.log")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands = {
        "nproc": ["nproc"],
        "lscpu": ["lscpu"],
        "free_h": ["free", "-h"],
        "df_h_root": ["df", "-h", "/"],
        "nvidia_smi": ["nvidia-smi"],
        "git_status_short": ["git", "status", "--short"],
    }
    rows: list[dict[str, object]] = []
    log_lines: list[str] = []
    for name, cmd in commands.items():
        rc, out = run(cmd)
        rows.append(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "item": name,
                "command": " ".join(cmd),
                "returncode": rc,
                "output": out,
            }
        )
        log_lines.extend([f"## {name}", f"$ {' '.join(cmd)}", out, ""])
    rows.extend(
        [
            {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "item": "OMP_NUM_THREADS", "command": "env", "returncode": 0, "output": os.environ.get("OMP_NUM_THREADS", "")},
            {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "item": "MKL_NUM_THREADS", "command": "env", "returncode": 0, "output": os.environ.get("MKL_NUM_THREADS", "")},
            {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "item": "OPENBLAS_NUM_THREADS", "command": "env", "returncode": 0, "output": os.environ.get("OPENBLAS_NUM_THREADS", "")},
            {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "item": "NUMEXPR_NUM_THREADS", "command": "env", "returncode": 0, "output": os.environ.get("NUMEXPR_NUM_THREADS", "")},
        ]
    )
    write_csv(ROOT / args.out, rows)
    log_path = ROOT / args.log
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
