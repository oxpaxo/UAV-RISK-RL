from __future__ import annotations

import argparse
import csv
from pathlib import Path


ASSET_ROWS = [
    {
        "category": "env",
        "module_or_file": "envs/dynamic_obstacle_env.py",
        "purpose": "legacy 3-obstacle Gymnasium environment, rich-motion patches, risk/distance info fields",
        "can_reuse_for_env_v2": "partial",
        "notes": "Reuse observation conventions, info metrics, and motion snippets; do not continue method work on old 3-ball environment.",
    },
    {
        "category": "train",
        "module_or_file": "train.py",
        "purpose": "SB3 PPO training entry, Monitor/VecEnv wiring, safety-cost wrapper, run config writing",
        "can_reuse_for_env_v2": "partial",
        "notes": "Useful later for Phase 2 baseline reproduction only; not executed in Phase 0/1.",
    },
    {
        "category": "eval",
        "module_or_file": "eval.py",
        "purpose": "checkpoint evaluation, episode metrics, trace export, attention snapshots",
        "can_reuse_for_env_v2": "partial",
        "notes": "Metric conventions are reusable; eval-style reaction_time is not true reaction latency.",
    },
    {
        "category": "policy",
        "module_or_file": "policies/obstacle_set_extractor.py",
        "purpose": "risk, mean, and learned-attention obstacle set aggregation feature extractor",
        "can_reuse_for_env_v2": "partial",
        "notes": "May be reused once Phase 2 trains PPO on Env V2; not modified for new methods here.",
    },
    {
        "category": "attention",
        "module_or_file": "policies/obstacle_set_extractor.py",
        "purpose": "attention_full implementation and optional risk bias hooks",
        "can_reuse_for_env_v2": "partial",
        "notes": "Attention tooling is frozen as engineering baseline; no temporal/risk-aware attention added.",
    },
    {
        "category": "cost",
        "module_or_file": "train.py",
        "purpose": "SafetyCostWrapper for distance_warning and risk_sum reward shaping",
        "can_reuse_for_env_v2": "partial",
        "notes": "Cost-scale/beta confounds must be controlled; no new risk formula in this phase.",
    },
    {
        "category": "cost",
        "module_or_file": "envs/dynamic_obstacle_env.py",
        "purpose": "legacy distance_warning_cost, risk_sum, and risk_max info fields",
        "can_reuse_for_env_v2": "partial",
        "notes": "Metric pattern reusable, but formulas are not promoted as primary innovation.",
    },
    {
        "category": "reaction metrics",
        "module_or_file": "eval.py",
        "purpose": "no-response/reaction-style lateral deviation metrics for sudden-turn eval",
        "can_reuse_for_env_v2": "partial",
        "notes": "Use as diagnostic lineage only; describe long-train degradation as response reliability/no_response oscillation.",
    },
    {
        "category": "reaction metrics",
        "module_or_file": "scripts/diagnose_sudden_turn.py",
        "purpose": "risk/reaction trace diagnostics around sudden-turn events",
        "can_reuse_for_env_v2": "partial",
        "notes": "Useful for later failure analysis after Env V2 baseline reproduction.",
    },
    {
        "category": "trace",
        "module_or_file": "eval.py",
        "purpose": "per-step trace CSV writing with positions, risk, attention, and deviation fields",
        "can_reuse_for_env_v2": "partial",
        "notes": "Trace infrastructure can be adapted after Phase 1 sanity passes.",
    },
    {
        "category": "trace",
        "module_or_file": "results/gate2b/traces",
        "purpose": "legacy trace artifact directory referenced by P0 analysis",
        "can_reuse_for_env_v2": "no",
        "notes": "Old evidence only; not a source of Env V2 results.",
    },
    {
        "category": "watcher",
        "module_or_file": "scripts/watch_p0_p1_completion.sh",
        "purpose": "legacy blocking watcher pattern for staged research runs",
        "can_reuse_for_env_v2": "yes",
        "notes": "Pattern reused for scripts/watch_phase0_phase1_completion.sh.",
    },
    {
        "category": "watcher",
        "module_or_file": "scripts/watch_p2_completion.sh",
        "purpose": "legacy Phase 2 completion verifier",
        "can_reuse_for_env_v2": "partial",
        "notes": "Verification style reusable; Phase 2 itself is not executed here.",
    },
    {
        "category": "report scripts",
        "module_or_file": "scripts/aggregate_results.py",
        "purpose": "legacy result aggregation helper",
        "can_reuse_for_env_v2": "partial",
        "notes": "Aggregation style reusable for future PPO results.",
    },
    {
        "category": "report scripts",
        "module_or_file": "scripts/summarize_risk_diagnostics.py",
        "purpose": "legacy diagnostic report writer",
        "can_reuse_for_env_v2": "partial",
        "notes": "Report pattern reusable; conclusions are preliminary diagnostics.",
    },
    {
        "category": "configs",
        "module_or_file": "configs",
        "purpose": "configuration directory for current and future environment settings",
        "can_reuse_for_env_v2": "yes",
        "notes": "Env V2 config is stored under configs/env_v2/.",
    },
]


REQUIRED_CONCLUSIONS = [
    "Old 3-ball Gym results are downgraded to preliminary diagnostic evidence.",
    "risk hard weighting / risk prior / risk_penalty are not pushed as the main innovation.",
    "d_warning=1.0 is a narrow/sparse weak baseline.",
    "d_warning=2.0 is a strong safety baseline.",
    "beta / cost scale is an important confound.",
    "Long-training degradation should be described as no_response_rate / response reliability oscillation, not true reaction latency becoming slower.",
    "The new environment must re-validate whether no-response / safety-margin erosion exists.",
    "The current restart does not add a new method; it rebuilds and sanity-checks the environment first.",
]


def status_for(root: Path, module_or_file: str) -> str:
    path = root / module_or_file
    if path.exists():
        return "found"
    return "not_found"


def discover_extra_assets(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    patterns = [
        ("watcher", "scripts/watch_*.sh", "blocking watcher shell scripts", "yes"),
        ("report scripts", "scripts/plot_*.py", "plotting helpers", "partial"),
        ("report scripts", "scripts/run_p*.py", "staged experiment orchestration scripts", "partial"),
        ("report scripts", "*REPORT*.md", "legacy markdown reports", "partial"),
        ("configs", "runs/**/run_config.json", "stored run configs", "partial"),
    ]
    seen: set[str] = {row["module_or_file"] for row in ASSET_ROWS}
    for category, pattern, purpose, reuse in patterns:
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                rel = path.relative_to(root).as_posix()
                if rel in seen:
                    continue
                seen.add(rel)
                rows.append(
                    {
                        "category": category,
                        "module_or_file": rel,
                        "status": "found",
                        "purpose": purpose,
                        "can_reuse_for_env_v2": reuse,
                        "notes": "Discovered during Phase 0 scan.",
                    }
                )
    return rows


def build_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in ASSET_ROWS:
        item = dict(row)
        item["status"] = status_for(root, item["module_or_file"])
        if item["status"] == "not_found":
            item["notes"] = item["notes"] + " Marked not_found; Phase 0 does not no-go on missing old assets."
        rows.append(item)
    rows.extend(discover_extra_assets(root))
    return rows


def write_csv(rows: list[dict[str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["category", "module_or_file", "status", "purpose", "can_reuse_for_env_v2", "notes"]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(rows: list[dict[str, str]], report_path: Path) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1

    lines: list[str] = [
        "# Old Experiments Assets Summary",
        "",
        "## Freeze Decision",
        "",
        "The legacy 3-ball Gym line is frozen. It remains useful only as hypothesis source, metric-system source, failure-mode clue, baseline design experience, and engineering toolchain foundation.",
        "",
        "## Required Conclusions",
        "",
    ]
    for conclusion in REQUIRED_CONCLUSIONS:
        lines.append(f"- {conclusion}")

    lines.extend(
        [
            "",
            "## Asset Scan Summary",
            "",
            f"- Total indexed assets: {len(rows)}",
            f"- found: {counts.get('found', 0)}",
            f"- not_found: {counts.get('not_found', 0)}",
            f"- needs_update: {counts.get('needs_update', 0)}",
            "",
            "| module_or_file | status | purpose | can_reuse_for_env_v2 | notes |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['module_or_file']} | {row['status']} | {row['purpose']} | "
            f"{row['can_reuse_for_env_v2']} | {row['notes']} |"
        )

    lines.extend(
        [
            "",
            "## Reusable Modules",
            "",
            "- Environment patterns: observation dict, info fields, rich-motion snippets, and risk/distance metric bookkeeping are reusable with updates.",
            "- Training/evaluation tooling: SB3 entrypoints, run configs, Monitor/VecEnv use, checkpoint evaluation, and CSV/report conventions are reusable later in Phase 2.",
            "- Policy tooling: obstacle-set feature extractor and attention/risk aggregation code can serve future baselines, but no new attention/risk method is added in Phase 0/1.",
            "- Watcher/report tooling: blocking watcher and staged report patterns are reusable for this restart.",
            "",
            "## Downgraded Or Rejected Routes",
            "",
            "- risk_penalty, risk hard weighting, and risk prior are not treated as a primary contribution without Env V2 re-validation.",
            "- d_warning=1.0 comparisons are weak because the warning band is narrow/sparse.",
            "- d_warning=2.0 is retained as a strong safety baseline candidate for later reproduction.",
            "- beta/cost-scale sweeps remain confounds that must be controlled before method claims.",
            "- Legacy sudden-turn reaction_time is an eval-style diagnostic and must not be described as physical reaction latency.",
            "",
            "## Phase 0 Output",
            "",
            "- `OLD_EXPERIMENTS_ASSETS_SUMMARY.md`",
            "- `results/restart_phase0_phase1/old_assets_index.csv`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--out_dir", type=str, default="results/restart_phase0_phase1")
    parser.add_argument("--report", type=str, default="OLD_EXPERIMENTS_ASSETS_SUMMARY.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    out_dir = root / args.out_dir
    rows = build_rows(root)
    write_csv(rows, out_dir / "old_assets_index.csv")
    write_report(rows, root / args.report)
    print(f"phase0_assets_indexed={len(rows)}")
    print(f"phase0_report={root / args.report}")
    print(f"phase0_csv={out_dir / 'old_assets_index.csv'}")


if __name__ == "__main__":
    main()
