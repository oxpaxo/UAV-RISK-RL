from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.eval_env_v2_gpsi_ppo_n3fz as base_eval
import scripts.eval_env_v2_gpsi_ppo_n3p as n3p_eval


RESULT_DIR = ROOT / "results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun"
SCENARIOS = [
    "eval_flow_id",
    "eval_flow_high_density",
    "eval_flow_high_speed",
    "eval_flow_high_threat",
    "eval_flow_mixed_ood",
    "eval_flow_sudden_threat",
]

STOP_FLAGS = {
    "eval_failed": "PHASE_N3PF_MS_AB_STOP_EVAL_FAILED.flag",
    "diagnostics_failed": "PHASE_N3PF_MS_AB_STOP_FEATURE_GPSI_DIAGNOSTICS_FAILED.flag",
}

CHECKPOINTS = {
    "checkpoint_250k": ("250k", "checkpoint_250k.zip", 250_000),
    "checkpoint_500k": ("500k", "checkpoint_500k.zip", 500_000),
    "checkpoint_750k": ("750k", "checkpoint_750k.zip", 750_000),
    "checkpoint_1000k": ("1000k", "checkpoint_1000k.zip", 1_000_000),
    "checkpoint_1250k": ("1250k", "checkpoint_1250k.zip", 1_250_000),
    "checkpoint_1500k": ("1500k", "checkpoint_1500k.zip", 1_500_000),
    "final": ("final", "final.zip", 1_500_000),
}


class EvalStop(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase N3PF-MS-AB Step A/B evaluation.")
    parser.add_argument("--mode", choices=["intermediate_eval", "rerun_eval"], required=True)
    parser.add_argument("--result-dir", default="results/env_v2_phase_n3pf_ms_seed2_ab_audit_rerun")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--source-result-dir", default="results/env_v2_phase_n3pf_ms_multiseed")
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--runs", nargs="*", default=["seed2_rerunA"])
    parser.add_argument("--checkpoints", nargs="+", default=[])
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[1000, 1001])
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--write-traces", action="store_true")
    parser.add_argument("--near-miss-distance", type=float, default=1.5)
    parser.add_argument("--raw-cpa-horizon", type=float, default=4.5)
    parser.add_argument("--raw-cpa-threshold", type=float, default=1.2)
    parser.add_argument("--raw-cpa-safe-threshold", type=float, default=1.5)
    parser.add_argument("--no-response-action-norm", type=float, default=0.05)
    parser.add_argument("--heartbeat-seconds", type=float, default=300.0)
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def ensure_dirs(result_dir: Path, out_dir: Path) -> None:
    for path in [result_dir, out_dir, out_dir / "logs", out_dir / "tables", out_dir / "plots", out_dir / "traces"]:
        path.mkdir(parents=True, exist_ok=True)


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
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    old: list[dict[str, Any]] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as handle:
            old = list(csv.DictReader(handle))
    write_csv(path, old + rows)


def write_stop(result_dir: Path, reason: str, detail: str) -> None:
    flag = STOP_FLAGS.get(reason, STOP_FLAGS["eval_failed"])
    write_text(result_dir / flag, f"{reason}\n{detail.strip()}\n")
    write_text(result_dir / "phase_n3pf_ms_seed2_ab_status.txt", f"stopped:{flag}\n")
    write_text(
        result_dir / "PHASE_N3PF_MS_SEED2_AB_AUDIT_RERUN_REPORT.md",
        "\n".join(
            [
                "# Phase N3PF-MS-AB Seed2 Collapse Audit + Minimal Rerun Sanity",
                "",
                f"`terminal_decision = phase_n3pf_ms_seed2_ab_stopped_{reason}`",
                "",
                "Partial report generated by eval.",
                "",
                "```text",
                detail.strip(),
                "```",
            ]
        )
        + "\n",
    )


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise EvalStop("eval_failed", f"config is not a mapping: {rel(path)}")
    return payload


def eval_args_for(args: argparse.Namespace, result_dir: str, seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        result_dir=result_dir,
        seed=int(seed),
        device=args.device,
        write_traces=bool(args.write_traces),
        near_miss_distance=float(args.near_miss_distance),
        raw_cpa_horizon=float(args.raw_cpa_horizon),
        raw_cpa_threshold=float(args.raw_cpa_threshold),
        raw_cpa_safe_threshold=float(args.raw_cpa_safe_threshold),
        no_response_action_norm=float(args.no_response_action_norm),
    )


def checkpoint_items(args: argparse.Namespace) -> list[str]:
    if args.checkpoints:
        items = args.checkpoints
    elif args.mode == "intermediate_eval":
        items = ["checkpoint_250k", "checkpoint_500k", "checkpoint_750k", "checkpoint_1250k"]
    else:
        items = ["checkpoint_1000k", "checkpoint_1500k", "final"]
    unknown = [item for item in items if item not in CHECKPOINTS]
    if unknown:
        raise EvalStop("eval_failed", f"unknown checkpoint spec(s): {unknown}")
    return items


def spec_for_original(seed: int, checkpoint_key: str) -> dict[str, Any]:
    label, filename, step = CHECKPOINTS[checkpoint_key]
    ckpt_dir = ROOT / f"checkpoints/env_v2_gpsi_heada_ppo_n3pf_ms_block_projected_s{seed}"
    config_path = ckpt_dir / "config_resolved.yaml"
    if not config_path.exists():
        config_path = ROOT / "configs/env_v2_gpsi_heada_ppo_n3pf_block_projected.yaml"
    method_key = f"p3_s{seed}_{label}"
    return {
        "run_name": f"original_seed{seed}",
        "training_seed": int(seed),
        "method_key": method_key,
        "method": f"P3_block_projected_seed{seed}_{label}_intermediate",
        "checkpoint": method_key,
        "checkpoint_label": label,
        "checkpoint_step": int(step),
        "checkpoint_path": ckpt_dir / filename,
        "checkpoint_dir": ckpt_dir,
        "config_path": config_path,
        "source_phase": "N3PF-MS-AB-StepA",
    }


def spec_for_rerun(run_name: str, checkpoint_key: str) -> dict[str, Any] | None:
    mapping = {
        "seed2_rerunA": (2, ROOT / "checkpoints/env_v2_gpsi_heada_ppo_n3pf_ms_block_projected_s2_rerunA"),
        "seed3_sanity": (3, ROOT / "checkpoints/env_v2_gpsi_heada_ppo_n3pf_ms_block_projected_s3_sanity"),
    }
    if run_name not in mapping:
        raise EvalStop("eval_failed", f"unknown run_name={run_name}")
    seed, ckpt_dir = mapping[run_name]
    if not ckpt_dir.exists():
        if run_name == "seed3_sanity":
            return None
        raise EvalStop("eval_failed", f"missing rerun checkpoint dir: {rel(ckpt_dir)}")
    label, filename, step = CHECKPOINTS[checkpoint_key]
    config_path = ckpt_dir / "config_resolved.yaml"
    if not config_path.exists():
        config_path = ROOT / "configs/env_v2_gpsi_heada_ppo_n3pf_block_projected.yaml"
    method_key = f"p3_s{seed}_{run_name}_{label}"
    return {
        "run_name": run_name,
        "training_seed": int(seed),
        "method_key": method_key,
        "method": f"P3_block_projected_{run_name}_{label}",
        "checkpoint": method_key,
        "checkpoint_label": label,
        "checkpoint_step": int(step),
        "checkpoint_path": ckpt_dir / filename,
        "checkpoint_dir": ckpt_dir,
        "config_path": config_path,
        "source_phase": "N3PF-MS-AB-StepB",
    }


def build_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for checkpoint_key in checkpoint_items(args):
        if args.mode == "intermediate_eval":
            for seed in args.training_seeds:
                specs.append(spec_for_original(int(seed), checkpoint_key))
        else:
            for run in args.runs:
                spec = spec_for_rerun(str(run), checkpoint_key)
                if spec is not None:
                    specs.append(spec)
    missing = [rel(Path(spec["checkpoint_path"])) for spec in specs if not Path(spec["checkpoint_path"]).exists()]
    if missing:
        raise EvalStop("eval_failed", "missing checkpoints:\n" + "\n".join(missing))
    return specs


def runtime_cfg(spec: dict[str, Any]) -> dict[str, Any]:
    cfg = load_yaml(Path(spec["config_path"]))
    cfg["method_key"] = "block_projected"
    cfg["method_name"] = str(spec["method"])
    cfg.setdefault("training", {})["seed"] = int(spec["training_seed"])
    cfg.setdefault("training", {})["no_shield"] = True
    cfg["training"]["action_filtering"] = False
    cfg["training"]["use_safety_cost"] = False
    return cfg


def register_default_config(spec: dict[str, Any]) -> None:
    n3p_eval.DEFAULT_CONFIGS[str(spec["method_key"])] = {
        "config": rel(Path(spec["config_path"])),
        "checkpoint_dir": rel(Path(spec["checkpoint_dir"])),
    }


def summarize_raw(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    group_cols = [
        "run_name",
        "training_seed",
        "method_key",
        "method",
        "checkpoint",
        "checkpoint_path",
        "checkpoint_step",
        "checkpoint_label",
        "eval_seed",
        "scenario",
        "motion_mode",
        "threat_class",
    ]
    group_cols = [col for col in group_cols if col in df.columns]
    out = (
        df.groupby(group_cols, dropna=False)
        .agg(
            steps=("step", "count"),
            raw_unsafe_rate=("raw_unsafe_action", "mean"),
            raw_safe_margin_unsafe_rate=("raw_safe_margin_unsafe_action", "mean"),
            raw_min_predicted_cpa=("raw_min_predicted_cpa", "mean"),
            raw_min_predicted_ttc=("raw_min_predicted_ttc", "mean"),
            action_norm=("action_norm", "mean"),
            action_delta=("action_delta", "mean"),
            no_response_rate=("no_response", "mean"),
        )
        .reset_index()
    )
    return out.to_dict("records")


def feature_summary_rows(feature_accum: dict[tuple[Any, ...], dict[str, base_eval.BlockAccumulator]], cfg_by_method: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    blocks = ["obs_i_12", "delta_hat_9_after_scale", "logvar_raw_9_clamped", "logvar_scaled_9_policy", "adapter_output_64", "full_aug_obs"]
    for key, groups in feature_accum.items():
        method_key, method, checkpoint, step, label, scenario, motion_mode, threat_class = key
        cfg = cfg_by_method.get(str(method_key), {})
        seed_match = re.search(r"p3_s(\d+)", str(method_key))
        run_name = "unknown"
        if "rerunA" in str(method_key):
            run_name = "seed2_rerunA"
        elif "sanity" in str(method_key):
            run_name = "seed3_sanity"
        elif seed_match:
            run_name = f"original_seed{seed_match.group(1)}"
        prefix = {
            "run_name": run_name,
            "training_seed": int(seed_match.group(1)) if seed_match else -1,
            "method_key": method_key,
            "method": method,
            "checkpoint": checkpoint,
            "checkpoint_step": int(step),
            "checkpoint_label": label,
            "scenario": scenario,
            "motion_mode": motion_mode,
            "threat_class": threat_class,
            "include_z": 0,
            "include_logvar": int(bool(cfg.get("gpsi", {}).get("include_logvar", True))),
            "logvar_output_scale": float(cfg.get("gpsi", {}).get("logvar_output_scale", 1.0)),
        }
        for block in blocks:
            if block in groups:
                rows.append(groups[block].row(prefix, block))
            else:
                rows.append(base_eval.BlockAccumulator().row(prefix, block, not_applicable=True))
    return rows


def aggregate_by_eval_seed(by_seed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not by_seed:
        return []
    df = pd.DataFrame(by_seed)
    group_cols = ["run_name", "training_seed", "method_key", "method", "checkpoint", "checkpoint_path", "checkpoint_step", "checkpoint_label", "source_phase"]
    group_cols = [col for col in group_cols if col in df.columns]
    metrics = [
        "success_rate",
        "collision_rate",
        "near_miss_rate",
        "progress",
        "mean_min_distance",
        "episode_min_distance",
        "episode_length",
        "episode_reward",
        "raw_unsafe_action_rate",
        "raw_safe_margin_unsafe_action_rate",
        "action_norm",
        "action_delta",
        "no_response_rate",
        "raw_min_predicted_cpa",
        "nan_or_crash",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row["num_eval_seeds"] = int(group["eval_seed"].nunique()) if "eval_seed" in group else 0
        row["num_episodes_total"] = int(pd.to_numeric(group.get("episodes", pd.Series(dtype=float)), errors="coerce").sum())
        for metric in metrics:
            if metric in group:
                values = pd.to_numeric(group[metric], errors="coerce")
                row[f"mean_{metric}"] = float(values.mean())
                row[f"std_{metric}"] = float(values.std(ddof=1)) if len(values.dropna()) > 1 else 0.0
        rows.append(row)
    return rows


def write_outputs(args: argparse.Namespace, out_dir: Path, episode_rows: list[dict[str, Any]], raw_rows: list[dict[str, Any]], gpsi_rows: list[dict[str, Any]], feature_rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(episode_rows)
    by_seed_cols = ["run_name", "training_seed", "method_key", "method", "checkpoint", "checkpoint_path", "checkpoint_step", "checkpoint_label", "eval_seed", "source_phase"]
    scenario_cols = by_seed_cols + ["scenario"]
    by_seed = summarize_episode_rows(df, by_seed_cols)
    scenario = summarize_episode_rows(df, scenario_cols)
    motion = summarize_episode_rows(df, by_seed_cols + ["threat_motion_mode"])
    threat = summarize_episode_rows(df, by_seed_cols + ["threat_class"])
    raw_summary = summarize_raw(raw_rows)
    aggregate = aggregate_by_eval_seed(by_seed)

    if args.mode == "intermediate_eval":
        write_csv(out_dir / "tables/phase_n3pf_ms_seed2a_intermediate_checkpoint_eval.csv", aggregate)
        write_csv(out_dir / "tables/phase_n3pf_ms_seed2a_intermediate_checkpoint_eval_by_seed.csv", by_seed)
    else:
        write_csv(out_dir / "tables/phase_n3pf_ms_seed2b_eval_summary_aggregate.csv", aggregate)
        write_csv(out_dir / "tables/phase_n3pf_ms_seed2b_eval_summary_by_seed.csv", by_seed)
        write_csv(out_dir / "tables/phase_n3pf_ms_seed2b_scenario_breakdown.csv", scenario)
        write_csv(out_dir / "tables/phase_n3pf_ms_seed2b_motion_mode_breakdown.csv", motion)
        write_csv(out_dir / "tables/phase_n3pf_ms_seed2b_threat_class_breakdown.csv", threat)
        write_csv(out_dir / "tables/phase_n3pf_ms_seed2b_raw_unsafe_action_summary.csv", raw_summary)
        write_csv(out_dir / "tables/phase_n3pf_ms_seed2b_gpsi_output_summary.csv", gpsi_rows)
        write_csv(out_dir / "tables/phase_n3pf_ms_seed2b_feature_block_stats.csv", feature_rows)


def summarize_episode_rows(df: pd.DataFrame, group_cols: list[str]) -> list[dict[str, Any]]:
    if df.empty:
        return []
    group_cols = [col for col in group_cols if col in df.columns]
    out = (
        df.groupby(group_cols, dropna=False)
        .agg(
            episodes=("episode_id", "count"),
            success_rate=("success", "mean"),
            collision_rate=("collision", "mean"),
            near_miss_rate=("near_miss", "mean"),
            progress=("progress", "mean"),
            mean_min_distance=("mean_min_distance", "mean"),
            episode_min_distance=("episode_min_distance", "mean"),
            episode_length=("episode_length", "mean"),
            episode_reward=("episode_reward", "mean"),
            raw_unsafe_action_rate=("raw_unsafe_action_rate", "mean"),
            raw_safe_margin_unsafe_action_rate=("raw_safe_margin_unsafe_action_rate", "mean"),
            action_norm=("action_norm", "mean"),
            action_delta=("action_delta", "mean"),
            no_response_rate=("no_response_rate", "mean"),
            raw_min_predicted_cpa=("raw_min_predicted_cpa", "mean"),
            nan_or_crash=("nan_or_crash", "sum"),
        )
        .reset_index()
    )
    return out.to_dict("records")


def main() -> None:
    args = parse_args()
    result_dir = ROOT / args.result_dir
    out_dir = ROOT / args.out_dir if args.out_dir else result_dir / ("step_a_audit" if args.mode == "intermediate_eval" else "step_b_rerun")
    ensure_dirs(result_dir, out_dir)
    try:
        specs = build_specs(args)
        append_csv(
            out_dir / "tables/phase_n3pf_ms_seed2ab_eval_command_manifest.csv",
            [
                {
                    "stage": args.mode,
                    "command": " ".join(["python", *sys.argv]),
                    "eval_seeds": json.dumps([int(seed) for seed in args.eval_seeds]),
                    "num_episodes": int(args.num_episodes),
                    "scenarios": json.dumps(args.scenarios),
                    "device": args.device,
                }
            ],
        )
        torch.manual_seed(0)
        np.random.seed(0)
        episode_rows: list[dict[str, Any]] = []
        raw_rows: list[dict[str, Any]] = []
        all_trace_files: list[tuple[Path, dict[str, Any]]] = []
        gpsi_accum: dict[tuple[Any, ...], dict[str, base_eval.ScalarAccumulator]] = {}
        feature_accum: dict[tuple[Any, ...], dict[str, base_eval.BlockAccumulator]] = {}
        cfg_by_method: dict[str, dict[str, Any]] = {}

        for spec in specs:
            register_default_config(spec)
            cfg = runtime_cfg(spec)
            cfg_by_method[str(spec["method_key"])] = cfg
            eval_args = eval_args_for(args, args.result_dir, int(spec["training_seed"]))
            model = PPO.load(str(spec["checkpoint_path"]), device=args.device)
            for eval_seed in args.eval_seeds:
                for scenario in args.scenarios:
                    env = n3p_eval.make_gpsi_env(cfg, scenario, eval_args)
                    start = time.time()
                    last = start
                    print(
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] N3PF_MS_AB_EVAL_START "
                        f"mode={args.mode} policy={spec['method_key']} run={spec['run_name']} eval_seed={eval_seed} "
                        f"scenario={scenario} episodes={args.num_episodes}",
                        flush=True,
                    )
                    for episode_id in range(int(args.num_episodes)):
                        ep_row, raw_step_rows, _trace_rows, trace_files = n3p_eval.evaluate_episode(
                            model=model,
                            env=env,
                            cfg=cfg,
                            method_key=str(spec["method_key"]),
                            method=str(spec["method"]),
                            checkpoint_name=str(spec["checkpoint"]),
                            checkpoint_step_value=int(spec["checkpoint_step"]),
                            checkpoint_label=str(spec["checkpoint_label"]),
                            scenario=scenario,
                            episode_id=(int(eval_seed) - min(args.eval_seeds)) * int(args.num_episodes) + episode_id,
                            episode_seed=int(eval_seed) + episode_id,
                            args=eval_args,
                            gpsi_accum=gpsi_accum,
                            feature_accum=feature_accum,
                        )
                        extra = {
                            "run_name": spec["run_name"],
                            "training_seed": int(spec["training_seed"]),
                            "eval_seed": int(eval_seed),
                            "checkpoint_path": rel(Path(spec["checkpoint_path"])),
                            "source_phase": spec["source_phase"],
                        }
                        ep_row.update(extra)
                        for row in raw_step_rows:
                            row.update(extra)
                        episode_rows.append(ep_row)
                        raw_rows.extend(raw_step_rows)
                        all_trace_files.extend(trace_files)
                        now = time.time()
                        if now - last >= float(args.heartbeat_seconds) or episode_id == int(args.num_episodes) - 1:
                            done = episode_id + 1
                            rate = done / max(now - start, 1.0e-6)
                            print(
                                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] N3PF_MS_AB_EVAL_HEARTBEAT "
                                f"policy={spec['method_key']} eval_seed={eval_seed} scenario={scenario} "
                                f"episodes={done}/{args.num_episodes} rate={rate:.2f} ep/s",
                                flush=True,
                            )
                            last = now
                    env.close()
            del model

        gpsi_rows = base_eval.gpsi_summary_rows(gpsi_accum)
        feature_rows = feature_summary_rows(feature_accum, cfg_by_method)
        write_outputs(args, out_dir, episode_rows, raw_rows, gpsi_rows, feature_rows)
        base_eval.copy_sample_traces(out_dir, all_trace_files)
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] N3PF_MS_AB_EVAL_END "
            f"mode={args.mode} episodes={len(episode_rows)} raw_steps={len(raw_rows)}",
            flush=True,
        )
    except EvalStop as exc:
        write_stop(result_dir, exc.reason, exc.detail)
        raise SystemExit(2) from exc
    except Exception:
        write_stop(result_dir, "eval_failed", traceback.format_exc())
        raise SystemExit(2)


if __name__ == "__main__":
    main()
