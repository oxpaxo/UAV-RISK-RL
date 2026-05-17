from __future__ import annotations

import argparse
import csv
import hashlib
import json
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


RESULT_DIR = ROOT / "results/env_v2_phase_n3pf_stab"
SCENARIOS = [
    "eval_flow_id",
    "eval_flow_high_density",
    "eval_flow_high_speed",
    "eval_flow_high_threat",
    "eval_flow_mixed_ood",
    "eval_flow_sudden_threat",
]
STOP_FLAGS = {
    "hard_error": "PHASE_N3PF_STAB_STOP_HARD_ERROR.flag",
    "validation_test_leakage": "PHASE_N3PF_STAB_STOP_VALIDATION_TEST_LEAKAGE.flag",
    "training_broken": "PHASE_N3PF_STAB_STOP_TRAINING_BROKEN.flag",
    "eval_failed": "PHASE_N3PF_STAB_STOP_EVAL_FAILED.flag",
}
CHECKPOINTS = {
    "500k": ("checkpoint_500k.zip", 500_000),
    "750k": ("checkpoint_750k.zip", 750_000),
    "1000k": ("checkpoint_1000k.zip", 1_000_000),
    "1250k": ("checkpoint_1250k.zip", 1_250_000),
    "1500k": ("checkpoint_1500k.zip", 1_500_000),
    "final": ("final.zip", -1),
}


class StabEvalStop(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Phase N3PF-STAB checkpoints.")
    parser.add_argument("--result-dir", default="results/env_v2_phase_n3pf_stab")
    parser.add_argument("--table-prefix", default="phase_n3pf_stab")
    parser.add_argument("--status-file", default="phase_n3pf_stab_status.txt")
    parser.add_argument("--report-file", default="PHASE_N3PF_STAB_REPORT.md")
    parser.add_argument("--terminal-prefix", default="phase_n3pf_stab")
    parser.add_argument("--stop-flag-mode", choices=["stab", "confirm"], default="stab")
    parser.add_argument("--eval-phase", choices=["validation", "test", "final_heldout"], required=True)
    parser.add_argument("--run", action="append", required=True, help="variant:training_seed:config_path:checkpoint_dir[:label1,label2]")
    parser.add_argument("--checkpoint-labels", nargs="+", default=["500k", "750k", "1000k", "final"])
    parser.add_argument("--eval-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--num-episodes", type=int, default=30)
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--write-traces", action="store_true")
    parser.add_argument("--skip-raw-step-table", action="store_true")
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


def ensure_dirs(result_dir: Path) -> None:
    for path in [result_dir, result_dir / "logs", result_dir / "tables", result_dir / "plots", result_dir / "traces"]:
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
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def merge_scalar_accumulator(dst: base_eval.ScalarAccumulator, src: base_eval.ScalarAccumulator) -> None:
    dst.values.extend(src.values)
    dst.nan_count += int(src.nan_count)
    dst.inf_count += int(src.inf_count)


def merge_block_accumulator(dst: base_eval.BlockAccumulator, src: base_eval.BlockAccumulator) -> None:
    merge_scalar_accumulator(dst.l2, src.l2)
    merge_scalar_accumulator(dst.max_abs, src.max_abs)
    dst.element_count += int(src.element_count)
    dst.nan_count += int(src.nan_count)
    dst.inf_count += int(src.inf_count)


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    old: list[dict[str, Any]] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as handle:
            old = list(csv.DictReader(handle))
    write_csv(path, old + rows)


def write_stop(result_dir: Path, args: argparse.Namespace, reason: str, detail: str) -> None:
    ensure_dirs(result_dir)
    flags = STOP_FLAGS
    if str(args.stop_flag_mode) == "confirm":
        flags = {
            "hard_error": "STOP_PREFLIGHT_FAILED.flag",
            "validation_test_leakage": "STOP_SELECTOR_CONTAMINATED.flag",
            "training_broken": "STOP_TRAINING_FAILED.flag",
            "eval_failed": "STOP_EVAL_FAILED.flag",
        }
    flag = flags.get(reason, flags["hard_error"])
    write_text(result_dir / flag, f"{reason}\n{detail.strip()}\n")
    write_text(result_dir / str(args.status_file), f"stopped:{flag}\n")
    write_text(
        result_dir / str(args.report_file),
        "\n".join(
            [
                "# Phase N3PF-STAB Eval Partial Report",
                "",
                f"`terminal_decision = {args.terminal_prefix}_stop_{reason}`",
                "",
                "Partial report generated by eval script.",
                "",
                "```text",
                detail.strip(),
                "```",
            ]
        )
        + "\n",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise StabEvalStop("hard_error", f"config is not a mapping: {rel(path)}")
    return payload


def parse_run(text: str) -> dict[str, Any]:
    parts = text.split(":", 4)
    if len(parts) not in {4, 5}:
        raise StabEvalStop("hard_error", f"--run must be variant:training_seed:config:checkpoint_dir[:labels], got {text!r}")
    variant, seed_text, config_path, checkpoint_dir = parts[:4]
    labels = None
    if len(parts) == 5 and parts[4].strip():
        labels = [item.strip() for item in parts[4].split(",") if item.strip()]
    return {
        "variant": variant,
        "training_seed": int(seed_text),
        "config_path": ROOT / config_path,
        "checkpoint_dir": ROOT / checkpoint_dir,
        "checkpoint_labels": labels,
    }


def runtime_args(args: argparse.Namespace, seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        result_dir=args.result_dir,
        seed=int(seed),
        device=args.device,
        write_traces=bool(args.write_traces),
        near_miss_distance=float(args.near_miss_distance),
        raw_cpa_horizon=float(args.raw_cpa_horizon),
        raw_cpa_threshold=float(args.raw_cpa_threshold),
        raw_cpa_safe_threshold=float(args.raw_cpa_safe_threshold),
        no_response_action_norm=float(args.no_response_action_norm),
    )


def checkpoint_path(run: dict[str, Any], label: str) -> tuple[Path, int]:
    if label not in CHECKPOINTS:
        raise StabEvalStop("hard_error", f"unknown checkpoint label: {label}")
    filename, step = CHECKPOINTS[label]
    path = Path(run["checkpoint_dir"]) / filename
    if label == "final":
        resolved = Path(run["checkpoint_dir"]) / "config_resolved.yaml"
        if resolved.exists():
            try:
                cfg = load_yaml(resolved)
                step = int(cfg.get("training", {}).get("total_steps", 1_000_000))
            except Exception:
                step = 1_000_000
        else:
            step = 1_000_000
    return path, int(step)


def write_checkpoint_manifest(result_dir: Path, table_prefix: str, rows: list[dict[str, Any]]) -> None:
    append_csv(result_dir / f"tables/{table_prefix}_checkpoint_manifest.csv", rows)


def feature_key(prefix: dict[str, Any], info: dict[str, Any]) -> tuple[Any, ...]:
    return (
        prefix["eval_phase"],
        prefix["variant"],
        prefix["training_seed"],
        prefix["method_key"],
        prefix["method"],
        prefix["checkpoint"],
        int(prefix["checkpoint_step"]),
        prefix["checkpoint_label"],
        prefix["scenario"],
        str(info.get("threat_motion_mode", "none")),
        str(info.get("threat_class", "none")),
    )


def feature_summary_rows(feature_accum: dict[tuple[Any, ...], dict[str, base_eval.BlockAccumulator]], cfg_by_method: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    blocks = [
        "obs_i_12",
        "delta_hat_9_after_scale",
        "logvar_raw_9_clamped",
        "logvar_scaled_9_policy",
        "adapter_output_64",
        "gated_base_branch_64",
        "gated_gpsi_branch_64",
        "gated_contribution_64",
        "gated_gate_value",
        "full_aug_obs",
    ]
    for key, groups in feature_accum.items():
        eval_phase, variant, training_seed, method_key, method, checkpoint, step, label, scenario, motion_mode, threat_class = key
        cfg = cfg_by_method.get(str(method_key), {})
        prefix = {
            "eval_phase": eval_phase,
            "variant": variant,
            "training_seed": int(training_seed),
            "method_key": method_key,
            "method": method,
            "checkpoint": checkpoint,
            "checkpoint_step": int(step),
            "checkpoint_label": label,
            "scenario": scenario,
            "motion_mode": motion_mode,
            "threat_class": threat_class,
            "feature_adapter": str(cfg.get("ppo", {}).get("feature_adapter", "")),
        }
        for block in blocks:
            if block in groups:
                rows.append(groups[block].row(prefix, block))
            else:
                rows.append(base_eval.BlockAccumulator().row(prefix, block, not_applicable=True))
    return rows


def gpsi_summary_rows(accum: dict[tuple[Any, ...], dict[str, base_eval.ScalarAccumulator]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, groups in accum.items():
        eval_phase, variant, training_seed, method_key, method, checkpoint, step, label, scenario, motion_mode, threat_class = key
        row: dict[str, Any] = {
            "eval_phase": eval_phase,
            "variant": variant,
            "training_seed": int(training_seed),
            "method_key": method_key,
            "method": method,
            "checkpoint": checkpoint,
            "checkpoint_step": int(step),
            "checkpoint_label": label,
            "scenario": scenario,
            "motion_mode": motion_mode,
            "threat_class": threat_class,
        }
        for name, acc in groups.items():
            row.update(acc.stats(name))
            if name.startswith("logvar_xy_"):
                values = np.asarray(acc.values, dtype=np.float64)
                row[f"{name}_span"] = float(np.max(values) - np.min(values)) if values.size else np.nan
        rows.append(row)
    return rows


def add_stab_diagnostics(
    *,
    accum: dict[tuple[Any, ...], dict[str, base_eval.ScalarAccumulator]],
    feature_accum: dict[tuple[Any, ...], dict[str, base_eval.BlockAccumulator]],
    cfg: dict[str, Any],
    env: Any,
    info: dict[str, Any],
    prefix: dict[str, Any],
) -> None:
    local_prefix = {
        "method_key": prefix["method_key"],
        "method": prefix["method"],
        "checkpoint": prefix["checkpoint"],
        "checkpoint_step": prefix["checkpoint_step"],
        "checkpoint_label": prefix["checkpoint_label"],
        "scenario": prefix["scenario"],
    }
    # Reuse the existing diagnostics, then remap its compact key into the STAB key.
    tmp_gpsi: dict[tuple[Any, ...], dict[str, base_eval.ScalarAccumulator]] = {}
    tmp_feat: dict[tuple[Any, ...], dict[str, base_eval.BlockAccumulator]] = {}
    n3p_eval.add_gpsi_diagnostics(accum=tmp_gpsi, feature_accum=tmp_feat, cfg=cfg, env=env, info=info, prefix=local_prefix)
    n3p_eval.add_adapter_output(feature_accum=tmp_feat, model=prefix["model"], cfg=cfg, env=env, info=info, prefix=local_prefix)
    src_key = n3p_eval.feature_key(local_prefix, info)
    dst_key = feature_key(prefix, info)
    if src_key in tmp_gpsi:
        if dst_key not in accum:
            accum[dst_key] = defaultdict(base_eval.ScalarAccumulator)
        for name, acc in tmp_gpsi[src_key].items():
            merge_scalar_accumulator(accum[dst_key][name], acc)
    if src_key in tmp_feat:
        if dst_key not in feature_accum:
            feature_accum[dst_key] = defaultdict(base_eval.BlockAccumulator)
        for name, acc in tmp_feat[src_key].items():
            merge_block_accumulator(feature_accum[dst_key][name], acc)


def summarize_raw(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    group_cols = ["eval_phase", "variant", "training_seed", "method_key", "method", "checkpoint", "checkpoint_path", "checkpoint_step", "checkpoint_label", "eval_seed", "scenario", "motion_mode", "threat_class"]
    out = (
        df.groupby(group_cols, dropna=False)
        .agg(
            steps=("step", "count"),
            raw_unsafe_action_rate=("raw_unsafe_action", "mean"),
            raw_safe_margin_unsafe_action_rate=("raw_safe_margin_unsafe_action", "mean"),
            raw_min_predicted_cpa=("raw_min_predicted_cpa", "mean"),
            raw_min_predicted_ttc=("raw_min_predicted_ttc", "mean"),
            action_norm=("action_norm", "mean"),
            action_delta=("action_delta", "mean"),
            no_response_rate=("no_response", "mean"),
        )
        .reset_index()
    )
    return out.to_dict("records")


def evaluate_episode(
    *,
    model: PPO,
    env: Any,
    cfg: dict[str, Any],
    prefix_base: dict[str, Any],
    scenario: str,
    eval_seed: int,
    episode_id: int,
    episode_seed: int,
    args: argparse.Namespace,
    runtime: SimpleNamespace,
    gpsi_accum: dict[tuple[Any, ...], dict[str, base_eval.ScalarAccumulator]],
    feature_accum: dict[tuple[Any, ...], dict[str, base_eval.BlockAccumulator]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    obs, info = env.reset(seed=episode_seed)
    done = False
    steps = 0
    episode_reward = 0.0
    min_distance_values: list[float] = []
    raw_rows: list[dict[str, Any]] = []
    last_info = info
    last_action = np.zeros(3, dtype=np.float32)
    raw_unsafe_count = 0
    raw_safe_unsafe_count = 0
    action_norms: list[float] = []
    action_deltas: list[float] = []
    no_response_values: list[float] = []
    cpa_values: list[float] = []

    while not done:
        flow_env = n3p_eval.flow_env_from(env)
        action, _ = model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32)
        prefix = {**prefix_base, "scenario": scenario, "model": model}
        raw = base_eval.raw_action_diagnostics(action, info, flow_env, runtime)
        action_norm = float(np.linalg.norm(action))
        action_delta = float(np.linalg.norm(action - last_action))
        no_response = int(action_norm < float(args.no_response_action_norm))
        action_norms.append(action_norm)
        action_deltas.append(action_delta)
        no_response_values.append(float(no_response))
        cpa_values.append(float(raw["raw_min_predicted_cpa"]))
        raw_unsafe_count += int(raw["raw_unsafe_action"])
        raw_safe_unsafe_count += int(raw["raw_safe_margin_unsafe_action"])
        raw_rows.append(
            {
                **{k: v for k, v in prefix_base.items() if k != "model"},
                "scenario": scenario,
                "episode_id": int(episode_id),
                "episode_seed": int(episode_seed),
                "eval_seed": int(eval_seed),
                "step": int(info.get("step", steps)),
                "time": float(info.get("time", steps * flow_env.dt)),
                "motion_mode": raw["raw_unsafe_motion_mode"],
                "threat_class": raw["raw_unsafe_threat_class"],
                "action_norm": action_norm,
                "action_delta": action_delta,
                "no_response": int(no_response),
                **raw,
            }
        )
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        steps += 1
        episode_reward += float(reward)
        min_distance_values.append(float(info["min_distance"]))
        last_info = info
        add_stab_diagnostics(accum=gpsi_accum, feature_accum=feature_accum, cfg=cfg, env=env, info=info, prefix=prefix)
        last_action = action.copy()

    near_miss = int(float(last_info["episode_min_distance"]) < args.near_miss_distance and not bool(last_info["is_collision"]))
    ep = {
        **{k: v for k, v in prefix_base.items() if k != "model"},
        "eval_seed": int(eval_seed),
        "scenario": scenario,
        "episode_id": int(episode_id),
        "episode_seed": int(episode_seed),
        "success": int(bool(last_info["is_success"])),
        "collision": int(bool(last_info["is_collision"])),
        "near_miss": near_miss,
        "mean_min_distance": float(np.mean(min_distance_values)) if min_distance_values else np.nan,
        "episode_min_distance": float(last_info["episode_min_distance"]),
        "episode_length": int(steps),
        "progress": float(last_info["progress"]),
        "planned_cpa": float(last_info.get("planned_cpa_to_threat", np.nan)),
        "planned_ttc": float(last_info.get("planned_ttc_to_threat", np.nan)),
        "threat_class": str(last_info.get("threat_class", "none")),
        "threat_motion_mode": str(last_info.get("threat_motion_mode", "none")),
        "episode_reward": float(episode_reward),
        "raw_unsafe_action_rate": float(raw_unsafe_count / max(steps, 1)),
        "raw_safe_margin_unsafe_action_rate": float(raw_safe_unsafe_count / max(steps, 1)),
        "raw_min_predicted_cpa": float(np.nanmean(cpa_values)) if cpa_values else np.nan,
        "action_norm": float(np.mean(action_norms)) if action_norms else np.nan,
        "action_delta": float(np.mean(action_deltas)) if action_deltas else np.nan,
        "no_response_rate": float(np.mean(no_response_values)) if no_response_values else np.nan,
        "nan_or_crash": 0,
    }
    return ep, raw_rows


def main() -> None:
    args = parse_args()
    result_dir = ROOT / args.result_dir
    table_prefix = str(args.table_prefix)
    ensure_dirs(result_dir)
    try:
        if args.eval_phase != "validation" and any(seed in {900, 901} for seed in args.eval_seeds):
            raise StabEvalStop("validation_test_leakage", f"{args.eval_phase} cannot use validation seeds: {args.eval_seeds}")
        if args.eval_phase == "validation" and any(seed not in {900, 901} for seed in args.eval_seeds):
            raise StabEvalStop("validation_test_leakage", f"validation phase can only use seeds 900/901, got {args.eval_seeds}")
        runs = [parse_run(item) for item in args.run]
        torch.manual_seed(0)
        np.random.seed(0)
        episode_rows: list[dict[str, Any]] = []
        raw_rows_all: list[dict[str, Any]] = []
        manifest_rows: list[dict[str, Any]] = []
        gpsi_accum: dict[tuple[Any, ...], dict[str, base_eval.ScalarAccumulator]] = {}
        feature_accum: dict[tuple[Any, ...], dict[str, base_eval.BlockAccumulator]] = {}
        cfg_by_method: dict[str, dict[str, Any]] = {}

        for run in runs:
            cfg = load_yaml(Path(run["config_path"]))
            cfg.setdefault("training", {})["seed"] = int(run["training_seed"])
            runtime = runtime_args(args, int(run["training_seed"]))
            labels = run.get("checkpoint_labels") or args.checkpoint_labels
            for label in labels:
                path, step = checkpoint_path(run, label)
                exists = path.exists() and path.stat().st_size > 0
                model_steps = np.nan
                if exists:
                    model_tmp = PPO.load(str(path), device="cpu")
                    model_steps = int(model_tmp.num_timesteps)
                    del model_tmp
                manifest_rows.append(
                    {
                        "eval_phase": args.eval_phase,
                        "variant": run["variant"],
                        "training_seed": int(run["training_seed"]),
                        "checkpoint_label": label,
                        "checkpoint_path": rel(path),
                        "exists": int(exists),
                        "size_bytes": int(path.stat().st_size) if exists else 0,
                        "sha256": sha256(path) if exists else "missing",
                        "checkpoint_step": int(step),
                        "model_num_timesteps": model_steps,
                    }
                )
                if not exists:
                    continue
                method_key = f"{run['variant']}_s{int(run['training_seed'])}_{label}"
                cfg_by_method[method_key] = cfg
                n3p_eval.DEFAULT_CONFIGS[method_key] = {"config": rel(Path(run["config_path"])), "checkpoint_dir": rel(Path(run["checkpoint_dir"]))}
                model = PPO.load(str(path), device=args.device)
                prefix_base = {
                    "eval_phase": args.eval_phase,
                    "variant": run["variant"],
                    "training_seed": int(run["training_seed"]),
                    "method_key": method_key,
                    "method": str(cfg.get("method_name", run["variant"])),
                    "checkpoint": method_key,
                    "checkpoint_path": rel(path),
                    "checkpoint_step": int(step),
                    "checkpoint_label": label,
                }
                for eval_seed in args.eval_seeds:
                    for scenario in args.scenarios:
                        env = n3p_eval.make_gpsi_env(cfg, scenario, runtime)
                        start = time.time()
                        last = start
                        print(
                            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] STAB_EVAL_START "
                            f"phase={args.eval_phase} method={method_key} eval_seed={eval_seed} scenario={scenario} episodes={args.num_episodes}",
                            flush=True,
                        )
                        for episode_id in range(int(args.num_episodes)):
                            episode_seed = int(eval_seed) + episode_id
                            ep, raw_rows = evaluate_episode(
                                model=model,
                                env=env,
                                cfg=cfg,
                                prefix_base=prefix_base,
                                scenario=scenario,
                                eval_seed=int(eval_seed),
                                episode_id=episode_id,
                                episode_seed=episode_seed,
                                args=args,
                                runtime=runtime,
                                gpsi_accum=gpsi_accum,
                                feature_accum=feature_accum,
                            )
                            episode_rows.append(ep)
                            raw_rows_all.extend(raw_rows)
                            now = time.time()
                            if now - last >= float(args.heartbeat_seconds) or episode_id == int(args.num_episodes) - 1:
                                done = episode_id + 1
                                print(
                                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] STAB_EVAL_HEARTBEAT "
                                    f"phase={args.eval_phase} method={method_key} eval_seed={eval_seed} scenario={scenario} "
                                    f"episodes={done}/{args.num_episodes} rate={done / max(now - start, 1e-6):.2f} ep/s",
                                    flush=True,
                                )
                                last = now
                        env.close()

        write_checkpoint_manifest(result_dir, table_prefix, manifest_rows)
        if not episode_rows:
            raise StabEvalStop("training_broken", "no checkpoint was available for eval")

        by_seed_cols = ["eval_phase", "variant", "training_seed", "method_key", "method", "checkpoint", "checkpoint_path", "checkpoint_step", "checkpoint_label", "eval_seed"]
        scenario_cols = by_seed_cols + ["scenario"]
        write_csv(result_dir / f"tables/{table_prefix}_{args.eval_phase}_episode_metrics.csv", episode_rows)
        write_csv(result_dir / f"tables/{table_prefix}_{args.eval_phase}_eval_summary_by_seed.csv", base_eval.summarize_episode_rows(episode_rows, by_seed_cols))
        write_csv(result_dir / f"tables/{table_prefix}_{args.eval_phase}_scenario_breakdown.csv", base_eval.summarize_episode_rows(episode_rows, scenario_cols))
        write_csv(result_dir / f"tables/{table_prefix}_{args.eval_phase}_motion_mode_breakdown.csv", base_eval.summarize_episode_rows(episode_rows, by_seed_cols + ["threat_motion_mode"]))
        write_csv(result_dir / f"tables/{table_prefix}_{args.eval_phase}_threat_class_breakdown.csv", base_eval.summarize_episode_rows(episode_rows, by_seed_cols + ["threat_class"]))
        if args.skip_raw_step_table:
            write_text(
                result_dir / f"tables/{table_prefix}_{args.eval_phase}_raw_unsafe_action_steps.README.txt",
                "Per-step raw unsafe table was intentionally not written for this phase to avoid large generated CSVs; use the summary table instead.\n",
            )
        else:
            write_csv(result_dir / f"tables/{table_prefix}_{args.eval_phase}_raw_unsafe_action_steps.csv", raw_rows_all)
        write_csv(result_dir / f"tables/{table_prefix}_{args.eval_phase}_raw_unsafe_action_summary.csv", summarize_raw(raw_rows_all))
        write_csv(result_dir / f"tables/{table_prefix}_{args.eval_phase}_gpsi_output_summary.csv", gpsi_summary_rows(gpsi_accum))
        write_csv(result_dir / f"tables/{table_prefix}_{args.eval_phase}_feature_block_stats.csv", feature_summary_rows(feature_accum, cfg_by_method))
        append_csv(
            result_dir / f"tables/{table_prefix}_eval_command_manifest.csv",
            [
                {
                    "eval_phase": args.eval_phase,
                    "command": " ".join(["python", *sys.argv]),
                    "runs": json.dumps(args.run),
                    "checkpoint_labels": json.dumps(args.checkpoint_labels),
                    "eval_seeds": json.dumps([int(s) for s in args.eval_seeds]),
                    "num_episodes": int(args.num_episodes),
                    "scenarios": json.dumps(args.scenarios),
                    "device": args.device,
                }
            ],
        )
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] STAB_EVAL_END phase={args.eval_phase} episodes={len(episode_rows)} raw_steps={len(raw_rows_all)}",
            flush=True,
        )
    except StabEvalStop as exc:
        write_stop(result_dir, args, exc.reason, exc.detail)
        raise SystemExit(2) from exc
    except Exception:
        detail = traceback.format_exc()
        write_stop(result_dir, args, "eval_failed", detail)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
