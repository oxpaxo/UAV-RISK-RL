from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.eval_env_v2_gpsi_ppo_n3fz as base_eval
import scripts.eval_env_v2_gpsi_ppo_n3p as n3p_eval
from envs.dynamic_obstacle_flow_env import DynamicObstacleFlowEnv


STOP_FLAGS = {
    "parent_missing": "PHASE_N3PF_STOP_PARENT_MISSING.flag",
    "resume_semantics_invalid": "PHASE_N3PF_STOP_RESUME_SEMANTICS_INVALID.flag",
    "config_mismatch": "PHASE_N3PF_STOP_CONFIG_MISMATCH.flag",
    "train_failed": "PHASE_N3PF_STOP_TRAIN_FAILED.flag",
    "eval_failed": "PHASE_N3PF_STOP_EVAL_FAILED.flag",
    "diagnostics_failed": "PHASE_N3PF_STOP_DIAGNOSTICS_FAILED.flag",
    "checkpoint_integrity_failed": "PHASE_N3PF_STOP_CHECKPOINT_INTEGRITY_FAILED.flag",
    "watcher_failed": "PHASE_N3PF_STOP_WATCHER_FAILED.flag",
}


class N3PFEvalStop(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Phase N3PF block-projected full checkpoints.")
    parser.add_argument("--result-dir", default="results/env_v2_phase_n3pf_block_projected_full")
    parser.add_argument("--checkpoint-dir", default="checkpoints/env_v2_gpsi_heada_ppo_n3pf_block_projected_s0")
    parser.add_argument("--noz-reference-dir", default="checkpoints/env_v2_gpsi_heada_ppo_n3f_no_z_s0")
    parser.add_argument("--z2-reference-dir", default="checkpoints/env_v2_gpsi_heada_ppo_n3z2cf_layernorm_alpha0p5_s0")
    parser.add_argument("--attention-checkpoint", default="checkpoints/env_v2_phase2/attention_full_s0/attention_full_s0_step1500000.zip")
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--scenarios", nargs="+", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=64)
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


def ensure_dirs(result_dir: Path) -> None:
    for path in [
        result_dir,
        result_dir / "logs",
        result_dir / "tables",
        result_dir / "plots",
        result_dir / "traces",
        result_dir / "traces/sampled_success_traces",
        result_dir / "traces/sampled_collision_traces",
        result_dir / "traces/sampled_near_miss_traces",
        result_dir / "traces/high_raw_unsafe_rate_traces",
    ]:
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
    ensure_dirs(result_dir)
    flag = STOP_FLAGS.get(reason, STOP_FLAGS["eval_failed"])
    write_text(result_dir / flag, f"{reason}\n{detail.strip()}\n")
    write_text(result_dir / "phase_n3pf_status.txt", f"stopped:{flag}\n")
    write_text(
        result_dir / "PHASE_N3PF_BLOCK_PROJECTED_FULL_REPORT.md",
        "\n".join(
            [
                "# Phase N3PF Block-Projected Full Report",
                "",
                f"`terminal_decision = phase_n3pf_stopped_{reason}`",
                "",
                "Partial report generated by N3PF eval.",
                "",
                "```text",
                detail.strip(),
                "```",
                "",
                "Can enter N4: no.",
            ]
        )
        + "\n",
    )


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise N3PFEvalStop("config_mismatch", f"missing config: {rel(path)}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise N3PFEvalStop("config_mismatch", f"config is not a mapping: {rel(path)}")
    return payload


def checkpoint_jobs(args: argparse.Namespace) -> list[tuple[str, str, dict[str, Any] | None, str, Path, int, str]]:
    p3_cfg = load_yaml(ROOT / "configs/env_v2_gpsi_heada_ppo_n3pf_block_projected.yaml")
    noz_cfg = load_yaml(ROOT / "configs/env_v2_gpsi_heada_ppo_n3f_no_z_full.yaml")
    z2_cfg = load_yaml(ROOT / "configs/env_v2_gpsi_heada_ppo_n3z2cf_layernorm_alpha0p5.yaml")
    ckpt = ROOT / args.checkpoint_dir
    noz = ROOT / args.noz_reference_dir
    z2 = ROOT / args.z2_reference_dir
    attention = ROOT / args.attention_checkpoint
    jobs: list[tuple[str, str, dict[str, Any] | None, str, Path, int, str]] = [
        ("attention_full", "attention_full_1500k", None, "attention_full_1500k", attention, 1_500_000, "attention_full_1500k"),
        ("n3f_no_z_full", "gpsi_no_z_full_1500k", noz_cfg, "n3f_no_z_full_final", noz / "final.zip", 1_500_000, "final"),
        ("z2_corrected_full", "gpsi_z_layernorm_alpha_0p5_corrected_full_1p5m", z2_cfg, "z2_corrected_full_final", z2 / "final.zip", 1_500_000, "z2_final"),
    ]
    for label, filename, step in [
        ("parent_500k", "parent_500k.zip", 500_000),
        ("750k", "checkpoint_750k.zip", 750_000),
        ("1000k", "checkpoint_1000k.zip", 1_000_000),
        ("1250k", "checkpoint_1250k.zip", 1_250_000),
        ("1500k", "checkpoint_1500k.zip", 1_500_000),
        ("final", "final.zip", 1_500_000),
        ("best_by_eval", "best_by_eval.zip", 1_500_000),
    ]:
        jobs.append(("block_projected_full", "n3pf_block_projected_no_z_full_1p5m", p3_cfg, f"block_projected_full_{label}", ckpt / filename, step, label))
    return jobs


def register_n3p_eval_checkpoint_paths(args: argparse.Namespace) -> None:
    n3p_eval.DEFAULT_CONFIGS["block_projected_full"] = {
        "config": "configs/env_v2_gpsi_heada_ppo_n3pf_block_projected.yaml",
        "checkpoint_dir": args.checkpoint_dir,
    }
    n3p_eval.DEFAULT_CONFIGS["n3f_no_z_full"] = {
        "config": "configs/env_v2_gpsi_heada_ppo_n3f_no_z_full.yaml",
        "checkpoint_dir": args.noz_reference_dir,
    }
    n3p_eval.DEFAULT_CONFIGS["z2_corrected_full"] = {
        "config": "configs/env_v2_gpsi_heada_ppo_n3z2cf_layernorm_alpha0p5.yaml",
        "checkpoint_dir": args.z2_reference_dir,
    }


def runtime_cfg_for_eval(method_key: str, cfg: dict[str, Any]) -> dict[str, Any]:
    if method_key == "block_projected_full":
        return {**cfg, "method_key": "block_projected"}
    return cfg


def main() -> None:
    args = parse_args()
    result_dir = ROOT / args.result_dir
    ensure_dirs(result_dir)
    try:
        append_csv(
            result_dir / "tables/phase_n3pf_command_manifest.csv",
            [
                {
                    "stage": "eval",
                    "command": " ".join(["python", *sys.argv]),
                    "episodes": int(args.num_episodes),
                    "scenarios": json.dumps(args.scenarios),
                    "eval_seed": int(args.eval_seed),
                    "checkpoint_dir": rel(ROOT / args.checkpoint_dir),
                    "noz_reference_dir": rel(ROOT / args.noz_reference_dir),
                    "z2_reference_dir": rel(ROOT / args.z2_reference_dir),
                    "attention_checkpoint": rel(ROOT / args.attention_checkpoint),
                }
            ],
        )
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        register_n3p_eval_checkpoint_paths(args)
        jobs = checkpoint_jobs(args)
        for method_key, _method, _cfg, _checkpoint_name, checkpoint_path, _step, _label in jobs:
            if not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0:
                raise N3PFEvalStop("eval_failed", f"missing checkpoint for {method_key}: {rel(checkpoint_path)}")

        cfg_by_method = {
            "block_projected_full": load_yaml(ROOT / "configs/env_v2_gpsi_heada_ppo_n3pf_block_projected.yaml"),
            "n3f_no_z_full": load_yaml(ROOT / "configs/env_v2_gpsi_heada_ppo_n3f_no_z_full.yaml"),
            "z2_corrected_full": load_yaml(ROOT / "configs/env_v2_gpsi_heada_ppo_n3z2cf_layernorm_alpha0p5.yaml"),
        }
        config_rows: list[dict[str, Any]] = []
        for key, cfg in cfg_by_method.items():
            config_rows.append(
                {
                    "method_key": key,
                    "method": cfg.get("method_name", key),
                    "obs_aug_dim": int(cfg.get("gpsi", {}).get("obs_aug_dim", -1)),
                    "include_z": int(bool(cfg.get("gpsi", {}).get("include_z", False))),
                    "include_logvar": int(bool(cfg.get("gpsi", {}).get("include_logvar", True))),
                    "logvar_output_scale": float(cfg.get("gpsi", {}).get("logvar_output_scale", 1.0)),
                    "feature_adapter": str(cfg.get("ppo", {}).get("feature_adapter", "raw_concat")),
                    "no_shield": int(bool(cfg.get("training", {}).get("no_shield", True))),
                    "action_filtering": int(bool(cfg.get("training", {}).get("action_filtering", False))),
                    "use_safety_cost": int(bool(cfg.get("training", {}).get("use_safety_cost", False))),
                }
            )
        write_csv(result_dir / "tables/phase_n3pf_config_manifest.csv", config_rows)

        all_episode_rows: list[dict[str, Any]] = []
        all_raw_rows: list[dict[str, Any]] = []
        all_trace_files: list[tuple[Path, dict[str, Any]]] = []
        gpsi_accum: dict[tuple[Any, ...], dict[str, base_eval.ScalarAccumulator]] = {}
        feature_accum: dict[tuple[Any, ...], dict[str, base_eval.BlockAccumulator]] = {}

        for method_key, method, cfg, checkpoint_name, checkpoint_path, step_value, checkpoint_label in jobs:
            if method_key == "attention_full":
                model = PPO.load(str(checkpoint_path), device=args.device, custom_objects={"policy_kwargs": base_eval.policy_kwargs_attention(args.hidden_dim)})
            else:
                model = PPO.load(str(checkpoint_path), device=args.device)
            for scenario in args.scenarios:
                eval_cfg = None if cfg is None else runtime_cfg_for_eval(method_key, cfg)
                env = DynamicObstacleFlowEnv(scenario=scenario) if eval_cfg is None else n3p_eval.make_gpsi_env(eval_cfg, scenario, args)
                start = time.time()
                last = start
                print(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] N3PF_EVAL_START "
                    f"method={method_key} checkpoint={checkpoint_label} scenario={scenario} episodes={args.num_episodes}",
                    flush=True,
                )
                for episode_id in range(args.num_episodes):
                    seed = int(args.eval_seed) + int(args.seed) * 10000 + episode_id
                    if cfg is None:
                        ep_row, raw_rows, _trace_rows, trace_files = base_eval.evaluate_episode(
                            model=model,
                            env=env,
                            cfg=None,
                            method_key=method_key,
                            method=method,
                            checkpoint_name=checkpoint_name,
                            checkpoint_step_value=step_value,
                            checkpoint_label=checkpoint_label,
                            scenario=scenario,
                            episode_id=episode_id,
                            episode_seed=seed,
                            args=args,
                            gpsi_accum=gpsi_accum,
                            feature_accum=feature_accum,
                        )
                    else:
                        ep_row, raw_rows, _trace_rows, trace_files = n3p_eval.evaluate_episode(
                            model=model,
                            env=env,
                            cfg=eval_cfg,
                            method_key=method_key,
                            method=method,
                            checkpoint_name=checkpoint_name,
                            checkpoint_step_value=step_value,
                            checkpoint_label=checkpoint_label,
                            scenario=scenario,
                            episode_id=episode_id,
                            episode_seed=seed,
                            args=args,
                            gpsi_accum=gpsi_accum,
                            feature_accum=feature_accum,
                        )
                    ep_row["checkpoint_path"] = rel(checkpoint_path)
                    for row in raw_rows:
                        row["checkpoint_path"] = rel(checkpoint_path)
                    all_episode_rows.append(ep_row)
                    all_raw_rows.extend(raw_rows)
                    all_trace_files.extend(trace_files)
                    now = time.time()
                    if now - last >= args.heartbeat_seconds or episode_id == args.num_episodes - 1:
                        done = episode_id + 1
                        rate = done / max(now - start, 1e-6)
                        eta = (args.num_episodes - done) / max(rate, 1e-6)
                        print(
                            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] N3PF_EVAL_HEARTBEAT "
                            f"method={method_key} checkpoint={checkpoint_label} scenario={scenario} "
                            f"episodes={done}/{args.num_episodes} rate={rate:.2f} ep/s eta={eta/60.0:.2f} min",
                            flush=True,
                        )
                        last = now
                env.close()

        group_cols = ["method_key", "method", "checkpoint", "checkpoint_path", "checkpoint_step", "checkpoint_label", "scenario"]
        eval_summary = base_eval.summarize_episode_rows(all_episode_rows, group_cols)
        write_csv(result_dir / "tables/phase_n3pf_episode_metrics.csv", all_episode_rows)
        write_csv(result_dir / "tables/phase_n3pf_checkpoint_eval_summary.csv", eval_summary)
        write_csv(result_dir / "tables/phase_n3pf_eval_summary.csv", eval_summary)
        write_csv(result_dir / "tables/phase_n3pf_scenario_breakdown.csv", eval_summary)
        write_csv(
            result_dir / "tables/phase_n3pf_motion_mode_breakdown.csv",
            base_eval.summarize_episode_rows(all_episode_rows, ["method_key", "method", "checkpoint", "checkpoint_path", "checkpoint_step", "checkpoint_label", "threat_motion_mode"]),
        )
        write_csv(
            result_dir / "tables/phase_n3pf_threat_class_breakdown.csv",
            base_eval.summarize_episode_rows(all_episode_rows, ["method_key", "method", "checkpoint", "checkpoint_path", "checkpoint_step", "checkpoint_label", "threat_class"]),
        )
        write_csv(result_dir / "tables/phase_n3pf_raw_unsafe_action_steps.csv", all_raw_rows)
        write_csv(result_dir / "tables/phase_n3pf_raw_unsafe_action_summary.csv", base_eval.summarize_raw(all_raw_rows))
        write_csv(result_dir / "tables/phase_n3pf_gpsi_output_summary.csv", base_eval.gpsi_summary_rows(gpsi_accum))
        write_csv(result_dir / "tables/phase_n3pf_feature_block_stats.csv", n3p_eval.feature_summary_rows(feature_accum, cfg_by_method))
        base_eval.copy_sample_traces(result_dir, all_trace_files)
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] N3PF_EVAL_END "
            f"episodes={len(all_episode_rows)} raw_steps={len(all_raw_rows)} gpsi_groups={len(gpsi_accum)} traces={len(all_trace_files)}",
            flush=True,
        )
    except N3PFEvalStop as exc:
        write_stop(result_dir, exc.reason, exc.detail)
        raise SystemExit(2) from exc
    except Exception:
        write_stop(result_dir, "eval_failed", traceback.format_exc())
        raise SystemExit(2)


if __name__ == "__main__":
    main()
