from __future__ import annotations

import argparse
import csv
import hashlib
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


SCENARIOS = [
    "eval_flow_id",
    "eval_flow_high_density",
    "eval_flow_high_speed",
    "eval_flow_high_threat",
    "eval_flow_mixed_ood",
    "eval_flow_sudden_threat",
]

STOP_FLAGS = {
    "required_checkpoint_missing": "PHASE_N3PFV_STOP_REQUIRED_CHECKPOINT_MISSING.flag",
    "reference_checkpoint_missing": "PHASE_N3PFV_STOP_REFERENCE_CHECKPOINT_MISSING.flag",
    "eval_failed": "PHASE_N3PFV_STOP_EVAL_FAILED.flag",
    "schema_invalid": "PHASE_N3PFV_STOP_SCHEMA_INVALID.flag",
    "diagnostics_failed": "PHASE_N3PFV_STOP_DIAGNOSTICS_FAILED.flag",
    "watcher_failed": "PHASE_N3PFV_STOP_WATCHER_FAILED.flag",
}


class N3PFVEvalStop(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase N3PF-V multi-seed checkpoint verification eval.")
    parser.add_argument("--result-dir", default="results/env_v2_phase_n3pfv_checkpoint_verification")
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[1000, 1001, 1002])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    parser.add_argument("--policies", nargs="+", default=["p3_1000k", "p3_1500k", "p3_final", "attention_full", "no_z_full", "z2_corrected_full"])
    parser.add_argument("--include-diagnostic-policies", nargs="*", default=["p3_parent_500k", "p3_1250k"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--write-traces", action="store_true")
    parser.add_argument("--near-miss-distance", type=float, default=1.5)
    parser.add_argument("--raw-cpa-horizon", type=float, default=4.5)
    parser.add_argument("--raw-cpa-threshold", type=float, default=1.2)
    parser.add_argument("--raw-cpa-safe-threshold", type=float, default=1.5)
    parser.add_argument("--no-response-action-norm", type=float, default=0.05)
    parser.add_argument("--heartbeat-seconds", type=float, default=300.0)
    parser.add_argument("--degraded-reason", default="")
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
    write_text(result_dir / "phase_n3pfv_status.txt", f"stopped:{flag}\n")
    write_text(
        result_dir / "PHASE_N3PFV_CHECKPOINT_VERIFICATION_REPORT.md",
        "\n".join(
            [
                "# Phase N3PF-V Checkpoint Verification Report",
                "",
                f"`terminal_decision = phase_n3pfv_stopped_{reason}`",
                "",
                "Partial report generated by N3PF-V eval.",
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise N3PFVEvalStop("schema_invalid", f"missing config: {rel(path)}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise N3PFVEvalStop("schema_invalid", f"config is not a mapping: {rel(path)}")
    return payload


def policy_specs() -> dict[str, dict[str, Any]]:
    return {
        "p3_1000k": {
            "method": "P3_block_projected_1000k",
            "cfg": "configs/env_v2_gpsi_heada_ppo_n3pf_block_projected.yaml",
            "path": "checkpoints/env_v2_gpsi_heada_ppo_n3pf_block_projected_s0/checkpoint_1000k.zip",
            "checkpoint_label": "1000k",
            "checkpoint_step": 1_000_000,
            "source_phase": "N3PF",
            "expected_success": 0.6300,
            "expected_collision": 0.3700,
            "required": 1,
            "diagnostic": 0,
        },
        "p3_1500k": {
            "method": "P3_block_projected_1500k",
            "cfg": "configs/env_v2_gpsi_heada_ppo_n3pf_block_projected.yaml",
            "path": "checkpoints/env_v2_gpsi_heada_ppo_n3pf_block_projected_s0/checkpoint_1500k.zip",
            "checkpoint_label": "1500k",
            "checkpoint_step": 1_500_000,
            "source_phase": "N3PF",
            "expected_success": 0.6200,
            "expected_collision": 0.3800,
            "required": 1,
            "diagnostic": 0,
        },
        "p3_final": {
            "method": "P3_block_projected_final",
            "cfg": "configs/env_v2_gpsi_heada_ppo_n3pf_block_projected.yaml",
            "path": "checkpoints/env_v2_gpsi_heada_ppo_n3pf_block_projected_s0/final.zip",
            "checkpoint_label": "final",
            "checkpoint_step": 1_500_000,
            "source_phase": "N3PF",
            "expected_success": 0.5967,
            "expected_collision": 0.4033,
            "required": 1,
            "diagnostic": 0,
        },
        "attention_full": {
            "method": "attention_full_1500k",
            "cfg": None,
            "path": "checkpoints/env_v2_phase2/attention_full_s0/attention_full_s0_step1500000.zip",
            "checkpoint_label": "attention_full_1500k",
            "checkpoint_step": 1_500_000,
            "source_phase": "Phase2",
            "expected_success": 0.6100,
            "expected_collision": 0.3900,
            "required": 1,
            "diagnostic": 0,
            "reference": 1,
        },
        "no_z_full": {
            "method": "N3F_no_z_full",
            "cfg": "configs/env_v2_gpsi_heada_ppo_n3f_no_z_full.yaml",
            "path": "checkpoints/env_v2_gpsi_heada_ppo_n3f_no_z_s0/final.zip",
            "checkpoint_label": "final",
            "checkpoint_step": 1_500_000,
            "source_phase": "N3FZ",
            "expected_success": 0.5633,
            "expected_collision": 0.4367,
            "required": 1,
            "diagnostic": 0,
            "reference": 1,
        },
        "z2_corrected_full": {
            "method": "N3Z2CF_corrected_Z2_full",
            "cfg": "configs/env_v2_gpsi_heada_ppo_n3z2cf_layernorm_alpha0p5.yaml",
            "path": "checkpoints/env_v2_gpsi_heada_ppo_n3z2cf_layernorm_alpha0p5_s0/final.zip",
            "checkpoint_label": "z2_final",
            "checkpoint_step": 1_500_000,
            "source_phase": "N3Z2CF",
            "expected_success": 0.5067,
            "expected_collision": 0.4933,
            "required": 1,
            "diagnostic": 0,
            "reference": 1,
        },
        "p3_parent_500k": {
            "method": "P3_block_projected_parent_500k",
            "cfg": "configs/env_v2_gpsi_heada_ppo_n3pf_block_projected.yaml",
            "path": "checkpoints/env_v2_gpsi_heada_ppo_n3p_block_projected_s0/checkpoint_500k.zip",
            "checkpoint_label": "parent_500k",
            "checkpoint_step": 500_000,
            "source_phase": "N3P",
            "expected_success": 0.5333,
            "expected_collision": 0.4667,
            "required": 0,
            "diagnostic": 1,
        },
        "p3_1250k": {
            "method": "P3_block_projected_1250k",
            "cfg": "configs/env_v2_gpsi_heada_ppo_n3pf_block_projected.yaml",
            "path": "checkpoints/env_v2_gpsi_heada_ppo_n3pf_block_projected_s0/checkpoint_1250k.zip",
            "checkpoint_label": "1250k",
            "checkpoint_step": 1_250_000,
            "source_phase": "N3PF",
            "expected_success": 0.4467,
            "expected_collision": 0.5533,
            "required": 0,
            "diagnostic": 1,
        },
    }


def selected_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs = policy_specs()
    keys: list[str] = []
    for key in list(args.policies) + list(args.include_diagnostic_policies):
        if key not in specs:
            raise N3PFVEvalStop("schema_invalid", f"unknown policy key: {key}")
        if key not in keys:
            keys.append(key)
    out = []
    for key in keys:
        spec = dict(specs[key])
        spec["policy_key"] = key
        spec["checkpoint_path"] = ROOT / str(spec["path"])
        spec["config_path"] = ROOT / str(spec["cfg"]) if spec.get("cfg") else None
        spec["selected_for_required_eval"] = int(key in set(args.policies))
        spec["selected_for_diagnostic_eval"] = int(key in set(args.include_diagnostic_policies))
        out.append(spec)
    return out


def write_checkpoint_manifest(result_dir: Path, specs: list[dict[str, Any]]) -> None:
    rows = []
    for spec in specs:
        path = Path(spec["checkpoint_path"])
        exists = path.exists() and path.stat().st_size > 0
        rows.append(
            {
                "policy_key": spec["policy_key"],
                "checkpoint_label": spec["checkpoint_label"],
                "checkpoint_path": rel(path),
                "exists": int(exists),
                "size_bytes": int(path.stat().st_size) if exists else 0,
                "sha256": sha256(path) if exists else "missing",
                "source_phase": spec["source_phase"],
                "expected_success_single_seed_if_known": spec["expected_success"],
                "expected_collision_single_seed_if_known": spec["expected_collision"],
                "selected_for_required_eval": int(spec["selected_for_required_eval"]),
                "selected_for_diagnostic_eval": int(spec["selected_for_diagnostic_eval"]),
            }
        )
    write_csv(result_dir / "tables/phase_n3pfv_checkpoint_manifest.csv", rows)


def validate_manifest(specs: list[dict[str, Any]]) -> None:
    missing_required = []
    missing_reference = []
    for spec in specs:
        path = Path(spec["checkpoint_path"])
        if path.exists() and path.stat().st_size > 0:
            continue
        if int(spec.get("selected_for_required_eval", 0)) and int(spec.get("reference", 0)):
            missing_reference.append(f"{spec['policy_key']}: {rel(path)}")
        elif int(spec.get("selected_for_required_eval", 0)):
            missing_required.append(f"{spec['policy_key']}: {rel(path)}")
    if missing_required:
        raise N3PFVEvalStop("required_checkpoint_missing", "\n".join(missing_required))
    if missing_reference:
        raise N3PFVEvalStop("reference_checkpoint_missing", "\n".join(missing_reference))


def register_n3p_paths(specs: list[dict[str, Any]]) -> None:
    for spec in specs:
        if spec.get("config_path") is None:
            continue
        checkpoint_path = Path(spec["checkpoint_path"])
        n3p_eval.DEFAULT_CONFIGS[str(spec["policy_key"])] = {
            "config": rel(Path(spec["config_path"])),
            "checkpoint_dir": rel(checkpoint_path.parent),
        }


def runtime_cfg(spec: dict[str, Any]) -> dict[str, Any] | None:
    if spec.get("config_path") is None:
        return None
    cfg = load_yaml(Path(spec["config_path"]))
    if str(spec["policy_key"]).startswith("p3_"):
        cfg = {**cfg, "method_key": "block_projected"}
    return cfg


def write_schema_check(result_dir: Path, specs: list[dict[str, Any]]) -> None:
    rows = []
    for spec in specs:
        cfg = runtime_cfg(spec)
        if cfg is None:
            rows.append(
                {
                    "policy_key": spec["policy_key"],
                    "checkpoint_label": spec["checkpoint_label"],
                    "config": "",
                    "obs_aug_dim": 12,
                    "include_z": "",
                    "include_logvar": "",
                    "feature_adapter": "attention_full",
                    "schema_ok": 1,
                }
            )
            continue
        gpsi = cfg.get("gpsi", {})
        ppo = cfg.get("ppo", {})
        rows.append(
            {
                "policy_key": spec["policy_key"],
                "checkpoint_label": spec["checkpoint_label"],
                "config": rel(Path(spec["config_path"])),
                "obs_aug_dim": int(gpsi.get("obs_aug_dim", -1)),
                "include_z": int(bool(gpsi.get("include_z", False))),
                "include_logvar": int(bool(gpsi.get("include_logvar", True))),
                "logvar_output_scale": float(gpsi.get("logvar_output_scale", 1.0)),
                "feature_adapter": str(ppo.get("feature_adapter", "raw_concat")),
                "no_shield": int(bool(cfg.get("training", {}).get("no_shield", True))),
                "action_filtering": int(bool(cfg.get("training", {}).get("action_filtering", False))),
                "use_safety_cost": int(bool(cfg.get("training", {}).get("use_safety_cost", False))),
                "schema_ok": int(
                    (not str(spec["policy_key"]).startswith("p3_") or (int(gpsi.get("obs_aug_dim", -1)) == 30 and str(ppo.get("feature_adapter", "")) == "block_projected_no_z"))
                    and not bool(cfg.get("training", {}).get("action_filtering", False))
                    and not bool(cfg.get("training", {}).get("use_safety_cost", False))
                ),
            }
        )
    write_csv(result_dir / "tables/phase_n3pfv_schema_check.csv", rows)
    bad = [row for row in rows if int(row["schema_ok"]) != 1]
    if bad:
        raise N3PFVEvalStop("schema_invalid", json.dumps(bad, indent=2))


def summarize_episode_rows(rows: list[dict[str, Any]], group_cols: list[str]) -> list[dict[str, Any]]:
    return base_eval.summarize_episode_rows(rows, group_cols)


def summarize_raw_by_seed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    import pandas as pd

    df = pd.DataFrame(rows)
    group_cols = ["method_key", "method", "checkpoint", "checkpoint_path", "checkpoint_step", "checkpoint_label", "eval_seed", "scenario", "motion_mode", "threat_class"]
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
    block_names = [
        "obs_i_12",
        "delta_hat_9_after_scale",
        "logvar_raw_9_clamped",
        "logvar_scaled_9_policy",
        "adapter_output_64",
        "full_aug_obs",
    ]
    for key, groups in feature_accum.items():
        method_key, method, checkpoint, step, label, scenario, motion_mode, threat_class = key
        prefix = {
            "method_key": method_key,
            "method": method,
            "checkpoint": checkpoint,
            "checkpoint_step": int(step),
            "checkpoint_label": label,
            "scenario": scenario,
            "motion_mode": motion_mode,
            "threat_class": threat_class,
        }
        gpsi_cfg = cfg_by_method.get(str(method_key), {}).get("gpsi", {})
        prefix.update(
            {
                "include_z": int(bool(gpsi_cfg.get("include_z", False))),
                "include_logvar": int(bool(gpsi_cfg.get("include_logvar", True))),
                "logvar_output_scale": float(gpsi_cfg.get("logvar_output_scale", 1.0)),
            }
        )
        for block in block_names:
            if block in groups:
                rows.append(groups[block].row(prefix, block))
            else:
                rows.append(base_eval.BlockAccumulator().row(prefix, block, not_applicable=True))
    return rows


def main() -> None:
    args = parse_args()
    result_dir = ROOT / args.result_dir
    ensure_dirs(result_dir)
    try:
        specs = selected_specs(args)
        write_checkpoint_manifest(result_dir, specs)
        validate_manifest(specs)
        write_schema_check(result_dir, specs)
        register_n3p_paths(specs)
        degraded = len(args.eval_seeds) < 3
        append_csv(
            result_dir / "tables/phase_n3pfv_command_manifest.csv",
            [
                {
                    "stage": "eval",
                    "command": " ".join(["python", *sys.argv]),
                    "eval_seeds": json.dumps([int(seed) for seed in args.eval_seeds]),
                    "num_episodes": int(args.num_episodes),
                    "scenarios": json.dumps(args.scenarios),
                    "policies": json.dumps(args.policies),
                    "diagnostic_policies": json.dumps(args.include_diagnostic_policies),
                    "verification_eval_degraded": int(degraded),
                    "degraded_reason": args.degraded_reason if degraded else "",
                    "device": args.device,
                    "write_traces": int(bool(args.write_traces)),
                }
            ],
        )
        torch.manual_seed(0)
        np.random.seed(0)
        all_episode_rows: list[dict[str, Any]] = []
        all_raw_rows: list[dict[str, Any]] = []
        all_trace_files: list[tuple[Path, dict[str, Any]]] = []
        gpsi_accum: dict[tuple[Any, ...], dict[str, base_eval.ScalarAccumulator]] = {}
        feature_accum: dict[tuple[Any, ...], dict[str, base_eval.BlockAccumulator]] = {}
        cfg_by_method: dict[str, dict[str, Any]] = {}

        for spec in specs:
            cfg = runtime_cfg(spec)
            if cfg is not None:
                cfg_by_method[str(spec["policy_key"])] = cfg
            checkpoint_path = Path(spec["checkpoint_path"])
            if cfg is None:
                model = PPO.load(str(checkpoint_path), device=args.device, custom_objects={"policy_kwargs": base_eval.policy_kwargs_attention(args.hidden_dim)})
            else:
                model = PPO.load(str(checkpoint_path), device=args.device)
            for eval_seed in args.eval_seeds:
                for scenario in args.scenarios:
                    env = DynamicObstacleFlowEnv(scenario=scenario) if cfg is None else n3p_eval.make_gpsi_env(cfg, scenario, args)
                    start = time.time()
                    last = start
                    print(
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] N3PFV_EVAL_START "
                        f"policy={spec['policy_key']} checkpoint={spec['checkpoint_label']} seed={eval_seed} "
                        f"scenario={scenario} episodes={args.num_episodes}",
                        flush=True,
                    )
                    for episode_id in range(int(args.num_episodes)):
                        episode_seed = int(eval_seed) + episode_id
                        episode_id_global = (int(eval_seed) - min(args.eval_seeds)) * int(args.num_episodes) + episode_id
                        if cfg is None:
                            ep_row, raw_rows, _trace_rows, trace_files = base_eval.evaluate_episode(
                                model=model,
                                env=env,
                                cfg=None,
                                method_key=str(spec["policy_key"]),
                                method=str(spec["method"]),
                                checkpoint_name=str(spec["method"]),
                                checkpoint_step_value=int(spec["checkpoint_step"]),
                                checkpoint_label=str(spec["checkpoint_label"]),
                                scenario=scenario,
                                episode_id=episode_id_global,
                                episode_seed=episode_seed,
                                args=args,
                                gpsi_accum=gpsi_accum,
                                feature_accum=feature_accum,
                            )
                        else:
                            ep_row, raw_rows, _trace_rows, trace_files = n3p_eval.evaluate_episode(
                                model=model,
                                env=env,
                                cfg=cfg,
                                method_key=str(spec["policy_key"]),
                                method=str(spec["method"]),
                                checkpoint_name=str(spec["method"]),
                                checkpoint_step_value=int(spec["checkpoint_step"]),
                                checkpoint_label=str(spec["checkpoint_label"]),
                                scenario=scenario,
                                episode_id=episode_id_global,
                                episode_seed=episode_seed,
                                args=args,
                                gpsi_accum=gpsi_accum,
                                feature_accum=feature_accum,
                            )
                        ep_row["eval_seed"] = int(eval_seed)
                        ep_row["checkpoint_path"] = rel(checkpoint_path)
                        ep_row["source_phase"] = spec["source_phase"]
                        ep_row["selected_for_required_eval"] = int(spec["selected_for_required_eval"])
                        ep_row["selected_for_diagnostic_eval"] = int(spec["selected_for_diagnostic_eval"])
                        for row in raw_rows:
                            row["eval_seed"] = int(eval_seed)
                            row["checkpoint_path"] = rel(checkpoint_path)
                            row["source_phase"] = spec["source_phase"]
                        all_episode_rows.append(ep_row)
                        all_raw_rows.extend(raw_rows)
                        all_trace_files.extend(trace_files)
                        now = time.time()
                        if now - last >= float(args.heartbeat_seconds) or episode_id == int(args.num_episodes) - 1:
                            done = episode_id + 1
                            rate = done / max(now - start, 1e-6)
                            print(
                                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] N3PFV_EVAL_HEARTBEAT "
                                f"policy={spec['policy_key']} seed={eval_seed} scenario={scenario} "
                                f"episodes={done}/{args.num_episodes} rate={rate:.2f} ep/s",
                                flush=True,
                            )
                            last = now
                    env.close()

        by_seed_cols = ["method_key", "method", "checkpoint", "checkpoint_path", "checkpoint_step", "checkpoint_label", "eval_seed"]
        scenario_cols = by_seed_cols + ["scenario"]
        write_csv(result_dir / "tables/phase_n3pfv_episode_metrics.csv", all_episode_rows)
        write_csv(result_dir / "tables/phase_n3pfv_eval_summary_by_seed.csv", summarize_episode_rows(all_episode_rows, by_seed_cols))
        write_csv(result_dir / "tables/phase_n3pfv_scenario_breakdown.csv", summarize_episode_rows(all_episode_rows, scenario_cols))
        write_csv(
            result_dir / "tables/phase_n3pfv_motion_mode_breakdown.csv",
            summarize_episode_rows(all_episode_rows, by_seed_cols + ["threat_motion_mode"]),
        )
        write_csv(
            result_dir / "tables/phase_n3pfv_threat_class_breakdown.csv",
            summarize_episode_rows(all_episode_rows, by_seed_cols + ["threat_class"]),
        )
        write_csv(result_dir / "tables/phase_n3pfv_raw_unsafe_action_steps.csv", all_raw_rows)
        write_csv(result_dir / "tables/phase_n3pfv_raw_unsafe_action_summary.csv", summarize_raw_by_seed(all_raw_rows))
        action_summary = summarize_episode_rows(all_episode_rows, by_seed_cols)
        write_csv(result_dir / "tables/phase_n3pfv_action_dynamics_summary.csv", action_summary)
        write_csv(result_dir / "tables/phase_n3pfv_gpsi_output_summary.csv", base_eval.gpsi_summary_rows(gpsi_accum))
        write_csv(result_dir / "tables/phase_n3pfv_feature_block_stats.csv", feature_summary_rows(feature_accum, cfg_by_method))
        base_eval.copy_sample_traces(result_dir, all_trace_files)
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] N3PFV_EVAL_END "
            f"episodes={len(all_episode_rows)} raw_steps={len(all_raw_rows)} gpsi_groups={len(gpsi_accum)} traces={len(all_trace_files)}",
            flush=True,
        )
    except N3PFVEvalStop as exc:
        write_stop(result_dir, exc.reason, exc.detail)
        raise SystemExit(2) from exc
    except Exception:
        write_stop(result_dir, "eval_failed", traceback.format_exc())
        raise SystemExit(2)


if __name__ == "__main__":
    main()
