from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.train_env_v2_gpsi_ppo_n3fz as base_train
from envs.dynamic_obstacle_flow_env import DynamicObstacleFlowEnv
from envs.wrappers.gpsi_obs_wrapper import GpsiObsWrapper
from models.gpsi_ppo_policy import GpsiBlockProjectedNoZExtractor, GpsiObstacleSetExtractor


RESULT_DIR = ROOT / "results/env_v2_phase_n3p_noz_representation_ablation"
N35_FLAG = ROOT / "results/env_v2_phase_n3_5_gpsi_wrapper_audit/PHASE_N3_5_GPSI_WRAPPER_AUDIT_COMPLETE.flag"
N3R_FLAG = ROOT / "results/env_v2_phase_n3r_gpsi_ppo_rerun/PHASE_N3R_GPSI_PPO_RERUN_COMPLETE.flag"
N3FZ_FLAG = ROOT / "results/env_v2_phase_n3fz_noz_full_z_screen/PHASE_N3FZ_NOZ_FULL_Z_SCREEN_COMPLETE.flag"
N3Z2CF_FLAG = ROOT / "results/env_v2_phase_n3z2cf_corrected_z2_full/PHASE_N3Z2CF_CORRECTED_Z2_FULL_COMPLETE.flag"
N3F_NOZ_FINAL = ROOT / "checkpoints/env_v2_gpsi_heada_ppo_n3f_no_z_s0/final.zip"
ATTENTION_FULL = ROOT / "checkpoints/env_v2_phase2/attention_full_s0/attention_full_s0_step1500000.zip"
Z2CF_REPORT = ROOT / "results/env_v2_phase_n3z2cf_corrected_z2_full/PHASE_N3Z2CF_CORRECTED_Z2_FULL_REPORT.md"

STOP_FLAGS = {
    "gpsi_checkpoint_missing": "PHASE_N3P_STOP_GPSI_CHECKPOINT_MISSING.flag",
    "baseline_artifacts_missing": "PHASE_N3P_STOP_BASELINE_ARTIFACTS_MISSING.flag",
    "config_invalid": "PHASE_N3P_STOP_CONFIG_INVALID.flag",
    "train_failed": "PHASE_N3P_STOP_TRAIN_FAILED.flag",
    "eval_failed": "PHASE_N3P_STOP_EVAL_FAILED.flag",
    "diagnostics_failed": "PHASE_N3P_STOP_DIAGNOSTICS_FAILED.flag",
    "feature_scale_invalid": "PHASE_N3P_STOP_FEATURE_SCALE_INVALID.flag",
    "watcher_failed": "PHASE_N3P_STOP_WATCHER_FAILED.flag",
}

VALID_METHODS = {"obs_delta_only", "logvar_scaled", "block_projected"}


class PhaseN3PTrainStop(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one Phase N3P no-z representation ablation config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-steps", type=int, default=500_000)
    parser.add_argument("--checkpoint-steps", nargs="*", type=int, default=[250_000, 500_000])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--heartbeat-seconds", type=float, default=300.0)
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


def ensure_dirs(out_dir: Path) -> None:
    for path in [out_dir, RESULT_DIR, RESULT_DIR / "logs", RESULT_DIR / "tables", RESULT_DIR / "plots"]:
        path.mkdir(parents=True, exist_ok=True)


def write_stop(reason: str, detail: str) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    flag = STOP_FLAGS.get(reason, STOP_FLAGS["train_failed"])
    write_text(RESULT_DIR / flag, f"{reason}\n{detail.strip()}\n")
    write_text(RESULT_DIR / "phase_n3p_status.txt", f"stopped:{flag}\n")
    write_text(
        RESULT_DIR / "PHASE_N3P_NOZ_REPRESENTATION_ABLATION_REPORT.md",
        "\n".join(
            [
                "# Phase N3P No-Z Representation Ablation Report",
                "",
                f"`terminal_decision = phase_n3p_stopped_{reason}`",
                "",
                "Partial report generated by the N3P training script.",
                "",
                "```text",
                detail.strip(),
                "```",
                "",
                "can_enter_N4: no",
            ]
        )
        + "\n",
    )


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise PhaseN3PTrainStop("config_invalid", f"config is not a mapping: {rel(path)}")
    return payload


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def sha256(path: Path) -> str:
    return base_train.sha256(path)


def run_command(label: str, command: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(command, check=False, text=True, capture_output=True, timeout=30)
        output = (proc.stdout + proc.stderr).strip()
        return {"item": label, "command": " ".join(command), "returncode": int(proc.returncode), "output": output}
    except Exception as exc:
        return {"item": label, "command": " ".join(command), "returncode": -1, "output": f"{type(exc).__name__}: {exc}"}


def write_resource_affinity(stage: str, method_key: str) -> None:
    rows = [
        run_command("nproc", ["nproc"]),
        run_command("nproc_all", ["nproc", "--all"]),
        run_command("taskset_current_process", ["taskset", "-pc", str(os.getpid())]),
        run_command("lscpu", ["lscpu"]),
        run_command("free_h", ["free", "-h"]),
        run_command("df_root", ["df", "-h", "/"]),
        run_command("nvidia_smi", ["nvidia-smi"]),
    ]
    for path in ["/sys/fs/cgroup/cpuset.cpus.effective", "/sys/fs/cgroup/cpu.max"]:
        p = Path(path)
        rows.append({"item": p.name, "command": f"read {path}", "returncode": 0 if p.exists() else 1, "output": p.read_text().strip() if p.exists() else "missing"})
    rows.append(
        {
            "item": "python_cpu_affinity",
            "command": "python runtime",
            "returncode": 0,
            "output": json.dumps(
                {
                    "stage": stage,
                    "method_key": method_key,
                    "os_cpu_count": os.cpu_count(),
                    "sched_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                    "sched_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                    "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
                    "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", ""),
                    "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", ""),
                    "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", ""),
                    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                },
                sort_keys=True,
            ),
        }
    )
    for row in rows:
        row["stage"] = stage
        row["method_key"] = method_key
        row["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    append_csv(RESULT_DIR / "tables/phase_n3p_resource_affinity.csv", rows)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_baselines(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    method_key = str(cfg.get("method_key", ""))
    if method_key not in VALID_METHODS:
        raise PhaseN3PTrainStop("config_invalid", f"unexpected method_key={method_key}; expected {sorted(VALID_METHODS)}")
    if int(args.train_steps) != 500_000:
        raise PhaseN3PTrainStop("config_invalid", f"N3P must train exactly 500k steps, got {args.train_steps}")
    missing_baselines = [path for path in [N35_FLAG, N3R_FLAG, N3FZ_FLAG, N3Z2CF_FLAG, N3F_NOZ_FINAL, ATTENTION_FULL, Z2CF_REPORT] if not path.exists()]
    if missing_baselines:
        raise PhaseN3PTrainStop("baseline_artifacts_missing", "missing baseline artifacts:\n" + "\n".join(rel(path) for path in missing_baselines))
    checkpoint = ROOT / str(cfg.get("gpsi", {}).get("checkpoint", "work_dirs/gpsi_heada_v1_nll/best.pth"))
    if not checkpoint.exists():
        raise PhaseN3PTrainStop("gpsi_checkpoint_missing", f"missing Gpsi checkpoint: {rel(checkpoint)}")


def validate_config_semantics(cfg: dict[str, Any]) -> None:
    method_key = str(cfg.get("method_key", ""))
    gpsi_cfg = cfg.get("gpsi", {})
    ppo_cfg = cfg.get("ppo", {})
    training = cfg.get("training", {})
    if bool(gpsi_cfg.get("include_z", False)):
        raise PhaseN3PTrainStop("config_invalid", f"{method_key} is no_z; include_z must be false")
    if method_key == "obs_delta_only":
        expected = {"include_logvar": False, "obs_aug_dim": 21, "logvar_output_scale": 1.0}
    elif method_key in {"logvar_scaled", "block_projected"}:
        expected = {"include_logvar": True, "obs_aug_dim": 30, "logvar_output_scale": 0.2}
    else:
        raise PhaseN3PTrainStop("config_invalid", f"unknown method_key={method_key}")
    if bool(gpsi_cfg.get("include_logvar", True)) != expected["include_logvar"]:
        raise PhaseN3PTrainStop("config_invalid", f"{method_key} include_logvar mismatch")
    if int(gpsi_cfg.get("obs_aug_dim", -1)) != int(expected["obs_aug_dim"]):
        raise PhaseN3PTrainStop("config_invalid", f"{method_key} obs_aug_dim mismatch")
    if abs(float(gpsi_cfg.get("logvar_output_scale", 1.0)) - float(expected["logvar_output_scale"])) > 1e-9:
        raise PhaseN3PTrainStop("config_invalid", f"{method_key} logvar_output_scale mismatch")
    if method_key == "block_projected" and str(ppo_cfg.get("feature_adapter", "")) != "block_projected_no_z":
        raise PhaseN3PTrainStop("config_invalid", "P3 must use feature_adapter=block_projected_no_z")
    if method_key != "block_projected" and str(ppo_cfg.get("feature_adapter", "raw_concat")) != "raw_concat":
        raise PhaseN3PTrainStop("config_invalid", f"{method_key} must use raw_concat adapter")
    if not bool(training.get("no_shield", True)) or bool(training.get("action_filtering", False)) or bool(training.get("use_safety_cost", False)):
        raise PhaseN3PTrainStop("config_invalid", "N3P requires no shield, no action filtering, no safety cost")


def make_wrapper(cfg: dict[str, Any], scenario: str, device: str) -> GpsiObsWrapper:
    gpsi_cfg = cfg.get("gpsi", {})
    return GpsiObsWrapper(
        DynamicObstacleFlowEnv(scenario=scenario),
        gpsi_checkpoint=ROOT / str(gpsi_cfg.get("checkpoint", "work_dirs/gpsi_heada_v1_nll/best.pth")),
        device=device,
        history_steps=int(gpsi_cfg.get("history_steps", 20)),
        delta_scale=float(gpsi_cfg.get("delta_scale", 5.0)),
        logvar_clamp=tuple(gpsi_cfg.get("logvar_clamp", [-5.0, 3.0])),
        normalize_z=bool(gpsi_cfg.get("normalize_z", False)),
        z_stats_path=(ROOT / str(gpsi_cfg.get("z_stats_path"))) if gpsi_cfg.get("z_stats_path") else None,
        z_std_floor=float(gpsi_cfg.get("z_std_floor", 1.0e-3)),
        include_z=bool(gpsi_cfg.get("include_z", False)),
        z_transform=gpsi_cfg.get("z_transform"),
        z_l2_target_norm=float(gpsi_cfg.get("z_l2_target_norm", 4.0)),
        z_l2_eps=float(gpsi_cfg.get("z_l2_eps", 1.0e-6)),
        z_layernorm_alpha=float(gpsi_cfg.get("z_layernorm_alpha", 0.5)),
        z_layernorm_eps=float(gpsi_cfg.get("z_layernorm_eps", 1.0e-5)),
        include_logvar=bool(gpsi_cfg.get("include_logvar", True)),
        logvar_output_scale=float(gpsi_cfg.get("logvar_output_scale", 1.0)),
        degenerate_std_threshold=float(gpsi_cfg.get("degenerate_std_threshold", 1.0e-5)),
        degenerate_std_floor=float(gpsi_cfg.get("degenerate_std_floor", 1.0)),
    )


def make_env_factory(cfg: dict[str, Any], args: argparse.Namespace, rank: int):
    scenario = str(cfg.get("env", {}).get("train_scenario", "train_flow_mixed"))
    expected_dim = int(cfg.get("gpsi", {}).get("obs_aug_dim", 30))

    def _factory() -> gym.Env:
        env = make_wrapper(cfg, scenario, args.device)
        freeze = env.freeze_check
        if freeze.training or freeze.requires_grad_any or freeze.trainable_parameters != 0:
            raise PhaseN3PTrainStop("config_invalid", f"Gpsi freeze check failed: {freeze}")
        actual_dim = int(env.observation_space["obs"].shape[-1])
        if actual_dim != expected_dim:
            raise PhaseN3PTrainStop("config_invalid", f"aug obs dim mismatch: env={actual_dim} config={expected_dim}")
        wrapped = Monitor(env)
        wrapped.reset(seed=int(args.seed) + rank)
        return wrapped

    return _factory


def make_policy_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    ppo_cfg = cfg.get("ppo", {})
    gpsi_cfg = cfg.get("gpsi", {})
    adapter = str(ppo_cfg.get("feature_adapter", "raw_concat"))
    common = {
        "net_arch": ppo_cfg.get("net_arch", {"pi": [128, 128], "vf": [128, 128]}),
        "activation_fn": torch.nn.Tanh,
    }
    if adapter == "block_projected_no_z":
        block_cfg = ppo_cfg.get("block_projector", {})
        return {
            **common,
            "features_extractor_class": GpsiBlockProjectedNoZExtractor,
            "features_extractor_kwargs": {
                "hidden_dim": int(ppo_cfg.get("hidden_dim", 64)),
                "obs_block_dim": int(block_cfg.get("obs_block_dim", 12)),
                "delta_block_dim": int(block_cfg.get("delta_block_dim", 9)),
                "logvar_block_dim": int(block_cfg.get("logvar_block_dim", 9)),
                "obs_project_dim": int(block_cfg.get("obs_project_dim", 32)),
                "delta_project_dim": int(block_cfg.get("delta_project_dim", 16)),
                "logvar_project_dim": int(block_cfg.get("logvar_project_dim", 16)),
                "activation": str(block_cfg.get("activation", "tanh")),
                "use_risk_bias": bool(ppo_cfg.get("use_risk_bias", False)),
                "lambda_bias": float(ppo_cfg.get("lambda_bias", 0.0)),
            },
        }
    return {
        **common,
        "features_extractor_class": GpsiObstacleSetExtractor,
        "features_extractor_kwargs": {
            "hidden_dim": int(ppo_cfg.get("hidden_dim", 64)),
            "obs_dim": int(gpsi_cfg.get("obs_aug_dim", 30)),
            "use_risk_bias": bool(ppo_cfg.get("use_risk_bias", False)),
            "lambda_bias": float(ppo_cfg.get("lambda_bias", 0.0)),
        },
    }


def finite_l2(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return np.asarray([], dtype=np.float32)
    return np.linalg.norm(arr.reshape(arr.shape[0], -1), axis=1)


def adapter_parameter_count(model: PPO) -> int:
    extractor = model.policy.features_extractor
    total = 0
    for name, parameter in extractor.named_parameters():
        if name.startswith(("obs_projector", "delta_projector", "logvar_projector", "block_fusion")):
            total += int(parameter.numel())
    return total


def validate_wrapper(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    try:
        method_key = str(cfg.get("method_key", ""))
        env = make_wrapper(cfg, "eval_flow_id", args.device)
        freeze = env.freeze_check
        obs, info = env.reset(seed=int(args.seed))
        expected_dim = int(cfg.get("gpsi", {}).get("obs_aug_dim", 30))
        debug = env.latest_gpsi_debug
        rows = [
            {"method_key": method_key, "check": "phase_n3_5_complete_flag", "value": str(N35_FLAG.exists()), "detail": rel(N35_FLAG)},
            {"method_key": method_key, "check": "phase_n3r_complete_flag", "value": str(N3R_FLAG.exists()), "detail": rel(N3R_FLAG)},
            {"method_key": method_key, "check": "phase_n3fz_complete_flag", "value": str(N3FZ_FLAG.exists()), "detail": rel(N3FZ_FLAG)},
            {"method_key": method_key, "check": "phase_n3z2cf_complete_flag", "value": str(N3Z2CF_FLAG.exists()), "detail": rel(N3Z2CF_FLAG)},
            {"method_key": method_key, "check": "gpsi_frozen", "value": str(not freeze.training and not freeze.requires_grad_any and freeze.trainable_parameters == 0), "detail": json.dumps(freeze.__dict__, sort_keys=True)},
            {"method_key": method_key, "check": "aug_obs_dim", "value": str(int(obs["obs"].shape[-1])), "detail": f"expected={expected_dim}"},
            {"method_key": method_key, "check": "include_z", "value": str(bool(debug.get("include_z", False))), "detail": "must be false"},
            {"method_key": method_key, "check": "include_logvar", "value": str(bool(debug.get("include_logvar", True))), "detail": f"scale={debug.get('logvar_output_scale', 1.0)}"},
            {"method_key": method_key, "check": "no_shield", "value": "true", "detail": "observation-only Gpsi wrapper"},
            {"method_key": method_key, "check": "no_action_filtering", "value": "true", "detail": "raw PPO action"},
            {"method_key": method_key, "check": "no_safety_cost", "value": "true", "detail": "EnvV2 original reward"},
        ]
        if int(obs["obs"].shape[-1]) != expected_dim:
            raise PhaseN3PTrainStop("config_invalid", f"obs dim mismatch: {obs['obs'].shape[-1]} vs {expected_dim}")
        if freeze.training or freeze.requires_grad_any or freeze.trainable_parameters != 0:
            raise PhaseN3PTrainStop("config_invalid", f"Gpsi freeze check failed: {freeze}")

        delta_1s: list[float] = []
        logvar_raw_l2: list[float] = []
        logvar_policy_l2: list[float] = []
        full_obs_l2: list[float] = []
        for step in range(80):
            goal = np.asarray(info["goal_position"], dtype=np.float32)
            uav = np.asarray(info["uav_position"], dtype=np.float32)
            vec = goal - uav
            vec[2] = 0.0
            norm = float(np.linalg.norm(vec))
            action = (vec / norm if norm > 1e-8 else env.action_space.sample()).astype(np.float32)
            obs, _reward, terminated, truncated, info = env.step(action)
            debug = env.latest_gpsi_debug
            active = np.asarray(debug.get("active_slots", []), dtype=np.int64)
            if active.size:
                delta = np.asarray(debug["delta_hat_raw"], dtype=np.float32)[active, 0, :]
                logvar_raw = np.asarray(debug["logvar_hat"], dtype=np.float32)[active].reshape(active.size, -1)
                logvar_policy = np.asarray(debug["logvar_hat_policy"], dtype=np.float32)[active].reshape(active.size, -1)
                delta_1s.extend(finite_l2(delta).tolist())
                logvar_raw_l2.extend(finite_l2(logvar_raw).tolist())
                logvar_policy_l2.extend(finite_l2(logvar_policy).tolist())
                full_obs_l2.extend(finite_l2(obs["obs"][active]).tolist())
            if terminated or truncated:
                obs, info = env.reset(seed=int(args.seed) + step + 1)
        env.close()
        rows.extend(
            [
                {"method_key": method_key, "check": "delta_norm_1s_p95_pretrain", "value": float(np.percentile(delta_1s, 95)) if delta_1s else np.nan, "detail": f"max={np.max(delta_1s) if delta_1s else np.nan}"},
                {"method_key": method_key, "check": "logvar_raw_l2_p95_pretrain", "value": float(np.percentile(logvar_raw_l2, 95)) if logvar_raw_l2 else np.nan, "detail": "diagnostic only"},
                {"method_key": method_key, "check": "logvar_policy_l2_p95_pretrain", "value": float(np.percentile(logvar_policy_l2, 95)) if logvar_policy_l2 else np.nan, "detail": "policy input if include_logvar=true"},
                {"method_key": method_key, "check": "full_aug_obs_l2_p95_pretrain", "value": float(np.percentile(full_obs_l2, 95)) if full_obs_l2 else np.nan, "detail": "actual PPO observation block"},
            ]
        )
        append_csv(RESULT_DIR / "tables/phase_n3p_schema_check.csv", rows)
        delta_p95 = float(np.percentile(delta_1s, 95)) if delta_1s else np.nan
        delta_max = float(np.max(delta_1s)) if delta_1s else np.nan
        if not np.isfinite(delta_p95) or delta_p95 > 100.0 or delta_max > 1000.0:
            raise PhaseN3PTrainStop("diagnostics_failed", f"pretrain Gpsi output scale invalid: delta_1s_p95={delta_p95} max={delta_max}")
        policy_p95 = float(np.percentile(logvar_policy_l2, 95)) if logvar_policy_l2 else np.nan
        if method_key in {"logvar_scaled", "block_projected"} and (not np.isfinite(policy_p95) or policy_p95 > 4.0):
            raise PhaseN3PTrainStop("feature_scale_invalid", f"{method_key} logvar_scaled l2 p95 is invalid: {policy_p95}")
    except PhaseN3PTrainStop:
        raise
    except Exception as exc:
        raise PhaseN3PTrainStop("diagnostics_failed", traceback.format_exc()) from exc


def write_config_manifest(cfg: dict[str, Any], args: argparse.Namespace, out_dir: Path) -> None:
    method_key = str(cfg.get("method_key", ""))
    ppo_cfg = cfg.get("ppo", {})
    row = {
        "method_key": method_key,
        "method": cfg.get("method_name", method_key),
        "config": rel(ROOT / args.config),
        "out_dir": rel(out_dir),
        "obs_aug_dim": int(cfg.get("gpsi", {}).get("obs_aug_dim", -1)),
        "include_z": int(bool(cfg.get("gpsi", {}).get("include_z", False))),
        "include_logvar": int(bool(cfg.get("gpsi", {}).get("include_logvar", True))),
        "logvar_output_scale": float(cfg.get("gpsi", {}).get("logvar_output_scale", 1.0)),
        "feature_adapter": str(ppo_cfg.get("feature_adapter", "raw_concat")),
        "seed": int(args.seed),
        "train_steps": int(args.train_steps),
        "n_envs": int(args.n_envs),
        "device": args.device,
        "no_shield": int(bool(cfg.get("training", {}).get("no_shield", True))),
        "action_filtering": int(bool(cfg.get("training", {}).get("action_filtering", False))),
        "use_safety_cost": int(bool(cfg.get("training", {}).get("use_safety_cost", False))),
    }
    append_csv(RESULT_DIR / "tables/phase_n3p_config_manifest.csv", [row])


def write_command_manifest(args: argparse.Namespace, cfg: dict[str, Any], out_dir: Path) -> None:
    checkpoint = ROOT / str(cfg.get("gpsi", {}).get("checkpoint", "work_dirs/gpsi_heada_v1_nll/best.pth"))
    row = {
        "stage": "train",
        "method_key": cfg.get("method_key", ""),
        "method": cfg.get("method_name", cfg.get("method_key", "")),
        "command": " ".join(["python", *sys.argv]),
        "config": rel(ROOT / args.config),
        "gpsi_checkpoint": rel(checkpoint),
        "gpsi_checkpoint_sha256": sha256(checkpoint) if checkpoint.exists() else "missing",
        "out_dir": rel(out_dir),
        "train_steps": int(args.train_steps),
        "seed": int(args.seed),
        "n_envs": int(args.n_envs),
        "device": args.device,
    }
    append_csv(RESULT_DIR / "tables/phase_n3p_command_manifest.csv", [row])


def train() -> None:
    args = parse_args()
    out_dir = ROOT / args.out_dir
    ensure_dirs(out_dir)
    try:
        cfg = load_config(ROOT / args.config)
        cfg.setdefault("env", {})["n_envs"] = int(args.n_envs)
        cfg.setdefault("env", {})["device"] = args.device
        cfg.setdefault("training", {})["seed"] = int(args.seed)
        cfg["training"]["total_steps"] = int(args.train_steps)
        cfg["training"]["checkpoint_steps"] = [int(step) for step in args.checkpoint_steps]
        validate_config_semantics(cfg)
        check_baselines(cfg, args)
        guard_action = base_train.prepare_training_guard(out_dir, cfg, args)
        if guard_action in {"skip", "wait"}:
            return
        save_yaml(out_dir / "config_resolved.yaml", cfg)
        write_resource_affinity("train_preflight", str(cfg.get("method_key", "")))
        write_config_manifest(cfg, args, out_dir)
        write_command_manifest(args, cfg, out_dir)
        set_seed(args.seed)
        validate_wrapper(cfg, args)

        env_fns = [make_env_factory(cfg, args, rank) for rank in range(int(args.n_envs))]
        vec_env = DummyVecEnv(env_fns)
        ppo_cfg = cfg.get("ppo", {})
        method_key = str(cfg.get("method_key", Path(args.config).stem))
        method_name = str(cfg.get("method_name", method_key))
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] N3P_TRAIN_CONFIG "
            f"config={method_key} steps={args.train_steps} seed={args.seed} n_envs={args.n_envs} device={args.device} "
            f"out_dir={rel(out_dir)} no_shield=true action_filtering=false safety_cost=false",
            flush=True,
        )
        model = PPO(
            "MultiInputPolicy",
            vec_env,
            seed=int(args.seed),
            device=args.device,
            verbose=1,
            tensorboard_log=str(out_dir / "tensorboard"),
            policy_kwargs=make_policy_kwargs(cfg),
            n_steps=int(ppo_cfg.get("n_steps", 1024)),
            batch_size=int(ppo_cfg.get("batch_size", 256)),
            n_epochs=int(ppo_cfg.get("n_epochs", 10)),
            learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
            gamma=float(ppo_cfg.get("gamma", 0.99)),
            gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
            clip_range=float(ppo_cfg.get("clip_range", 0.2)),
            ent_coef=float(ppo_cfg.get("ent_coef", 0.01)),
            vf_coef=float(ppo_cfg.get("vf_coef", 0.5)),
            max_grad_norm=float(ppo_cfg.get("max_grad_norm", 0.5)),
        )
        if method_key == "block_projected":
            append_csv(
                RESULT_DIR / "tables/phase_n3p_config_manifest.csv",
                [
                    {
                        "method_key": method_key,
                        "method": method_name,
                        "feature_adapter": "block_projected_no_z",
                        "adapter_parameter_count": adapter_parameter_count(model),
                    }
                ],
            )
        callback = base_train.N3FZTrainCallback(args.train_steps, method_key, method_name, args.heartbeat_seconds)
        targets = [step for step in args.checkpoint_steps if step <= args.train_steps]
        checkpoint_cb = base_train.FixedNameCheckpointCallback(targets, out_dir)
        model.learn(
            total_timesteps=int(args.train_steps),
            callback=[callback, checkpoint_cb],
            progress_bar=False,
            reset_num_timesteps=True,
        )
        final_path = out_dir / "final.zip"
        model.save(str(final_path))
        if not (out_dir / "best_by_eval.zip").exists():
            shutil.copyfile(final_path, out_dir / "best_by_eval.zip")
        final_step_checkpoint = out_dir / f"checkpoint_{int(args.train_steps) // 1000}k.zip"
        if not final_step_checkpoint.exists():
            shutil.copyfile(final_path, final_step_checkpoint)
        append_csv(RESULT_DIR / "tables/phase_n3p_train_curve.csv", callback.episode_rows)
        append_csv(RESULT_DIR / "tables/phase_n3p_train_heartbeat.csv", callback.heartbeat_rows)
        vec_env.close()
        base_train.complete_path(out_dir).write_text(f"completed\ntrain_steps={int(args.train_steps)}\n", encoding="utf-8")
        base_train.write_train_status(out_dir, "completed", cfg, args, "training completed successfully")
        base_train.lock_path(out_dir).unlink(missing_ok=True)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] N3P_TRAIN_END config={method_key} final={rel(final_path)}", flush=True)
    except PhaseN3PTrainStop as exc:
        write_stop(exc.reason, exc.detail)
        raise SystemExit(2) from exc
    except base_train.PhaseN3FZTrainStop as exc:
        reason = "train_failed" if exc.reason == "train_failed" else "config_invalid"
        write_stop(reason, exc.detail)
        raise SystemExit(2) from exc
    except Exception:
        detail = traceback.format_exc()
        try:
            if "out_dir" in locals() and "cfg" in locals() and "args" in locals():
                base_train.write_train_status(out_dir, "failed", cfg, args, detail)
        except Exception:
            pass
        write_stop("train_failed", detail)
        raise SystemExit(2)


if __name__ == "__main__":
    train()
