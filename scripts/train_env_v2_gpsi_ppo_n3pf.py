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

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.train_env_v2_gpsi_ppo_n3p as n3p_train
import scripts.train_env_v2_gpsi_ppo_n3z2c as cont_train


RESULT_DIR = ROOT / "results/env_v2_phase_n3pf_block_projected_full"
REQUIRED_PARENT = ROOT / "checkpoints/env_v2_gpsi_heada_ppo_n3p_block_projected_s0/checkpoint_500k.zip"
FORBIDDEN_PARENTS = {
    (ROOT / "checkpoints/env_v2_gpsi_heada_ppo_n3p_block_projected_s0/final.zip").resolve(),
    (ROOT / "checkpoints/env_v2_gpsi_heada_ppo_n3p_block_projected_s0/best_by_eval.zip").resolve(),
}
N35_FLAG = ROOT / "results/env_v2_phase_n3_5_gpsi_wrapper_audit/PHASE_N3_5_GPSI_WRAPPER_AUDIT_COMPLETE.flag"
N3P_FLAG = ROOT / "results/env_v2_phase_n3p_noz_representation_ablation/PHASE_N3P_NOZ_REPRESENTATION_ABLATION_COMPLETE.flag"
N3FZ_FLAG = ROOT / "results/env_v2_phase_n3fz_noz_full_z_screen/PHASE_N3FZ_NOZ_FULL_Z_SCREEN_COMPLETE.flag"
N3Z2CF_FLAG = ROOT / "results/env_v2_phase_n3z2cf_corrected_z2_full/PHASE_N3Z2CF_CORRECTED_Z2_FULL_COMPLETE.flag"

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


class N3PFTrainStop(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phase N3PF block-projected no-z full continuation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--additional-steps", type=int, default=1_000_000)
    parser.add_argument("--target-total-steps", type=int, default=1_500_000)
    parser.add_argument("--parent-total-steps", type=int, default=500_000)
    parser.add_argument("--checkpoint-total-steps", nargs="*", type=int, default=[750_000, 1_000_000, 1_250_000, 1_500_000])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--reset-num-timesteps", default="false")
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
    write_text(RESULT_DIR / "phase_n3pf_status.txt", f"stopped:{flag}\n")
    write_text(
        RESULT_DIR / "PHASE_N3PF_BLOCK_PROJECTED_FULL_REPORT.md",
        "\n".join(
            [
                "# Phase N3PF Block-Projected Full Report",
                "",
                f"`terminal_decision = phase_n3pf_stopped_{reason}`",
                "",
                "Partial report generated by the N3PF training script.",
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


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise N3PFTrainStop("config_mismatch", f"config is not a mapping: {rel(path)}")
    return payload


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def run_command(label: str, command: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(command, check=False, text=True, capture_output=True, timeout=30)
        output = (proc.stdout + proc.stderr).strip()
        return {"item": label, "command": " ".join(command), "returncode": int(proc.returncode), "output": output}
    except Exception as exc:
        return {"item": label, "command": " ".join(command), "returncode": -1, "output": f"{type(exc).__name__}: {exc}"}


def write_resource_affinity() -> None:
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
    write_csv(RESULT_DIR / "tables/phase_n3pf_resource_affinity.csv", rows)


def sha256(path: Path) -> str:
    return cont_train.sha256(path)


def parse_false(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"false", "0", "no"}:
        return False
    if normalized in {"true", "1", "yes"}:
        return True
    raise N3PFTrainStop("resume_semantics_invalid", f"invalid --reset-num-timesteps value: {value}")


def schedule_value(schedule: Any, progress_remaining: float) -> Any:
    try:
        return float(schedule(progress_remaining)) if callable(schedule) else float(schedule)
    except Exception as exc:
        return f"unavailable:{type(exc).__name__}:{exc}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_config(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    missing = [path for path in [N35_FLAG, N3P_FLAG, N3FZ_FLAG, N3Z2CF_FLAG] if not path.exists()]
    if missing:
        raise N3PFTrainStop("parent_missing", "missing prerequisite flags: " + ", ".join(rel(path) for path in missing))
    resume = (ROOT / args.resume).resolve()
    if resume in FORBIDDEN_PARENTS:
        raise N3PFTrainStop("resume_semantics_invalid", f"forbidden parent selected: {rel(resume)}")
    if resume != REQUIRED_PARENT.resolve():
        raise N3PFTrainStop("resume_semantics_invalid", f"N3PF must use fixed parent {rel(REQUIRED_PARENT)}, got {rel(resume)}")
    if not resume.exists() or resume.stat().st_size == 0:
        raise N3PFTrainStop("parent_missing", f"missing fixed parent checkpoint: {rel(resume)}")
    if parse_false(args.reset_num_timesteps):
        raise N3PFTrainStop("resume_semantics_invalid", "reset_num_timesteps=True is forbidden")
    if int(args.parent_total_steps) != 500_000 or int(args.additional_steps) != 1_000_000 or int(args.target_total_steps) != 1_500_000:
        raise N3PFTrainStop("resume_semantics_invalid", "expected parent=500000 additional=1000000 target=1500000")
    required_steps = {750_000, 1_000_000, 1_250_000, 1_500_000}
    if set(int(step) for step in args.checkpoint_total_steps) != required_steps:
        raise N3PFTrainStop("config_mismatch", f"checkpoint steps must be {sorted(required_steps)}")
    if str(cfg.get("method_key", "")) != "block_projected_full":
        raise N3PFTrainStop("config_mismatch", f"expected method_key=block_projected_full, got {cfg.get('method_key')}")
    gpsi = cfg.get("gpsi", {})
    if bool(gpsi.get("include_z", True)):
        raise N3PFTrainStop("config_mismatch", "N3PF must use include_z=false")
    if not bool(gpsi.get("include_logvar", True)):
        raise N3PFTrainStop("config_mismatch", "N3PF must include scaled logvar")
    if int(gpsi.get("obs_aug_dim", -1)) != 30:
        raise N3PFTrainStop("config_mismatch", f"expected obs_aug_dim=30, got {gpsi.get('obs_aug_dim')}")
    if abs(float(gpsi.get("logvar_output_scale", 1.0)) - 0.2) > 1e-9:
        raise N3PFTrainStop("config_mismatch", f"expected logvar_output_scale=0.2, got {gpsi.get('logvar_output_scale')}")
    ppo = cfg.get("ppo", {})
    if str(ppo.get("feature_adapter", "")) != "block_projected_no_z":
        raise N3PFTrainStop("config_mismatch", "feature_adapter must be block_projected_no_z")
    train = cfg.get("training", {})
    if not bool(train.get("no_shield", True)) or bool(train.get("action_filtering", False)) or bool(train.get("use_safety_cost", False)):
        raise N3PFTrainStop("config_mismatch", "expected no shield, no action filtering, no safety cost")
    checkpoint = ROOT / str(gpsi.get("checkpoint", "work_dirs/gpsi_heada_v1_nll/best.pth"))
    if not checkpoint.exists():
        raise N3PFTrainStop("parent_missing", f"missing Gpsi checkpoint: {rel(checkpoint)}")


def copy_parent(out_dir: Path, parent: Path) -> None:
    target = out_dir / "parent_500k.zip"
    if target.exists() and target.stat().st_size > 0:
        if sha256(target) != sha256(parent):
            raise N3PFTrainStop("parent_missing", f"existing parent_500k.zip hash mismatch: {rel(target)}")
        return
    shutil.copyfile(parent, target)


def target_ready(out_dir: Path, target_total_steps: int) -> bool:
    return (out_dir / "final.zip").exists() and (out_dir / f"checkpoint_{target_total_steps // 1000}k.zip").exists()


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def wait_for_existing_training(out_dir: Path, args: argparse.Namespace, pid: int, status: Path) -> str:
    status.write_text(
        json.dumps(
            {
                "status": "waiting_for_existing_training",
                "existing_pid": int(pid),
                "target_total_steps": int(args.target_total_steps),
                "out_dir": rel(out_dir),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    while pid_alive(pid):
        if target_ready(out_dir, int(args.target_total_steps)):
            (out_dir / "TRAIN_COMPLETE.flag").write_text(
                f"completed_wait\ntarget_total_steps={args.target_total_steps}\nexisting_pid={pid}\n",
                encoding="utf-8",
            )
            status.write_text(
                json.dumps(
                    {
                        "status": "completed_wait",
                        "existing_pid": int(pid),
                        "target_total_steps": int(args.target_total_steps),
                        "out_dir": rel(out_dir),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            return "skip"
        time.sleep(30.0)
    if target_ready(out_dir, int(args.target_total_steps)):
        (out_dir / "TRAIN_COMPLETE.flag").write_text(
            f"completed_wait\ntarget_total_steps={args.target_total_steps}\nexisting_pid={pid}\n",
            encoding="utf-8",
        )
        status.write_text(
            json.dumps(
                {
                    "status": "completed_wait",
                    "existing_pid": int(pid),
                    "target_total_steps": int(args.target_total_steps),
                    "out_dir": rel(out_dir),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return "skip"
    existing = sorted(path.name for path in out_dir.glob("*.zip") if path.name != "parent_500k.zip")
    raise N3PFTrainStop("train_failed", f"existing training pid {pid} exited before target completion; partial checkpoints: {existing}")


def guard_start(out_dir: Path, cfg: dict[str, Any], args: argparse.Namespace) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = out_dir / "TRAIN_RUNNING.lock"
    complete = out_dir / "TRAIN_COMPLETE.flag"
    status = out_dir / "TRAIN_STATUS.json"
    if target_ready(out_dir, int(args.target_total_steps)):
        complete.write_text(f"completed_skip\ntarget_total_steps={args.target_total_steps}\n", encoding="utf-8")
        status.write_text(json.dumps({"status": "completed_skip", "target_total_steps": int(args.target_total_steps), "out_dir": rel(out_dir)}, indent=2) + "\n", encoding="utf-8")
        return "skip"
    if lock.exists():
        try:
            payload = json.loads(lock.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        pid = int(payload.get("pid", -1))
        if pid_alive(pid):
            return wait_for_existing_training(out_dir, args, pid, status)
        existing = sorted(path.name for path in out_dir.glob("*.zip") if path.name != "parent_500k.zip")
        if existing:
            raise N3PFTrainStop("train_failed", f"stale lock and partial checkpoints exist: {existing}")
        lock.unlink(missing_ok=True)
    existing = sorted(path.name for path in out_dir.glob("*.zip") if path.name != "parent_500k.zip")
    if existing:
        raise N3PFTrainStop("train_failed", f"partial checkpoint artifacts exist without completion: {existing}")
    lock.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "config": args.config,
                "method_key": cfg.get("method_key", ""),
                "out_dir": rel(out_dir),
                "resume": rel(ROOT / args.resume),
                "parent_total_steps": int(args.parent_total_steps),
                "additional_steps": int(args.additional_steps),
                "target_total_steps": int(args.target_total_steps),
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "hostname": socket.gethostname(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    status.write_text(json.dumps({"status": "running", "pid": os.getpid(), "out_dir": rel(out_dir)}, indent=2) + "\n", encoding="utf-8")
    return "start"


def write_command_manifest(args: argparse.Namespace, cfg: dict[str, Any], out_dir: Path) -> None:
    checkpoint = ROOT / str(cfg.get("gpsi", {}).get("checkpoint", "work_dirs/gpsi_heada_v1_nll/best.pth"))
    append_csv(
        RESULT_DIR / "tables/phase_n3pf_command_manifest.csv",
        [
            {
                "stage": "train",
                "method_key": cfg.get("method_key", ""),
                "method": cfg.get("method_name", cfg.get("method_key", "")),
                "command": " ".join(["python", *sys.argv]),
                "config": rel(ROOT / args.config),
                "gpsi_checkpoint": rel(checkpoint),
                "gpsi_checkpoint_sha256": sha256(checkpoint) if checkpoint.exists() else "missing",
                "out_dir": rel(out_dir),
                "resume": rel(ROOT / args.resume),
                "parent_sha256": sha256(ROOT / args.resume),
                "parent_total_steps": int(args.parent_total_steps),
                "additional_steps": int(args.additional_steps),
                "target_total_steps": int(args.target_total_steps),
                "checkpoint_total_steps": json.dumps([int(step) for step in args.checkpoint_total_steps]),
                "seed": int(args.seed),
                "n_envs": int(args.n_envs),
                "device": args.device,
                "reset_num_timesteps": False,
            }
        ],
    )


def write_config_manifest(args: argparse.Namespace, cfg: dict[str, Any], out_dir: Path, adapter_count: int) -> None:
    gpsi = cfg.get("gpsi", {})
    ppo = cfg.get("ppo", {})
    append_csv(
        RESULT_DIR / "tables/phase_n3pf_config_manifest.csv",
        [
            {
                "method_key": cfg.get("method_key", ""),
                "method": cfg.get("method_name", cfg.get("method_key", "")),
                "config": rel(ROOT / args.config),
                "out_dir": rel(out_dir),
                "obs_aug_dim": int(gpsi.get("obs_aug_dim", -1)),
                "include_z": int(bool(gpsi.get("include_z", False))),
                "include_logvar": int(bool(gpsi.get("include_logvar", True))),
                "logvar_output_scale": float(gpsi.get("logvar_output_scale", 1.0)),
                "feature_adapter": str(ppo.get("feature_adapter", "")),
                "adapter_parameter_count": int(adapter_count),
                "n_envs": int(args.n_envs),
                "device": args.device,
                "no_shield": int(bool(cfg.get("training", {}).get("no_shield", True))),
                "action_filtering": int(bool(cfg.get("training", {}).get("action_filtering", False))),
                "use_safety_cost": int(bool(cfg.get("training", {}).get("use_safety_cost", False))),
            }
        ],
    )


def adapter_parameter_count(model: PPO) -> int:
    extractor = model.policy.features_extractor
    total = 0
    for name, parameter in extractor.named_parameters():
        if name.startswith(("obs_projector", "delta_projector", "logvar_projector", "block_fusion")):
            total += int(parameter.numel())
    return total


def validate_wrapper_to_schema(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    class SchemaArgs:
        seed = int(args.seed)
        device = args.device

    n3p_train.validate_config_semantics({**cfg, "method_key": "block_projected"})
    old_result_dir = n3p_train.RESULT_DIR
    n3p_train.RESULT_DIR = RESULT_DIR
    try:
        n3p_train.validate_wrapper({**cfg, "method_key": "block_projected"}, SchemaArgs)
    finally:
        n3p_train.RESULT_DIR = old_result_dir
    src = RESULT_DIR / "tables/phase_n3p_schema_check.csv"
    if src.exists():
        rows: list[dict[str, Any]] = []
        with src.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if row.get("method_key") == "block_projected":
                row["method_key"] = "block_projected_full"
        append_csv(RESULT_DIR / "tables/phase_n3pf_schema_check.csv", rows)
        src.unlink(missing_ok=True)


def train() -> None:
    args = parse_args()
    out_dir = ROOT / args.out_dir
    ensure_dirs(out_dir)
    try:
        write_resource_affinity()
        cfg = load_config(ROOT / args.config)
        cfg.setdefault("env", {})["n_envs"] = int(args.n_envs)
        cfg.setdefault("env", {})["device"] = str(args.device)
        cfg.setdefault("training", {})["seed"] = int(args.seed)
        cfg["training"]["parent_total_steps"] = int(args.parent_total_steps)
        cfg["training"]["additional_steps"] = int(args.additional_steps)
        cfg["training"]["total_steps"] = int(args.target_total_steps)
        cfg["training"]["checkpoint_steps"] = [int(step) for step in args.checkpoint_total_steps]
        cfg["training"]["resume"] = rel(ROOT / args.resume)
        cfg["training"]["reset_num_timesteps"] = False
        validate_config(cfg, args)
        copy_parent(out_dir, ROOT / args.resume)
        action = guard_start(out_dir, cfg, args)
        if action in {"skip", "wait"}:
            return
        save_yaml(out_dir / "config_resolved.yaml", cfg)
        write_command_manifest(args, cfg, out_dir)
        set_seed(args.seed)
        validate_wrapper_to_schema(cfg, args)

        env_fns = [n3p_train.make_env_factory({**cfg, "method_key": "block_projected"}, args, rank) for rank in range(int(args.n_envs))]
        vec_env = DummyVecEnv(env_fns)
        ppo_cfg = cfg.get("ppo", {})
        method_key = str(cfg.get("method_key", "block_projected_full"))
        method_name = str(cfg.get("method_name", method_key))
        model = PPO.load(str(ROOT / args.resume), env=vec_env, device=args.device, seed=int(args.seed), tensorboard_log=str(out_dir / "tensorboard"))
        model.verbose = 1
        before_steps = int(model.num_timesteps)
        if before_steps != int(args.parent_total_steps):
            raise N3PFTrainStop("resume_semantics_invalid", f"parent model.num_timesteps={before_steps}, expected {args.parent_total_steps}")
        loaded_hparams = {
            "n_steps": int(model.n_steps),
            "batch_size": int(model.batch_size),
            "n_epochs": int(model.n_epochs),
            "gamma": float(model.gamma),
            "gae_lambda": float(model.gae_lambda),
            "ent_coef": float(model.ent_coef),
            "vf_coef": float(model.vf_coef),
            "max_grad_norm": float(model.max_grad_norm),
        }
        expected_hparams = {
            "n_steps": int(ppo_cfg.get("n_steps", loaded_hparams["n_steps"])),
            "batch_size": int(ppo_cfg.get("batch_size", loaded_hparams["batch_size"])),
            "n_epochs": int(ppo_cfg.get("n_epochs", loaded_hparams["n_epochs"])),
            "gamma": float(ppo_cfg.get("gamma", loaded_hparams["gamma"])),
            "gae_lambda": float(ppo_cfg.get("gae_lambda", loaded_hparams["gae_lambda"])),
            "ent_coef": float(ppo_cfg.get("ent_coef", loaded_hparams["ent_coef"])),
            "vf_coef": float(ppo_cfg.get("vf_coef", loaded_hparams["vf_coef"])),
            "max_grad_norm": float(ppo_cfg.get("max_grad_norm", loaded_hparams["max_grad_norm"])),
        }
        mismatched = {key: {"loaded": loaded_hparams[key], "config": expected_hparams[key]} for key in loaded_hparams if loaded_hparams[key] != expected_hparams[key]}
        if mismatched:
            raise N3PFTrainStop("config_mismatch", f"loaded PPO hyperparameters differ from config: {json.dumps(mismatched, sort_keys=True)}")
        optimizer_state = model.policy.optimizer.state_dict()
        if not optimizer_state.get("state"):
            raise N3PFTrainStop("resume_semantics_invalid", "optimizer state is empty after SB3 load")
        adapter_count = adapter_parameter_count(model)
        write_config_manifest(args, cfg, out_dir, adapter_count)
        progress_before = 1.0 - float(before_steps) / max(float(args.target_total_steps), 1.0)
        append_csv(
            RESULT_DIR / "tables/phase_n3pf_resume_semantics.csv",
            [
                {
                    "phase": "before_learn",
                    "selected_parent_path": rel(ROOT / args.resume),
                    "selected_parent_sha256": sha256(ROOT / args.resume),
                    "selected_parent_success": 0.5333,
                    "selected_parent_collision": 0.4667,
                    "forbidden_final_parent_used": 0,
                    "parent_total_steps": int(args.parent_total_steps),
                    "additional_steps": int(args.additional_steps),
                    "target_total_steps": int(args.target_total_steps),
                    "reset_num_timesteps": False,
                    "model_num_timesteps": before_steps,
                    "model_parent_step_match": int(before_steps == int(args.parent_total_steps)),
                    "optimizer_state_entries": int(len(optimizer_state.get("state", {}))),
                    "optimizer_state_restored": int(bool(optimizer_state.get("state"))),
                    "learning_rate_current": schedule_value(model.lr_schedule, progress_before),
                    "clip_range_current": schedule_value(model.clip_range, progress_before),
                    "progress_remaining_assuming_global_total": float(progress_before),
                    "n_envs": int(args.n_envs),
                    "n_steps": int(model.n_steps),
                    "batch_size": int(model.batch_size),
                    "obs_dim": int(vec_env.observation_space["obs"].shape[-1]),
                    "feature_extractor": type(model.policy.features_extractor).__name__,
                    "adapter_parameter_count": int(adapter_count),
                    "hostname": socket.gethostname(),
                }
            ],
        )
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] N3PF_TRAIN_CONFIG "
            f"parent_total_steps={args.parent_total_steps} additional_steps={args.additional_steps} "
            f"target_total_steps={args.target_total_steps} seed={args.seed} n_envs={args.n_envs} device={args.device} "
            f"resume={rel(ROOT / args.resume)} out_dir={rel(out_dir)} reset_num_timesteps=false",
            flush=True,
        )
        callback = cont_train.N3Z2CTrainCallback(args.additional_steps, args.target_total_steps, args.parent_total_steps, method_key, method_name, args.heartbeat_seconds)
        checkpoint_cb = cont_train.TotalStepCheckpointCallback([int(step) for step in args.checkpoint_total_steps], out_dir, args.parent_total_steps)
        model.learn(total_timesteps=int(args.additional_steps), callback=[callback, checkpoint_cb], progress_bar=False, reset_num_timesteps=False)
        final_path = out_dir / "final.zip"
        model.save(str(final_path))
        final_step_checkpoint = out_dir / f"checkpoint_{int(args.target_total_steps) // 1000}k.zip"
        if not final_step_checkpoint.exists():
            shutil.copyfile(final_path, final_step_checkpoint)
        if not (out_dir / "best_by_eval.zip").exists():
            shutil.copyfile(final_path, out_dir / "best_by_eval.zip")
        append_csv(RESULT_DIR / "tables/phase_n3pf_train_curve.csv", callback.episode_rows)
        append_csv(RESULT_DIR / "tables/phase_n3pf_train_heartbeat.csv", callback.heartbeat_rows)
        after_steps = int(model.num_timesteps)
        progress_after = 1.0 - float(after_steps) / max(float(args.target_total_steps), 1.0)
        append_csv(
            RESULT_DIR / "tables/phase_n3pf_resume_semantics.csv",
            [
                {
                    "phase": "after_learn",
                    "selected_parent_path": rel(ROOT / args.resume),
                    "selected_parent_sha256": sha256(ROOT / args.resume),
                    "parent_total_steps": int(args.parent_total_steps),
                    "additional_steps": int(args.additional_steps),
                    "target_total_steps": int(args.target_total_steps),
                    "reset_num_timesteps": False,
                    "model_num_timesteps": after_steps,
                    "model_expected_target_steps": int(args.target_total_steps),
                    "model_target_step_match": int(after_steps >= int(args.target_total_steps)),
                    "optimizer_state_entries": int(len(model.policy.optimizer.state_dict().get("state", {}))),
                    "optimizer_state_restored": int(bool(model.policy.optimizer.state_dict().get("state"))),
                    "learning_rate_current": schedule_value(model.lr_schedule, progress_after),
                    "clip_range_current": schedule_value(model.clip_range, progress_after),
                    "progress_remaining_assuming_global_total": float(progress_after),
                    "n_envs": int(args.n_envs),
                    "n_steps": int(model.n_steps),
                    "batch_size": int(model.batch_size),
                    "obs_dim": int(vec_env.observation_space["obs"].shape[-1]),
                    "feature_extractor": type(model.policy.features_extractor).__name__,
                    "adapter_parameter_count": int(adapter_count),
                    "hostname": socket.gethostname(),
                }
            ],
        )
        vec_env.close()
        (out_dir / "TRAIN_COMPLETE.flag").write_text(
            f"completed\nparent_total_steps={int(args.parent_total_steps)}\nadditional_steps={int(args.additional_steps)}\n"
            f"target_total_steps={int(args.target_total_steps)}\nreset_num_timesteps=false\n",
            encoding="utf-8",
        )
        (out_dir / "TRAIN_STATUS.json").write_text(
            json.dumps({"status": "completed", "target_total_steps": int(args.target_total_steps), "model_num_timesteps": int(after_steps)}, indent=2) + "\n",
            encoding="utf-8",
        )
        (out_dir / "TRAIN_RUNNING.lock").unlink(missing_ok=True)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] N3PF_TRAIN_END final={rel(final_path)} model_num_timesteps={after_steps}", flush=True)
    except N3PFTrainStop as exc:
        write_stop(exc.reason, exc.detail)
        raise SystemExit(2) from exc
    except Exception:
        write_stop("train_failed", traceback.format_exc())
        raise SystemExit(2)


if __name__ == "__main__":
    train()
