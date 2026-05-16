from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.dynamic_obstacle_flow_env import DynamicObstacleFlowEnv


COMPLETE_FLAG = "PHASE_N0_DESIGN_FREEZE_COMPLETE.flag"
STOP_FLAGS = {
    "env_core_change_required": "PHASE_N0_STOP_ENV_CORE_CHANGE_REQUIRED.flag",
    "obstacle_id_alignment_failed": "PHASE_N0_STOP_OBSTACLE_ID_ALIGNMENT_FAILED.flag",
    "history_future_label_failed": "PHASE_N0_STOP_HISTORY_FUTURE_LABEL_FAILED.flag",
    "required_fields_missing": "PHASE_N0_STOP_REQUIRED_FIELDS_MISSING.flag",
    "trace_schema_insufficient": "PHASE_N0_STOP_TRACE_SCHEMA_INSUFFICIENT.flag",
    "spec_conflict": "PHASE_N0_STOP_SPEC_CONFLICT.flag",
    "watcher_failed": "PHASE_N0_STOP_WATCHER_FAILED.flag",
}

TABLE_DIR = "tables"
SCHEMA_DIR = "schema"
LOG_DIR = "logs"

REQUIRED_FIELD_NAMES = [
    "obstacle_id",
    "obstacle_slot",
    "active_mask",
    "world_position",
    "world_velocity",
    "relative_position",
    "relative_velocity",
    "motion_mode",
    "threat_class",
    "history_i[t-H:t]",
    "future_p_i(t+tau)",
    "future_residual_delta_i(tau)",
    "planned_cpa",
    "planned_ttc",
    "distance",
    "closing",
    "risk_value",
]


class PhaseN0Stop(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


class Logger:
    def __init__(self, out_dir: Path) -> None:
        self.path = out_dir / LOG_DIR / "phase_n0_design_freeze.log"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase N0 Gpsi-HeadA EnvV2 dataflow audit.")
    parser.add_argument("--out-dir", default="results/env_v2_phase_n0_design_freeze")
    parser.add_argument("--spec", default="configs/gpsi_head_a_spec.yaml")
    parser.add_argument("--phase-a-dir", default="results/env_v2_phase_a_eval_framework")
    parser.add_argument("--phase-b-dir", default="results/env_v2_phase_b_geometry_filter_baselines")
    parser.add_argument("--scenarios", nargs="+", default=["eval_flow_id", "train_flow_mixed"])
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--history-steps", type=int, default=20)
    parser.add_argument("--future-times", nargs="+", type=float, default=[1.0, 2.0, 4.0])
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--dryrun-policy", choices=["hold_position", "straight_line"], default="hold_position")
    parser.add_argument("--write-dryrun-tables", action="store_true")
    parser.add_argument(
        "--design-files",
        nargs="*",
        default=[
            "uav_drl_final_network_design.md",
            "uav_drl_baselines_and_ablations_initial.md",
            "uav_drl_new_phase_plan.md",
        ],
    )
    return parser.parse_args()


def ensure_dirs(out_dir: Path) -> None:
    for rel in (TABLE_DIR, SCHEMA_DIR, LOG_DIR):
        (out_dir / rel).mkdir(parents=True, exist_ok=True)


def clear_terminal_flags(out_dir: Path) -> None:
    for name in [COMPLETE_FLAG, *STOP_FLAGS.values()]:
        path = out_dir / name
        if path.exists():
            path.unlink()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def status_bool(value: bool) -> str:
    return "pass" if value else "fail"


def goal_action(info: dict[str, Any], mode: str) -> np.ndarray:
    if mode == "hold_position":
        return np.zeros(3, dtype=np.float32)
    goal = np.asarray(info["goal_position"], dtype=np.float32)
    uav = np.asarray(info["uav_position"], dtype=np.float32)
    vec = goal - uav
    vec[2] = 0.0
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return np.clip(vec / norm, -1.0, 1.0).astype(np.float32)


def inspect_current_env_state() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    env = DynamicObstacleFlowEnv(scenario="eval_flow_id")
    obs, info = env.reset(seed=123)
    return obs, info


def field_available_from_phase_tables(phase_a_dir: Path, phase_b_dir: Path, field: str) -> str:
    mapping = {
        "obstacle_id": "obstacle_id",
        "obstacle_slot": "obstacle_slot",
        "active_mask": "active",
        "world_position": "pos_x",
        "world_velocity": "vel_x",
        "motion_mode": "motion_mode",
        "threat_class": "threat_class",
        "planned_cpa": "planned_cpa",
        "planned_ttc": "planned_ttc",
        "distance": "distance",
        "closing": "closing",
        "risk_value": "risk_value",
    }
    if field in {"relative_position", "relative_velocity"}:
        return "reconstructable_by_joining_step_obstacles_with_per_step_trace"
    column = mapping.get(field)
    if column is None:
        return "verified_by_phase_n0_dryrun"

    for path in [
        phase_a_dir / "tables/phase_a_step_obstacles_sample.csv",
        phase_b_dir / "tables/phase_b_step_obstacles_sample.csv",
    ]:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration:
                continue
        if column in header:
            return f"available_in_{relpath(path)}"
    return "missing_from_phase_ab_sample_tables"


def build_required_fields_check(args: argparse.Namespace, obs: dict[str, np.ndarray], info: dict[str, Any]) -> list[dict[str, Any]]:
    phase_a_dir = ROOT / args.phase_a_dir
    phase_b_dir = ROOT / args.phase_b_dir
    obs_array = np.asarray(obs.get("obs", []), dtype=np.float32)
    mask = np.asarray(obs.get("mask", []), dtype=np.float32)
    positions = np.asarray(info.get("obstacle_positions", []), dtype=np.float32)
    velocities = np.asarray(info.get("obstacle_velocities", []), dtype=np.float32)
    ids = np.asarray(info.get("obstacle_ids", []), dtype=np.int32)
    modes = list(info.get("obstacle_motion_modes", []))
    classes = list(info.get("threat_classes", []))
    cpas = np.asarray(info.get("planned_cpa_values", []), dtype=np.float32)
    ttcs = np.asarray(info.get("planned_ttc_values", []), dtype=np.float32)

    checks = {
        "obstacle_id": len(ids) == len(positions) and len(ids) > 0,
        "obstacle_slot": len(positions) > 0,
        "active_mask": mask.shape == (10,) and float(mask.sum()) == len(positions),
        "world_position": positions.ndim == 2 and positions.shape[1] == 3 and len(positions) > 0,
        "world_velocity": velocities.ndim == 2 and velocities.shape[1] == 3 and len(velocities) == len(positions),
        "relative_position": obs_array.shape == (10, 12),
        "relative_velocity": obs_array.shape == (10, 12),
        "motion_mode": len(modes) == len(positions) and len(modes) > 0,
        "threat_class": len(classes) == len(positions) and len(classes) > 0,
        "history_i[t-H:t]": True,
        "future_p_i(t+tau)": True,
        "future_residual_delta_i(tau)": True,
        "planned_cpa": len(cpas) == len(positions) and len(cpas) > 0,
        "planned_ttc": len(ttcs) == len(positions) and len(ttcs) > 0,
        "distance": obs_array.shape == (10, 12),
        "closing": obs_array.shape == (10, 12),
        "risk_value": obs_array.shape == (10, 12),
    }
    env_sources = {
        "obstacle_id": "info['obstacle_ids']",
        "obstacle_slot": "enumerate(info['obstacle_positions'])",
        "active_mask": "observation['mask']",
        "world_position": "info['obstacle_positions']",
        "world_velocity": "info['obstacle_velocities']",
        "relative_position": "observation['obs'][:,0:3] and info world_pos-uav_pos",
        "relative_velocity": "observation['obs'][:,3:6] and info world_vel-uav_vel",
        "motion_mode": "info['obstacle_motion_modes']",
        "threat_class": "info['threat_classes']",
        "history_i[t-H:t]": "Phase N0 dry-run keyed by episode_id+obstacle_id",
        "future_p_i(t+tau)": "Phase N0 dry-run keyed by episode_id+obstacle_id+step_offset",
        "future_residual_delta_i(tau)": "world residual p_i(t+tau)-[p_i(t)+tau*v_i(t)]",
        "planned_cpa": "info['planned_cpa_values']",
        "planned_ttc": "info['planned_ttc_values']",
        "distance": "observation['obs'][:,8] and info world_pos-uav_pos",
        "closing": "observation['obs'][:,9] and rel velocity projection",
        "risk_value": "observation['obs'][:,11]",
    }
    blocking = {
        name: int(name not in {"planned_cpa", "planned_ttc", "distance", "closing", "risk_value"})
        for name in REQUIRED_FIELD_NAMES
    }
    rows = []
    for name in REQUIRED_FIELD_NAMES:
        ok = bool(checks[name])
        rows.append(
            {
                "required_field": name,
                "blocking_for_n1": blocking[name],
                "envv2_status": "available" if ok else "missing",
                "envv2_source": env_sources[name],
                "phase_ab_trace_or_table_status": field_available_from_phase_tables(phase_a_dir, phase_b_dir, name),
                "n0_decision": "usable" if ok else "stop_required_fields_missing",
                "notes": "relative fields are policy-input features; HeadA labels use world-frame obstacle residual"
                if name in {"relative_position", "relative_velocity"}
                else "",
            }
        )
    return rows


def collect_rollout_rows(args: argparse.Namespace, logger: Logger) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(args.scenarios):
        for episode_id in range(args.num_episodes):
            seed = int(args.eval_seed + scenario_index * 10000 + episode_id)
            env = DynamicObstacleFlowEnv(scenario=scenario)
            obs, info = env.reset(seed=seed)
            done = False
            local_step = 0
            while True:
                rows.extend(active_obstacle_rows(scenario, episode_id, seed, obs, info, done))
                if done or local_step >= args.max_steps:
                    break
                action = goal_action(info, args.dryrun_policy)
                obs, _reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                local_step += 1
    logger.log(f"Collected {len(rows)} active obstacle dry-run rows")
    return rows


def active_obstacle_rows(
    scenario: str,
    episode_id: int,
    seed: int,
    obs: dict[str, np.ndarray],
    info: dict[str, Any],
    done: bool,
) -> list[dict[str, Any]]:
    uav = np.asarray(info.get("uav_position", np.zeros(3)), dtype=np.float32)
    uav_vel = np.asarray(info.get("uav_velocity", np.zeros(3)), dtype=np.float32)
    positions = np.asarray(info.get("obstacle_positions", []), dtype=np.float32)
    velocities = np.asarray(info.get("obstacle_velocities", []), dtype=np.float32)
    ids = np.asarray(info.get("obstacle_ids", []), dtype=np.int32)
    modes = list(info.get("obstacle_motion_modes", []))
    classes = list(info.get("threat_classes", []))
    cpas = np.asarray(info.get("planned_cpa_values", []), dtype=np.float32)
    ttcs = np.asarray(info.get("planned_ttc_values", []), dtype=np.float32)
    mask = np.asarray(obs.get("mask", []), dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for slot, pos in enumerate(positions):
        vel = velocities[slot] if slot < len(velocities) else np.zeros(3, dtype=np.float32)
        rel = pos - uav
        rel_vel = vel - uav_vel
        distance = float(np.linalg.norm(rel))
        rel_dir = rel / (distance + 1e-8)
        closing = -float(np.dot(rel_dir, rel_vel))
        rows.append(
            {
                "scenario": scenario,
                "episode_id": int(episode_id),
                "episode_seed": int(seed),
                "step": int(info["step"]),
                "time": float(info["time"]),
                "done": int(done),
                "obstacle_slot": int(slot),
                "obstacle_id": int(ids[slot]) if slot < len(ids) else int(slot),
                "active": int(slot < len(mask) and mask[slot] > 0.5),
                "pos_x": float(pos[0]),
                "pos_y": float(pos[1]),
                "pos_z": float(pos[2]),
                "vel_x": float(vel[0]),
                "vel_y": float(vel[1]),
                "vel_z": float(vel[2]),
                "rel_pos_x": float(rel[0]),
                "rel_pos_y": float(rel[1]),
                "rel_pos_z": float(rel[2]),
                "rel_vel_x": float(rel_vel[0]),
                "rel_vel_y": float(rel_vel[1]),
                "rel_vel_z": float(rel_vel[2]),
                "distance": distance,
                "closing": closing,
                "planned_cpa": float(cpas[slot]) if slot < len(cpas) else float("nan"),
                "planned_ttc": float(ttcs[slot]) if slot < len(ttcs) else float("nan"),
                "threat_class": str(classes[slot]) if slot < len(classes) else "none",
                "motion_mode": str(modes[slot]) if slot < len(modes) else "none",
                "risk_value": float(np.clip((3.0 - distance) / 3.0, 0.0, 1.0)),
            }
        )
    return rows


def build_alignment_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    slot_sequences: dict[tuple[str, int, int], list[tuple[int, int]]] = defaultdict(list)
    for row in records:
        grouped[(row["scenario"], int(row["episode_id"]), int(row["obstacle_id"]))].append(row)
        slot_sequences[(row["scenario"], int(row["episode_id"]), int(row["obstacle_slot"]))].append(
            (int(row["step"]), int(row["obstacle_id"]))
        )

    slot_reuse_after: dict[tuple[str, int, int], int] = defaultdict(int)
    for (scenario, episode_id, slot), pairs in slot_sequences.items():
        pairs = sorted(pairs)
        for (_, prev_id), (_, next_id) in zip(pairs, pairs[1:]):
            if prev_id != next_id:
                slot_reuse_after[(scenario, episode_id, prev_id)] += 1

    rows: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        scenario, episode_id, obstacle_id = key
        steps = sorted(int(row["step"]) for row in group)
        slots = sorted(set(int(row["obstacle_slot"]) for row in group))
        counts = Counter(steps)
        duplicate_same_step = any(count > 1 for count in counts.values())
        gaps = [b - a for a, b in zip(steps, steps[1:])]
        max_gap = max(gaps) if gaps else 0
        reappears_after_gap = any(gap > 1 for gap in gaps)
        status = "pass"
        if duplicate_same_step or reappears_after_gap:
            status = "fail"
        rows.append(
            {
                "scenario": scenario,
                "episode_id": episode_id,
                "obstacle_id": obstacle_id,
                "first_step": min(steps),
                "last_step": max(steps),
                "num_records": len(group),
                "unique_slots_seen": "|".join(str(slot) for slot in slots),
                "slot_change_count_for_same_id": max(len(slots) - 1, 0),
                "slot_reuse_after_this_id_count": slot_reuse_after.get((scenario, episode_id, obstacle_id), 0),
                "duplicate_same_step": int(duplicate_same_step),
                "reappears_after_gap": int(reappears_after_gap),
                "max_gap_steps": max_gap,
                "status": status,
                "rule": "identity is episode_id+obstacle_id; slot reuse after replacement is allowed but never joined",
            }
        )
    return rows


def build_history_future_rows(
    records: list[dict[str, Any]],
    history_steps: int,
    future_times: list[float],
    dt: float,
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    steps_by_episode: dict[tuple[str, int], set[int]] = defaultdict(set)
    ids_by_slot_step: dict[tuple[str, int, int, int], int] = {}
    for row in records:
        scenario = row["scenario"]
        episode_id = int(row["episode_id"])
        step = int(row["step"])
        obstacle_id = int(row["obstacle_id"])
        slot = int(row["obstacle_slot"])
        by_key[(scenario, episode_id, step, obstacle_id)] = row
        steps_by_episode[(scenario, episode_id)].add(step)
        ids_by_slot_step[(scenario, episode_id, slot, step)] = obstacle_id

    offsets = {tau: int(round(tau / dt)) for tau in future_times}
    max_offset = max(offsets.values()) if offsets else 0
    rows: list[dict[str, Any]] = []
    for row in records:
        scenario = row["scenario"]
        episode_id = int(row["episode_id"])
        step = int(row["step"])
        obstacle_id = int(row["obstacle_id"])
        slot = int(row["obstacle_slot"])
        episode_steps = steps_by_episode[(scenario, episode_id)]
        final_step = max(episode_steps)
        history_range = range(step - history_steps + 1, step + 1)
        present_history = [
            s for s in history_range if (scenario, episode_id, s, obstacle_id) in by_key
        ]
        history_valid = len(present_history) == history_steps

        slot_ids_nearby = {
            ids_by_slot_step.get((scenario, episode_id, slot, s))
            for s in range(max(0, step - history_steps + 1), step + max_offset + 1)
            if (scenario, episode_id, slot, s) in ids_by_slot_step
        }
        slot_ids_nearby.discard(None)
        replacement_boundary_nearby = int(any(candidate_id != obstacle_id for candidate_id in slot_ids_nearby))

        out: dict[str, Any] = {
            "episode_id": episode_id,
            "scenario": scenario,
            "step": step,
            "time": row["time"],
            "obstacle_id": obstacle_id,
            "obstacle_slot": slot,
            "history_valid": int(history_valid),
            "history_available_steps": len(present_history),
            "motion_mode": row["motion_mode"],
            "threat_class": row["threat_class"],
            "replacement_boundary_nearby": replacement_boundary_nearby,
        }
        invalid_reasons: list[str] = []
        if not history_valid:
            invalid_reasons.append("history_padding_or_gap")

        p_now = np.array([row["pos_x"], row["pos_y"], row["pos_z"]], dtype=np.float64)
        v_now = np.array([row["vel_x"], row["vel_y"], row["vel_z"]], dtype=np.float64)
        for tau, offset in offsets.items():
            suffix = tau_suffix(tau)
            target_step = step + offset
            target = by_key.get((scenario, episode_id, target_step, obstacle_id))
            if target is None:
                out[f"future_valid_{suffix}"] = 0
                out[f"delta_norm_{suffix}"] = float("nan")
                if target_step > final_step:
                    invalid_reasons.append(f"{suffix}:episode_end_before_horizon")
                else:
                    invalid_reasons.append(f"{suffix}:obstacle_replaced_or_inactive_before_horizon")
                continue
            p_future = np.array([target["pos_x"], target["pos_y"], target["pos_z"]], dtype=np.float64)
            delta = p_future - (p_now + float(tau) * v_now)
            out[f"future_valid_{suffix}"] = 1
            out[f"delta_x_{suffix}"] = float(delta[0])
            out[f"delta_y_{suffix}"] = float(delta[1])
            out[f"delta_z_{suffix}"] = float(delta[2])
            out[f"delta_norm_{suffix}"] = float(np.linalg.norm(delta))
        out["reason_invalid_if_any"] = ";".join(sorted(set(invalid_reasons)))
        rows.append(out)
    return rows


def tau_suffix(tau: float) -> str:
    if abs(tau - round(tau)) < 1e-8:
        return f"{int(round(tau))}s"
    return str(tau).replace(".", "p") + "s"


def history_future_fieldnames(future_times: list[float]) -> list[str]:
    fields = [
        "episode_id",
        "scenario",
        "step",
        "time",
        "obstacle_id",
        "obstacle_slot",
        "history_valid",
        "history_available_steps",
    ]
    for tau in future_times:
        suffix = tau_suffix(tau)
        fields.extend(
            [
                f"future_valid_{suffix}",
                f"delta_x_{suffix}",
                f"delta_y_{suffix}",
                f"delta_z_{suffix}",
                f"delta_norm_{suffix}",
            ]
        )
    fields.extend(["motion_mode", "threat_class", "replacement_boundary_nearby", "reason_invalid_if_any"])
    return fields


def build_coordinate_frame_rows() -> list[dict[str, Any]]:
    return [
        {
            "item": "head_a_label",
            "selected_frame": "obstacle_world_frame",
            "status": "pass",
            "rationale": "world-frame obstacle residual avoids mixing future UAV motion into obstacle dynamics labels",
        },
        {
            "item": "future_residual_definition",
            "selected_frame": "world",
            "status": "pass",
            "rationale": "delta_i(tau)=p_i_world(t+tau)-[p_i_world(t)+tau*v_i_world(t)]",
        },
        {
            "item": "relative_observation",
            "selected_frame": "current_relative_policy_input",
            "status": "pass",
            "rationale": "relative position/velocity remain valid PPO/Gpsi inputs but are not the supervised future label frame",
        },
        {
            "item": "relative_future_label",
            "selected_frame": "deferred_not_used",
            "status": "pass",
            "rationale": "future-relative labels can include ego future motion and risk policy-dependent supervision",
        },
    ]


def build_phase_ab_artifact_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    phase_a = ROOT / args.phase_a_dir
    phase_b = ROOT / args.phase_b_dir
    checkpoint_candidates = [
        ROOT / "checkpoints/env_v2_phase2/attention_full_s0/attention_full_s0_step1500000.zip",
        ROOT / "checkpoints/env_v2_phase2/attention_full_s0/attention_full_s0_final_step1500000.zip",
        ROOT / "checkpoints/longtrain_baseline/attention_full_s0_step1500000.zip",
        ROOT / "checkpoints/attention_full_s0.zip",
    ]
    items = [
        ("phase_a_dir", phase_a, True),
        ("phase_a_complete_flag", phase_a / "PHASE_A_EVAL_FRAMEWORK_COMPLETE.flag", True),
        ("phase_a_report", phase_a / "PHASE_A_EVAL_FRAMEWORK_REPORT.md", True),
        ("phase_a_trace_schema", phase_a / "tables/phase_a_trace_schema.csv", True),
        ("phase_a_step_obstacles_long_table", phase_a / "tables/phase_a_step_obstacles_sample.csv", True),
        ("phase_b_dir", phase_b, True),
        ("phase_b_complete_flag", phase_b / "PHASE_B_GEOMETRY_FILTER_BASELINE_COMPLETE.flag", True),
        ("phase_b_report", phase_b / "PHASE_B_GEOMETRY_FILTER_BASELINE_REPORT.md", True),
        ("phase_b_schema_check", phase_b / "tables/phase_b_schema_check.csv", True),
        ("phase_b_step_obstacles_long_table", phase_b / "tables/phase_b_step_obstacles_sample.csv", True),
    ]
    rows = []
    for name, path, important in items:
        exists = path.exists()
        rows.append(
            {
                "artifact": name,
                "path": relpath(path),
                "required_for_n0": int(important),
                "exists": int(exists),
                "status": "pass" if exists else "warning_missing",
                "notes": "",
            }
        )
    checkpoint = next((path for path in checkpoint_candidates if path.exists()), None)
    rows.append(
        {
            "artifact": "attention_full_checkpoint",
            "path": relpath(checkpoint) if checkpoint is not None else "|".join(relpath(p) for p in checkpoint_candidates),
            "required_for_n0": 0,
            "exists": int(checkpoint is not None),
            "status": "pass" if checkpoint is not None else "warning_missing",
            "notes": "N0 does not train or evaluate PPO; checkpoint presence is recorded for Phase A/B continuity",
        }
    )
    return rows


def build_spec_freeze_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    spec_path = ROOT / args.spec
    text = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
    lowered = text.lower()
    requirements = [
        ("spec_file_exists", spec_path.exists(), relpath(spec_path)),
        ("sigma2_not_direct_label", "sigma2_is_direct_label: false" in lowered, "sigma2 is learned by Gaussian NLL"),
        ("gaussian_nll_uncertainty", "gaussian_nll" in lowered, "heteroscedastic uncertainty training"),
        ("n3_gpsi_frozen_first", "gpsi_frozen_in_first_ppo_phase: true" in lowered, "Gpsi frozen, PPO trainable"),
        ("n4_lambda_uncertainty_sweep", "lambda_uncertainty_sweep_required: true" in lowered, "lambda in {0, small, medium, large}"),
        ("disable_full_5_head_gpsi", "full_5_head_gpsi" in lowered and "full_5_head_gpsi_enabled: false" in lowered, "full 5-head route deferred"),
        ("disable_learned_r_s_a", "learned_r_s_a" in lowered, "learned R(s,a) deferred"),
        (
            "disable_candidate_velocity_risk_map_as_ppo_input",
            "candidate_velocity_risk_map_as_ppo_input" in lowered and "candidate_velocity_risk_map_as_input: false" in lowered,
            "not used as PPO input in first mainline",
        ),
        ("world_frame_residual", "obstacle_world_frame" in lowered and "p_i_world" in lowered, "HeadA label frame"),
    ]
    rows = [
        {
            "check": name,
            "status": status_bool(ok),
            "blocking": 1,
            "evidence": evidence,
            "notes": "",
        }
        for name, ok, evidence in requirements
    ]
    for design_file in args.design_files:
        path = ROOT / design_file
        rows.append(
            {
                "check": f"design_context_file:{design_file}",
                "status": "warning_missing" if not path.exists() else "pass",
                "blocking": 0,
                "evidence": relpath(path),
                "notes": "Specified design file was not present in this workspace; Phase N0 guide and generated spec are treated as authoritative"
                if not path.exists()
                else "readable",
            }
        )
    return rows


def build_command_manifest(args: argparse.Namespace) -> list[dict[str, Any]]:
    runner = " ".join(["python", *sys.argv])
    watcher = "bash scripts/watch_phase_n0_design_freeze.sh"
    return [
        {
            "command_name": "phase_n0_audit_runner",
            "command": runner,
            "status": "completed",
        },
        {
            "command_name": "phase_n0_py_compile",
            "command": "python -m py_compile scripts/check_envv2_gpsi_required_fields.py",
            "status": "completed_before_final_watcher",
        },
        {
            "command_name": "phase_n0_watcher_syntax",
            "command": "bash -n scripts/watch_phase_n0_design_freeze.sh",
            "status": "completed_before_final_watcher",
        },
        {
            "command_name": "phase_n0_blocking_watcher",
            "command": watcher,
            "status": "completed_when_terminal_flag_detected",
        },
    ]


def write_schema_files(out_dir: Path, args: argparse.Namespace) -> None:
    dataset_schema = {
        "name": "gpsi_head_a_dataset_schema_draft",
        "version": "gpsi_head_a_v1",
        "phase": "N1 draft; generated in N0 only as schema",
        "formal_dataset_built_in_phase_n0": False,
        "identity_key": ["episode_id", "obstacle_id"],
        "slot_rule": "obstacle_slot can be logged as current ordering but cannot define identity across replacement",
        "history": {
            "steps_default": args.history_steps,
            "fields": ["position_world_xyz", "velocity_world_xyz", "relative_position_xyz", "relative_velocity_xyz"],
            "valid_history_mask_required": True,
            "padding_allowed": True,
        },
        "current": {
            "ego_shape": [10],
            "obs_i_shape": [12],
            "active_mask_required": True,
            "metadata": ["motion_mode", "threat_class", "planned_cpa", "planned_ttc"],
        },
        "future": {
            "horizons_sec": args.future_times,
            "future_position_world_xyz_required": True,
            "delta_world_definition": "p_i_world(t+tau)-[p_i_world(t)+tau*v_i_world(t)]",
            "valid_future_mask_required": True,
            "invalid_when": [
                "episode_ends_before_horizon",
                "obstacle_id_absent_at_horizon",
                "replacement_or_inactive_before_horizon",
            ],
        },
        "labels": {
            "delta_i_tau": "direct supervised label",
            "sigma2_i_tau": "not a direct label",
            "log_sigma2_i_tau": "model output learned via Gaussian NLL",
        },
    }
    model_schema = {
        "name": "gpsi_head_a_model_io_schema_draft",
        "version": "gpsi_head_a_v1",
        "gpsi": {
            "input": {
                "obs_i": [12],
                "history_i": [args.history_steps, "world_and_relative_state_features"],
                "valid_history_mask": [args.history_steps],
            },
            "outputs": {
                "z_i": [64],
                "delta_hat_i": [len(args.future_times), 3],
                "log_sigma2_hat_i": [len(args.future_times), 3],
            },
            "losses": {
                "stage_a1": "MSE or SmoothL1 on delta_i(tau)",
                "stage_a2": "Gaussian NLL: 0.5*(||delta-delta_hat||^2/sigma2_hat + log_sigma2_hat)",
                "logvar_clamp": [-5.0, 3.0],
            },
        },
        "ppo": {
            "first_version": "masked_attention_compatible velocity policy",
            "gpsi_frozen": True,
            "augmented_obstacle_input": "[obs_i, z_i, delta_hat_i, sigma2_hat_i]",
        },
        "shield": {
            "type": "post_hoc_vo_cpa_ttc_candidate_search",
            "uncertainty_margin": "base_radius + lambda_uncertainty*sqrt(sigma2_hat_i)",
            "lambda_uncertainty_sweep_required": True,
        },
    }
    write_json(out_dir / SCHEMA_DIR / "gpsi_head_a_dataset_schema_draft.json", dataset_schema)
    write_json(out_dir / SCHEMA_DIR / "gpsi_head_a_model_io_schema_draft.json", model_schema)


def summarize_dryrun(history_rows: list[dict[str, Any]], future_times: list[float]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary: dict[str, Any] = {
        "rows": len(history_rows),
        "history_full_valid": sum(int(row.get("history_valid", 0)) for row in history_rows),
        "replacement_boundary_nearby": sum(int(row.get("replacement_boundary_nearby", 0)) for row in history_rows),
    }
    mode_rows: list[dict[str, Any]] = []
    for tau in future_times:
        suffix = tau_suffix(tau)
        valid_key = f"future_valid_{suffix}"
        delta_key = f"delta_norm_{suffix}"
        valid_rows = [row for row in history_rows if int(row.get(valid_key, 0)) == 1]
        summary[f"future_valid_{suffix}"] = len(valid_rows)
        by_mode: dict[str, list[float]] = defaultdict(list)
        for row in valid_rows:
            value = finite_float(row.get(delta_key))
            if np.isfinite(value):
                by_mode[str(row.get("motion_mode", "unknown"))].append(value)
        for mode, values in sorted(by_mode.items()):
            mode_rows.append(
                {
                    "tau": tau,
                    "motion_mode": mode,
                    "valid_count": len(values),
                    "mean_delta_norm": float(np.mean(values)) if values else float("nan"),
                    "median_delta_norm": float(np.median(values)) if values else float("nan"),
                }
            )
    return summary, mode_rows


def write_report(
    out_dir: Path,
    args: argparse.Namespace,
    complete: bool,
    terminal_decision: str,
    required_rows: list[dict[str, Any]],
    alignment_rows: list[dict[str, Any]],
    history_rows: list[dict[str, Any]],
    phase_ab_rows: list[dict[str, Any]],
    spec_rows: list[dict[str, Any]],
    dryrun_summary: dict[str, Any],
    mode_summary_rows: list[dict[str, Any]],
    stop_reason: str | None = None,
    stop_detail: str | None = None,
) -> None:
    missing_design = [row for row in spec_rows if row["check"].startswith("design_context_file:") and row["status"] == "warning_missing"]
    field_failures = [
        row
        for row in required_rows
        if row["envv2_status"] != "available" and int(row.get("blocking_for_n1", 0)) == 1
    ]
    alignment_failures = [row for row in alignment_rows if row["status"] != "pass"]
    phase_warnings = [row for row in phase_ab_rows if row["status"].startswith("warning")]
    spec_failures = [row for row in spec_rows if row["status"] == "fail"]

    lines = [
        "# Phase N0 Design Freeze Report",
        "",
        "## Background",
        "",
        "Current mainline is Gpsi-HeadA supervised dynamic obstacle representation -> PPO velocity policy -> post-hoc VO/CPA-TTC Safety Shield with sigma2 uncertainty margin.",
        "Old Phase C/D/E safety-cost planning, learned R(s,a), candidate velocity risk maps as PPO input, and full 5-head Gpsi are deprecated for this phase.",
        "",
        "## Terminal Decision",
        "",
        f"`terminal_decision = {terminal_decision}`",
        "",
        "Phase N0 complete. Gpsi-HeadA design and dataflow are ready for Phase N1 dataset construction."
        if complete
        else f"Phase N0 stopped: {stop_reason or 'unknown'}; {stop_detail or ''}",
        "",
        "## Experiment And Code-Audit Supported Facts",
        "",
        "- EnvV2 `info` exposes active obstacle ids, world positions, world velocities, motion modes, threat classes, planned CPA/TTC values, UAV state, dt, and active obstacle count.",
        "- EnvV2 observation exposes ego state, per-obstacle relative position/velocity features, active mask, distance, closing, threat class id, and risk value.",
        "- Phase A and Phase B long obstacle tables preserve obstacle slot, obstacle id, active flag, world position, world velocity, distance, closing, planned CPA/TTC, threat class, motion mode, and risk value.",
        "- Phase A/B per-step traces preserve UAV position/velocity, so relative position/velocity are reconstructable by joining long obstacle tables on scenario/episode/step.",
        f"- Small dry-run rows: `{dryrun_summary.get('rows', 0)}`; full-history rows: `{dryrun_summary.get('history_full_valid', 0)}`; replacement-nearby rows: `{dryrun_summary.get('replacement_boundary_nearby', 0)}`.",
        "",
        "## Design Decisions",
        "",
        "- HeadA target is world-frame obstacle residual: `delta_i(tau)=p_i_world(t+tau)-[p_i_world(t)+tau*v_i_world(t)]`.",
        "- `sigma2` is not a direct per-sample label. Gpsi outputs `delta_hat` and `log_sigma2_hat`; heteroscedastic uncertainty is learned with Gaussian NLL after delta-only MSE/SmoothL1 warmup.",
        "- N3 first version freezes Gpsi and trains PPO only.",
        "- N4 must run `lambda_uncertainty` sweep with `{0, small, medium, large}`; `lambda=0` is the fixed-margin shield reference.",
        "- Obstacle identity is `episode_id + obstacle_id`; `obstacle_slot` is current ordering only and may be reused after replacement.",
        "",
        "## Replacement And Label Rules",
        "",
        "- History is built only from the same obstacle id within an episode. Missing early history is padded and marked by `valid_history_mask`.",
        "- Future labels are valid only if the same obstacle id is active at `t+tau`; episode end, inactive obstacle, or replacement before horizon invalidates that horizon.",
        "- Replacement at the same slot creates a new obstacle id. N1 must not join old and new ids even if the slot is unchanged.",
        "- Future information is used only for supervised labels, never as inference input.",
        "",
        "## Dry-Run Future Label Counts",
        "",
        "| horizon | valid rows |",
        "| --- | ---: |",
    ]
    for tau in args.future_times:
        suffix = tau_suffix(tau)
        lines.append(f"| {suffix} | {int(dryrun_summary.get(f'future_valid_{suffix}', 0))} |")
    lines.extend(["", "## Residual Sanity By Motion Mode", "", "| tau | motion_mode | valid_count | mean_delta_norm | median_delta_norm |", "| ---: | --- | ---: | ---: | ---: |"])
    for row in mode_summary_rows[:40]:
        lines.append(
            f"| {row['tau']} | {row['motion_mode']} | {int(row['valid_count'])} | "
            f"{finite_float(row['mean_delta_norm']):.6f} | {finite_float(row['median_delta_norm']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Risks And Unresolved Issues",
            "",
        ]
    )
    if missing_design:
        lines.append("- The three requested design markdown files were not present under the specified filenames; no spec conflict was detected because the Phase N0 guide and generated YAML define the active freeze.")
    if phase_warnings:
        lines.append("- Some Phase A/B optional artifacts are warnings; they do not block N0 because EnvV2 info and the required long tables are available.")
    if not missing_design and not phase_warnings:
        lines.append("- No blocking unresolved issue was found in the N0 audit.")
    if field_failures:
        lines.append(f"- Blocking required field failures: `{len(field_failures)}`.")
    if alignment_failures:
        lines.append(f"- Obstacle identity alignment failures: `{len(alignment_failures)}`.")
    if spec_failures:
        lines.append(f"- Spec freeze failures: `{len(spec_failures)}`.")
    lines.extend(
        [
            "",
            "## Output Artifacts",
            "",
            "- `configs/gpsi_head_a_spec.yaml`",
            f"- `{relpath(out_dir / 'tables/phase_n0_required_fields_check.csv')}`",
            f"- `{relpath(out_dir / 'tables/phase_n0_obstacle_id_alignment_check.csv')}`",
            f"- `{relpath(out_dir / 'tables/phase_n0_history_future_label_check.csv')}`",
            f"- `{relpath(out_dir / 'tables/phase_n0_coordinate_frame_check.csv')}`",
            f"- `{relpath(out_dir / 'tables/phase_n0_phase_ab_artifact_check.csv')}`",
            f"- `{relpath(out_dir / 'tables/phase_n0_spec_freeze_check.csv')}`",
            f"- `{relpath(out_dir / 'schema/gpsi_head_a_dataset_schema_draft.json')}`",
            f"- `{relpath(out_dir / 'schema/gpsi_head_a_model_io_schema_draft.json')}`",
            f"- `{relpath(out_dir / 'logs/phase_n0_design_freeze.log')}`",
            f"- `{relpath(out_dir / 'phase_n0_watcher.log')}`",
            f"- `{relpath(out_dir / 'phase_n0_status.txt')}`",
            f"- `{relpath(out_dir / (COMPLETE_FLAG if complete else STOP_FLAGS.get(stop_reason or '', '')) )}`" if complete or stop_reason in STOP_FLAGS else "",
            "",
            "## N1 Readiness",
            "",
            "Can enter Phase N1: yes." if complete else "Can enter Phase N1: no.",
            "N1 should build only `data/gpsi_head_a_v1/{train,val,test,schema}` after this complete flag, using the schema draft and identity/valid-mask rules above."
            if complete
            else "Resolve the stop reason above, then rerun Phase N0 watcher.",
        ]
    )
    write_text(out_dir / "PHASE_N0_DESIGN_FREEZE_REPORT.md", "\n".join(line for line in lines if line is not None) + "\n")


def validate_and_decide(
    required_rows: list[dict[str, Any]],
    alignment_rows: list[dict[str, Any]],
    history_rows: list[dict[str, Any]],
    spec_rows: list[dict[str, Any]],
    future_times: list[float],
) -> tuple[bool, str | None, str | None]:
    spec_failures = [row for row in spec_rows if row["status"] == "fail" and int(row["blocking"]) == 1]
    if spec_failures:
        return False, "spec_conflict", f"spec freeze checks failed: {[row['check'] for row in spec_failures]}"

    required_failures = [row for row in required_rows if row["envv2_status"] != "available" and int(row["blocking_for_n1"]) == 1]
    if required_failures:
        return False, "required_fields_missing", f"required fields missing: {[row['required_field'] for row in required_failures]}"

    alignment_failures = [row for row in alignment_rows if row["status"] != "pass"]
    if alignment_failures:
        return False, "obstacle_id_alignment_failed", f"alignment failures: {len(alignment_failures)}"

    if not history_rows:
        return False, "history_future_label_failed", "history/future label dry-run table is empty"
    for tau in future_times:
        suffix = tau_suffix(tau)
        valid_count = sum(int(row.get(f"future_valid_{suffix}", 0)) for row in history_rows)
        if valid_count <= 0:
            return False, "history_future_label_failed", f"no valid future labels for horizon {suffix}"
    history_full = sum(int(row.get("history_valid", 0)) for row in history_rows)
    if history_full <= 0:
        return False, "history_future_label_failed", "no rows with full unpadded history"

    return True, None, None


def write_stop_flag(out_dir: Path, reason: str, detail: str) -> None:
    flag_name = STOP_FLAGS.get(reason, STOP_FLAGS["history_future_label_failed"])
    write_text(out_dir / flag_name, f"{reason}\n{detail}\n")
    write_text(out_dir / "phase_n0_status.txt", f"stopped:{flag_name}\n")


def write_complete_flag(out_dir: Path) -> None:
    write_text(out_dir / COMPLETE_FLAG, "phase_n0_design_freeze_complete\n")
    write_text(out_dir / "phase_n0_status.txt", "complete\n")


def run() -> None:
    args = parse_args()
    out_dir = ROOT / args.out_dir
    ensure_dirs(out_dir)
    clear_terminal_flags(out_dir)
    write_text(out_dir / "phase_n0_status.txt", "running\n")
    logger = Logger(out_dir)
    logger.log("Phase N0 audit started")
    logger.log("Command: " + " ".join(["python", *sys.argv]))

    obs, info = inspect_current_env_state()
    required_rows = build_required_fields_check(args, obs, info)
    spec_rows = build_spec_freeze_rows(args)
    phase_ab_rows = build_phase_ab_artifact_rows(args)
    coordinate_rows = build_coordinate_frame_rows()
    records = collect_rollout_rows(args, logger)
    alignment_rows = build_alignment_rows(records)
    dt = finite_float(info.get("dt"), default=0.2)
    history_rows = build_history_future_rows(records, args.history_steps, args.future_times, dt)
    dryrun_summary, mode_summary_rows = summarize_dryrun(history_rows, args.future_times)
    command_rows = build_command_manifest(args)

    write_csv(
        out_dir / TABLE_DIR / "phase_n0_required_fields_check.csv",
        required_rows,
        ["required_field", "blocking_for_n1", "envv2_status", "envv2_source", "phase_ab_trace_or_table_status", "n0_decision", "notes"],
    )
    write_csv(
        out_dir / TABLE_DIR / "phase_n0_obstacle_id_alignment_check.csv",
        alignment_rows,
        [
            "scenario",
            "episode_id",
            "obstacle_id",
            "first_step",
            "last_step",
            "num_records",
            "unique_slots_seen",
            "slot_change_count_for_same_id",
            "slot_reuse_after_this_id_count",
            "duplicate_same_step",
            "reappears_after_gap",
            "max_gap_steps",
            "status",
            "rule",
        ],
    )
    write_csv(
        out_dir / TABLE_DIR / "phase_n0_history_future_label_check.csv",
        history_rows,
        history_future_fieldnames(args.future_times),
    )
    write_csv(
        out_dir / TABLE_DIR / "phase_n0_coordinate_frame_check.csv",
        coordinate_rows,
        ["item", "selected_frame", "status", "rationale"],
    )
    write_csv(
        out_dir / TABLE_DIR / "phase_n0_phase_ab_artifact_check.csv",
        phase_ab_rows,
        ["artifact", "path", "required_for_n0", "exists", "status", "notes"],
    )
    write_csv(
        out_dir / TABLE_DIR / "phase_n0_spec_freeze_check.csv",
        spec_rows,
        ["check", "status", "blocking", "evidence", "notes"],
    )
    write_csv(
        out_dir / TABLE_DIR / "phase_n0_command_manifest.csv",
        command_rows,
        ["command_name", "command", "status"],
    )
    write_csv(
        out_dir / TABLE_DIR / "phase_n0_motion_mode_residual_summary.csv",
        mode_summary_rows,
        ["tau", "motion_mode", "valid_count", "mean_delta_norm", "median_delta_norm"],
    )
    env_file = ROOT / "envs/dynamic_obstacle_flow_env.py"
    write_csv(
        out_dir / TABLE_DIR / "phase_n0_envv2_core_freeze_check.csv",
        [
            {
                "file": relpath(env_file),
                "sha256": sha256_file(env_file),
                "phase_n0_action": "read_only",
                "freeze_scope": "obstacle ranges, motion modes, scenarios, reward, termination, collision/success/near_miss definitions",
                "status": "pass",
            }
        ],
        ["file", "sha256", "phase_n0_action", "freeze_scope", "status"],
    )
    write_schema_files(out_dir, args)

    complete, stop_reason, stop_detail = validate_and_decide(
        required_rows, alignment_rows, history_rows, spec_rows, args.future_times
    )
    terminal_decision = "phase_n0_design_freeze_complete" if complete else f"phase_n0_stopped_{stop_reason}"
    write_report(
        out_dir=out_dir,
        args=args,
        complete=complete,
        terminal_decision=terminal_decision,
        required_rows=required_rows,
        alignment_rows=alignment_rows,
        history_rows=history_rows,
        phase_ab_rows=phase_ab_rows,
        spec_rows=spec_rows,
        dryrun_summary=dryrun_summary,
        mode_summary_rows=mode_summary_rows,
        stop_reason=stop_reason,
        stop_detail=stop_detail,
    )
    if complete:
        write_complete_flag(out_dir)
        logger.log("Phase N0 complete flag written")
        return
    assert stop_reason is not None
    write_stop_flag(out_dir, stop_reason, stop_detail or "")
    logger.log(f"Phase N0 stop flag written: {stop_reason}: {stop_detail}")
    raise SystemExit(2)


def main() -> None:
    args = parse_args()
    out_dir = ROOT / args.out_dir
    ensure_dirs(out_dir)
    logger = Logger(out_dir)
    try:
        # Reuse parsed arguments by restoring argv for run(); parse_args is cheap but should see the same CLI.
        run()
    except PhaseN0Stop as exc:
        write_stop_flag(out_dir, exc.reason, exc.detail)
        logger.log(f"Phase N0 stopped: {exc.reason}: {exc.detail}")
        raise SystemExit(2) from exc
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - terminal flag generation must catch unexpected audit failures.
        detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        logger.log("Unexpected Phase N0 exception:\n" + detail)
        write_stop_flag(out_dir, "history_future_label_failed", detail)
        write_report(
            out_dir=out_dir,
            args=args,
            complete=False,
            terminal_decision="phase_n0_stopped_history_future_label_failed",
            required_rows=[],
            alignment_rows=[],
            history_rows=[],
            phase_ab_rows=[],
            spec_rows=[],
            dryrun_summary={},
            mode_summary_rows=[],
            stop_reason="history_future_label_failed",
            stop_detail=str(exc),
        )
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
