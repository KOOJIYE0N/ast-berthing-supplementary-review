"""Deterministic rollout and safety logging utilities for the AST revision."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> dict[str, Any]:
    """Seed Python, NumPy, and Torch for a reproducible policy-sampling stream."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return {
        "python_seed": seed,
        "numpy_seed": seed,
        "torch_seed": seed,
        "cuda_seed": seed if torch.cuda.is_available() else None,
        "deterministic_algorithms": deterministic,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def reset_environment_exact(pnp: Any, env_seed: int) -> None:
    """Reset all MuJoCo data plus the scene randomizer using exactly env_seed."""
    # BerthingEnv.reset() positions the joints and randomizes the model, but it does
    # not call mj_resetData. Without this call, qacc_warmstart, ctrl, solver state,
    # time, and other MjData fields leak from the preceding episode.
    pnp.env.reset(step=False)
    if hasattr(pnp, "_scene_reset_idx"):
        pnp._scene_reset_idx = 0
    pnp.reset(seed=int(env_seed))


def runtime_metadata() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "mujoco": mujoco.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _body_id(model: mujoco.MjModel, name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))


def _body_pose_velocity(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> dict[str, np.ndarray]:
    bid = _body_id(model, name)
    if bid < 0:
        return {}
    velocity = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, bid, velocity, 0)
    return {
        "position": np.asarray(data.xpos[bid], dtype=np.float64).copy(),
        "rotation": np.asarray(data.xmat[bid], dtype=np.float64).reshape(3, 3).copy(),
        "quaternion": np.asarray(data.xquat[bid], dtype=np.float64).copy(),
        "angular_velocity": velocity[:3].copy(),
        "linear_velocity": velocity[3:].copy(),
    }


def _safe_angle(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)))


@dataclass
class SafetyTraceLogger:
    hz: int
    rows: list[dict[str, Any]] = field(default_factory=list)
    cumulative_contact_impulse_n_s: float = 0.0
    peak_contact_force_all_sim_steps_n: float = 0.0
    cumulative_docking_contact_impulse_n_s: float = 0.0
    peak_docking_contact_force_n: float = 0.0
    signed_actuator_work_j: float = 0.0
    positive_actuator_work_j: float = 0.0
    absolute_actuator_work_j: float = 0.0
    peak_absolute_actuator_power_w: float = 0.0

    sim_steps_observed: int = 0
    integrated_duration_s: float = 0.0
    last_observed_time_s: float | None = None
    first_observed_contact_s: float | None = None
    post_contact_steps_observed: int = 0
    post_contact_steps_with_contact: int = 0
    all_dof_signed_work_j: float = 0.0
    all_dof_positive_work_j: float = 0.0
    all_dof_absolute_work_j: float = 0.0

    marker_names = (
        "docking_marker_lower",
        "docking_marker_upper",
        "wrist_docking_marker_lower",
        "wrist_docking_marker_upper",
    )

    def observe_sim_step(self, pnp: Any) -> None:
        """Accumulate contact load and generalized actuator work each simulator step."""
        model, data = pnp.env.model, pnp.env.data
        total_force = 0.0
        docking_force = 0.0
        docking_now = False
        docking_pairs = (
            frozenset(("docking_marker_lower", "wrist_docking_marker_lower")),
            frozenset(("docking_marker_upper", "wrist_docking_marker_upper")),
        )
        for idx in range(int(data.ncon)):
            con = data.contact[idx]
            wrench = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, idx, wrench)
            force_norm = float(np.linalg.norm(wrench[:3]))
            total_force += force_norm
            self.peak_contact_force_all_sim_steps_n = max(
                self.peak_contact_force_all_sim_steps_n, force_norm
            )
            b1, b2 = int(model.geom_bodyid[con.geom1]), int(model.geom_bodyid[con.geom2])
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or f"body_{b1}"
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or f"body_{b2}"
            if frozenset((n1, n2)) in docking_pairs:
                docking_now = True
                docking_force += force_norm
                self.peak_docking_contact_force_n = max(self.peak_docking_contact_force_n, force_norm)
        dt = float(model.opt.timestep)
        t = float(data.time)
        if self.last_observed_time_s is not None:
            assert abs((t - self.last_observed_time_s) - dt) < 1e-8, "Missed or duplicated physics sample"
        self.last_observed_time_s = t
        self.sim_steps_observed += 1
        self.integrated_duration_s += dt
        if docking_now and self.first_observed_contact_s is None:
            self.first_observed_contact_s = t
        if self.first_observed_contact_s is not None:
            self.post_contact_steps_observed += 1
            self.post_contact_steps_with_contact += int(docking_now)
        full_power = np.asarray(data.qfrc_actuator) * np.asarray(data.qvel)
        self.all_dof_signed_work_j += float(full_power.sum()) * dt
        self.all_dof_positive_work_j += float(np.maximum(full_power, 0).sum()) * dt
        self.all_dof_absolute_work_j += float(np.abs(full_power).sum()) * dt
        self.cumulative_contact_impulse_n_s += total_force * dt
        self.cumulative_docking_contact_impulse_n_s += docking_force * dt
        joint_power = []
        for name in pnp.joint_names:
            jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
            dadr = int(model.jnt_dofadr[jid])
            joint_power.append(float(data.qfrc_actuator[dadr] * data.qvel[dadr]))
        signed_power = float(np.sum(joint_power))
        absolute_power = float(np.sum(np.abs(joint_power)))
        self.signed_actuator_work_j += signed_power * dt
        self.positive_actuator_work_j += float(np.sum(np.maximum(joint_power, 0.0))) * dt
        self.absolute_actuator_work_j += absolute_power * dt
        self.peak_absolute_actuator_power_w = max(self.peak_absolute_actuator_power_w, absolute_power)

    def observe_contacts(self, pnp: Any) -> None:
        """Backward-compatible alias used by revision_eval_v1."""
        self.observe_sim_step(pnp)

    def tick(self, pnp: Any, policy_step: int, action: np.ndarray) -> None:
        parser = pnp.env
        model, data = parser.model, parser.data
        bodies = {name: _body_pose_velocity(model, data, name) for name in ("gripper_link", *self.marker_names)}

        dl, du = bodies["docking_marker_lower"], bodies["docking_marker_upper"]
        wl, wu = bodies["wrist_docking_marker_lower"], bodies["wrist_docking_marker_upper"]
        target_mid = 0.5 * (dl["position"] + du["position"])
        wrist_mid = 0.5 * (wl["position"] + wu["position"])
        target_v = 0.5 * (dl["linear_velocity"] + du["linear_velocity"])
        wrist_v = 0.5 * (wl["linear_velocity"] + wu["linear_velocity"])
        target_w = 0.5 * (dl["angular_velocity"] + du["angular_velocity"])
        wrist_w = 0.5 * (wl["angular_velocity"] + wu["angular_velocity"])

        contacts: list[dict[str, Any]] = []
        total_force = 0.0
        max_force = 0.0
        for idx in range(int(data.ncon)):
            con = data.contact[idx]
            wrench = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, idx, wrench)
            b1 = int(model.geom_bodyid[con.geom1])
            b2 = int(model.geom_bodyid[con.geom2])
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or f"body_{b1}"
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or f"body_{b2}"
            force_norm = float(np.linalg.norm(wrench[:3]))
            total_force += force_norm
            max_force = max(max_force, force_norm)
            contacts.append({
                "body_1": n1,
                "body_2": n2,
                "force_torque_contact_frame": wrench.tolist(),
                "force_norm_n": force_norm,
                "distance_m": float(con.dist),
            })

        contact_pairs = {frozenset((c["body_1"], c["body_2"])) for c in contacts}
        docking_contact_now = any(pair in contact_pairs for pair in (
            frozenset(("docking_marker_lower", "wrist_docking_marker_lower")),
            frozenset(("docking_marker_upper", "wrist_docking_marker_upper")),
        ))
        docking_contact_force_now = max(
            (c["force_norm_n"] for c in contacts if frozenset((c["body_1"], c["body_2"])) in (
                frozenset(("docking_marker_lower", "wrist_docking_marker_lower")),
                frozenset(("docking_marker_upper", "wrist_docking_marker_upper")),
            )), default=0.0,
        )

        joint_q = np.asarray(pnp.get_joint_state(), dtype=np.float64).copy()
        joint_qvel = []
        joint_margins = []
        joint_tau = []
        joint_power = []
        for name in pnp.joint_names:
            jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
            qadr, dadr = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
            joint_qvel.append(float(data.qvel[dadr]))
            tau = float(data.qfrc_actuator[dadr])
            joint_tau.append(tau)
            joint_power.append(tau * float(data.qvel[dadr]))
            if bool(model.jnt_limited[jid]):
                lo, hi = model.jnt_range[jid]
                joint_margins.append(float(min(data.qpos[qadr] - lo, hi - data.qpos[qadr])))
            else:
                joint_margins.append(float("nan"))

        position_error = wrist_mid - target_mid
        rel_rot = dl["rotation"].T @ wl["rotation"]
        attitude_error = float(np.arccos(np.clip((np.trace(rel_rot) - 1.0) / 2.0, -1.0, 1.0)))
        self.rows.append({
            "policy_step": int(policy_step),
            "sim_time_s": float(data.time),
            "action": np.asarray(action, dtype=np.float64).tolist(),
            "joint_position_rad": joint_q.tolist(),
            "joint_velocity_rad_s": joint_qvel,
            "joint_generalized_actuator_torque_nm": joint_tau,
            "joint_mechanical_power_w": joint_power,
            "actuator_ctrl": np.asarray(data.ctrl, dtype=np.float64).tolist(),
            "actuator_force": np.asarray(data.actuator_force, dtype=np.float64).tolist(),
            "joint_limit_margin_rad": joint_margins,
            "ee_position_m": bodies["gripper_link"]["position"].tolist(),
            "ee_quaternion_wxyz": bodies["gripper_link"]["quaternion"].tolist(),
            "ee_rotation_matrix": bodies["gripper_link"]["rotation"].tolist(),
            "ee_linear_velocity_m_s": bodies["gripper_link"]["linear_velocity"].tolist(),
            "ee_angular_velocity_rad_s": bodies["gripper_link"]["angular_velocity"].tolist(),
            "target_marker_midpoint_m": target_mid.tolist(),
            "wrist_marker_midpoint_m": wrist_mid.tolist(),
            "marker_midpoint_distance_m": float(np.linalg.norm(wrist_mid - target_mid)),
            "relative_position_error_xyz_m": position_error.tolist(),
            "marker_pair_distance_lower_m": float(np.linalg.norm(wl["position"] - dl["position"])),
            "marker_pair_distance_upper_m": float(np.linalg.norm(wu["position"] - du["position"])),
            "marker_axis_alignment_error_rad": _safe_angle(du["position"] - dl["position"], wu["position"] - wl["position"]),
            "relative_marker_attitude_error_rad": attitude_error,
            "relative_marker_linear_velocity_m_s": (wrist_v - target_v).tolist(),
            "relative_marker_speed_m_s": float(np.linalg.norm(wrist_v - target_v)),
            "relative_marker_angular_velocity_rad_s": (wrist_w - target_w).tolist(),
            "relative_marker_angular_speed_rad_s": float(np.linalg.norm(wrist_w - target_w)),
            "geometric_contact_success_flag": bool(pnp.success),
            "docking_marker_contact_now": docking_contact_now,
            "docking_contact_force_n": docking_contact_force_now,
            "contact_count": int(data.ncon),
            "total_contact_force_n": total_force,
            "max_contact_force_n": max_force,
            "contacts": contacts,
        })

    def terminal_metrics(self) -> dict[str, Any]:
        if not self.rows:
            return {}
        last = self.rows[-1]
        return {
            "terminal_marker_midpoint_distance_m": last["marker_midpoint_distance_m"],
            "terminal_marker_axis_alignment_error_rad": last["marker_axis_alignment_error_rad"],
            "terminal_relative_marker_speed_m_s": last["relative_marker_speed_m_s"],
            "terminal_ee_linear_speed_m_s": float(np.linalg.norm(last["ee_linear_velocity_m_s"])),
            "terminal_ee_angular_speed_rad_s": float(np.linalg.norm(last["ee_angular_velocity_rad_s"])),
            "terminal_max_contact_force_n": last["max_contact_force_n"],
            "peak_contact_force_n": self.peak_contact_force_all_sim_steps_n,
            "contact_impulse_magnitude_sum_n_s": self.cumulative_contact_impulse_n_s,
            "docking_contact_impulse_magnitude_sum_n_s": self.cumulative_docking_contact_impulse_n_s,
            "peak_docking_contact_force_n": self.peak_docking_contact_force_n,
            "signed_actuator_work_j": self.signed_actuator_work_j,
            "positive_actuator_work_j": self.positive_actuator_work_j,
            "absolute_actuator_work_j": self.absolute_actuator_work_j,
            "peak_absolute_actuator_power_w": self.peak_absolute_actuator_power_w,
            "minimum_joint_limit_margin_rad": self.minimum_joint_limit_margin(),
            "physics_steps_observed": self.sim_steps_observed,
            "physics_integrated_duration_s": self.integrated_duration_s,
            "post_contact_physics_samples": self.post_contact_steps_observed,
            "post_contact_physics_samples_with_contact": self.post_contact_steps_with_contact,
            "post_contact_physics_contact_fraction": (self.post_contact_steps_with_contact / self.post_contact_steps_observed if self.post_contact_steps_observed else None),
            "all_six_dof_signed_actuator_work_j": self.all_dof_signed_work_j,
            "all_six_dof_positive_actuator_work_j": self.all_dof_positive_work_j,
            "all_six_dof_absolute_actuator_work_j": self.all_dof_absolute_work_j,
            "velocity_reference": "body origin in world axes, mjOBJ_XBODY; consistent with xpos",
            "legacy_work_scope": "five commanded joints; wrist_rotate excluded",
            "all_dof_work_scope": "all six model hinge DOFs, including wrist_rotate holding actuator",
            "force_peak_scope": "maximum single-contact force magnitude, not sum of simultaneous forces",
            "impulse_scope": "time integral of sum of contact force magnitudes, not a vector impulse",
        }

    def minimum_joint_limit_margin(self) -> float:
        values = np.asarray([m for r in self.rows for m in r["joint_limit_margin_rad"]], dtype=float)
        finite = values[np.isfinite(values)]
        return float(np.min(finite)) if finite.size else float("nan")

    def action_sha256(self) -> str:
        actions = np.asarray([r["action"] for r in self.rows], dtype=np.float32)
        return hashlib.sha256(actions.tobytes()).hexdigest()

    def quantized_action_sha256(self, decimals: int = 5) -> str:
        """Stable fingerprint that ignores sub-micro numerical noise from CUDA/EGL."""
        actions = np.asarray([r["action"] for r in self.rows], dtype=np.float32)
        quantized = np.round(actions, decimals=decimals)
        return hashlib.sha256(quantized.tobytes()).hexdigest()

    def save_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in self.rows:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
