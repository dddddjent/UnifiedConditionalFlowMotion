"""Decode the two motion representations and export a canonical BVH skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from unified_motion.config import ModelConfig


PARENTS = np.array(
    (-1, 0, 1, 2, 3, 4, 5, 3, 7, 8, 9, 3, 11, 12, 13, 0, 15, 16, 17, 18, 15, 20, 21, 22),
    dtype=np.int64,
)
JOINT_NAMES = (
    "ROOT",
    "C_spine0001_bind_JNT",
    "C_spine0002_bind_JNT",
    "C_spine0003_bind_JNT",
    "C_neck0001_bind_JNT",
    "C_neck0002_bind_JNT",
    "C_head_bind_JNT",
    "L_clavicle_bind_JNT",
    "L_armUpper0001_bind_JNT",
    "L_armLower0001_bind_JNT",
    "L_hand0001_bind_JNT",
    "R_clavicle_bind_JNT",
    "R_armUpper0001_bind_JNT",
    "R_armLower0001_bind_JNT",
    "R_hand0001_bind_JNT",
    "C_pelvis0001_bind_JNT",
    "L_legUpper0001_bind_JNT",
    "L_legLower0001_bind_JNT",
    "L_foot0001_bind_JNT",
    "L_foot0002_bind_JNT",
    "R_legUpper0001_bind_JNT",
    "R_legLower0001_bind_JNT",
    "R_foot0001_bind_JNT",
    "R_foot0002_bind_JNT",
)


@dataclass
class DecodedMotion:
    direct_positions: np.ndarray
    fk_positions: np.ndarray
    local_rotations: np.ndarray
    root_positions: np.ndarray
    offsets: np.ndarray


def rotation6d(value: np.ndarray, *, interleaved: bool) -> np.ndarray:
    """SMG stores concatenated columns; AnyTop stores row-interleaved columns."""
    if interleaved:
        columns = value.reshape(*value.shape[:-1], 3, 2)
        first, second = columns[..., 0], columns[..., 1]
        first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-8)
        second = second - (first * second).sum(-1, keepdims=True) * first
        assert (np.linalg.norm(second, axis=-1) > 1e-8).all(), "Degenerate AnyTop 6D rotation."
        second = second / np.linalg.norm(second, axis=-1, keepdims=True)
        third = np.cross(first, second)
    else:
        first, second = value[..., :3], value[..., 3:]
        first = first / (np.linalg.norm(first, axis=-1, keepdims=True) + 1e-8)
        third = np.cross(first, second)
        assert (np.linalg.norm(third, axis=-1) > 1e-8).all(), "Degenerate SMG 6D rotation."
        third = third / (np.linalg.norm(third, axis=-1, keepdims=True) + 1e-8)
        second = np.cross(third, first)
    return np.stack((first, second, third), -1)


def skeleton_offsets(skeleton: np.ndarray) -> np.ndarray:
    positions = skeleton.reshape(24, 12)[:, :3]
    offsets = positions.copy()
    offsets[1:] -= positions[PARENTS[1:]]
    offsets[0] = 0
    return offsets


def forward_kinematics(
    local_rotations: np.ndarray, root: np.ndarray, offsets: np.ndarray
) -> np.ndarray:
    frames = len(root)
    global_rotations = np.empty_like(local_rotations)
    positions = np.empty((frames, 24, 3), dtype=np.float64)
    positions[:, 0], global_rotations[:, 0] = root, local_rotations[:, 0]
    for joint in range(1, 24):
        parent = PARENTS[joint]
        global_rotations[:, joint] = global_rotations[:, parent] @ local_rotations[:, joint]
        positions[:, joint] = positions[:, parent] + np.einsum(
            "fij,j->fi", global_rotations[:, parent], offsets[joint]
        )
    return positions.astype(np.float32)


def decode_smg(features: np.ndarray, skeleton: np.ndarray) -> DecodedMotion:
    frames = len(features)
    angles = np.cumsum(np.concatenate(([0.0], features[:-1, 0])))
    rotation_vectors = np.zeros((frames, 3))
    rotation_vectors[:, 1] = -angles
    root_world = Rotation.from_rotvec(rotation_vectors).as_matrix()
    root_delta = np.zeros((frames, 3))
    root_delta[1:, 0], root_delta[1:, 2] = features[:-1, 1], features[:-1, 2]
    root = np.cumsum(np.einsum("fij,fj->fi", root_world, root_delta), 0)
    root[:, 1] = features[:, 3]
    positions = features[:, 148:220].reshape(frames, 24, 3)
    direct = np.einsum("fij,fkj->fki", root_world, positions)
    direct[..., (0, 2)] += root[:, None, (0, 2)]
    global_rotations = root_world[:, None] @ rotation6d(
        features[:, 4:148].reshape(frames, 24, 6), interleaved=False
    )
    local = global_rotations.copy()
    local[:, 1:] = (
        global_rotations[:, PARENTS[1:]].swapaxes(-1, -2) @ global_rotations[:, 1:]
    )
    offsets = skeleton_offsets(skeleton)
    return DecodedMotion(
        direct.astype(np.float32), forward_kinematics(local, root, offsets),
        local, root, offsets,
    )


def decode_anytop(features: np.ndarray, skeleton: np.ndarray) -> DecodedMotion:
    frames = len(features)
    structured = features.reshape(frames, 24, 12)
    positions, rotations, velocities = structured[..., :3], structured[..., 3:9], structured[..., 9:12]
    matrices = rotation6d(rotations, interleaved=True)
    alignment = matrices[:, 0]
    world = alignment.swapaxes(-1, -2)
    future_inverse = np.concatenate((world[1:], world[-1:]))
    delta = np.einsum("fij,fj->fi", future_inverse, velocities[:, 0])
    root = np.zeros((frames, 3))
    root[:, 1] = positions[:, 0, 1]
    root[1:, 0] = np.cumsum(delta[:-1, 0])
    root[1:, 2] = np.cumsum(delta[:-1, 2])
    direct = np.einsum("fij,fkj->fki", world, positions)
    direct[..., (0, 2)] += root[:, None, (0, 2)]
    direct[:, 0] = root
    # The training exporter stores parent-joint local rotations in child slots.
    local = np.broadcast_to(np.eye(3), (frames, 24, 3, 3)).copy()
    local[:, 0] = world
    for joint in range(1, 24):
        local[:, PARENTS[joint]] = matrices[:, joint]
    offsets = skeleton_offsets(skeleton)
    return DecodedMotion(
        direct.astype(np.float32), forward_kinematics(local, root, offsets),
        local, root, offsets,
    )


def decode_motion(
    features: np.ndarray, skeleton: np.ndarray, config: ModelConfig
) -> DecodedMotion:
    assert features.ndim == 2 and features.shape[1] == config.feature_dim and len(features) > 0
    assert skeleton.size == config.skeleton_dim
    assert np.isfinite(features).all() and np.isfinite(skeleton).all()
    if config.representation == "combined":
        return decode_anytop(np.asarray(features[:, config.smg_dim:], dtype=np.float64), skeleton)
    return decode_smg(np.asarray(features, dtype=np.float64), skeleton)


def write_bvh(path: Path, motion: DecodedMotion, fps: float = 30.0) -> None:
    assert fps > 0
    lines = ["HIERARCHY"]
    traversal: list[int] = []

    def emit(joint: int, level: int) -> None:
        indent = "  " * level
        traversal.append(joint)
        lines.extend((
            f"{indent}{'ROOT' if joint == 0 else 'JOINT'} {JOINT_NAMES[joint]}",
            f"{indent}{{",
        ))
        offset = " ".join(f"{value:.8f}" for value in motion.offsets[joint])
        lines.append(f"{indent}  OFFSET {offset}")
        prefix = "6 Xposition Yposition Zposition" if joint == 0 else "3"
        lines.append(f"{indent}  CHANNELS {prefix} Zrotation Yrotation Xrotation")
        children = np.flatnonzero(PARENTS == joint)
        for child in children:
            emit(int(child), level + 1)
        if not len(children):
            lines.extend((
                f"{indent}  End Site", f"{indent}  {{",
                f"{indent}    OFFSET 0 0 0", f"{indent}  }}",
            ))
        lines.append(f"{indent}}}")

    emit(0, 0)
    rotations = Rotation.from_matrix(
        motion.local_rotations.reshape(-1, 3, 3)
    ).as_euler("ZYX", degrees=True).reshape(-1, 24, 3)
    channels = np.concatenate((
        motion.root_positions, rotations[:, traversal].reshape(len(rotations), -1)
    ), -1)
    lines.extend(("MOTION", f"Frames: {len(channels)}", f"Frame Time: {1 / fps:.12f}"))
    lines.extend(" ".join(f"{value:.8f}" for value in row) for row in channels)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
