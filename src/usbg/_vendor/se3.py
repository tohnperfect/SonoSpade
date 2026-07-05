"""
geometry.py -- SE(3) / SO(3) conventions and math. (numpy-only; imports on a bare Mac CPU.)

THE coordinate conventions for the whole project live here (requirements doc Section 1.3).
Ported verbatim, where possible, from the *verified* predecessor `learning_probe_pose_cld`
(report/probe_pose_report.pdf Section 7); the AprilTag pose path built on these utilities
was the one method measured to lock cleanly on the real probe.

Notation lock (requirements Section 1.3, RFC-2119 normative):
    `T_a_b` is the pose of frame `b` expressed in frame `a` -- a 4x4 homogeneous rigid
    SE(3) matrix with
        p_a = T_a_b @ p_b          (points as homogeneous column vectors)
        T_a_c = T_a_b @ T_b_c      (adjacent subscripts cancel)
        inv(T_a_b) = T_b_a         (R^T, -R^T t)   <- invert_T

Camera frame (OpenCV): +X right, +Y down, +Z forward (into the scene).
Tag frame (ArUco):     origin at tag centre, +X right, +Y up, +Z out of the face.
Probe frame:           origin at the transducer-face centre; +Z along the beam.

Units: meters and radians everywhere internally and on disk.
Rotation interchange: quaternion [qx, qy, qz, qw] (scalar-LAST), canonicalized to qw >= 0.

This module is intentionally numpy-only so it imports with no scipy/torch/cv2.
"""
from __future__ import annotations

import numpy as np

# ======================================================================================
# Quaternion <-> rotation matrix.  Quaternions are ALWAYS [qx, qy, qz, qw] (scalar last).
# ======================================================================================


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """[qx,qy,qz,qw] -> 3x3 rotation matrix. Input need not be normalized."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ], dtype=np.float64)


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> [qx,qy,qz,qw], normalized, with qw >= 0 (canonical sign)."""
    R = np.asarray(R, dtype=np.float64)[:3, :3]
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    if q[3] < 0:  # canonical hemisphere (double-cover, R-GEN-6 / R-OUT-5)
        q = -q
    return q


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return q / (np.linalg.norm(q) + 1e-12)


def quat_canonical(q: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    """Flip sign so q is in the same hemisphere as `ref` (default: qw >= 0).

    q and -q are the same rotation; for averaging/interpolation we must keep a continuous
    sign relative to a reference sample (R-GEN-6, R-OUT-5).
    """
    q = quat_normalize(q)
    if ref is None:
        return q if q[3] >= 0 else -q
    ref = quat_normalize(ref)
    return q if float(np.dot(q, ref)) >= 0.0 else -q


def quat_slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    """Spherical linear interpolation, u in [0,1]."""
    q0 = quat_normalize(q0)
    q1 = quat_canonical(q1, q0)
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot > 0.9995:  # nearly identical -> linear
        return quat_normalize(q0 + u * (q1 - q0))
    theta0 = np.arccos(dot)
    theta = theta0 * u
    q2 = quat_normalize(q1 - q0 * dot)
    return q0 * np.cos(theta) + q2 * np.sin(theta)


def geodesic_angle(R0: np.ndarray, R1: np.ndarray) -> float:
    """Geodesic SO(3) angle (radians) between two rotation matrices."""
    R = np.asarray(R0)[:3, :3].T @ np.asarray(R1)[:3, :3]
    cos = (np.trace(R) - 1.0) * 0.5
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


# ======================================================================================
# Homogeneous transforms.
# ======================================================================================


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)[:3, :3]
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def T_from_quat_trans(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    return make_T(quat_to_matrix(q), t)


def quat_trans_from_T(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T = np.asarray(T, dtype=np.float64)
    return matrix_to_quat(T[:3, :3]), T[:3, 3].copy()


def invert_T(T: np.ndarray) -> np.ndarray:
    """Rigid inverse inv(T_a_b) = T_b_a, i.e. (R^T, -R^T t)."""
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def is_valid_T(T) -> bool:
    """True iff T is a finite 4x4 with an orthonormal, right-handed rotation block."""
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        return False
    R = T[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-6):   # orthonormal to <= 1e-6 (R-CAL-1)
        return False
    if not np.allclose(T[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        return False
    return bool(np.linalg.det(R) > 0)


# ======================================================================================
# SO(3) / SE(3) exp & log -- exact screw-linear interpolation (requirements R-MC-16a).
#
# Temporal alignment across cameras / to the arm clock MUST use screw-linear interpolation
# (exp/log on SE(3): SLERP on rotation + geodesic-consistent translation), NOT independent
# quaternion+linear. These primitives implement it.
# ======================================================================================


def _skew(w: np.ndarray) -> np.ndarray:
    wx, wy, wz = np.asarray(w, dtype=np.float64).reshape(3)
    return np.array([[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]], dtype=np.float64)


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (axis * angle), radians."""
    R = np.asarray(R, dtype=np.float64)[:3, :3]
    cos = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos))
    if theta < 1e-9:
        # First-order: skew part of R recovers the small rotation vector.
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * 0.5
    if abs(theta - np.pi) < 1e-6:
        # Near pi: derive axis from the largest diagonal of (R+I)/2.
        A = (R + np.eye(3)) * 0.5
        k = int(np.argmax(np.diag(A)))
        axis = A[:, k] / np.sqrt(max(A[k, k], 0.0) + 1e-12)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return axis * theta
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (theta / (2.0 * np.sin(theta)))


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle) -> rotation matrix (Rodrigues)."""
    w = np.asarray(w, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(w))
    if theta < 1e-9:
        W = _skew(w)
        return np.eye(3) + W + 0.5 * W @ W
    k = w / theta
    K = _skew(k)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def se3_log(T: np.ndarray) -> np.ndarray:
    """SE(3) -> 6-vector twist [v(3); w(3)] with T = se3_exp(twist)."""
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    p = T[:3, 3]
    w = so3_log(R)
    theta = float(np.linalg.norm(w))
    if theta < 1e-9:
        return np.concatenate([p, w])
    K = _skew(w / theta)
    # Inverse of the SE(3) left-Jacobian V (Murray, Li & Sastry).
    A = np.sin(theta) / theta
    B = (1.0 - np.cos(theta)) / (theta * theta)
    Vinv = np.eye(3) - 0.5 * _skew(w) + (1.0 / (theta * theta)) * (1.0 - A / (2.0 * B)) * (_skew(w) @ _skew(w))
    v = Vinv @ p
    return np.concatenate([v, w])


def se3_exp(twist: np.ndarray) -> np.ndarray:
    """6-vector twist [v(3); w(3)] -> SE(3)."""
    twist = np.asarray(twist, dtype=np.float64).reshape(6)
    v = twist[:3]
    w = twist[3:]
    theta = float(np.linalg.norm(w))
    R = so3_exp(w)
    if theta < 1e-9:
        return make_T(R, v)
    K = _skew(w / theta)
    V = (np.eye(3) + (1.0 - np.cos(theta)) / (theta * theta) * _skew(w)
         + (theta - np.sin(theta)) / (theta ** 3) * (_skew(w) @ _skew(w)))
    return make_T(R, V @ v)


def se3_interpolate(T0: np.ndarray, T1: np.ndarray, u: float) -> np.ndarray:
    """Screw-linear SE(3) interpolation: T0 @ exp(u * log(inv(T0) @ T1)), u in [0,1].

    This is the constant-twist geodesic on SE(3) (R-MC-16a). At u=0 returns T0, at u=1
    returns T1; SLERP on rotation with geodesic-consistent translation coupling.
    """
    T0 = np.asarray(T0, dtype=np.float64).reshape(4, 4)
    T1 = np.asarray(T1, dtype=np.float64).reshape(4, 4)
    rel = invert_T(T0) @ T1
    return T0 @ se3_exp(float(u) * se3_log(rel))


# ======================================================================================
# Projection (overlay + reprojection-residual checks; OpenCV pinhole + distortion).
# ======================================================================================


def project_points(P_cam: np.ndarray, K: np.ndarray, dist: np.ndarray | None = None) -> np.ndarray:
    """Project Nx3 camera-frame points to Nx2 pixels (OpenCV pinhole + optional distortion).

    `dist` is the OpenCV distortion vector [k1,k2,p1,p2,k3,...] or None.
    Points behind the camera (Z<=0) project to NaN.
    """
    P = np.asarray(P_cam, dtype=np.float64).reshape(-1, 3)
    K = np.asarray(K, dtype=np.float64)
    z = P[:, 2]
    xn = P[:, 0] / np.where(np.abs(z) < 1e-9, np.nan, z)
    yn = P[:, 1] / np.where(np.abs(z) < 1e-9, np.nan, z)
    if dist is not None and np.any(np.asarray(dist) != 0):
        d = np.asarray(dist, dtype=np.float64).reshape(-1)
        k1, k2, p1, p2 = d[0], d[1], d[2], d[3]
        k3 = d[4] if d.size > 4 else 0.0
        r2 = xn * xn + yn * yn
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        x_t = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
        y_t = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
        xn, yn = x_t, y_t
    u = K[0, 0] * xn + K[0, 2]
    v = K[1, 1] * yn + K[1, 2]
    uv = np.stack([u, v], axis=1)
    uv[z <= 0] = np.nan
    return uv
