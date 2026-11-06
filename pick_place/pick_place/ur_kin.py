"""
UR5e 정기구학(FK) / 수치 역기구학(IK) — 외부 라이브러리 없이 numpy만 사용
기준: base → tool0 (플랜지). 공식 UR5e DH 파라미터.
"""
import numpy as np

# UR5e DH (Universal Robots 공식)
D = [0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996]
A = [0.0, -0.425, -0.3922, 0.0, 0.0, 0.0]
ALPHA = [np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0]


def _dh(theta, d, a, alpha):
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st * ca, st * sa, a * ct],
                     [st, ct * ca, -ct * sa, a * st],
                     [0.0, sa, ca, d],
                     [0.0, 0.0, 0.0, 1.0]])


def fk(q):
    """q: 6 관절 (rad, shoulder_pan..wrist_3 순서) → 4x4 base→tool0 변환"""
    T = np.eye(4)
    for i in range(6):
        T = T @ _dh(q[i], D[i], A[i], ALPHA[i])
    return T


def _pose_err(T, T_goal):
    dp = T_goal[:3, 3] - T[:3, 3]
    R = T_goal[:3, :3] @ T[:3, :3].T                 # 회전 오차 (axis-angle)
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    if ang < 1e-9:
        dr = np.zeros(3)
    else:
        dr = ang / (2 * np.sin(ang)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return np.r_[dp, dr]


def ik_near(T_goal, q_seed, iters=200, tol=1e-5, damping=1e-3):
    """seed 근처의 해를 찾는 감쇠 최소제곱 IK (자세 크게 안 바뀜 → 손목 회전 문제 없음). 실패 시 None"""
    q = np.array(q_seed, dtype=np.float64)
    for _ in range(iters):
        T = fk(q)
        e = _pose_err(T, T_goal)
        if np.linalg.norm(e[:3]) < tol and np.linalg.norm(e[3:]) < tol * 10:
            return q
        J = np.zeros((6, 6))
        h = 1e-6
        for j in range(6):
            dq = np.zeros(6); dq[j] = h
            J[:, j] = _pose_err(T, fk(q + dq)) / h
        J = -J                                         # e = goal - current 이므로 부호 맞춤
        dq = np.linalg.solve(J.T @ J + damping * np.eye(6), J.T @ e)
        q = q - dq
    return None


def offset_z(q, dz):
    """현재 자세에서 tool 자세(방향)는 그대로 두고 base 기준 z로 dz[m] 이동한 관절값"""
    T = fk(q).copy()
    T[2, 3] += dz
    return ik_near(T, q)


def vertical_path(q, dz, step=0.02):
    """그리퍼 방향을 유지한 채 수직으로 dz[m] 이동하는 관절 웨이포인트 목록 (step 간격으로 IK → 직선 이동, 손목 뒤집힘 없음)
    실패하면 None"""
    n = max(1, int(np.ceil(abs(dz) / step)))
    wp = [np.array(q, dtype=np.float64)]
    for _ in range(n):
        nq = offset_z(wp[-1], dz / n)
        if nq is None or np.abs(nq - wp[-1]).max() > np.radians(15):
            return None
        wp.append(nq)
    return wp
