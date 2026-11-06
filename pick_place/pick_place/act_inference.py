#!/usr/bin/env python3
"""
학습된 ACT 모델로 UR5e Pick&Place 실행 (CPU용)

동작 방식:
  카메라 화면 + 관절값 → 모델이 앞으로 100스텝(3.3초) 동작 예측
  → 추론에 걸린 시간만큼 앞부분은 건너뛰고, 그다음 일부 구간을 궤적으로 로봇에 전송
  → 로봇이 그 구간을 실행하는 동안 바로 다음 추론 (멈춤 없이 이어짐)

키 조작:
  o      관찰 자세로 이동
  SPACE  모델 실행 시작 / 정지
  g      그리퍼 수동 토글 (실행 중이 아닐 때)
  q      종료

안전장치:
  - 관절 속도 제한 (max_joint_vel, 기본 0.5 rad/s ≈ 29°/s) → 넘으면 궤적을 느리게 늘림
  - 모델 예측이 현재 자세에서 max_jump 이상 떨어져 있으면 즉시 정지
  - max_duration 초 지나면 정지
  - 잡았다 놓은 뒤 관찰 자세 근처로 돌아오면 자동 종료
"""
import json
import math
import os
import select
import sys
import termios
import threading
import time
import tty

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Float64, Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration as DurationMsg

from pick_place.data_collector import (DataCollector, ARM_JOINTS, TRAJ_CONTROLLERS,
                                       GRIPPER_OPEN, GRIPPER_CLOSE)
from pick_place.act_model import ACTPolicy
from pick_place.ur_kin import vertical_path, fk

FWD_CTRL = 'forward_position_controller'
from control_msgs.action import FollowJointTrajectory


# ─────────────────────────── 궤적 생성 (ROS 의존 없음, 테스트 가능) ───────────────────────────
def build_segment(chunk, skip, n_steps, dt, max_vel, q_cur):
    """
    chunk:   (K, 7) 모델 예측 (관절 6 + 그리퍼), chunk[i]는 관측 시점 기준 (i+1)*dt 뒤 목표
    skip:    추론 지연 스텝 수 (이 근처까지만 시작점을 찾는다)
    q_cur:   (6,) 궤적을 보내는 순간의 실제 관절값
    시작점: chunk[0 .. skip+5] 중 현재 자세에 가장 가까운 점
            (로봇이 속도 제한/슬라이더 때문에 모델 계획보다 느리면, 시간 기준으로 건너뛰면 앞으로 튐)
    속도 제한: 현재 자세 → 첫 점, 그리고 점과 점 사이 모두 max_vel 이하가 되도록 시간을 늘림
    returns: positions (N,6), times (N,), velocities (N,6), gripper (N,), 시작 인덱스, 시작점까지 거리[rad]
    """
    K = len(chunk)
    q_cur = np.asarray(q_cur, dtype=np.float64)
    P = chunk[:, :6].astype(np.float64)
    win = int(min(max(skip, 0) + 6, K - 1))
    d = np.abs(P[:win] - q_cur).max(axis=1)
    s = int(np.argmin(d))
    # 시작점을 점 사이에서 연속으로: q_cur를 (s-1,s), (s,s+1) 선분에 투영 → 소수 인덱스 f
    # (가장 가까운 '점'에서 시작하면 반 스텝까지 어긋나서, 33ms 안에 그만큼 움직이라는 순간 속도 튐이 생김)
    best_f, best_e = float(s), d[s]
    for a in (s - 1, s):
        b = a + 1
        if a < 0 or b >= K:
            continue
        seg_v = P[b] - P[a]
        L2 = float(seg_v @ seg_v)
        u = 0.0 if L2 < 1e-12 else float(np.clip((q_cur - P[a]) @ seg_v / L2, 0.0, 1.0))
        e = float(np.abs(P[a] + u * seg_v - q_cur).max())
        if e < best_e:
            best_f, best_e = a + u, e
    idx = best_f + 1.0 + np.arange(n_steps)                         # 시작점 다음 스텝부터
    idx = np.minimum(idx, K - 1)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, K - 1)
    fr = (idx - lo)[:, None]
    pos = (1 - fr) * P[lo] + fr * P[hi]
    grip = ((1 - fr) * chunk[lo, 6:7] + fr * chunk[hi, 6:7])[:, 0].astype(np.float64)
    s = int(round(best_f))
    d = np.full(K, np.inf); d[s] = best_e
    step_dt = np.full(len(pos), dt)
    step_dt[0] = max(dt, float(np.abs(pos[0] - q_cur).max()) / max_vel)
    if len(pos) > 1:
        dq = np.abs(np.diff(pos, axis=0)).max(axis=1)                # 스텝당 최대 관절 변화
        step_dt[1:] = np.maximum(dt, dq / max_vel)
    times = np.cumsum(step_dt)
    # 속도: 현재 자세를 시각 0의 점으로 포함해 중앙차분 → 첫 점도 실제 진행 속도 (0으로 두면 궤적이 바뀔 때마다 급정지)
    full_p = np.vstack([q_cur, pos])
    full_t = np.r_[0.0, times]
    vel = np.zeros_like(pos)
    if len(pos) > 1:
        vel[:-1] = (full_p[2:] - full_p[:-2]) / (full_t[2:] - full_t[:-2])[:, None]
    vel[-1] = 0.0                                                     # 다음 궤적이 늦으면 부드럽게 정지
    return pos, times, vel, grip, s, float(d[s])


def eval_segment(seg, t):
    """보낸 궤적 seg=(시작 자세, times, pos)의 시각 t 위치 = 컨트롤러가 지금 따라가고 있는 명령 위치(근사)"""
    q0, times, pos = seg
    tt = np.r_[0.0, times]
    pp = np.vstack([q0, pos])
    t = float(np.clip(t, 0.0, tt[-1]))
    return np.array([np.interp(t, tt, pp[:, j]) for j in range(pp.shape[1])])


class CubicSeg:
    """q(t) = q0 + v0 t + a t² + b t³  (0 ≤ t ≤ T, 이후는 끝점에서 정지) — 모델 구간 궤적 평활화용"""

    def __init__(self, q0, v0, a, b, T):
        self.q0, self.v0, self.a, self.b, self.T = q0, v0, a, b, T

    def pos(self, t):
        t = min(max(t, 0.0), self.T)
        return self.q0 + self.v0 * t + self.a * t ** 2 + self.b * t ** 3

    def vel(self, t):
        if t >= self.T:
            return np.zeros_like(self.q0)
        t = max(t, 0.0)
        return self.v0 + 2 * self.a * t + 3 * self.b * t ** 2


def fit_cubic(q0, v0, times, pos, max_vel):
    """
    현재 (위치 q0, 속도 v0)에서 시작해 목표점들(times, pos)을 최소제곱으로 따라가는 3차 곡선.
    → 궤적을 바꿀 때 위치·속도가 끊기지 않고, 0.3초마다 바뀌는 예측의 작은 흔들림은 곡선이 걸러냄.
    끝 속도가 max_vel을 넘으면 시간을 늘림.
    """
    q0 = np.asarray(q0, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    A = np.c_[t ** 2, t ** 3]
    rhs = pos - q0 - np.outer(t, v0)
    coef, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    seg = CubicSeg(q0, v0, coef[0], coef[1], float(t[-1]))
    ts = np.linspace(0, seg.T, 20)
    vmax = max(np.abs(seg.vel(x)).max() for x in ts[:-1])
    if vmax > max_vel * 1.05:                      # 시간 스케일링 (같은 경로, 더 천천히)
        k = vmax / max_vel
        seg = CubicSeg(q0, v0, coef[0] / k ** 2, coef[1] / k ** 3, seg.T * k)
        seg.v0 = v0                                # 시작 속도는 연속성 유지
    return seg


def min_jerk_path(waypoints, max_vel, dt=0.04, min_T=0.8):
    """
    관절 공간 경로(웨이포인트를 잇는 꺾은선)를 minimum-jerk 속도 프로파일로 시간 할당.
    정지 → 부드럽게 가속 → 부드럽게 감속 → 정지 (좌표 이동처럼).
    최고 속도 = 1.875 × 거리 / 시간 ≤ max_vel
    returns positions (N,6), times (N,), velocities (N,6)
    """
    W = np.asarray(waypoints, dtype=np.float64)
    seglen = np.abs(np.diff(W, axis=0)).max(axis=1)
    S = np.r_[0.0, np.cumsum(seglen)]
    total = S[-1]
    if total < 1e-6:
        return W[-1:], np.array([min_T]), np.zeros((1, 6))
    T = max(min_T, 1.875 * total / max_vel)
    n = int(np.ceil(T / dt))
    ts = np.linspace(dt, T, n) if n > 1 else np.array([T])
    tau = ts / T
    s = total * (10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5)
    pos = np.stack([np.interp(s, S, W[:, j]) for j in range(6)], axis=1)
    full_p = np.vstack([W[0], pos])
    full_t = np.r_[0.0, ts]
    vel = np.zeros_like(pos)
    if len(pos) > 1:
        vel[:-1] = (full_p[2:] - full_p[:-2]) / (full_t[2:] - full_t[:-2])[:, None]
    vel[-1] = 0.0
    return pos, ts, vel


def bezier(p0, c, p2, n=30):
    """2차 베지어 곡선 (p0 → p2, c 쪽으로 휘어짐) — 경유점에서 꺾이지 않는 매끄러운 운반 경로"""
    s = np.linspace(0, 1, n)[:, None]
    return (1 - s) ** 2 * p0 + 2 * (1 - s) * s * c + s ** 2 * p2


def to_duration(t):
    sec = int(math.floor(t))
    return DurationMsg(sec=sec, nanosec=int((t - sec) * 1e9))


class ActionEnsembler:
    """
    원본 ACT의 temporal ensembling (팔 6축 + 그리퍼 모두).
    추론마다 (관측 시각, chunk 전체)를 저장하고, 어떤 시각 τ의 목표 = τ를 예측했던 모든 과거 chunk 값의 가중 평균.
      - 팔: 매 추론마다 예측이 조금씩 달라서 생기는 흔들림을 평균으로 줄임
      - 그리퍼: "0.5초 뒤 닫기" 예측이 시간이 지나 '지금'이 되면 반영됨 (최신 chunk만 보면 계속 뒤로 밀림)
    """

    def __init__(self, dt, chunk_size, k=0.01):
        from collections import deque
        self.dt, self.chunk, self.k = dt, chunk_size, k
        self.hist = deque()

    def reset(self):
        self.hist.clear()

    def add(self, t_obs, chunk):
        self.hist.append((t_obs, np.asarray(chunk, dtype=np.float64)))
        while self.hist and t_obs - self.hist[0][0] > (self.chunk + 1) * self.dt:
            self.hist.popleft()

    def value(self, tau):
        vals = []
        for t0, c in self.hist:                         # 오래된 것부터 (원본 ACT와 같은 가중 순서)
            j = int(round((tau - t0) / self.dt)) - 1    # chunk[j]는 t0 + (j+1)*dt 시점 목표
            if 0 <= j < len(c):
                vals.append(c[j])
        if not vals:
            return None
        w = np.exp(-self.k * np.arange(len(vals)))
        return (np.array(vals) * w[:, None]).sum(0) / w.sum()

    def plan_aligned(self, q_cur, t_now, n, search_ahead=6, tie_tol=np.radians(0.5)):
        """
        팔 궤적 앙상블 (n,6) — 시간이 아니라 '로봇이 지금 각 예측 궤적의 어디쯤에 있는지' 기준으로 맞춘 뒤 평균.
        각 chunk에서 현재 자세와 가장 가까운 점 p를 찾고(시간상 예상 위치 + search_ahead 까지만),
        그 다음 점들 c[p+1 : p+1+n]을 평균 → 로봇이 계획보다 늦어도 오차가 쌓이지 않음.
        가까운 점이 여러 개(정지 구간)면 가장 뒤쪽 점을 골라 정지 구간을 시간대로 지나가게 함.
        """
        q_cur = np.asarray(q_cur, dtype=np.float64)
        segs = []
        for t0, c in self.hist:                                   # 오래된 것부터
            K = len(c)
            j_t = int(round((t_now - t0) / self.dt)) - 1
            if j_t >= K - 2:
                continue
            hi = int(min(max(j_t, 0) + search_ahead, K - 2))
            d = np.abs(c[:hi + 1, :6] - q_cur).max(axis=1)
            near = np.where(d <= d.min() + tie_tol)[0]
            p = int(near.max())
            s = c[p + 1:p + 1 + n, :6]
            if len(s) < n:
                s = np.concatenate([s, np.repeat(s[-1:], n - len(s), axis=0)])
            segs.append(s)
        if not segs:
            return None
        w = np.exp(-self.k * np.arange(len(segs)))
        return np.tensordot(w / w.sum(), np.stack(segs), axes=1)


class GripperLogic:
    """모델의 그리퍼 출력(연속값)을 히스테리시스로 열림/닫힘 결정"""

    def __init__(self, close_th=0.6, open_th=0.4, min_hold=1.0):
        self.close_th, self.open_th, self.min_hold = close_th, open_th, min_hold
        self.last_change = -1e9

    def decide(self, value, current, now):
        if now - self.last_change < self.min_hold:
            return current
        if current == GRIPPER_OPEN and value > self.close_th:
            self.last_change = now
            return GRIPPER_CLOSE
        if current == GRIPPER_CLOSE and value < self.open_th:
            self.last_change = now
            return GRIPPER_OPEN
        return current


# ─────────────────────────────── 노드 ───────────────────────────────
class ACTRunner(DataCollector):
    def __init__(self):
        super().__init__('act_runner')
        self.declare_parameter('ckpt', '~/act_data/policy_best.ckpt')
        self.declare_parameter('max_joint_vel', 0.5)
        self.declare_parameter('max_jump', 0.2)
        self.declare_parameter('run_max_duration', 150.0)
        self.declare_parameter('horizon_margin', 0.3)
        self.declare_parameter('torch_threads', 0)
        self.declare_parameter('grip_close_th', 0.5)
        self.declare_parameter('grip_open_th', 0.4)
        self.declare_parameter('save_runs', True)
        self.declare_parameter('ens_k', 0.01)
        self.declare_parameter('post_release_sec', 1.5)
        self.declare_parameter('blend_steps', 10)
        self.declare_parameter('stall_sec', 8.0)
        self.declare_parameter('stall_deg', 4.0)
        self.declare_parameter('place_mode', 'scripted')      # scripted: 잡은 뒤는 좌표 이동 / model: 끝까지 모델
        self.declare_parameter('place_pose_file', '')         # 기본: save_dir/place_pose.json
        self.declare_parameter('lift_height', 0.25)           # 잡은 뒤 수직으로 들어 올릴 높이 [m]
        self.declare_parameter('place_vel', 0.5)              # 좌표 이동 최고 관절 속도 [rad/s]
        self.declare_parameter('grasp_settle', 1.0)           # 그리퍼 닫고 들어 올리기 전 대기 [s]
        gp = self.get_parameter
        self.max_joint_vel = float(gp('max_joint_vel').value)
        self.max_jump = float(gp('max_jump').value)
        self.run_max_duration = float(gp('run_max_duration').value)
        self.horizon_margin = float(gp('horizon_margin').value)
        self.save_runs = bool(gp('save_runs').value)
        self.run_dir = os.path.join(self.save_dir, 'runs')

        import torch
        nt = int(gp('torch_threads').value)
        if nt > 0:
            torch.set_num_threads(nt)
        ckpt = os.path.expanduser(gp('ckpt').value)
        self.policy = ACTPolicy(ckpt, device='cpu')
        self.dt = 1.0 / float(self.policy.cfg.get('fps', 30.0))
        hw = tuple(self.policy.cfg['image_hw'])
        if hw != (self.img_h, self.img_w):
            self.get_logger().warn(f'모델 이미지 크기 {hw} ≠ 수집 설정 {(self.img_h, self.img_w)} → 자동 리사이즈')

        self.traj_pubs = {c: self.create_publisher(JointTrajectory, f'/{c}/joint_trajectory', 10)
                          for c in TRAJ_CONTROLLERS}
        self.grip_logic = GripperLogic(close_th=float(gp('grip_close_th').value),
                                       open_th=float(gp('grip_open_th').value))
        self.ens = ActionEnsembler(self.dt, self.policy.chunk_size, k=float(gp('ens_k').value))
        self.post_release_sec = float(gp('post_release_sec').value)
        self.ref_tol = math.radians(5.0)   # 명령 위치와 실제 위치가 이보다 멀면 실제 위치 기준으로 다시 시작
        self.blend_steps = int(gp('blend_steps').value)
        self.stall_sec = float(gp('stall_sec').value)
        self.stall_rad = math.radians(float(gp('stall_deg').value))
        self._last_cubic = None       # (보낸 시각, CubicSeg) — 모델 구간 궤적 연속성용
        self.declare_parameter('control_mode', 'stream')     # stream: forward_position_controller / jtc: 궤적 컨트롤러
        self.declare_parameter('stream_hz', 200.0)
        self.control_mode = str(gp('control_mode').value)
        self.stream_hz = float(gp('stream_hz').value)
        self.fwd_pub = self.create_publisher(Float64MultiArray, f'/{FWD_CTRL}/commands', 10)
        self._stream_lock = threading.Lock()
        self._stream_seg = None
        self._streaming = False
        self._stream_thread = None
        self._last_cmd = None
        self._last_cmd_t = 0.0
        self.place_mode = str(gp('place_mode').value)
        self.lift_height = float(gp('lift_height').value)
        self.place_vel = float(gp('place_vel').value)
        self.grasp_settle = float(gp('grasp_settle').value)
        pf = gp('place_pose_file').value or os.path.join(self.save_dir, 'place_pose.json')
        self.place = self._load_place(os.path.expanduser(pf))
        self.phase = ''
        self._js_log = None           # 실행 중 관절값 100Hz 기록 (부드러움 분석용)
        self.create_timer(0.01, self._js_log_tick, callback_group=self.cb)
        self.speed_scaling = None     # 티치펜던트 속도 슬라이더 (%)
        self.create_subscription(Float64, '/speed_scaling_state_broadcaster/speed_scaling',
                                 self._scaling_cb, 10, callback_group=self.cb)
        self.grip_now = 0.0           # 현재 시점 그리퍼 예측값 (ensemble, 화면 표시용)
        self.grip_ahead = 0.0         # 앞으로 1초 안의 최대 예측값
        self.last_run_file = None
        self.returning = False
        self._grasped = False
        self._grasp_time = None
        self._release_time = None
        self.running = False
        self.stop_reason = None
        self.latency = None
        self.n_exec = 15
        self.n_infer = 0
        self._thread = None

    # ── 준비 ──
    # ── 스트리밍 제어 (모델 구간) ──
    # 궤적 컨트롤러는 새 궤적을 받을 때마다 명령을 '실제 위치'로 되돌려서 0.3초마다 툭 튐 →
    # 모델 구간은 forward_position_controller에 3차 곡선 위의 목표값을 초당 stream_hz번 연속으로 보냄
    def _stream_set(self, seg, t0):
        with self._stream_lock:
            self._stream_seg = (t0, seg)

    def stream_state(self, now):
        with self._stream_lock:
            t0, seg = self._stream_seg
        return seg.pos(now - t0), seg.vel(now - t0)

    def _stream_loop(self):
        period = 1.0 / self.stream_hz
        nxt = time.monotonic()
        while self._streaming:
            now = time.monotonic()
            q, _ = self.stream_state(now)
            if self._last_cmd is not None:            # 안전: 명령 사이 최대 변화량 제한
                lim = self.max_joint_vel * 1.5 * max(now - self._last_cmd_t, 1e-3)
                q = self._last_cmd + np.clip(q - self._last_cmd, -lim, lim)
            self.fwd_pub.publish(Float64MultiArray(data=[float(x) for x in q]))
            self._last_cmd, self._last_cmd_t = q, now
            nxt += period
            d = nxt - time.monotonic()
            if d > 0:
                time.sleep(d)
            elif d < -0.1:
                nxt = time.monotonic()

    def stream_begin(self):
        """궤적 컨트롤러 → forward_position_controller 전환 후 현재 자세 유지부터 시작"""
        with self.lock:
            q = self.qpos6.copy()
        z = np.zeros(6)
        self._stream_set(CubicSeg(q, z, z, z, 0.0), time.monotonic())
        self._last_cmd = None
        self._switch([FWD_CTRL], [self.traj_ctrl])
        self._streaming = True
        self._stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._stream_thread.start()

    def stream_end(self, amax=1.5):
        """현재 명령 속도에서 부드럽게 감속·정지한 뒤 궤적 컨트롤러로 되돌림"""
        if not self._streaming:
            return
        now = time.monotonic()
        q0, v0 = self.stream_state(now)
        T = max(0.25, float(np.abs(v0).max()) / amax)
        self._stream_set(CubicSeg(q0, v0, -v0 / T, v0 / (3 * T * T), T), now)   # 끝에서 속도·가속도 0
        time.sleep(T + 0.15)
        self._streaming = False
        if self._stream_thread:
            self._stream_thread.join(timeout=1.0)
        self._switch([self.traj_ctrl], [FWD_CTRL])

    def warmup(self):
        q = np.array(self.policy.act_mean, dtype=np.float32)
        img = np.zeros((self.img_h, self.img_w, 3), np.uint8)
        self.policy.predict(q, img)
        ts = []
        for _ in range(5):
            t = time.monotonic()
            self.policy.predict(q, img)
            ts.append(time.monotonic() - t)
        self.latency = float(np.median(ts))
        # 한 번에 보낼 구간 길이: 다음 추론이 끝날 때까지 + 여유
        self.n_exec = int(min(self.policy.chunk_size - 5,
                              max(10, math.ceil((self.latency + self.horizon_margin) / self.dt))))
        return self.latency, self.n_exec

    def ensure_traj_controller(self):
        states = self.controller_states()
        if states.get(self.traj_ctrl) != 'active':
            self._switch([self.traj_ctrl], [])

    def observation(self):
        with self.lock:
            if self.qpos6 is None or self.image is None:
                return None
            if time.monotonic() - self.img_time > 0.5:
                return None
            return np.append(self.qpos6, self.gripper_cmd).astype(np.float32), self.image, self.qpos6.copy()

    # ── 궤적 전송 ──
    def publish_segment(self, pos, times, vel):
        msg = JointTrajectory()
        msg.joint_names = ARM_JOINTS                 # header.stamp=0 → 받는 즉시 시작 (이전 궤적 대체)
        for p, t, v in zip(pos, times, vel):
            pt = JointTrajectoryPoint()
            pt.positions = p.tolist()
            pt.velocities = v.tolist()
            pt.time_from_start = to_duration(float(t))
            msg.points.append(pt)
        self.traj_pubs[self.traj_ctrl].publish(msg)

    def hold_position(self):
        if self._streaming:
            self.stream_end()
            return
        with self.lock:
            q = None if self.qpos6 is None else self.qpos6.copy()
        if q is None:
            return
        self.publish_segment(q[None], np.array([0.3]), np.zeros((1, 6)))

    # ── 실행 루프 (별도 스레드) ──
    def start_run(self):
        if self.running:
            return
        self.require_program()
        self.freedrive_off()
        self.ensure_traj_controller()
        if self.observation() is None:
            raise RuntimeError('카메라 이미지 또는 joint_states가 안 들어옵니다')
        self.stop_reason = None
        self.n_infer = 0
        if self.control_mode == 'stream':
            if FWD_CTRL not in self.controller_states():
                raise RuntimeError(f'{FWD_CTRL}가 없습니다 (-p control_mode:=jtc 로 실행하세요)')
            self.stream_begin()
        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop_run(self, reason='사용자 정지'):
        if not self.running:
            return
        self.stop_reason = reason
        self.running = False
        if self._thread:
            self._thread.join(timeout=5.0)        # 부드러운 정지(stream_end)가 끝날 때까지
        if not self._streaming:
            self.hold_position()

    def manual_gripper(self):
        """실행 중 사람이 직접 그리퍼 토글. 3초간 모델의 그리퍼 판단을 막는다"""
        v = self.toggle_gripper()
        self.grip_logic.last_change = time.monotonic() + 2.0
        self._on_gripper_change(v)
        return v

    def _on_gripper_change(self, v):
        if v == GRIPPER_CLOSE:
            self._grasped = True
            if self._grasp_time is None:
                self._grasp_time = time.monotonic()
        elif self._grasped and self._release_time is None:
            self._release_time = time.monotonic()

    def _scaling_cb(self, msg):
        self.speed_scaling = float(msg.data)

    # ── 좌표 이동 (잡기 이후) ──
    def _load_place(self, path):
        try:
            with open(path, encoding='utf-8') as f:
                d = json.load(f)
            out = {k: np.array(d[k], dtype=np.float64) for k in ('release', 'pre_place', 'via')}
            down = vertical_path(out['pre_place'], -float(d.get('above', 0.10)))
            if down is None:
                raise ValueError('pre_place → release 수직 경로 계산 실패')
            out['down'] = down
            out['path'] = path
            return out
        except Exception as e:
            self.get_logger().warn(f'놓기 자세 파일을 못 읽음 ({path}): {e} → 잡은 뒤에도 모델이 움직입니다')
            return None

    def _js_log_tick(self):
        if self._js_log is not None and self.qpos6 is not None:
            self._js_log.append((time.monotonic(), self.qpos6.copy()))

    def execute_traj(self, pos, times, vel, what):
        """FollowJointTrajectory로 궤적 실행 후 끝날 때까지 대기 (중간에 SPACE로 멈추면 중단)"""
        cli = self.traj_clients[self.traj_ctrl]
        if not cli.wait_for_server(timeout_sec=3.0):
            raise RuntimeError(f'{self.traj_ctrl} 액션 서버 없음')
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ARM_JOINTS
        for p, t, v in zip(pos, times, vel):
            pt = JointTrajectoryPoint()
            pt.positions = [float(x) for x in p]
            pt.velocities = [float(x) for x in v]
            pt.time_from_start = to_duration(float(t))
            goal.trajectory.points.append(pt)
        fut = cli.send_goal_async(goal)
        while not fut.done():
            time.sleep(0.01)
        gh = fut.result()
        if not gh.accepted:
            raise RuntimeError(f'{what}: 궤적 거부됨')
        res = gh.get_result_async()
        end = time.monotonic() + float(times[-1]) / max(0.05, (self.speed_scaling or 100.0) / 100.0) + 10.0
        while not res.done():
            if not self.running:
                gh.cancel_goal_async()
                raise RuntimeError(f'{what} 중 정지됨')
            if time.monotonic() > end:
                gh.cancel_goal_async()
                raise RuntimeError(f'{what} 시간 초과')
            time.sleep(0.02)
        code = res.result().result.error_code
        if code != 0:
            raise RuntimeError(f'{what} 실패 (error_code={code})')

    def scripted_place(self):
        """잡은 자세 → 수직 들어올리기 → 컵 위로 매끄럽게 운반 → 수직 내려놓기 → 놓기 → 살짝 위로 → 관찰 자세"""
        P = self.place
        with self.lock:
            q = self.qpos6.copy()
        lift = vertical_path(q, self.lift_height)
        if lift is None:
            raise RuntimeError('이 자세에서는 수직으로 들어 올릴 수 없음 (역기구학 실패)')
        steps = [('들어올리기', lift),
                 ('운반', bezier(lift[-1], P['via'], P['pre_place'])),
                 ('내려놓기', P['down'])]
        t0 = time.monotonic()
        for name, wp in steps:
            self.phase = name
            pos, ts, vel = min_jerk_path(wp, self.place_vel)
            self.execute_traj(pos, ts, vel, name)
        self.phase = '놓기'
        self.set_gripper(GRIPPER_OPEN)
        self._on_gripper_change(GRIPPER_OPEN)
        time.sleep(0.8)
        self.phase = '빠지기'
        with self.lock:
            q = self.qpos6.copy()
        up = vertical_path(q, 0.05)
        if up is not None:
            pos, ts, vel = min_jerk_path(up, self.place_vel)
            self.execute_traj(pos, ts, vel, '빠지기')
        self.phase = '관찰 자세 복귀'
        dur = self.move_to_observe()
        return time.monotonic() - t0

    def _save_run(self, log):
        if not self.save_runs or not log['t']:
            return None
        import h5py
        os.makedirs(self.run_dir, exist_ok=True)
        path = os.path.join(self.run_dir, time.strftime('run_%Y%m%d_%H%M%S.hdf5'))
        imgs = np.stack(log['img'])
        with h5py.File(path, 'w') as f:
            f.attrs['stop_reason'] = str(self.stop_reason)
            f.attrs['ckpt'] = str(self.get_parameter('ckpt').value)
            f.attrs['camera_name'] = self.cam_name
            f.create_dataset('t', data=np.array(log['t']))
            f.create_dataset('qpos', data=np.array(log['q'], np.float32))          # 관측 (관절6 + 그리퍼 상태)
            f.create_dataset('chunk', data=np.array(log['chunk'], np.float32))     # 모델 예측 (N,100,7)
            f.create_dataset('latency', data=np.array(log['lat']))
            js = log.get('js') or []
            if js:
                f.create_dataset('js_t', data=np.array([a for a, _ in js]) - log['t0'])   # 관절값 100Hz
                f.create_dataset('js_q', data=np.array([b for _, b in js], np.float32))
            f.create_dataset('skip', data=np.array(log['skip']))
            f.create_dataset('images', data=imgs, chunks=(1,) + imgs.shape[1:], compression='gzip',
                             compression_opts=4)
        return path

    def _run_loop(self):
        t_start = time.monotonic()
        self.ens.reset()
        self._grasped = False
        self._grasp_time = None
        self._release_time = None
        self._last_cubic = None
        self.phase = '모델'
        self._js_log = []
        from collections import deque
        stall_hist = deque()
        log = {'t': [], 'q': [], 'chunk': [], 'lat': [], 'skip': [], 'img': [], 't0': t_start}
        try:
            while self.running:
                now = time.monotonic()
                if now - t_start > self.run_max_duration:
                    self.stop_reason = f'최대 실행 시간 {self.run_max_duration:.0f}s 초과'
                    break

                # 잡은 뒤 (좌표 이동 모드): 그리퍼가 다 닫히길 기다렸다가 들어올리기~놓기~복귀를 좌표 이동으로
                if self._grasp_time is not None and self.place_mode == 'scripted' and self.place is not None:
                    if now - self._grasp_time > self.grasp_settle:
                        self.returning = True
                        self.hold_position()          # 스트리밍이면 부드럽게 멈추고 궤적 컨트롤러로 전환
                        dur = self.scripted_place()
                        self.stop_reason = f'완료 (모델로 잡기 → 좌표 이동으로 옮기기·놓기·복귀 {dur:.1f}s)'
                        break
                    time.sleep(0.05)
                    continue

                # 놓은 뒤 post_release_sec 지나면: 모델 대신 데이터 수집 때와 같은 자동 이동으로 관찰 자세 복귀
                if self._release_time is not None and now - self._release_time > self.post_release_sec:
                    self.returning = True
                    self.hold_position()
                    dur = self.move_to_observe()
                    self.stop_reason = f'완료 (잡기 → 놓기 → 관찰 자세 자동 복귀 {dur:.1f}s)'
                    break

                # 그리퍼: 과거 chunk들이 '지금'에 대해 예측한 값의 가중 평균으로 결정
                ev = self.ens.value(now)
                if ev is not None:
                    self.grip_now = float(ev[6])
                    new = self.grip_logic.decide(self.grip_now, self.gripper_cmd, now)
                    if new != self.gripper_cmd:
                        self.set_gripper(new)
                        self._on_gripper_change(new)

                obs = self.observation()
                if obs is None:
                    self.stop_reason = '센서 데이터 끊김'
                    break
                q7, img, q_now = obs
                t_obs = time.monotonic()

                # 진행 멈춤 감지: stall_sec 동안 모든 관절이 stall_deg 이내로만 움직이고 그리퍼도 그대로면 정지
                # (학습 데이터에 없던 상황에서 모델이 다음 동작을 못 정하고 맴도는 경우)
                stall_hist.append((t_obs, q7.copy()))
                while stall_hist and t_obs - stall_hist[0][0] > self.stall_sec:
                    stall_hist.popleft()
                if self.stall_sec > 0 and t_obs - t_start > self.stall_sec and len(stall_hist) > 3:
                    qs = np.array([h[1] for h in stall_hist])
                    if t_obs - stall_hist[0][0] > self.stall_sec * 0.75 and \
                            np.ptp(qs[:, :6], axis=0).max() < self.stall_rad and np.ptp(qs[:, 6]) == 0:
                        self.stop_reason = (f'진행 멈춤: {self.stall_sec:.0f}초 동안 거의 움직이지 않음 '
                                            f'(학습 때 없던 상황일 가능성 — 실행 기록 확인)')
                        break
                chunk = self.policy.predict(q7, img)
                lat = time.monotonic() - t_obs
                self.n_infer += 1
                skip = int(round(lat / self.dt))
                s = min(skip, len(chunk) - 1)
                self.ens.add(t_obs, chunk)
                self.grip_ahead = float(chunk[s:s + int(1.0 / self.dt), 6].max())
                log['t'].append(t_obs - t_start); log['q'].append(q7); log['chunk'].append(chunk)
                log['lat'].append(lat); log['skip'].append(skip); log['img'].append(img)

                # 팔: 과거 예측들을 로봇 위치 기준으로 맞춰 평균한 궤적 → 시작점·속도 제한
                # 기준 위치 = 방금 보낸 궤적에서 지금 있어야 할 명령 위치 (실제 위치는 빠를수록 명령보다 뒤처져서,
                #             실제 위치에서 다시 시작하면 명령이 뒤로 당겨졌다 앞으로 가며 떨림)
                t_now = time.monotonic()
                with self.lock:
                    q_meas = self.qpos6.copy()
                q_ref, v_ref, use_prev = q_meas, np.zeros(6), False
                scale = 1.0 if self.speed_scaling is None else float(np.clip(self.speed_scaling / 100.0, 0.05, 1.0))
                if self._streaming:
                    # 스트리밍: 지금 보내고 있는 명령 위치·속도에서 바로 이어감 (컨트롤러가 되돌리지 않음)
                    q_ref, v_ref = self.stream_state(t_now)
                    cs, el, use_prev = self._stream_seg[1], t_now - self._stream_seg[0], self._last_cubic is not None
                    if np.abs(q_ref - q_meas).max() > self.ref_tol:
                        self.stop_reason = (f'명령과 실제 위치 차이가 {math.degrees(np.abs(q_ref - q_meas).max()):.1f}° '
                                            '(로봇이 명령을 못 따라감) → 안전 정지')
                        break
                elif self._last_cubic is not None:
                    # 직전에 보낸 곡선에서 '지금' 명령 위치·속도 (슬라이더가 낮으면 컨트롤러 시간도 느리게 흐름)
                    el = (t_now - self._last_cubic[0]) * scale
                    cs = self._last_cubic[1]
                    q_cmd = cs.pos(el)
                    if np.abs(q_cmd - q_meas).max() < self.ref_tol:
                        q_ref, v_ref, use_prev = q_cmd, cs.vel(el), True
                n_plan = self.n_exec + 6
                plan = self.ens.plan_aligned(q_ref, t_now, n_plan)
                if use_prev and self.blend_steps > 0:
                    # 크로스페이드: 앞쪽은 직전 계획 위주 → blend_steps에 걸쳐 새 계획으로
                    prev = np.stack([cs.pos(el + (m + 1) * self.dt) for m in range(n_plan)])
                    w = np.minimum(1.0, (np.arange(n_plan) + 1) / self.blend_steps)[:, None]
                    plan = w * plan + (1 - w) * prev
                plan7 = np.concatenate([plan, np.zeros((len(plan), 1))], axis=1)   # build_segment는 7열 입력
                pos, times, vel, grip, s_idx, jump = build_segment(plan7, 0, self.n_exec, self.dt,
                                                                    self.max_joint_vel, q_ref)
                jump = max(jump, float(np.abs(q_ref - q_meas).max()))
                if jump > self.max_jump:
                    self.stop_reason = (f'모델 예측 궤적이 현재 자세에서 최소 {math.degrees(jump):.1f}° 떨어짐 '
                                        f'(제한 {math.degrees(self.max_jump):.0f}°) → 안전 정지')
                    break
                if not self.running:
                    break
                # 3차 곡선으로 다듬기: 현재 위치·속도에서 이어지고, 예측의 잔떨림은 걸러냄
                cs_new = fit_cubic(q_ref, v_ref, times, pos, self.max_joint_vel)
                if self._streaming:
                    self._stream_set(cs_new, t_now)       # 스트리밍 스레드가 이 곡선을 따라 명령을 보냄
                    self._last_cubic = (t_now, cs_new)
                else:
                    ts = np.linspace(cs_new.T / 12, cs_new.T, 12)
                    self.publish_segment(np.stack([cs_new.pos(x) for x in ts]), ts,
                                         np.stack([cs_new.vel(x) for x in ts[:-1]] + [np.zeros(6)]))
                    self._last_cubic = (time.monotonic(), cs_new)

                # 추론이 아주 빠르면 너무 자주 보내지 않도록 최소 간격
                wait = 2 * self.dt - (time.monotonic() - t_obs)
                if wait > 0:
                    time.sleep(wait)
        except Exception as e:
            self.stop_reason = f'오류: {e}'
        finally:
            self.returning = False
            self.phase = ''
            try:
                if self._streaming:
                    self.stream_end()                   # 오류·정지로 끝나도 부드럽게 멈추고 궤적 컨트롤러 복구
            except Exception as e:
                self.get_logger().error(f'스트리밍 종료 중 오류: {e}')
            log['js'], self._js_log = self._js_log, None
            if self.running:
                self.running = False
                if not (self.stop_reason or '').startswith('완료'):
                    self.hold_position()
            try:
                self.last_run_file = self._save_run(log)
            except Exception as e:
                self.get_logger().error(f'실행 기록 저장 실패: {e}')
                self.last_run_file = None


# ─────────────────────────────── 키보드 UI ───────────────────────────────
HELP = """
┌───────────────── ACT 실행 ─────────────────┐
│ o : 관찰 자세로 이동                       │
│ SPACE : 모델 실행 시작 / 정지              │
│ g : 그리퍼 토글 (실행 중에도 수동 개입 가능)│
│ q : 종료                                   │
│ ※ 비상시 티치펜던트 비상정지 버튼          │
└────────────────────────────────────────────┘"""


def main(args=None):
    try:
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    except (ImportError, TypeError):
        rclpy.init(args=args)
    node = ACTRunner()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    def say(msg):
        print('\r\033[K' + msg, flush=True)

    time.sleep(1.0)
    try:
        node.check_setup()
    except Exception as e:
        say(f'[ERROR] {e}')
        node.destroy_node(); rclpy.shutdown()
        return
    say('모델 준비 중 (CPU 속도 측정)...')
    lat, n = node.warmup()
    say(f'[OK] 모델: {node.get_parameter("ckpt").value}')
    say(f'[OK] CPU 추론 {lat * 1000:.0f}ms → 한 번에 {n}스텝({n * node.dt:.2f}s) 구간 전송')
    if lat > 1.0:
        say('[WARN] 추론이 1초 이상 걸립니다. 동작이 끊길 수 있습니다')
    say(f'[OK] 속도 제한 {math.degrees(node.max_joint_vel):.0f}°/s, 이탈 제한 {math.degrees(node.max_jump):.0f}°, '
        f'최대 {node.run_max_duration:.0f}s, 그리퍼 닫기 기준 {node.grip_logic.close_th:.2f}')
    say('[OK] External Control 실행 중' if node.program_running else
        '[WARN] External Control이 재생 중이 아닙니다 → 티치펜던트에서 ▶')
    if node.observe_joints is None:
        say('[WARN] 관찰 자세 파일이 없습니다 (data_collector에서 p로 저장한 observe_pose.json 필요)')
    if node.speed_scaling is not None and node.speed_scaling < 95:
        say(f'[WARN] 티치펜던트 속도 슬라이더 {node.speed_scaling:.0f}% → 100%로 올리세요. '
            '낮으면 로봇이 모델 계획보다 늦어져 흔들립니다 (속도는 노드가 max_joint_vel로 제한함)')
    if node.control_mode == 'stream':
        st = node.controller_states().get(FWD_CTRL)
        say(f'[OK] 모델 구간 제어: {FWD_CTRL}에 {node.stream_hz:.0f}Hz 연속 명령 (상태: {st})' if st else
            f'[WARN] {FWD_CTRL}가 없습니다 → -p control_mode:=jtc 로 실행하세요')
    else:
        say('[OK] 모델 구간 제어: 궤적 컨트롤러 (0.3초마다 궤적 교체)')
    if node.place_mode == 'scripted' and node.place is not None:
        say(f'[OK] 잡은 뒤: 좌표 이동 (수직 {node.lift_height * 100:.0f}cm 들어올리기 → 컵 위 → 수직 내려놓기 → 복귀) '
            f'— {node.place["path"]}')
    else:
        say(f'[OK] 잡은 뒤에도 모델이 이동, 놓고 {node.post_release_sec:.1f}s 후 관찰 자세로 자동 복귀')
    print(HELP, flush=True)

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    was_running = False
    last_status = 0.0
    t_run = 0.0
    shown_file = None
    try:
        while True:
            if was_running and not node.running:
                say(f'[STOP] {node.stop_reason} (추론 {node.n_infer}회)')
                was_running = False
            if node.running and time.monotonic() - last_status > 0.5:
                last_status = time.monotonic()
                g = 'CLOSE' if node.gripper_cmd == GRIPPER_CLOSE else 'OPEN '
                if node.returning:
                    print(f'\r\033[K▶ {time.monotonic() - t_run:5.1f}s | 좌표 이동: {node.phase}',
                          end='', flush=True)
                else:
                    print(f'\r\033[K▶ {time.monotonic() - t_run:5.1f}s | 추론 {node.n_infer}회 | gripper {g} | '
                          f'예측 지금 {node.grip_now:.2f} / 1초내 최대 {node.grip_ahead:.2f} '
                          f'(닫기 기준 {node.grip_logic.close_th:.2f})', end='', flush=True)
            if node.last_run_file and node.last_run_file != shown_file:
                shown_file = node.last_run_file
                say(f'[LOG] 실행 기록 저장: {shown_file}')

            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not r:
                continue
            key = sys.stdin.read(1)
            try:
                if key == 'q':
                    break
                elif key == ' ':
                    if node.running:
                        node.stop_run('사용자 정지 (SPACE)')
                    else:
                        if node.observe_joints is not None and node.qpos6 is not None:
                            d = np.abs(node.qpos6 - np.array(node.observe_joints)).max()
                            if d > 0.1:
                                say(f'[WARN] 관찰 자세에서 {math.degrees(d):.0f}° 떨어져 있습니다. '
                                    '학습 데이터는 관찰 자세에서 시작했으니 o로 먼저 이동하세요')
                                continue
                        if node.speed_scaling is not None and node.speed_scaling < 95:
                            say(f'[WARN] 속도 슬라이더 {node.speed_scaling:.0f}% (100% 권장, 흔들림 원인)')
                        node.start_run()
                        t_run = time.monotonic()
                        was_running = True
                        say('[RUN] 모델 실행 시작 (SPACE로 정지)')
                elif key == 'o':
                    node.stop_run('관찰 자세 이동')
                    say('관찰 자세로 이동 중...')
                    dur = node.move_to_observe()
                    say(f'[OK] 관찰 자세 도착 ({dur:.1f}s)')
                elif key == 'g':
                    if node.running:
                        v = node.manual_gripper()
                        say(f'[GRIPPER 수동] {"CLOSE" if v == GRIPPER_CLOSE else "OPEN"} '
                            '(3초간 모델 그리퍼 판단 중지, 모델은 계속 실행)')
                    else:
                        v = node.toggle_gripper()
                        say(f'[GRIPPER] {"CLOSE" if v == GRIPPER_CLOSE else "OPEN"}')
                elif key in ('h', '?'):
                    print(HELP, flush=True)
            except Exception as e:
                say(f'[ERROR] {e}')
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        print()
        node.stop_run('종료')
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
