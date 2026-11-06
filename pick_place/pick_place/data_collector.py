#!/usr/bin/env python3
"""
ACT 모방학습용 시연 데이터 수집 노드 (UR5e 프리드라이브)

녹화 항목 (fps 주기):
  observations/qpos            (T, 7)  관절 6개 + 그리퍼(0=열림, 1=닫힘)
  observations/qvel            (T, 7)
  observations/images/<cam>    (T, H, W, 3) uint8 RGB
  action                       (T, 7)  다음 스텝의 qpos (프리드라이브 시연 표준)
  timestamp                    (T,)
→ 원본 ACT(tonyzhaozh/act) HDF5 형식과 동일. LeRobot 변환도 이 파일에서 함.

키 조작 (이 노드를 실행한 터미널에서):
  o      관찰 자세로 자동 이동 (프리드라이브 자동 해제)
  f      프리드라이브 ON / OFF
  g      그리퍼 열기 / 닫기 토글
  SPACE  녹화 시작 / 정지+저장
  d      현재 녹화 버리기
  p      현재 로봇 자세를 관찰 자세로 저장 (save_dir/observe_pose.json)
  q      종료

에피소드 1회 순서:
  o → (물체를 새 위치에 놓기) → f → SPACE → 손으로 시연(g로 집고 g로 놓기) → 손 떼고 o
  (녹화 중 o: 녹화를 유지한 채 관찰 자세로 자동 복귀, 도착하면 자동 저장)
"""
import glob
import json
import os
import re
import select
import sys
import termios
import threading
import time
import tty

import numpy as np
import cv2
import h5py

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Bool
from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from controller_manager_msgs.srv import SwitchController, ListControllers
from ur_msgs.srv import SetIO
from ur_msgs.msg import IOStates
from cv_bridge import CvBridge

ARM_JOINTS = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
              'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
OBSERVE_POSE_FILE = 'observe_pose.json'   # p 키로 저장, 시작 시 자동 로드

TRAJ_CONTROLLERS = ['scaled_joint_trajectory_controller', 'joint_trajectory_controller']
FREEDRIVE_CTRL = 'freedrive_mode_controller'

GRIPPER_OPEN = 0.0
GRIPPER_CLOSE = 1.0


# ─────────────────────────── 저장 유틸 (ROS 의존 없음) ───────────────────────────
def next_episode_index(save_dir):
    idx = [int(m.group(1)) for f in glob.glob(os.path.join(save_dir, 'episode_*.hdf5'))
           if (m := re.search(r'episode_(\d+)\.hdf5$', f))]
    return max(idx) + 1 if idx else 0


def save_episode_hdf5(path, qpos, qvel, images, timestamps, camera_name, fps, attrs=None):
    """qpos/qvel: (T,7), images: (T,H,W,3) uint8, timestamps: (T,)"""
    qpos = np.asarray(qpos, dtype=np.float32)
    qvel = np.asarray(qvel, dtype=np.float32)
    images = np.asarray(images, dtype=np.uint8)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    T = len(qpos)
    assert len(qvel) == T and len(images) == T and len(timestamps) == T, '길이 불일치'

    # action[t] = qpos[t+1]  (마지막은 그대로 유지)
    action = np.concatenate([qpos[1:], qpos[-1:]], axis=0)

    tmp = path + '.tmp'
    _, H, W, _ = images.shape
    with h5py.File(tmp, 'w') as f:
        f.attrs['sim'] = False
        f.attrs['fps'] = float(fps)
        f.attrs['camera_names'] = [camera_name]
        f.attrs['joint_names'] = ARM_JOINTS + ['gripper']
        for k, v in (attrs or {}).items():
            f.attrs[k] = v
        obs = f.create_group('observations')
        obs.create_dataset('qpos', data=qpos)
        obs.create_dataset('qvel', data=qvel)
        img_grp = obs.create_group('images')
        img_grp.create_dataset(camera_name, data=images, chunks=(1, H, W, 3),
                               compression='gzip', compression_opts=4)
        f.create_dataset('action', data=action)
        f.create_dataset('timestamp', data=timestamps)
    os.replace(tmp, path)   # 저장 중 중단돼도 깨진 파일이 남지 않게
    return T


# ─────────────────────────────── ROS2 노드 ───────────────────────────────
class DataCollector(Node):
    def __init__(self, node_name='act_data_collector'):
        super().__init__(node_name)
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('save_dir', '~/act_data/pick_place')
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('img_width', 320)
        self.declare_parameter('img_height', 240)
        self.declare_parameter('camera_name', 'cam_main')
        self.declare_parameter('max_duration', 90.0)
        self.declare_parameter('gripper_pin', 16)

        gp = self.get_parameter
        self.image_topic = gp('image_topic').value
        self.save_dir = os.path.expanduser(gp('save_dir').value)
        self.fps = float(gp('fps').value)
        self.img_w = int(gp('img_width').value)
        self.img_h = int(gp('img_height').value)
        self.cam_name = gp('camera_name').value
        self.max_duration = float(gp('max_duration').value)
        self.gripper_pin = int(gp('gripper_pin').value)
        os.makedirs(self.save_dir, exist_ok=True)
        self.observe_joints = self._load_observe_pose()

        self.cb = ReentrantCallbackGroup()
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        # 최신 센서값
        self.qpos6 = None
        self.qvel6 = np.zeros(6)
        self.js_time = 0.0
        self.image = None
        self.img_time = 0.0
        self.gripper_cmd = GRIPPER_OPEN
        self.gripper_io = None

        # 상태
        self.freedrive = False
        self.traj_ctrl = None
        self.recording = False
        self.auto_stop = False
        self._reset_buffers()

        self.create_subscription(JointState, '/joint_states', self._js_cb, 50, callback_group=self.cb)
        self.create_subscription(Image, self.image_topic, self._img_cb, 5, callback_group=self.cb)
        self.create_subscription(IOStates, '/io_and_status_controller/io_states', self._io_cb, 10,
                                 callback_group=self.cb)
        # 티치펜던트 External Control 재생 여부 (latched 토픽)
        self.program_running = None
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Bool, '/io_and_status_controller/robot_program_running',
                                 self._prog_cb, latched, callback_group=self.cb)

        self.fd_pub = self.create_publisher(Bool, f'/{FREEDRIVE_CTRL}/enable_freedrive_mode', 10)
        self.switch_cli = self.create_client(SwitchController, '/controller_manager/switch_controller',
                                             callback_group=self.cb)
        self.list_cli = self.create_client(ListControllers, '/controller_manager/list_controllers',
                                           callback_group=self.cb)
        self.io_cli = self.create_client(SetIO, '/io_and_status_controller/set_io', callback_group=self.cb)
        self.traj_clients = {c: ActionClient(self, FollowJointTrajectory, f'/{c}/follow_joint_trajectory',
                                             callback_group=self.cb) for c in TRAJ_CONTROLLERS}

        self.create_timer(0.1, self._freedrive_keepalive, callback_group=self.cb)   # inactive_timeout=1s
        self.create_timer(1.0 / self.fps, self._record_tick, callback_group=self.cb)

    # ── 관찰 자세 저장/로드 ──
    def _load_observe_pose(self):
        path = os.path.join(self.save_dir, OBSERVE_POSE_FILE)
        try:
            with open(path) as f:
                q = json.load(f)['joints']
            if len(q) == 6:
                return [float(v) for v in q]
        except (OSError, KeyError, ValueError):
            pass
        return None

    def save_observe_pose(self):
        with self.lock:
            if self.qpos6 is None:
                raise RuntimeError('joint_states 수신 전입니다')
            q = [round(float(v), 4) for v in self.qpos6]
        path = os.path.join(self.save_dir, OBSERVE_POSE_FILE)
        with open(path, 'w') as f:
            json.dump({'joint_names': ARM_JOINTS, 'joints': q,
                       'saved': time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2)
        self.observe_joints = q
        return q, path

    # ── 콜백 ──
    def _js_cb(self, msg):
        idx = {n: i for i, n in enumerate(msg.name)}
        if not all(j in idx for j in ARM_JOINTS):
            return
        pos = [msg.position[idx[j]] for j in ARM_JOINTS]   # /joint_states 순서는 알파벳순이라 재정렬 필수
        vel = [msg.velocity[idx[j]] for j in ARM_JOINTS] if len(msg.velocity) == len(msg.name) else [0.0] * 6
        with self.lock:
            self.qpos6 = np.array(pos)
            self.qvel6 = np.array(vel)
            self.js_time = time.monotonic()

    def _img_cb(self, msg):
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().error(f'이미지 변환 실패: {e}', throttle_duration_sec=5.0)
            return
        if rgb.shape[1] != self.img_w or rgb.shape[0] != self.img_h:
            rgb = cv2.resize(rgb, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
        with self.lock:
            self.image = rgb
            self.img_time = time.monotonic()

    def _io_cb(self, msg):
        for d in msg.digital_out_states:
            if d.pin == self.gripper_pin:
                self.gripper_io = bool(d.state)

    def _prog_cb(self, msg):
        if self.program_running is True and not msg.data:
            self.get_logger().error('External Control 프로그램이 멈췄습니다! 티치펜던트에서 다시 재생(▶)하세요')
            self.freedrive = False
        self.program_running = bool(msg.data)

    def require_program(self):
        if self.program_running is not True:
            raise RuntimeError('티치펜던트에서 External Control 프로그램이 재생(▶) 중이 아닙니다. '
                               '재생 후 드라이버 터미널에 "Ready to receive control commands"가 뜨면 다시 시도하세요')

    def _freedrive_keepalive(self):
        if self.freedrive:
            self.fd_pub.publish(Bool(data=True))

    def _record_tick(self):
        if not self.recording:
            return
        now = time.monotonic()
        with self.lock:
            if self.qpos6 is None or self.image is None:
                return
            if now - self.img_time > 0.5 or now - self.js_time > 0.5:
                self.stale_count += 1
                return
            g = self.gripper_cmd
            self.buf_qpos.append(np.append(self.qpos6, g))
            self.buf_qvel.append(np.append(self.qvel6, 0.0))
            self.buf_img.append(self.image)          # _img_cb가 매번 새 배열을 만들므로 copy 불필요
            self.buf_t.append(now - self.t0)
        if now - self.t0 > self.max_duration:
            self.recording = False
            self.auto_stop = True

    def _reset_buffers(self):
        self.buf_qpos, self.buf_qvel, self.buf_img, self.buf_t = [], [], [], []
        self.stale_count = 0
        self.t0 = time.monotonic()

    # ── 서비스 헬퍼 ──
    def _call(self, cli, req, timeout=3.0):
        if not cli.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f'서비스 없음: {cli.srv_name}')
        fut = cli.call_async(req)
        end = time.monotonic() + timeout
        while not fut.done():
            if time.monotonic() > end:
                raise RuntimeError(f'서비스 응답 시간 초과: {cli.srv_name}')
            time.sleep(0.01)
        return fut.result()

    def controller_states(self):
        res = self._call(self.list_cli, ListControllers.Request())
        return {c.name: c.state for c in res.controller}

    def _switch(self, activate, deactivate):
        req = SwitchController.Request()
        req.activate_controllers = activate
        req.deactivate_controllers = deactivate
        req.strictness = SwitchController.Request.STRICT
        req.timeout = DurationMsg(sec=3)
        if not self._call(self.switch_cli, req, timeout=5.0).ok:
            raise RuntimeError(f'컨트롤러 전환 실패 (activate={activate}, deactivate={deactivate})')

    def check_setup(self):
        """시작 시 드라이버 구성 확인"""
        states = self.controller_states()
        if FREEDRIVE_CTRL not in states:
            raise RuntimeError(
                f"'{FREEDRIVE_CTRL}'가 없습니다. ur_robot_driver 2.5.0 이상이 필요합니다.\n"
                "  sudo apt update && sudo apt install --only-upgrade ros-humble-ur*")
        active = [c for c in TRAJ_CONTROLLERS if states.get(c) == 'active']
        self.traj_ctrl = active[0] if active else 'scaled_joint_trajectory_controller'
        return states

    # ── 동작 ──
    def freedrive_on(self):
        if self.freedrive:
            return
        self.require_program()
        states = self.controller_states()
        active_traj = [c for c in TRAJ_CONTROLLERS if states.get(c) == 'active']
        if active_traj:
            self.traj_ctrl = active_traj[0]
        if states.get(FREEDRIVE_CTRL) != 'active':
            self._switch([FREEDRIVE_CTRL], active_traj)
        self.freedrive = True
        self.fd_pub.publish(Bool(data=True))

    def freedrive_off(self):
        if not self.freedrive:
            states = self.controller_states()
            if states.get(FREEDRIVE_CTRL) != 'active':
                return
        self.freedrive = False
        self.fd_pub.publish(Bool(data=False))
        time.sleep(0.3)
        self._switch([self.traj_ctrl], [FREEDRIVE_CTRL])

    def set_gripper(self, value):
        req = SetIO.Request()
        req.fun = SetIO.Request.FUN_SET_DIGITAL_OUT
        req.pin = self.gripper_pin
        req.state = float(value)
        if not self._call(self.io_cli, req).success:
            raise RuntimeError('그리퍼 IO 설정 실패')
        with self.lock:
            self.gripper_cmd = float(value)

    def toggle_gripper(self):
        self.set_gripper(GRIPPER_CLOSE if self.gripper_cmd == GRIPPER_OPEN else GRIPPER_OPEN)
        return self.gripper_cmd

    def move_to_observe(self, max_vel=0.4):
        if self.observe_joints is None:
            raise RuntimeError('관찰 자세가 아직 없습니다. f로 로봇을 원하는 관찰 자세에 옮긴 뒤 p를 눌러 저장하세요')
        self.require_program()
        self.freedrive_off()
        if self.controller_states().get(self.traj_ctrl) != 'active':
            self._switch([self.traj_ctrl], [])
        with self.lock:
            cur = None if self.qpos6 is None else self.qpos6.copy()
        if cur is None:
            raise RuntimeError('joint_states 수신 전입니다')
        dur = max(3.0, float(np.max(np.abs(np.array(self.observe_joints) - cur))) / max_vel)

        cli = self.traj_clients[self.traj_ctrl]
        if not cli.wait_for_server(timeout_sec=3.0):
            raise RuntimeError(f'{self.traj_ctrl} 액션 서버 없음')
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ARM_JOINTS
        pt = JointTrajectoryPoint()
        pt.positions = list(self.observe_joints)
        pt.velocities = [0.0] * 6
        pt.time_from_start = DurationMsg(sec=int(dur), nanosec=int((dur % 1) * 1e9))
        goal.trajectory.points = [pt]

        fut = cli.send_goal_async(goal)
        while not fut.done():
            time.sleep(0.01)
        gh = fut.result()
        if not gh.accepted:
            raise RuntimeError('궤적 목표가 거부됨')
        res_fut = gh.get_result_async()
        end = time.monotonic() + dur + 10.0
        while not res_fut.done():
            if time.monotonic() > end:
                gh.cancel_goal_async()
                raise RuntimeError('관찰 자세 이동 시간 초과')
            time.sleep(0.02)
        code = res_fut.result().result.error_code
        if code != 0:
            raise RuntimeError(f'관찰 자세 이동 실패 (error_code={code})')
        return dur

    def start_recording(self):
        with self.lock:
            if self.qpos6 is None:
                raise RuntimeError('joint_states가 안 들어옵니다')
            if self.image is None:
                raise RuntimeError(f'카메라 이미지가 안 들어옵니다 ({self.image_topic})')
            self._reset_buffers()
        self.auto_stop = False
        self.recording = True

    def stop_and_save(self, min_frames=30):
        self.recording = False
        time.sleep(1.5 / self.fps)
        with self.lock:
            qpos, qvel, imgs, ts = self.buf_qpos, self.buf_qvel, self.buf_img, self.buf_t
            stale = self.stale_count
            self._reset_buffers()
        if len(qpos) < min_frames:
            return None, len(qpos), stale
        idx = next_episode_index(self.save_dir)
        path = os.path.join(self.save_dir, f'episode_{idx}.hdf5')
        save_episode_hdf5(path, np.stack(qpos), np.stack(qvel), np.stack(imgs), np.array(ts),
                          self.cam_name, self.fps,
                          attrs={'task': 'pick and place', 'created': time.strftime('%Y-%m-%d %H:%M:%S'),
                                 'observe_joints': self.observe_joints or []})
        return path, len(qpos), stale

    def discard(self):
        self.recording = False
        with self.lock:
            n = len(self.buf_qpos)
            self._reset_buffers()
        return n


# ─────────────────────────────── 키보드 UI ───────────────────────────────
HELP = """
┌──────────────── ACT 데이터 수집 ────────────────┐
│ o : 관찰 자세로 이동     f : 프리드라이브 ON/OFF │
│ g : 그리퍼 토글          SPACE : 녹화 시작/저장  │
│ d : 녹화 버리기          p : 현재 자세=관찰 자세 │
│ q : 종료                                         │
└──────────────────────────────────────────────────┘"""


def main(args=None):
    try:
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    except (ImportError, TypeError):
        rclpy.init(args=args)
    node = DataCollector()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    def say(msg):
        print('\r\033[K' + msg, flush=True)

    # 초기 확인
    time.sleep(1.0)
    try:
        states = node.check_setup()
        say(f'[OK] 궤적 컨트롤러: {node.traj_ctrl} ({states.get(node.traj_ctrl)}), '
            f'프리드라이브 컨트롤러: {states.get(FREEDRIVE_CTRL)}')
    except Exception as e:
        say(f'[ERROR] {e}')
        node.destroy_node()
        rclpy.shutdown()
        return
    if node.program_running is not True:
        say('[WARN] External Control 프로그램이 재생 중이 아닙니다 → 티치펜던트에서 ▶ 누르세요 (f, o 동작 안 함)')
    else:
        say('[OK] External Control 실행 중')
    if node.image is None:
        say(f'[WARN] 아직 카메라 이미지가 없습니다: {node.image_topic}')
    say(f'저장 위치: {node.save_dir}  (다음 에피소드 번호: {next_episode_index(node.save_dir)})')
    say(f'녹화: {node.fps:.0f}Hz, 이미지 {node.img_w}x{node.img_h}')
    if node.observe_joints is None:
        say('[WARN] 관찰 자세가 없습니다 → f로 원하는 관찰 자세로 옮기고 p로 저장하세요 (그 전엔 o 동작 안 함)')
    else:
        say('[OK] 관찰 자세: [' + ', '.join(f'{v:.3f}' for v in node.observe_joints) + ']')
    print(HELP, flush=True)

    fd = sys.stdin.fileno()
    old_term = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    last_status = 0.0
    try:
        while True:
            if node.auto_stop:
                node.auto_stop = False
                say(f'[AUTO] 최대 녹화 시간({node.max_duration:.0f}s) 도달 → 저장')
                path, n, stale = node.stop_and_save()
                say(f'[SAVED] {path} ({n} frames)' if path else f'[DISCARD] 프레임 부족 ({n})')

            if node.recording and time.monotonic() - last_status > 0.5:
                last_status = time.monotonic()
                g = 'CLOSE' if node.gripper_cmd == GRIPPER_CLOSE else 'OPEN '
                el = time.monotonic() - node.t0
                print(f'\r\033[K● REC {el:5.1f}s | {len(node.buf_qpos):4d} frames | gripper {g} | '
                      f'freedrive {"ON" if node.freedrive else "OFF"}', end='', flush=True)

            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not r:
                continue
            key = sys.stdin.read(1)
            try:
                if key == 'q':
                    break
                elif key == 'o':
                    if node.recording:
                        # 녹화 유지한 채로 관찰 자세 복귀 → 도착하면 자동 저장
                        say('[REC] 녹화 계속하며 관찰 자세로 복귀 중... (손 떼세요 / 비상시 티치펜던트 정지)')
                        try:
                            dur = node.move_to_observe()
                        except Exception as e:
                            say(f'[ERROR] 복귀 실패: {e} → 녹화는 계속됩니다 (SPACE 저장 / d 버리기)')
                            continue
                        time.sleep(0.3)   # 정지 상태 몇 프레임 포함
                        path, n, stale = node.stop_and_save()
                        say(f'[SAVED] {path} ({n} frames, {n / node.fps:.1f}s, 복귀 {dur:.1f}s 포함)'
                            if path else f'[DISCARD] 너무 짧음 ({n} frames)')
                        continue
                    say('관찰 자세로 이동 중... (비상시 티치펜던트 정지 버튼)')
                    dur = node.move_to_observe()
                    say(f'[OK] 관찰 자세 도착 ({dur:.1f}s)')
                elif key == 'f':
                    if node.freedrive:
                        node.freedrive_off()
                        say('[FREEDRIVE OFF] 로봇이 현재 위치에 고정됩니다')
                    else:
                        node.freedrive_on()
                        say('[FREEDRIVE ON] 손으로 로봇을 움직일 수 있습니다')
                elif key == 'g':
                    v = node.toggle_gripper()
                    say(f'[GRIPPER] {"CLOSE" if v == GRIPPER_CLOSE else "OPEN"}')
                elif key == ' ':
                    if node.recording:
                        say('저장 중...')
                        path, n, stale = node.stop_and_save()
                        if path:
                            say(f'[SAVED] {path} ({n} frames, {n / node.fps:.1f}s'
                                + (f', 센서 끊김 {stale}회' if stale else '') + ')')
                        else:
                            say(f'[DISCARD] 너무 짧음 ({n} frames)')
                    else:
                        if not node.freedrive:
                            say('[WARN] 프리드라이브가 꺼져 있습니다 (f로 켜세요). 그래도 녹화 시작합니다')
                        node.start_recording()
                        say(f'[REC START] episode_{next_episode_index(node.save_dir)}')
                elif key == 'p':
                    if node.recording:
                        say('[WARN] 녹화 중에는 관찰 자세를 저장할 수 없습니다')
                        continue
                    had = node.observe_joints is not None
                    q, path = node.save_observe_pose()
                    say('[SAVED] 관찰 자세: [' + ', '.join(f'{v:.3f}' for v in q) + f'] → {path}')
                    if had and next_episode_index(node.save_dir) > 0:
                        say('[WARN] 이미 녹화한 에피소드가 있습니다. 관찰 자세가 바뀌면 시작 자세가 달라지니 '
                            '기존 에피소드와 섞지 않는 게 좋습니다')
                elif key == 'd':
                    if node.recording:
                        n = node.discard()
                        say(f'[DISCARD] 녹화 {n} frames 버림')
                elif key in ('h', '?'):
                    print(HELP, flush=True)
            except Exception as e:
                say(f'[ERROR] {e}')
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
        print()
        if node.recording:
            n = node.discard()
            print(f'[종료] 저장 안 된 녹화 {n} frames 버림')
        try:
            node.freedrive_off()
            print('[종료] 프리드라이브 해제, 궤적 컨트롤러 복구')
        except Exception as e:
            print(f'[종료] 프리드라이브 해제 중 오류: {e}')
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
