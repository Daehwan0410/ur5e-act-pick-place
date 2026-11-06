#!/usr/bin/env python3
"""
수집한 에피소드 점검 (ROS 없이 실행 가능)

  python3 inspect_episode.py ~/act_data/pick_place              # 폴더 전체 요약
  python3 inspect_episode.py ~/act_data/pick_place/episode_0.hdf5 --video   # 미리보기 mp4 생성
"""
import argparse
import glob
import os
import re

import numpy as np
import h5py

JOINT_SHORT = ['pan', 'lift', 'elbow', 'w1', 'w2', 'w3', 'grip']


def summarize(path, verbose=True):
    with h5py.File(path, 'r') as f:
        qpos = f['observations/qpos'][()]
        action = f['action'][()]
        ts = f['timestamp'][()]
        fps = float(f.attrs.get('fps', 30.0))
        cams = list(f['observations/images'].keys())
        img_shape = f[f'observations/images/{cams[0]}'].shape

    T = len(qpos)
    dur = ts[-1] - ts[0] if T > 1 else 0.0
    dt = np.diff(ts) if T > 1 else np.array([1 / fps])
    grip = qpos[:, 6]
    toggles = np.where(np.diff(grip) != 0)[0] + 1
    warnings = []

    real_fps = (T - 1) / dur if dur > 0 else 0
    if real_fps < fps * 0.85:
        warnings.append(f'실제 fps {real_fps:.1f} < 설정 {fps:.0f} (프레임 누락)')
    if dt.max() > 3.0 / fps:
        warnings.append(f'최대 프레임 간격 {dt.max() * 1000:.0f}ms (센서 끊김)')
    n_close = int(np.sum(np.diff(grip) > 0))
    n_open = int(np.sum(np.diff(grip) < 0))
    if n_close == 0:
        warnings.append('그리퍼를 한 번도 닫지 않음')
    if n_close > 1 or n_open > 1:
        warnings.append(f'그리퍼 토글이 여러 번 (닫기 {n_close}, 열기 {n_open}) → 재시도 시연이면 버리는 게 나음')
    if grip[-1] == 1.0:
        warnings.append('그리퍼가 닫힌 채로 끝남 (놓기 전에 녹화 종료?)')
    joint_range = qpos[:, :6].max(0) - qpos[:, :6].min(0)
    if joint_range.max() > np.pi:
        j = JOINT_SHORT[int(joint_range.argmax())]
        warnings.append(f'{j} 관절이 {np.degrees(joint_range.max()):.0f}° 회전 (과도한 회전)')
    if np.abs(action - qpos).max() == 0:
        warnings.append('로봇이 전혀 움직이지 않음')

    if verbose:
        name = os.path.basename(path)
        print(f'\n■ {name}: {T} frames, {dur:.1f}s, fps {real_fps:.1f}, 이미지 {img_shape[1:]} ({", ".join(cams)})')
        print('  관절 이동 범위(°): ' + '  '.join(f'{n}={np.degrees(r):.0f}' for n, r in zip(JOINT_SHORT, joint_range)))
        print(f'  그리퍼 변화 시점(s): {[round(float(ts[i]), 1) for i in toggles]}')
        for w in warnings:
            print(f'  ⚠ {w}')
        if not warnings:
            print('  ✓ 이상 없음')
    return dict(T=T, dur=dur, warnings=warnings)


def make_video(path, out=None):
    import cv2
    out = out or os.path.splitext(path)[0] + '_preview.mp4'
    with h5py.File(path, 'r') as f:
        cam = list(f['observations/images'].keys())[0]
        imgs = f[f'observations/images/{cam}']
        qpos = f['observations/qpos'][()]
        ts = f['timestamp'][()]
        fps = float(f.attrs.get('fps', 30.0))
        T, H, W, _ = imgs.shape
        scale = max(1, 480 // H)
        vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W * scale, H * scale))
        for t in range(T):
            frame = cv2.cvtColor(imgs[t], cv2.COLOR_RGB2BGR)
            if scale > 1:
                frame = cv2.resize(frame, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
            g = 'CLOSE' if qpos[t, 6] > 0.5 else 'OPEN'
            color = (0, 0, 255) if g == 'CLOSE' else (0, 200, 0)
            cv2.putText(frame, f'{ts[t]:5.1f}s  #{t}  gripper {g}', (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            js = ' '.join(f'{np.degrees(v):.0f}' for v in qpos[t, :6])
            cv2.putText(frame, js, (8, H * scale - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            vw.write(frame)
        vw.release()
    print(f'  ▶ 미리보기: {out}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path', help='episode_N.hdf5 파일 또는 폴더')
    ap.add_argument('--video', action='store_true', help='미리보기 mp4 생성')
    a = ap.parse_args()
    p = os.path.expanduser(a.path)

    if os.path.isdir(p):
        files = sorted(glob.glob(os.path.join(p, 'episode_*.hdf5')),
                       key=lambda s: int(re.search(r'episode_(\d+)', s).group(1)))
        if not files:
            print('에피소드 파일이 없습니다')
            return
        stats = [summarize(f) for f in files]
        bad = [os.path.basename(f) for f, s in zip(files, stats) if s['warnings']]
        durs = [s['dur'] for s in stats]
        print(f'\n━━ 총 {len(files)}개 에피소드, 평균 {np.mean(durs):.1f}s '
              f'(최소 {min(durs):.1f}s / 최대 {max(durs):.1f}s), 경고 있는 에피소드 {len(bad)}개')
        if bad:
            print('   확인 필요: ' + ', '.join(bad))
        if a.video:
            for f in files:
                make_video(f)
    else:
        summarize(p)
        if a.video:
            make_video(p)


if __name__ == '__main__':
    main()
