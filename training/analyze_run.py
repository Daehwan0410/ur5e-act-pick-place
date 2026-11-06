#!/usr/bin/env python3
"""
로봇 실행 기록(run_*.hdf5) 분석

  python analyze_run.py C:\\Users\\user1\\urPick\\act_runs\\run_20261106_101500.hdf5

결과 (같은 폴더, 파일 이름_analysis.png / _frames.png):
  analysis.png  시간에 따른 관절 움직임, 그리퍼 예측값(지금 / 1초내 최대), 멈춰 있던 구간 표시
  frames.png    실행 중 카메라 화면 12장 (머뭇거린 구간 포함)
"""
import argparse
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3']


def set_korean_font():
    from matplotlib import font_manager
    for name in ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'Noto Sans CJK KR']:
        if any(name == f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams['font.family'] = name
            break
    plt.rcParams['axes.unicode_minus'] = False


def stall_segments(t, q, speed_th=np.radians(1.0), min_len=3.0):
    """관절 최대 속도가 speed_th(1°/s) 미만인 구간이 min_len초 이상 이어지면 '머뭇거림'"""
    if len(t) < 3:
        return []
    v = np.abs(np.diff(q, axis=0)).max(axis=1) / np.maximum(np.diff(t), 1e-3)
    slow = np.r_[v < speed_th, False]
    segs, start = [], None
    for i, s in enumerate(slow):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if t[i] - t[start] >= min_len:
                segs.append((start, i))
            start = None
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    a = ap.parse_args()
    set_korean_font()

    with h5py.File(a.run, 'r') as f:
        t = f['t'][()]
        q = f['qpos'][()]
        chunk = f['chunk'][()]
        lat = f['latency'][()]
        skip = f['skip'][()]
        imgs = f['images']
        reason = f.attrs.get('stop_reason', '')
        n = len(t)
        idx = np.clip(skip, 0, chunk.shape[1] - 1)
        g_now = chunk[np.arange(n), idx, 6]
        g_ahead = np.array([chunk[i, idx[i]:idx[i] + 30, 6].max() for i in range(n)])
        segs = stall_segments(t, q[:, :6])

        js_t = f['js_t'][()] if 'js_t' in f else None
        js_q = f['js_q'][()] if 'js_q' in f else None

        # 화면 12장: 균등 간격 + 머뭇거림 구간 중앙
        picks = set(np.linspace(0, n - 1, 9).astype(int).tolist())
        for s, e in segs[:3]:
            picks.add((s + e) // 2)
        picks = sorted(picks)[:12]
        frames = [(i, imgs[i]) for i in picks]

    stem = os.path.splitext(a.run)[0]
    print(f'\n■ {os.path.basename(a.run)}: {t[-1]:.1f}s, 추론 {n}회, 평균 지연 {lat.mean() * 1000:.0f}ms')
    print(f'  정지 이유: {reason}')
    print(f'  그리퍼 상태: {"한 번이라도 닫힘" if q[:, 6].max() > 0.5 else "끝까지 열림"}')
    print(f'  그리퍼 예측(지금) 최대 {g_now.max():.2f} @ {t[g_now.argmax()]:.1f}s, '
          f'1초내 최대값의 최대 {g_ahead.max():.2f} @ {t[g_ahead.argmax()]:.1f}s')
    for s, e in segs:
        print(f'  머뭇거림: {t[s]:.1f}~{t[e]:.1f}s ({t[e] - t[s]:.1f}s 동안 거의 정지), '
              f'그 동안 그리퍼 예측 평균 {g_now[s:e].mean():.2f}, 최대 {g_now[s:e].max():.2f}')
    if not segs:
        print('  머뭇거림 구간 없음 (계속 움직임)')

    speed = None
    if js_t is not None and len(js_t) > 50:
        # 100Hz 관절 기록으로 부드러움 측정 (0.1초 이동평균으로 센서 잡음 제거 후 미분)
        k = 10
        qs = np.apply_along_axis(lambda x: np.convolve(x, np.ones(k) / k, mode='valid'), 0, js_q)
        ts = js_t[k // 2: k // 2 + len(qs)]
        v = np.diff(qs, axis=0) / np.diff(ts)[:, None]
        acc = np.diff(v, axis=0) / np.diff(ts)[1:, None]
        speed = (ts[1:], np.degrees(np.abs(v).max(axis=1)))
        moving = speed[1][1:] > 2.0
        if moving.any():
            am = np.degrees(np.abs(acc).max(axis=1))[moving]
            print(f'  [100Hz 기록] 움직이는 동안 관절 가속도: 중간값 {np.median(am):.0f}°/s², '
                  f'상위 1% {np.percentile(am, 99):.0f}°/s² (낮을수록 부드러움)')
            sp = speed[1][1:][moving]
            print(f'  [100Hz 기록] 관절 최고 속도 {sp.max():.0f}°/s, 움직이는 동안 중간값 {np.median(sp):.0f}°/s')

    nrow = 4 if speed is not None else 3
    fig, axes = plt.subplots(nrow, 1, figsize=(11, 11 if speed is not None else 9), sharex=True,
                             gridspec_kw={'height_ratios': [3, 2, 1, 2][:nrow]})
    for j in range(6):
        axes[0].plot(t, np.degrees(q[:, j] - q[0, j]), label=JOINTS[j])
    axes[0].set_ylabel('시작 자세 대비 변화 (°)')
    axes[0].legend(fontsize=8, ncol=3)
    axes[1].plot(t, g_now, label='그리퍼 예측 (지금)', color='#d62728')
    axes[1].plot(t, g_ahead, label='그리퍼 예측 (1초내 최대)', color='#ff9896', ls='--')
    axes[1].step(t, q[:, 6], label='실제 그리퍼 상태', color='k', where='post', lw=1)
    axes[1].axhline(0.5, color='gray', ls=':', lw=1)
    axes[1].set_ylim(-0.05, 1.05); axes[1].set_ylabel('그리퍼'); axes[1].legend(fontsize=8)
    axes[2].plot(t, lat * 1000, color='#7f7f7f'); axes[2].set_ylabel('추론 지연 (ms)')
    if speed is not None:
        axes[3].plot(*speed, color='#1f77b4', lw=1)
        axes[3].set_ylabel('관절 최고 속도 (°/s)\n100Hz 기록')
    axes[-1].set_xlabel('시간 (s)')
    for ax in axes:
        for s, e in segs:
            ax.axvspan(t[s], t[e], color='orange', alpha=0.15)
        ax.grid(alpha=0.3)
    fig.suptitle(f'{os.path.basename(a.run)} — 주황 영역: 머뭇거림 / 정지 이유: {reason}', fontsize=10)
    fig.tight_layout()
    fig.savefig(stem + '_analysis.png', dpi=100)
    plt.close(fig)

    cols = 4
    rows = int(np.ceil(len(frames) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.6))
    for ax in np.ravel(axes):
        ax.axis('off')
    for ax, (i, im) in zip(np.ravel(axes), frames):
        ax.imshow(im)
        stalled = any(s <= i <= e for s, e in segs)
        ax.set_title(f'{t[i]:.1f}s  그리퍼예측 {g_now[i]:.2f}' + ('  [머뭇]' if stalled else ''),
                     fontsize=8, color='#d62728' if stalled else 'k')
    fig.tight_layout()
    fig.savefig(stem + '_frames.png', dpi=100)
    plt.close(fig)
    print(f'  그래프: {stem}_analysis.png, {stem}_frames.png')


if __name__ == '__main__':
    main()
