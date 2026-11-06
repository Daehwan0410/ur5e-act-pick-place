#!/usr/bin/env python3
"""
학습된 ACT 모델 평가·시각화 (로봇 없이, 학습에 안 쓴 검증 에피소드로)

  python eval_act.py --ckpt C:\\...\\act_runs\\run1\\policy_best.ckpt --data_dir C:\\...\\act_data\\pick_place

결과 (ckpt와 같은 폴더의 eval/):
  summary.png              관절별 평균 오차(°), 그리퍼 정확도 한눈에
  episode_N_joints.png     에피소드별: 실제 시연 vs 모델 예측 (관절 6개 + 그리퍼)
  metrics.csv              에피소드별 수치

보는 법:
  - 매 스텝 "현재 화면+관절값 → 다음 동작" 예측이 실제 시연과 얼마나 같은지 비교 (temporal ensembling 적용, 실제 로봇 실행과 같은 방식)
  - 관절 오차가 1~2° 이내면 매우 좋음, 5° 넘으면 실제 로봇에서 빗나갈 가능성 큼
  - 그리퍼 타이밍 오차: 잡기/놓기를 실제보다 몇 초 빠르게/늦게 예측했는지
  ※ 오프라인 평가는 "시연을 얼마나 잘 따라 하는지"만 봅니다. 실제 성공률은 로봇에서 확인해야 합니다.
"""
import argparse
import csv
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
# act_model.py는 ROS 패키지(pick_place/pick_place/)에 하나만 둠 — 같은 폴더에 없으면 거기서 찾음
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pick_place', 'pick_place'))
from act_model import ACTPolicy, TemporalEnsembler

JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3', 'gripper']


def toggle_times(g, ts):
    d = np.diff((g > 0.5).astype(int))
    return ts[1:][d > 0], ts[1:][d < 0]          # 닫힘 시각, 열림 시각


def eval_episode(pol, path, k):
    with h5py.File(path, 'r') as f:
        qpos = f['observations/qpos'][()]
        action = f['action'][()]
        ts = f['timestamp'][()]
        cam = pol.cfg['camera']
        imgs = f[f'observations/images/{cam}']
        ens = TemporalEnsembler(pol.chunk_size, k=k)
        pred = np.zeros_like(action)
        for t in range(len(qpos)):
            pred[t] = ens.add_and_get(pol.predict(qpos[t], imgs[t]))
    return qpos, action, pred, ts


def plot_episode(name, action, pred, ts, out_png, metrics):
    fig, axes = plt.subplots(7, 1, figsize=(10, 13), sharex=True)
    for j, ax in enumerate(axes):
        if j < 6:
            ax.plot(ts, np.degrees(action[:, j]), label='실제 시연', color='#1f77b4', lw=2)
            ax.plot(ts, np.degrees(pred[:, j]), label='모델 예측', color='#ff7f0e', lw=1.5, ls='--')
            ax.set_ylabel(f'{JOINTS[j]}\n(°)', fontsize=9)
            lo, hi = ax.get_ylim()
            if hi - lo < 2.0:                      # 거의 안 움직인 관절은 확대해서 과장되지 않게 최소 ±1°
                c = (hi + lo) / 2
                ax.set_ylim(c - 1.0, c + 1.0)
            ax.set_title(f'오차 평균 {metrics[f"{JOINTS[j]}_mae_deg"]:.2f}°  최대 {metrics[f"{JOINTS[j]}_max_deg"]:.2f}°',
                         fontsize=8, loc='right')
        else:
            ax.step(ts, action[:, j], label='실제 시연', color='#1f77b4', lw=2, where='post')
            ax.plot(ts, pred[:, j], label='모델 예측', color='#ff7f0e', lw=1.5, ls='--')
            ax.axhline(0.5, color='gray', lw=0.5, ls=':')
            ax.set_ylabel('gripper\n(0열림 1닫힘)', fontsize=9)
            ax.set_title(f'그리퍼 정확도 {metrics["gripper_acc"] * 100:.1f}%  '
                         f'잡기 타이밍 {metrics["close_dt"]:+.2f}s  놓기 타이밍 {metrics["open_dt"]:+.2f}s',
                         fontsize=8, loc='right')
        ax.grid(alpha=0.3)
    axes[0].legend(loc='upper left', fontsize=8)
    axes[-1].set_xlabel('시간 (s)')
    fig.suptitle(f'{name}: 실제 시연 vs 모델 예측', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=100)
    plt.close(fig)


def compute_metrics(action, pred, ts):
    m = {}
    err = np.abs(pred[:, :6] - action[:, :6])
    for j in range(6):
        m[f'{JOINTS[j]}_mae_deg'] = float(np.degrees(err[:, j].mean()))
        m[f'{JOINTS[j]}_max_deg'] = float(np.degrees(err[:, j].max()))
    m['joint_mae_deg'] = float(np.degrees(err.mean()))
    m['gripper_acc'] = float(((pred[:, 6] > 0.5) == (action[:, 6] > 0.5)).mean())
    gc, go = toggle_times(action[:, 6], ts)
    pc, po = toggle_times(pred[:, 6], ts)
    m['close_dt'] = float(pc[0] - gc[0]) if len(gc) and len(pc) else float('nan')   # +면 늦게 잡음
    m['open_dt'] = float(po[0] - go[0]) if len(go) and len(po) else float('nan')
    m['pred_close_count'] = int(len(pc))
    return m


def plot_summary(rows, out_png):
    names = [r['episode'] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={'width_ratios': [2, 1]})
    x = np.arange(6)
    w = 0.8 / len(rows)
    for i, r in enumerate(rows):
        ax1.bar(x + i * w - 0.4 + w / 2, [r[f'{j}_mae_deg'] for j in JOINTS[:6]], w, label=r['episode'])
    mean = [np.mean([r[f'{j}_mae_deg'] for r in rows]) for j in JOINTS[:6]]
    for xi, v in zip(x, mean):
        ax1.text(xi, v, f'{v:.1f}°', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax1.axhline(2, color='green', ls=':', lw=1); ax1.text(5.45, 2, ' 2°', color='green', va='center', fontsize=8)
    ax1.axhline(5, color='red', ls=':', lw=1); ax1.text(5.45, 5, ' 5°', color='red', va='center', fontsize=8)
    ax1.set_xticks(x); ax1.set_xticklabels(JOINTS[:6], fontsize=9)
    ax1.set_ylabel('평균 오차 (°)'); ax1.set_title('관절별 평균 오차 (검증 에피소드)')
    ax1.legend(fontsize=7, ncol=2); ax1.grid(axis='y', alpha=0.3)

    acc = [r['gripper_acc'] * 100 for r in rows]
    ax2.bar(range(len(rows)), acc, color='#2ca02c')
    for i, a in enumerate(acc):
        ax2.text(i, a, f'{a:.0f}%', ha='center', va='bottom', fontsize=9)
    ax2.set_xticks(range(len(rows))); ax2.set_xticklabels([n.replace('episode_', 'ep') for n in names], fontsize=8)
    ax2.set_ylim(0, 105); ax2.set_ylabel('그리퍼 열림/닫힘 일치율 (%)'); ax2.set_title('그리퍼 정확도')
    ax2.grid(axis='y', alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)


def set_korean_font():
    from matplotlib import font_manager
    for name in ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'Noto Sans CJK KR', 'Noto Sans KR']:
        if any(name == f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams['font.family'] = name
            break
    plt.rcParams['axes.unicode_minus'] = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--episodes', nargs='*', help='평가할 파일 이름 (기본: 학습 때 검증용으로 뺀 에피소드)')
    ap.add_argument('--k', type=float, default=0.01, help='temporal ensembling 가중치')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    set_korean_font()

    pol = ACTPolicy(a.ckpt)
    eps = a.episodes or pol.cfg['val_episodes']
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.ckpt)), 'eval')
    os.makedirs(out, exist_ok=True)
    print(f'모델: {a.ckpt} ({pol.device}), 평가 에피소드: {", ".join(eps)}')

    rows = []
    for name in eps:
        path = os.path.join(a.data_dir, name)
        qpos, action, pred, ts = eval_episode(pol, path, a.k)
        m = compute_metrics(action, pred, ts)
        stem = os.path.splitext(name)[0]
        plot_episode(stem, action, pred, ts, os.path.join(out, f'{stem}_joints.png'), m)
        rows.append({'episode': stem, **m})
        print(f'  {stem}: 관절 평균 오차 {m["joint_mae_deg"]:.2f}°, 그리퍼 {m["gripper_acc"] * 100:.1f}%, '
              f'잡기 {m["close_dt"]:+.2f}s, 놓기 {m["open_dt"]:+.2f}s')

    with open(os.path.join(out, 'metrics.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    plot_summary(rows, os.path.join(out, 'summary.png'))

    per_joint = {j: np.mean([r[f'{j}_mae_deg'] for r in rows]) for j in JOINTS[:6]}
    worst = max(per_joint, key=per_joint.get)
    wv = per_joint[worst]
    ga = np.mean([r['gripper_acc'] for r in rows]) * 100
    verdict = '매우 좋음' if wv < 2 else '괜찮음' if wv < 5 else '부족 (데이터 추가 또는 더 학습 필요)'
    print('\n관절별 평균 오차: ' + ', '.join(f'{j} {v:.2f}°' for j, v in per_joint.items()))
    print(f'가장 큰 오차: {worst} {wv:.2f}° → {verdict}')
    print(f'그리퍼 정확도 {ga:.1f}%')
    print(f'그래프: {out}')


if __name__ == '__main__':
    main()
