#!/usr/bin/env python3
"""
ACT 학습 (Windows / Linux, RTX 3050 4GB 기준 설정)

  conda activate act
  python train_act.py --data_dir C:\\act_data\\pick_place --out_dir C:\\act_runs\\run1

  # 먼저 짧게 돌아가는지 확인
  python train_act.py --data_dir C:\\act_data\\pick_place --out_dir C:\\act_runs\\test --steps 300 --val_every 100

  # 중간에 끊겼으면 이어서
  python train_act.py --data_dir C:\\act_data\\pick_place --out_dir C:\\act_runs\\run1 --resume

결과 (out_dir):
  policy_best.ckpt   검증 손실이 가장 낮았던 모델  ← 로봇에 쓸 파일
  policy_last.ckpt   마지막 모델 (이어서 학습용)
  loss.png           손실 그래프
  log.csv            손실 기록
"""
import argparse
import csv
import glob
import os
import random
import re
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import sys
# act_model.py는 ROS 패키지(pick_place/pick_place/)에 하나만 둠 — 같은 폴더에 없으면 거기서 찾음
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pick_place', 'pick_place'))
from act_model import ACT, act_loss


# ─────────────────────────────── 데이터 ───────────────────────────────
def list_episodes(data_dir):
    files = glob.glob(os.path.join(data_dir, 'episode_*.hdf5'))
    return sorted(files, key=lambda s: int(re.search(r'episode_(\d+)\.hdf5$', s).group(1)))


def compute_stats(files):
    qs, acts = [], []
    for f in files:
        with h5py.File(f, 'r') as h:
            qs.append(h['observations/qpos'][()])
            acts.append(h['action'][()])
    q = np.concatenate(qs).astype(np.float32)
    a = np.concatenate(acts).astype(np.float32)
    return {
        'qpos_mean': q.mean(0).tolist(), 'qpos_std': np.clip(q.std(0), 1e-2, None).tolist(),
        'action_mean': a.mean(0).tolist(), 'action_std': np.clip(a.std(0), 1e-2, None).tolist(),
    }


class EpisodeDataset(Dataset):
    """모든 (에피소드, 시점) 쌍을 샘플로 사용. h5 파일은 워커마다 처음 쓸 때 연다 (Windows spawn 대응)."""

    def __init__(self, files, chunk_size, stats, camera):
        self.files = files
        self.chunk = chunk_size
        self.cam = camera
        self.q_mean = np.array(stats['qpos_mean'], np.float32)
        self.q_std = np.array(stats['qpos_std'], np.float32)
        self.a_mean = np.array(stats['action_mean'], np.float32)
        self.a_std = np.array(stats['action_std'], np.float32)
        self.index = []
        for fi, f in enumerate(files):
            with h5py.File(f, 'r') as h:
                T = len(h['observations/qpos'])
            self.index += [(fi, t) for t in range(T)]
        self._h5 = {}

    def __getstate__(self):
        d = self.__dict__.copy()
        d['_h5'] = {}
        return d

    def __len__(self):
        return len(self.index)

    def _file(self, fi):
        if fi not in self._h5:
            self._h5[fi] = h5py.File(self.files[fi], 'r')
        return self._h5[fi]

    def __getitem__(self, i):
        fi, t = self.index[i]
        h = self._file(fi)
        qpos = h['observations/qpos'][t]
        act = h['action'][t:t + self.chunk]
        img = h[f'observations/images/{self.cam}'][t]                # (H,W,3) uint8 RGB
        n = len(act)
        is_pad = np.zeros(self.chunk, dtype=bool)
        if n < self.chunk:
            act = np.concatenate([act, np.repeat(act[-1:], self.chunk - n, axis=0)])
            is_pad[n:] = True
        qpos = (qpos - self.q_mean) / self.q_std
        act = (act - self.a_mean) / self.a_std
        return (torch.from_numpy(qpos.astype(np.float32)),
                torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1),   # uint8 (3,H,W)
                torch.from_numpy(act.astype(np.float32)),
                torch.from_numpy(is_pad))


def gpu_augment(img, max_shift=8):
    """img: (B,3,H,W) 0~1. 작은 평행이동 + 밝기/대비 변화 (조명·카메라 미세 흔들림에 강하게)"""
    B, C, H, W = img.shape
    p = max_shift
    padded = F.pad(img, (p, p, p, p), mode='replicate')
    out = torch.empty_like(img)
    xs = torch.randint(0, 2 * p + 1, (B,)).tolist()
    ys = torch.randint(0, 2 * p + 1, (B,)).tolist()
    for i in range(B):
        out[i] = padded[i, :, ys[i]:ys[i] + H, xs[i]:xs[i] + W]
    bright = 1 + (torch.rand(B, 1, 1, 1, device=img.device) - 0.5) * 0.4
    contrast = 1 + (torch.rand(B, 1, 1, 1, device=img.device) - 0.5) * 0.4
    mean = out.mean(dim=(1, 2, 3), keepdim=True)
    return (((out - mean) * contrast + mean) * bright).clamp(0, 1)


def infinite(loader):
    while True:
        for b in loader:
            yield b


# ─────────────────────────────── 학습 ───────────────────────────────
def save_ckpt(path, model, opt, scaler, step, best_val, cfg, stats, with_optimizer=True):
    ck = {'model': model.state_dict(), 'step': step, 'best_val': best_val, 'config': cfg, 'stats': stats}
    if with_optimizer:   # 이어서 학습용 (용량 3배). 로봇용 best 파일에는 불필요
        ck['optimizer'] = opt.state_dict()
        ck['scaler'] = scaler.state_dict()
    torch.save(ck, path + '.tmp')
    os.replace(path + '.tmp', path)


def plot_log(log_path, png_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows = list(csv.DictReader(open(log_path, encoding='utf-8')))
    tr = [(int(r['step']), float(r['train_l1'])) for r in rows if r['train_l1']]
    va = [(int(r['step']), float(r['val_l1'])) for r in rows if r['val_l1']]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if tr:
        ax.plot(*zip(*tr), label='train L1', alpha=0.7)
    if va:
        ax.plot(*zip(*va), 'o-', label='val L1')
    ax.set_xlabel('step'); ax.set_ylabel('L1 (normalized)'); ax.set_yscale('log')
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(png_path, dpi=110); plt.close(fig)


@torch.no_grad()
def evaluate(model, loader, device, kl_weight, use_amp, max_batches=50):
    model.eval()
    tot_l1, tot_kl, n = 0.0, 0.0, 0
    for i, (q, img, act, pad) in enumerate(loader):
        if i >= max_batches:
            break
        q, act, pad = q.to(device), act.to(device), pad.to(device)
        img = img.to(device).float() / 255.0
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            pred, mu, logvar = model(q, img, act, pad)
        _, l1, kl = act_loss(pred.float(), act, pad, mu.float(), logvar.float(), kl_weight)
        tot_l1 += l1.item(); tot_kl += kl.item(); n += 1
    model.train()
    return tot_l1 / max(n, 1), tot_kl / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--steps', type=int, default=50000)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--chunk', type=int, default=100, help='한 번에 예측할 스텝 수 (30Hz면 100=3.3초)')
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--lr_backbone', type=float, default=1e-5)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--kl', type=float, default=10.0)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--val_ratio', type=float, default=0.1)
    ap.add_argument('--val_every', type=int, default=1000)
    ap.add_argument('--log_every', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--no_aug', action='store_true')
    ap.add_argument('--no_amp', action='store_true')
    ap.add_argument('--no_pretrained', action='store_true', help='ImageNet 가중치 다운로드 안 함 (테스트용)')
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda' and not a.no_amp
    print(f'장치: {device}' + (f' ({torch.cuda.get_device_name(0)})' if device.type == 'cuda' else ' ← GPU 안 잡힘!'))

    # 데이터
    files = list_episodes(a.data_dir)
    if len(files) < 2:
        raise SystemExit(f'에피소드가 부족합니다: {a.data_dir} ({len(files)}개)')
    with h5py.File(files[0], 'r') as h:
        cam = list(h['observations/images'].keys())[0]
        _, H, W, _ = h[f'observations/images/{cam}'].shape
        state_dim = h['observations/qpos'].shape[1]
        action_dim = h['action'].shape[1]
        fps = float(h.attrs.get('fps', 30.0))
    rng = random.Random(a.seed)
    shuffled = files[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(files) * a.val_ratio)))
    val_files, train_files = sorted(shuffled[:n_val]), sorted(shuffled[n_val:])

    ckpt_last = os.path.join(a.out_dir, 'policy_last.ckpt')
    ckpt_best = os.path.join(a.out_dir, 'policy_best.ckpt')
    resume_ck = torch.load(ckpt_last, map_location='cpu', weights_only=False) if a.resume and os.path.exists(ckpt_last) else None
    stats = resume_ck['stats'] if resume_ck else compute_stats(train_files)

    train_ds = EpisodeDataset(train_files, a.chunk, stats, cam)
    val_ds = EpisodeDataset(val_files, a.chunk, stats, cam)
    kw = dict(num_workers=a.workers, pin_memory=device.type == 'cuda', persistent_workers=a.workers > 0)
    train_dl = DataLoader(train_ds, batch_size=a.batch, shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, batch_size=a.batch, shuffle=True, **kw)
    print(f'데이터: 학습 {len(train_files)}개 에피소드({len(train_ds)} 샘플), 검증 {len(val_files)}개({len(val_ds)} 샘플)')
    print(f'이미지 {W}x{H} ({cam}), 상태 {state_dim}차원, 동작 {action_dim}차원, chunk {a.chunk} ({a.chunk / fps:.1f}s)')

    # 모델
    model_cfg = dict(state_dim=state_dim, action_dim=action_dim, chunk_size=a.chunk)
    cfg = {'model': model_cfg, 'image_hw': [H, W], 'camera': cam, 'fps': fps, 'kl_weight': a.kl,
           'train_episodes': [os.path.basename(f) for f in train_files],
           'val_episodes': [os.path.basename(f) for f in val_files]}
    model = ACT(**model_cfg, pretrained=not a.no_pretrained and resume_ck is None).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'모델 파라미터: {n_params / 1e6:.1f}M')

    bb = [p for n, p in model.named_parameters() if n.startswith('backbone') and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if not n.startswith('backbone') and p.requires_grad]
    opt = torch.optim.AdamW([{'params': rest, 'lr': a.lr}, {'params': bb, 'lr': a.lr_backbone}],
                            weight_decay=a.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    step, best_val = 0, float('inf')
    if resume_ck:
        model.load_state_dict(resume_ck['model'])
        opt.load_state_dict(resume_ck['optimizer'])
        scaler.load_state_dict(resume_ck['scaler'])
        step, best_val = resume_ck['step'], resume_ck['best_val']
        print(f'이어서 학습: step {step}부터 (best val {best_val:.4f})')
    elif a.resume:
        print('[WARN] policy_last.ckpt가 없어서 처음부터 학습합니다')

    log_path = os.path.join(a.out_dir, 'log.csv')
    new_log = not (resume_ck and os.path.exists(log_path))
    log_f = open(log_path, 'w' if new_log else 'a', newline='', encoding='utf-8')
    log = csv.writer(log_f)
    if new_log:
        log.writerow(['step', 'train_loss', 'train_l1', 'train_kl', 'val_l1', 'val_kl', 'sec'])

    model.train()
    it = infinite(train_dl)
    t0 = time.time()
    t_win, acc = time.time(), np.zeros(3)
    n_acc = 0
    try:
        while step < a.steps:
            q, img, act, pad = next(it)
            q, act, pad = q.to(device, non_blocking=True), act.to(device, non_blocking=True), pad.to(device)
            img = img.to(device, non_blocking=True).float() / 255.0
            if not a.no_aug:
                img = gpu_augment(img)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                pred, mu, logvar = model(q, img, act, pad)
            loss, l1, kl = act_loss(pred.float(), act, pad, mu.float(), logvar.float(), a.kl)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            acc += [loss.item(), l1.item(), kl.item()]
            n_acc += 1

            if step % a.log_every == 0 or step == a.steps:
                m = acc / n_acc
                ips = n_acc / (time.time() - t_win)
                eta = (a.steps - step) / max(ips, 1e-6)
                mem = f' | GPU {torch.cuda.max_memory_allocated() / 2**30:.2f}GB' if device.type == 'cuda' else ''
                print(f'step {step:6d}/{a.steps} | loss {m[0]:.4f} (L1 {m[1]:.4f}, KL {m[2]:.4f}) | '
                      f'{ips:.1f} it/s | 남은 시간 {eta / 3600:.1f}h{mem}', flush=True)
                log.writerow([step, f'{m[0]:.5f}', f'{m[1]:.5f}', f'{m[2]:.5f}', '', '', f'{time.time() - t0:.0f}'])
                acc[:] = 0; n_acc = 0; t_win = time.time()

            if step % a.val_every == 0 or step == a.steps:
                vl1, vkl = evaluate(model, val_dl, device, a.kl, use_amp)
                tag = ''
                if vl1 < best_val:
                    best_val = vl1
                    save_ckpt(ckpt_best, model, opt, scaler, step, best_val, cfg, stats, with_optimizer=False)
                    tag = '  ★ best 저장'
                save_ckpt(ckpt_last, model, opt, scaler, step, best_val, cfg, stats)
                print(f'  [val] step {step}: L1 {vl1:.4f}, KL {vkl:.4f}{tag}', flush=True)
                log.writerow([step, '', '', '', f'{vl1:.5f}', f'{vkl:.5f}', f'{time.time() - t0:.0f}'])
                log_f.flush()
                plot_log(log_path, os.path.join(a.out_dir, 'loss.png'))
    except KeyboardInterrupt:
        print('\n중단됨 → 현재 상태 저장 (--resume으로 이어서 가능)')
        save_ckpt(ckpt_last, model, opt, scaler, step, best_val, cfg, stats)
    finally:
        log_f.close()
        plot_log(log_path, os.path.join(a.out_dir, 'loss.png'))

    print(f'\n완료: {step} steps, {(time.time() - t0) / 60:.0f}분, best val L1 {best_val:.4f}')
    print(f'로봇에 쓸 파일: {ckpt_best}')


if __name__ == '__main__':
    main()
