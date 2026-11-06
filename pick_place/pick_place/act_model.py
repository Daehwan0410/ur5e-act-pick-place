"""
ACT (Action Chunking with Transformers) 모델 - 학습(Windows)과 추론(Ubuntu/ROS2) 공용

구조 (원본 ACT, Zhao et al. 2023 기준):
  - CVAE 인코더: [CLS, qpos, action chunk] → 잠재변수 z (학습 때만 사용, 추론 때 z=0)
  - ResNet18(ImageNet 사전학습, FrozenBN) → 이미지 특징맵
  - Transformer 인코더: [z 토큰, qpos 토큰, 이미지 토큰들]
  - Transformer 디코더: chunk_size개 쿼리 → 앞으로의 동작 chunk_size 스텝 예측
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.ops.misc import FrozenBatchNorm2d

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sinusoid_1d(n, d):
    pos = torch.arange(n, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def sinusoid_2d(h, w, d, device):
    """DETR 방식 2D sine 위치 임베딩 → (h*w, d)"""
    npf = d // 2
    y = torch.arange(1, h + 1, device=device, dtype=torch.float32)[:, None].expand(h, w)
    x = torch.arange(1, w + 1, device=device, dtype=torch.float32)[None, :].expand(h, w)
    y = y / (h + 1e-6) * 2 * math.pi
    x = x / (w + 1e-6) * 2 * math.pi
    dim_t = torch.arange(npf, device=device, dtype=torch.float32)
    dim_t = 10000 ** (2 * (dim_t // 2) / npf)
    px = x[..., None] / dim_t
    py = y[..., None] / dim_t
    px = torch.stack((px[..., 0::2].sin(), px[..., 1::2].cos()), dim=3).flatten(2)
    py = torch.stack((py[..., 0::2].sin(), py[..., 1::2].cos()), dim=3).flatten(2)
    return torch.cat((py, px), dim=2).flatten(0, 1)


def _enc_layer(d, nh, ff, dp):
    return nn.TransformerEncoderLayer(d, nh, ff, dp, activation='relu', batch_first=True)


class ACT(nn.Module):
    def __init__(self, state_dim=7, action_dim=7, chunk_size=100, hidden=512, nheads=8, ff=3200,
                 enc_layers=4, dec_layers=1, vae_enc_layers=4, latent_dim=32, dropout=0.1, pretrained=True):
        super().__init__()
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        self.hidden = hidden

        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        rn = torchvision.models.resnet18(weights=weights, norm_layer=FrozenBatchNorm2d)
        self.backbone = nn.Sequential(*list(rn.children())[:-2])          # (B,512,H/32,W/32)
        self.input_proj = nn.Conv2d(512, hidden, kernel_size=1)
        self.register_buffer('img_mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('img_std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        # CVAE 인코더
        self.cls_embed = nn.Embedding(1, hidden)
        self.vae_qpos_proj = nn.Linear(state_dim, hidden)
        self.vae_action_proj = nn.Linear(action_dim, hidden)
        self.vae_encoder = nn.TransformerEncoder(_enc_layer(hidden, nheads, ff, dropout), vae_enc_layers,
                                                 enable_nested_tensor=False)
        self.latent_proj = nn.Linear(hidden, latent_dim * 2)
        self.register_buffer('vae_pos', sinusoid_1d(chunk_size + 2, hidden)[None])

        # 메인 트랜스포머
        self.latent_out_proj = nn.Linear(latent_dim, hidden)
        self.qpos_proj = nn.Linear(state_dim, hidden)
        self.extra_pos = nn.Embedding(2, hidden)                         # [z, qpos] 토큰 위치
        self.encoder = nn.TransformerEncoder(_enc_layer(hidden, nheads, ff, dropout), enc_layers,
                                             enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden, nheads, ff, dropout, activation='relu', batch_first=True), dec_layers)
        self.query_embed = nn.Embedding(chunk_size, hidden)
        self.action_head = nn.Linear(hidden, action_dim)

    def forward(self, qpos, image, actions=None, is_pad=None):
        """
        qpos:    (B, state_dim)       정규화된 값
        image:   (B, 3, H, W)         0~1 RGB
        actions: (B, K, action_dim)   정규화된 값 (학습 때만)
        is_pad:  (B, K) bool          에피소드 끝 넘어간 부분 True
        """
        B = qpos.shape[0]
        if actions is not None:
            tok = torch.cat([self.cls_embed.weight[None].expand(B, -1, -1),
                             self.vae_qpos_proj(qpos)[:, None],
                             self.vae_action_proj(actions)], dim=1) + self.vae_pos
            pad = torch.cat([torch.zeros(B, 2, dtype=torch.bool, device=qpos.device), is_pad], dim=1)
            out = self.vae_encoder(tok, src_key_padding_mask=pad)[:, 0]
            mu, logvar = self.latent_proj(out).chunk(2, dim=-1)
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        else:
            mu = logvar = None
            z = torch.zeros(B, self.latent_dim, device=qpos.device, dtype=qpos.dtype)

        img = (image - self.img_mean) / self.img_std
        feat = self.input_proj(self.backbone(img))                       # (B, hidden, h, w)
        h, w = feat.shape[-2:]
        img_tok = feat.flatten(2).transpose(1, 2) + sinusoid_2d(h, w, self.hidden, feat.device).to(feat.dtype)
        z_tok = (self.latent_out_proj(z) + self.extra_pos.weight[0])[:, None]
        q_tok = (self.qpos_proj(qpos) + self.extra_pos.weight[1])[:, None]
        memory = self.encoder(torch.cat([z_tok, q_tok, img_tok], dim=1))

        tgt = self.query_embed.weight[None].expand(B, -1, -1)
        hs = self.decoder(tgt, memory)
        return self.action_head(hs), mu, logvar


def act_loss(pred, actions, is_pad, mu, logvar, kl_weight):
    valid = (~is_pad).unsqueeze(-1).to(pred.dtype)
    l1 = (torch.abs(pred - actions) * valid).sum() / (valid.sum() * pred.shape[-1]).clamp(min=1)
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
    return l1 + kl_weight * kl, l1, kl


# ───────────────────────── 추론용 래퍼 (Ubuntu ROS2 노드에서 사용) ─────────────────────────
class ACTPolicy:
    """
    pol = ACTPolicy('policy_best.ckpt', device='cuda')
    actions = pol.predict(qpos7, rgb_image)   # (chunk_size, 7) 실제 단위(rad, 그리퍼 0/1)
    """

    def __init__(self, ckpt_path, device=None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        self.cfg = ck['config']
        self.model = ACT(**self.cfg['model'], pretrained=False)
        self.model.load_state_dict(ck['model'])
        self.model.to(self.device).eval()
        s = ck['stats']
        self.qpos_mean = torch.tensor(s['qpos_mean'], device=self.device)
        self.qpos_std = torch.tensor(s['qpos_std'], device=self.device)
        self.act_mean = np.asarray(s['action_mean'], dtype=np.float32)
        self.act_std = np.asarray(s['action_std'], dtype=np.float32)
        self.img_h, self.img_w = self.cfg['image_hw']
        self.chunk_size = self.cfg['model']['chunk_size']

    @torch.no_grad()
    def predict(self, qpos, image_rgb):
        import cv2
        if image_rgb.shape[:2] != (self.img_h, self.img_w):
            image_rgb = cv2.resize(image_rgb, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
        q = (torch.tensor(np.asarray(qpos, dtype=np.float32), device=self.device) - self.qpos_mean) / self.qpos_std
        img = torch.from_numpy(np.ascontiguousarray(image_rgb)).to(self.device).permute(2, 0, 1).float() / 255.0
        pred, _, _ = self.model(q[None], img[None])
        return pred[0].float().cpu().numpy() * self.act_std + self.act_mean


class TemporalEnsembler:
    """원본 ACT의 temporal ensembling: 매 스텝 추론한 chunk들을 지수 가중 평균 → 부드러운 동작"""

    def __init__(self, chunk_size, k=0.01):
        from collections import deque
        self.k = k
        self.chunk = chunk_size
        self.history = deque(maxlen=chunk_size)   # (시작 스텝, chunk) - 최근 chunk_size개만 유효
        self.t = 0

    def reset(self):
        self.history.clear()
        self.t = 0

    def add_and_get(self, chunk_actions):
        self.history.append((self.t, np.asarray(chunk_actions)))
        # 오래된 예측 → 최신 예측 순서, 가중치 exp(-k*i) (원본 ACT와 동일: 오래된 것에 더 큰 가중치)
        acts = np.stack([c[self.t - s] for s, c in self.history if self.t - s < self.chunk])
        w = np.exp(-self.k * np.arange(len(acts)))
        w = w / w.sum()
        self.t += 1
        return (acts * w[:, None]).sum(0)
