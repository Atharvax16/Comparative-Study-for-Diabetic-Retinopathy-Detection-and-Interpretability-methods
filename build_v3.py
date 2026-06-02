"""
Build Thesis_optimized_final_version3.ipynb from version2.

Adds Phase 4 Diffusion #3: a PATHOLOGY-PRESERVING conditional DDPM, kept
ALONGSIDE the existing vanilla DDPM (Step 4) so the recovery table shows an
ablation: does pathology-conditioning help restoration vs. a plain DDPM?

What this adds (all resume-guarded, trains exactly once):
  Step 7 : Pathology-preserving residual conditional DDPM
             - self-attention + FiLM U-Net (concat degraded condition, 6ch)
             - pathology-aware perceptual loss in the feature space of OUR
               trained EfficientNetV2-S DR classifier (build_model_v3), NOT
               VGG/ImageNet
             - input-fidelity L1 loss (anchors reconstruction to evidence)
             - sampling-time anti-hallucination clamp (blend x0 -> degraded)
             - new checkpoint ddpm_pathology_v1.pt, enhancer key 'ddpm_path'
  Step 8 : Hallucination detection report (grade-change / pixel-deviation /
           risk score) saved to P4/metrics — CSV only, no new plots.

The existing recovery-eval cell iterates ENHANCERS dynamically, so adding
'ddpm_path' to that tuple is enough for it to appear in recovery_accuracy.csv
and every downstream summary table/plot the v2 notebook already produces.

Why these conventions (vs. the original Phase-4 prompt, which assumed a flat
`efficientnet_b3_v2_best.pt` and ImageNet-normalised [-1,1] data):
  - classifier ckpt is CKPT_DIR_V3/efficientnet_b3_v3_best.pt, weights under
    ckpt['state_dict'], loaded via build_model_v3 (a MultiScaleClassifier
    wrapping a features_only EfficientNetV2-S backbone) -> perceptual loss
    hooks model.backbone(x), not raw timm .conv_stem/.blocks (which 404 here)
  - DDPM data is in [-1,1] via Normalize([0.5]*3,[0.5]*3) (the only norm that
    makes the [-1,1] clamp valid); degradation is generated ON THE FLY with the
    Phase-1 DEGRADERS, matching the existing _DDPMDataset
  - checkpoint name bumped to ddpm_pathology_v1.pt (ddpm_fundus_v1.pt is taken)
"""
import json
from pathlib import Path

SRC = Path(r"C:\Dissertation\Thesis_optimized_final_version2.ipynb")
DST = Path(r"C:\Dissertation\Thesis_optimized_final_version3.ipynb")


def md_cell(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": src.splitlines(keepends=True)}


def code_cell(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


def find_cell(cells: list, marker: str) -> int:
    """Index of the first cell whose joined source contains `marker`."""
    for i, c in enumerate(cells):
        if marker in "".join(c.get("source", [])):
            return i
    raise RuntimeError(f"marker not found: {marker!r}")


def patch_cell(cells: list, idx: int, old: str, new: str) -> None:
    """Substring replace inside a cell's source; resets code outputs."""
    c = cells[idx]
    s = "".join(c.get("source", []))
    if old not in s:
        raise RuntimeError(f"substring not found in cell {idx}: {old!r}")
    c["source"] = s.replace(old, new).splitlines(keepends=True)
    if c.get("cell_type") == "code":
        c["execution_count"] = None
        c["outputs"] = []


# ---------------------------------------------------------------------------
# Step 7 — pathology-preserving conditional DDPM
# ---------------------------------------------------------------------------
DDPM_PATH_MD = '''## V3 — Step 7 — Pathology-preserving conditional DDPM (Phase 4 Diffusion #3)

Upgrades the vanilla DDPM (Step 4) with three explicit pathology-preservation
mechanisms, so the restorer recovers image quality **without inventing or
erasing lesions** (microaneurysms, haemorrhages, exudates):

1. **Pathology-aware perceptual loss** — computed in the feature space of *our*
   trained EfficientNetV2-S DR classifier (`build_model_v3('efficientnet_b3')`),
   not VGG/ImageNet. The model is penalised for changing the features the DR
   classifier treats as diagnostic.
2. **Input-fidelity loss** — an L1 term anchoring the reconstructed clean image
   to the degraded observation.
3. **Sampling-time anti-hallucination clamp** — at every reverse step the
   predicted `x0` is blended toward the degraded input (`clamp_strength`),
   capping how far the sampler may drift from the evidence.

The forward/denoising objective is still standard DDPM (Gaussian noise,
epsilon-prediction) — identical to Step 4 — so vanilla-vs-pathology is a fair
ablation. Kept as a **separate** enhancer `'ddpm_path'` ("DDPM (Pathology)")
with its own checkpoint `ddpm_pathology_v1.pt`; the recovery table now reports
both DDPMs side by side. Training is resume-guarded (runs once).
'''

DDPM_PATH_IMPL = r'''# === V3 PATCH (Step 7): pathology-preserving conditional DDPM ===
# Phase 4 Diffusion #3. Vanilla DDPM objective (Step 4) + three pathology
# preservation mechanisms: classifier-feature perceptual loss, input-fidelity
# L1, and a sampling-time anti-hallucination clamp. Kept alongside 'ddpm'.
import math, random
import torch, torch.nn as nn, torch.nn.functional as F

DDPM_PATH_CKPT = CHECKPOINTS_DIR / 'ddpm_pathology_v1.pt'
DDPM_PATH_SIZE = 256
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class _PathSinPE(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000.0) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class _PathAttn(nn.Module):
    """Self-attention at the bottleneck for global context."""
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv  = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.scale = ch ** -0.5
    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, C, H * W).unbind(1)
        attn = torch.softmax((q.transpose(-1, -2) @ k) * self.scale, dim=-1)
        out  = (v @ attn.transpose(-1, -2)).reshape(B, C, H, W)
        return x + self.proj(out)


class _PathUNet(nn.Module):
    """Conditional U-Net: concat(noisy_clean, degraded)=6ch -> predicted noise 3ch.
    FiLM modulation on the timestep + self-attention at the bottleneck."""
    def __init__(self, in_ch=6, out_ch=3, base_ch=64, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            _PathSinPE(time_dim),
            nn.Linear(time_dim, time_dim), nn.GELU(),
            nn.Linear(time_dim, time_dim))
        def blk(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU(),
                nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU())
        self.enc1 = blk(in_ch, base_ch)
        self.enc2 = blk(base_ch, base_ch * 2)
        self.enc3 = blk(base_ch * 2, base_ch * 4)
        self.mid  = blk(base_ch * 4, base_ch * 4)
        self.mid_attn = _PathAttn(base_ch * 4)
        self.film = nn.Linear(time_dim, base_ch * 4 * 2)
        self.dec3 = blk(base_ch * 8, base_ch * 2)
        self.dec2 = blk(base_ch * 4, base_ch)
        self.dec1 = blk(base_ch * 2, base_ch)
        self.final = nn.Conv2d(base_ch, out_ch, 1)
        self.pool = nn.MaxPool2d(2)
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self, x, t):
        te = self.time_mlp(t)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        m  = self.mid(self.pool(e3))
        scale, shift = self.film(te).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        m  = self.mid_attn(m * (1 + scale) + shift)
        d3 = self.dec3(torch.cat([self.up(m),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        return self.final(d1)


class PathologyPerceptualLoss(nn.Module):
    """Perceptual loss in the feature space of OUR trained DR classifier
    (MultiScaleClassifier wrapping a features_only EfficientNetV2-S).
    Inputs are in DDPM space ([-1,1]); converted to the classifier's
    ImageNet-normalised 384px space internally."""
    def __init__(self, device=DEVICE):
        super().__init__()
        clf = build_model_v3('efficientnet_b3', pretrained=False).to(device)
        ck  = CKPT_DIR_V3 / 'efficientnet_b3_v3_best.pt'
        sd  = torch.load(ck, map_location=device, weights_only=False)
        clf.load_state_dict(sd['state_dict'] if 'state_dict' in sd else sd, strict=False)
        clf.eval()
        self.backbone = clf.backbone               # features_only -> list of maps
        self.n_stages = min(3, clf.n_stages)
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer('mean', _IMAGENET_MEAN.to(device))
        self.register_buffer('std',  _IMAGENET_STD.to(device))
    def _feats(self, x):
        x01 = (x * 0.5 + 0.5).clamp(0, 1)
        x01 = F.interpolate(x01, size=IMAGE_SIZE_V3, mode='bilinear', align_corners=False)
        x01 = (x01 - self.mean) / self.std
        return list(self.backbone(x01))[-self.n_stages:]
    def forward(self, restored, clean):
        fr, fc = self._feats(restored), self._feats(clean)
        return sum(F.l1_loss(a, b) for a, b in zip(fr, fc)) / len(fr)


class PathologyDDPM:
    """Standard DDPM (Ho et al. 2020) conditioned on the degraded image, with a
    pathology-preservation clamp during sampling."""
    def __init__(self, model, T=1000, beta_start=1e-4, beta_end=0.02, device=DEVICE):
        self.model = model; self.T = T; self.device = device
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.acp = torch.cumprod(self.alphas, dim=0)
        self.sqrt_acp = torch.sqrt(self.acp)
        self.sqrt_omacp = torch.sqrt(1 - self.acp)
    def q_sample(self, x0, t, noise):
        a = self.sqrt_acp[t].view(-1, 1, 1, 1)
        b = self.sqrt_omacp[t].view(-1, 1, 1, 1)
        return a * x0 + b * noise
    def _x0_from_eps(self, xt, t, eps):
        a = self.sqrt_acp[t].view(-1, 1, 1, 1)
        b = self.sqrt_omacp[t].view(-1, 1, 1, 1)
        return (xt - b * eps) / a
    @torch.no_grad()
    def restore(self, degraded, n_steps=50, clamp_strength=0.15):
        self.model.eval()
        B = degraded.size(0)
        x = torch.randn(B, 3, degraded.size(2), degraded.size(3), device=self.device)
        step = max(self.T // n_steps, 1)
        ts = list(range(self.T - 1, 0, -step))
        for i, tv in enumerate(ts):
            t = torch.full((B,), tv, device=self.device, dtype=torch.long)
            eps = self.model(torch.cat([x, degraded], dim=1), t)
            x0 = self._x0_from_eps(x, t, eps)
            # ---- pathology-preservation clamp: anchor x0 toward evidence ----
            x0 = ((1 - clamp_strength) * x0 + clamp_strength * degraded).clamp(-1, 1)
            tv_next = ts[i + 1] if i + 1 < len(ts) else 0
            if tv_next > 0:
                a_next = self.acp[tv_next]
                x = torch.sqrt(a_next) * x0 + torch.sqrt(1 - a_next) * torch.randn_like(x)
            else:
                x = x0
        return x.clamp(-1, 1)


class _PathDataset(Dataset):
    """(degraded, clean) pairs; degradation generated on the fly with the
    Phase-1 operators, matching the vanilla DDPM dataset. Data in [-1,1]."""
    def __init__(self, df, n_per=4):
        self.df = df.reset_index(drop=True); self.n_per = n_per
        self.tfm = T.Compose([T.Resize((DDPM_PATH_SIZE, DDPM_PATH_SIZE)),
                              T.ToTensor(),
                              T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    def __len__(self): return len(self.df) * self.n_per
    def __getitem__(self, i):
        row = self.df.iloc[i // self.n_per]
        try:
            img = Image.open(resolve_image(row['id_code'])).convert('RGB')
        except FileNotFoundError:
            img = Image.new('RGB', (DDPM_PATH_SIZE, DDPM_PATH_SIZE))
        img = img.resize((DDPM_PATH_SIZE, DDPM_PATH_SIZE), Image.BILINEAR)
        kind  = random.choice(['blur', 'exposure', 'noise'])
        level = random.choice(['low', 'mid', 'high'])
        deg   = DEGRADERS[kind](img, DEGRADATION_PARAMS[kind][level])
        return self.tfm(deg), self.tfm(img)


def train_ddpm_pathology(epochs=15, batch_size=8, lr=1e-4, T=1000,
                         lambda_perc=0.1, lambda_fid=0.05):
    """Resume-guarded. Loss = MSE(eps) + lambda_perc * PathologyPerceptual(x0, clean)
                                       + lambda_fid  * L1(x0, degraded)."""
    if DDPM_PATH_CKPT.exists():
        print(f"[skip] {DDPM_PATH_CKPT.name} already on Drive — not retraining.")
        return
    print(f"Training pathology DDPM ({epochs} ep, T={T}, bs={batch_size}, lr={lr}, "
          f"lambda_perc={lambda_perc}, lambda_fid={lambda_fid})...")
    df = pd.read_csv(PRISTINE_CSV)
    dl = DataLoader(_PathDataset(df, n_per=4), batch_size=batch_size, shuffle=True,
                    num_workers=2, pin_memory=True, drop_last=True)
    net  = _PathUNet().to(DEVICE)
    ddpm = PathologyDDPM(net, T=T, device=DEVICE)
    try:
        perc = PathologyPerceptualLoss(DEVICE)
        print("  pathology perceptual loss: ENABLED (EfficientNetV2-S DR features)")
    except Exception as e:
        perc = None
        print(f"  WARNING: perceptual loss disabled ({e}); using MSE + fidelity only")
    opt   = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = GradScaler(enabled=True)
    for ep in range(epochs):
        net.train()
        run = dict(tot=0.0, mse=0.0, perc=0.0, fid=0.0, n=0)
        pbar = tqdm(dl, desc=f"path-ddpm ep {ep+1}/{epochs}", leave=False)
        for deg, clean in pbar:
            deg   = deg.to(DEVICE, non_blocking=True)
            clean = clean.to(DEVICE, non_blocking=True)
            B = clean.size(0)
            t = torch.randint(0, T, (B,), device=DEVICE)
            noise = torch.randn_like(clean)
            noisy = ddpm.q_sample(clean, t, noise)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                eps = net(torch.cat([noisy, deg], dim=1), t)
                l_mse = F.mse_loss(eps, noise)
                x0 = ddpm._x0_from_eps(noisy, t, eps).clamp(-1, 1)
                l_perc = perc(x0, clean) if (perc is not None and lambda_perc > 0) \
                         else torch.zeros((), device=DEVICE)
                l_fid  = F.l1_loss(x0, deg) if lambda_fid > 0 \
                         else torch.zeros((), device=DEVICE)
                loss = l_mse + lambda_perc * l_perc + lambda_fid * l_fid
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            run['tot']  += loss.item() * B
            run['mse']  += l_mse.item() * B
            run['perc'] += float(l_perc) * B
            run['fid']  += float(l_fid) * B
            run['n']    += B
            pbar.set_postfix(tot=f"{run['tot']/run['n']:.3f}",
                             mse=f"{run['mse']/run['n']:.3f}")
        sched.step()
        n = run['n']
        print(f"  ep {ep+1}: tot={run['tot']/n:.4f} mse={run['mse']/n:.4f} "
              f"perc={run['perc']/n:.4f} fid={run['fid']/n:.4f}")
    torch.save({'state_dict': net.state_dict(), 'T': T}, DDPM_PATH_CKPT)
    print(f"Saved -> {DDPM_PATH_CKPT}")


train_ddpm_pathology(epochs=15, batch_size=8, lr=1e-4, T=1000)


# ---- Inference ----
_path_net  = None
_path_ddpm = None
def _load_path_ddpm(T=1000):
    global _path_net, _path_ddpm
    if _path_ddpm is not None:
        return _path_ddpm
    _path_net = _PathUNet().to(DEVICE).eval()
    sd = torch.load(DDPM_PATH_CKPT, map_location=DEVICE, weights_only=False)
    _path_net.load_state_dict(sd['state_dict'])
    _path_ddpm = PathologyDDPM(_path_net, T=sd.get('T', T), device=DEVICE)
    return _path_ddpm


@torch.no_grad()
def enhance_ddpm_path(img_pil, n_steps=50, clamp_strength=0.15):
    """Restore via the pathology-preserving DDPM. Same PIL->PIL signature as
    enhance_ddpm so the build_enhanced registry call is a drop-in."""
    ddpm = _load_path_ddpm()
    orig = img_pil.size
    src  = img_pil.convert('RGB').resize((DDPM_PATH_SIZE, DDPM_PATH_SIZE), Image.BILINEAR)
    tfm  = T.Compose([T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    x = tfm(src).unsqueeze(0).to(DEVICE)
    y = ddpm.restore(x, n_steps=n_steps, clamp_strength=clamp_strength)
    arr = (y.squeeze().permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
    arr = (arr * 255).clip(0, 255).astype('uint8')
    return Image.fromarray(arr).resize(orig)


print("Pathology-preserving DDPM restorer ready.")
'''


# ---------------------------------------------------------------------------
# Step 8 — hallucination detection report
# ---------------------------------------------------------------------------
HALL_MD = '''## V3 — Step 8 — Hallucination detection (pathology DDPM)

A restoration *hallucinates* when it changes the diagnostic content rather than
just the image quality. The clearest operational signal is a **change in the
predicted DR grade** between the degraded input and the restored output (a
lesion added or erased). For a capped sample of high-severity test images we
log, per degradation kind:

- DR grade + confidence before/after restoration, and the grade-change rate
- mean pixel deviation (L1 in [0,1]) between restored and degraded
- a 0–3 risk score → low / medium / high

Saved as a CSV under the Phase-4 metrics folder (no new plots).
'''

HALL_IMPL = r'''# === V3 PATCH (Step 8): hallucination detection for the pathology DDPM ===
import numpy as np
import torch.nn.functional as F

P4 = PHASE_DIRS['phase4_genai_enhancement']
(P4 / 'metrics').mkdir(parents=True, exist_ok=True)

_hall_clf = load_classifier_v3('efficientnet_b3')   # our trained DR grader
_hall_clf.eval()


def _clf_logits(pil):
    x = TFM_EVAL_V3(pil.convert('RGB')).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return _hall_clf(x)


def hallucination_check(degraded_pil, restored_pil):
    lb = _clf_logits(degraded_pil); la = _clf_logits(restored_pil)
    gb, ga = int(lb.argmax(1)), int(la.argmax(1))
    cb = float(F.softmax(lb, 1).max()); ca = float(F.softmax(la, 1).max())
    db = np.asarray(degraded_pil.resize((256, 256)), dtype=np.float32) / 255.0
    dr = np.asarray(restored_pil.resize((256, 256)), dtype=np.float32) / 255.0
    pix = float(np.abs(dr - db).mean())
    score = (2 if gb != ga else 0) + (1 if pix > 0.15 else 0)
    risk = 'low' if score <= 1 else ('medium' if score == 2 else 'high')
    return dict(grade_before=gb, grade_after=ga, grade_changed=gb != ga,
                conf_before=round(cb, 3), conf_after=round(ca, 3),
                pixel_deviation=round(pix, 3),
                risk_score=score, hallucination_risk=risk)


# Run over a capped sample of high-severity test images (time budget).
_hall_rows = []
_sample = set(map(str, list(test_id_set)[:60]))
for k in DEGRADATION_TYPES:
    l = 'high'
    src_dir = DEGRADED_DIR / k / l
    if not (src_dir / 'manifest.csv').exists():
        continue
    mani = pd.read_csv(src_dir / 'manifest.csv')
    mani = mani[mani['id_code'].astype(str).isin(_sample)]
    for _, row in tqdm(mani.iterrows(), total=len(mani),
                       desc=f'hallucination {k}/{l}', leave=False):
        deg = Image.open(src_dir / row['rel_path']).convert('RGB')
        res = enhance_ddpm_path(deg)
        r = hallucination_check(deg, res)
        r.update(id_code=row['id_code'], kind=k, level=l)
        _hall_rows.append(r)

hall_df = pd.DataFrame(_hall_rows)
_hall_csv = P4 / 'metrics' / 'ddpm_pathology_hallucination.csv'
hall_df.to_csv(_hall_csv, index=False)
print(f"Saved hallucination report -> {_hall_csv}")
if len(hall_df):
    print(f"  grade-change (hallucination) rate: {hall_df['grade_changed'].mean()*100:.1f}%")
    print("  risk distribution:")
    print(hall_df['hallucination_risk'].value_counts().to_string())
'''


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def main():
    nb = json.loads(SRC.read_text(encoding='utf-8'))
    cells = nb['cells']
    print(f'  cells in v2: {len(cells)}')

    # Clean start: strip every code cell's outputs.
    for c in cells:
        if c.get('cell_type') == 'code':
            c['outputs'] = []
            c['execution_count'] = None

    # 1) Patch the enhancer registry: add 'ddpm_path' to ENHANCERS + a label.
    reg_idx = find_cell(cells, "ENHANCERS = ('clahe', 'genai', 'cold_diff'")
    patch_cell(cells, reg_idx,
               "'cyclegan', 'ddpm')",
               "'cyclegan', 'ddpm', 'ddpm_path')")
    patch_cell(cells, reg_idx,
               "    'ddpm':        'DDPM (Vanilla)',\n",
               "    'ddpm':        'DDPM (Vanilla)',\n"
               "    'ddpm_path':   'DDPM (Pathology)',\n")
    print(f'  patched enhancer registry @ cell {reg_idx}')

    # 2) Patch the build_enhanced caller: generate the 'ddpm_path' set + bump cache.
    call_idx = find_cell(cells, "build_enhanced('ddpm',       lambda im, k, l: enhance_ddpm(im))")
    patch_cell(cells, call_idx,
               "build_enhanced('ddpm',       lambda im, k, l: enhance_ddpm(im))       # V2 NEW (Step 4)\n",
               "build_enhanced('ddpm',       lambda im, k, l: enhance_ddpm(im))       # V2 NEW (Step 4)\n"
               "build_enhanced('ddpm_path',  lambda im, k, l: enhance_ddpm_path(im))  # V3 NEW (Step 7)\n")
    patch_cell(cells, call_idx,
               "print('Enhanced sets ready (incl. CycleGAN-CBAM + DDPM).')",
               "print('Enhanced sets ready (incl. CycleGAN-CBAM + vanilla & pathology DDPM).')")
    # Bump cache filename so the new 'ddpm_path' tree is persisted to Drive.
    patch_cell(cells, call_idx, "cache_enhanced.tar.gz", "cache_enhanced_v3.tar.gz")
    print(f'  patched build_enhanced caller @ cell {call_idx}')

    # 3) Insert the Step-7 pathology DDPM cells right AFTER the vanilla DDPM cell
    #    (so enhance_ddpm_path is defined before the build_enhanced caller runs).
    ddpm_idx = find_cell(cells, "# === V2 PATCH (Step 4): conditional vanilla DDPM ===")
    block = [md_cell(DDPM_PATH_MD), code_cell(DDPM_PATH_IMPL)]
    cells[ddpm_idx + 1:ddpm_idx + 1] = block
    print(f'  inserted Step-7 pathology DDPM after cell {ddpm_idx}')

    # 4) Append Step-8 hallucination detection at the very end.
    cells.extend([md_cell(HALL_MD), code_cell(HALL_IMPL)])
    print('  appended Step-8 hallucination detection')

    # 5) Refresh the top marker.
    if cells[0].get('cell_type') == 'markdown':
        cells[0]['source'] = [
            '# V3 PATCH MARKER (do not delete)\n',
            '\n',
            'V3-extended notebook built by `build_v3.py` from version2. Adds:\n',
            '\n',
            '- Step 7: pathology-preserving conditional DDPM (Phase 4 Diffusion #3) — '
            'classifier-feature perceptual loss + input-fidelity loss + sampling-time '
            "anti-hallucination clamp. New checkpoint `ddpm_pathology_v1.pt`, enhancer key `'ddpm_path'`.\n",
            '- Step 8: hallucination detection report (grade-change / pixel-deviation / risk), '
            'saved to `phase4/metrics/ddpm_pathology_hallucination.csv`.\n',
            '\n',
            'The vanilla DDPM (Step 4) is retained so the recovery table reports both DDPMs '
            '(vanilla vs pathology) as an ablation. All new training is resume-guarded.\n',
        ]

    nb['cells'] = cells
    test_json = json.dumps(nb)
    print(f'  cells in v3: {len(cells)} | serialised: {len(test_json)/1024:.1f} KB')
    DST.write_text(json.dumps(nb, indent=1), encoding='utf-8')
    print(f'Wrote v3 -> {DST}')


if __name__ == '__main__':
    main()
