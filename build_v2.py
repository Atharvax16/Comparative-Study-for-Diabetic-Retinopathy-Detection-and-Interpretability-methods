"""
Build Thesis_optimized_final_version2.ipynb from version1.

Applies the master-prompt changes:
  Step 1a: load_classifier_v3 no longer prefers undertrained distilled ckpts
  Step 1b: phantom EyeQ filter removed (explicit "full APTOS")
  Step 1c: Cold Diffusion retrained — 15 epochs, cosine LR, noise-biased sampling, new ckpt v6
  Step 1d: Phase 5 quality classifier calibrated with threshold-based routing
  Step 1e: SwinIR-GAN retrained — 8 epochs, adv_w=0.005, freeze D after epoch 4, new ckpt v6
  Step 2 : SHAP via GradientExplainer across all 3 models + faithfulness/sufficiency/consistency
  Step 3 : CycleGAN-CBAM (Phase 4 GAN #2)
  Step 4 : Vanilla DDPM conditional on degraded input (Phase 4 Diffusion #2)
  Step 5+6: Per-class diagnostics + failure-mode analysis + regen markers

All new training cells include resume guards. All new cells have execution_count=None
and outputs=[] so the v2 notebook is "clean" and ready to run.
"""
import json
import sys
from pathlib import Path

SRC = Path(r"C:\Dissertation\Thesis_optimized_final_version1.ipynb")
DST = Path(r"C:\Dissertation\Thesis_optimized_final_version2.ipynb")


def code_cell(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def md_cell(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def replace_cell(cells: list, idx: int, new_source: str, cell_type: str = "code") -> None:
    """In-place replace cell source; resets execution_count and outputs for code cells."""
    c = cells[idx]
    c["source"] = new_source.splitlines(keepends=True)
    if cell_type == "code":
        c["cell_type"] = "code"
        c["execution_count"] = None
        c["outputs"] = []
        c.setdefault("metadata", {})


# ---------------------------------------------------------------------------
# Replacement sources
# ---------------------------------------------------------------------------

CELL_17_NEW = r'''P1 = PHASE_DIRS['phase1_data_engineering']

# 1.1 V2 PATCH — phantom EyeQ filter removed.
# The original filter_pristine() tried to join EyeQ quality labels onto APTOS
# but the join key never matched in any observed Colab run — the routine
# silently fell back to "use all APTOS rows". To keep the methodology honest
# we now explicitly use the full APTOS dataset and document "APTOS only" in
# the dissertation. EyeQ is still loaded separately for the Phase 5 quality
# classifier training (Q_CKPT).
def filter_pristine(aptos_csv, eyeq_csv=None, out_csv=None, **kwargs):
    """V2 patch: always returns the full APTOS dataset."""
    aptos = pd.read_csv(aptos_csv)
    if out_csv is not None:
        aptos.to_csv(out_csv, index=False)
    return aptos

PRISTINE_CSV = P1 / 'metrics' / 'pristine_split.csv'
pristine_df = filter_pristine(APTOS_CSV, EYEQ_CSV, PRISTINE_CSV)
print(f'Using full APTOS dataset: {len(pristine_df)} images')
pristine_df.head()
'''

CELL_39_NEW = r'''def load_classifier_v3(name):
    """Load a v3 multi-scale classifier from disk.

    V2 PATCH: prefers `{name}_v3_best.pt` (fully-trained) over the undertrained
    `{name}_v3_distilled.pt`. The original V4 distillation run was interrupted
    at epoch 3 of the resnet50 student, leaving two ~3-epoch distilled
    checkpoints on Drive. The previous prefer-distilled logic silently routed
    those under-trained students into the V4 ensemble and clean accuracy
    regressed 0.814 -> 0.725. Set USE_DISTILLED = True below only after you
    have re-run distillation to the full 10 epochs and verified val kappa.
    """
    USE_DISTILLED = False   # flip back to True once distillation reruns to convergence
    m = build_model_v3(name, pretrained=False).to(DEVICE)
    distilled = CKPT_DIR_V3 / f"{name}_v3_distilled.pt"
    best      = CKPT_DIR_V3 / f"{name}_v3_best.pt"
    src = distilled if (USE_DISTILLED and distilled.exists()) else best
    ckpt = torch.load(src, map_location=DEVICE, weights_only=False)
    m.load_state_dict(ckpt["state_dict"]); m.eval()
    return m


@torch.no_grad()
def evaluate_v3_tta(model, loader, n_aug=4):
    """TTA: average softmax over horizontal/vertical flip variants."""
    model.eval(); ys, probs = [], []
    for x, y, _ in loader:
        x = x.to(DEVICE, non_blocking=True)
        views = [x,
                 torch.flip(x, [3]),
                 torch.flip(x, [2]),
                 torch.flip(x, [2, 3])][:n_aug]
        avg = None
        for v in views:
            p = torch.softmax(model(v), 1)
            avg = p if avg is None else avg + p
        avg /= len(views)
        probs.append(avg.cpu().numpy()); ys.append(y.numpy())
    y  = np.concatenate(ys)
    pr = np.concatenate(probs)
    p  = pr.argmax(1)
    out = {
        "accuracy": accuracy_score(y, p),
        "f1_macro": f1_score(y, p, average="macro"),
        "kappa_qw": cohen_kappa_score(y, p, weights="quadratic"),
    }
    try:
        out["auc_macro_ovr"] = roc_auc_score(y, pr, average="macro", multi_class="ovr")
    except ValueError:
        out["auc_macro_ovr"] = float("nan")
    return out


# Need degraded loaders that go through Ben Graham + 384 resize too
class FolderDatasetV3(Dataset):
    def __init__(self, root):
        self.root = Path(root); self.df = pd.read_csv(self.root / "manifest.csv")
        self.transform = TFM_EVAL_V3
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.root / row["rel_path"]).convert("RGB")
        return self.transform(img), int(row["diagnosis"]), row["id_code"]


test_id_set = set(map(str, [full_eval_v3.df.iloc[i]["id_code"] for i in te_idx]))


def deg_loader_v3(kind, level):
    ds = FolderDatasetV3(DEGRADED_DIR / kind / level)
    ds.df = ds.df[ds.df["id_code"].astype(str).isin(test_id_set)].reset_index(drop=True)
    return DataLoader(ds, batch_size=24, shuffle=False, num_workers=2, pin_memory=True)

# [RESUME GUARD] Skip v3 stress-test loop if the saved CSV already exists.
_v3_stress_csv = P2 / "metrics" / "v3" / "stress_test_results_v3.csv"
if _v3_stress_csv.exists():
    print(f"[skip] v3 stress-test CSV already exists -> {_v3_stress_csv}")
    stress_df_v3 = pd.read_csv(_v3_stress_csv)
    pivot_v3 = stress_df_v3.pivot_table(index=["degradation", "level"],
                                         columns="model", values="accuracy").round(3)
    pivot_v3
else:
    print("[run] Running v3 stress test (this takes ~30 min on A100)...")
    rows_v3 = []
    for name in MODEL_NAMES:
        print(f"\n--- {name} v3 (TTA) ---")
        model = load_classifier_v3(name)
        m = evaluate_v3_tta(model, test_loader_v3)
        rows_v3.append({"model": name, "degradation": "clean", "level": "none", **m})
        print(f"  clean: acc={m['accuracy']:.4f}  f1={m['f1_macro']:.4f}  kappa={m['kappa_qw']:.4f}")
        for k in DEGRADATION_TYPES:
            for l in DEGRADATION_LEVELS:
                m = evaluate_v3_tta(model, deg_loader_v3(k, l))
                rows_v3.append({"model": name, "degradation": k, "level": l, **m})
                print(f"  {k}/{l}: acc={m['accuracy']:.4f}  f1={m['f1_macro']:.4f}  kappa={m['kappa_qw']:.4f}")
        del model; torch.cuda.empty_cache()
    stress_df_v3 = pd.DataFrame(rows_v3)
    stress_df_v3.to_csv(_v3_stress_csv, index=False)
    pivot_v3 = stress_df_v3.pivot_table(index=["degradation", "level"],
                                         columns="model", values="accuracy").round(3)
    pivot_v3
'''


CELL_53_NEW = r'''# 3.2 Run XAI benchmark — for each (image, model, method, condition)
# V2 PATCH: the METHOD_REGISTRY is now built by upstream cells (real SHAP added
# in the post-IG cell, see Step 2). We no longer hardcode it here — that would
# overwrite the SHAP rebind. Print the current dispatch table for transparency.
print('XAI methods in benchmark:',
      {k: (v['fn'].__name__, v['models']) for k, v in METHOD_REGISTRY.items()})

rows = []
for method_name, spec in METHOD_REGISTRY.items():
    fn = spec['fn']
    for model_name in spec['models']:
        model = models[model_name]
        print(f'\n[{method_name} | {model_name}]')
        for id_code in tqdm(EXPLAIN_IDS, desc=f'{method_name}/{model_name}'):
            label = int(labels_df[id_code])
            mask  = load_mask(id_code)
            x_clean = load_tensor(resolve_image(id_code))
            try:    h_clean = fn(model, x_clean, target_class=label)
            except Exception as e: print('  skip clean:', e); continue
            rows.append({'id_code': id_code, 'model': model_name, 'method': method_name,
                         'degradation': 'clean', 'level': 'none',
                         'deletion_auc':  deletion_auc(model, x_clean, h_clean, label),
                         'insertion_auc': insertion_auc(model, x_clean, h_clean, label),
                         'stability':     1.0,
                         'iou':           localization_iou(h_clean, mask)})
            for k in DEGRADATION_TYPES:
                for l in DEGRADATION_LEVELS:
                    src = DEGRADED_DIR / k / l / f'{id_code}{DEG_SAVE_EXT}'
                    if not src.exists(): continue
                    x_deg = load_tensor(src)
                    try: h_deg = fn(model, x_deg, target_class=label)
                    except Exception: continue
                    rows.append({'id_code': id_code, 'model': model_name, 'method': method_name,
                                 'degradation': k, 'level': l,
                                 'deletion_auc':  deletion_auc(model, x_deg, h_deg, label),
                                 'insertion_auc': insertion_auc(model, x_deg, h_deg, label),
                                 'stability':     stability_spearman(h_clean, h_deg),
                                 'iou':           localization_iou(h_deg, mask)})

xai_df = pd.DataFrame(rows)
xai_df.to_csv(P3 / 'metrics' / 'xai_results.csv', index=False)
xai_df.head()
'''


CELL_61_NEW = r'''# === V2 PATCH: relabel cell neutralised ===
# In V1 this cell renamed every 'shap' row in xai_results.csv to 'IG' because
# the V4 patch had repurposed the registry key 'shap' to point at IG. In V2
# the registry has BOTH 'shap' (real SHAP via GradientExplainer) and 'ig'
# (renamed from the legacy 'shap'). Renaming would now corrupt the labels.
# We leave the CSV untouched.
import pandas as pd
xai_csv = P3 / 'metrics' / 'xai_results.csv'
if xai_csv.exists():
    df = pd.read_csv(xai_csv)
    print('Method label counts in xai_results.csv (untouched):')
    print(df['method'].value_counts().to_string())
else:
    print('[skip] xai_results.csv not yet written')
'''


CELL_67_NEW = r'''# === V5 / V2 PATCH: Cold Diffusion for fundus restoration ===
# Bansal et al. 2022 (https://arxiv.org/abs/2208.09392): the diffusion forward
# process can be any deterministic degradation D(x, t). Train R_theta so that
# R(D(x, t), t) ≈ x for all t, then sample via Algorithm 2:
#
#       x_{s-1} = x_s - D(R(x_s, s), s) + D(R(x_s, s), s-1)
#
# V2 patch (Step 1c) fixes the noise-catastrophe documented in OBSERVATIONS:
#   - 15 epochs with CosineAnnealingLR (was 4 epochs, constant LR)
#   - Kind sampling biased to noise (was uniform)
#   - New checkpoint name cold_diffusion_v6.pt so the resume guard fires fresh

import math, random

COLD_DIFF_CKPT = CHECKPOINTS_DIR / 'cold_diffusion_v6.pt'   # V2 patch: bumped from v5
COLD_DIFF_SIZE = 256        # work at 256 px to fit budget; resize back at end
T_STEPS        = 8          # train + inference steps

# Severity schedule: t in [0, T_STEPS] -> degradation parameter
SEV_HIGH = {'blur': 9.0, 'exposure_gain_low': 0.2, 'noise': 0.12}

def _severity(t, kind):
    f = t / T_STEPS  # in [0, 1]
    if kind == 'blur':
        return max(0.01, f * SEV_HIGH['blur'])
    if kind == 'exposure':
        return 1.0 - f * (1.0 - SEV_HIGH['exposure_gain_low'])
    if kind == 'noise':
        return f * SEV_HIGH['noise']
    raise ValueError(kind)

def _degrade_t(img_pil, kind, t):
    """Forward operator at severity t using the Phase-1 functions."""
    if t <= 0:
        return img_pil
    p = _severity(t, kind)
    return DEGRADERS[kind](img_pil, p)


# ---- Conditional U-Net (FiLM on kind + t/T) ----
class CondUNet(nn.Module):
    def __init__(self, ch=48, n_kinds=3, t_emb=64):
        super().__init__()
        self.kind_emb = nn.Embedding(n_kinds, t_emb)
        self.t_proj   = nn.Sequential(
            nn.Linear(1, t_emb), nn.SiLU(), nn.Linear(t_emb, t_emb),
        )
        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU(),
                nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU(),
            )
        self.in_proj = nn.Conv2d(3, ch, 3, padding=1)
        self.e1 = block(ch,   ch)
        self.e2 = block(ch,   ch*2)
        self.e3 = block(ch*2, ch*4)
        self.mid = block(ch*4, ch*4)
        self.film = nn.Linear(t_emb*2, ch*4*2)
        self.up2 = nn.ConvTranspose2d(ch*4, ch*2, 2, stride=2); self.d2 = block(ch*4, ch*2)
        self.up1 = nn.ConvTranspose2d(ch*2, ch,   2, stride=2); self.d1 = block(ch*2, ch)
        self.out = nn.Conv2d(ch, 3, 1)
    def forward(self, x, kind_id, t_norm):
        cond = torch.cat([self.kind_emb(kind_id),
                          self.t_proj(t_norm.unsqueeze(-1))], dim=-1)
        gb = self.film(cond)
        g, b = gb.chunk(2, dim=-1)
        h0 = self.in_proj(x)
        e1 = self.e1(h0)
        e2 = self.e2(F.avg_pool2d(e1, 2))
        e3 = self.e3(F.avg_pool2d(e2, 2))
        m  = self.mid(e3)
        m  = m * (1 + g[:, :, None, None]) + b[:, :, None, None]
        d2 = self.d2(torch.cat([self.up2(m),  e2], 1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
        return torch.sigmoid(self.out(d1))


KIND_TO_ID = {'blur': 0, 'exposure': 1, 'noise': 2}
# V2 patch (Step 1c): weight noise samples higher in training to fix the noise catastrophe
KIND_SAMPLE_WEIGHTS = {'blur': 0.25, 'exposure': 0.25, 'noise': 0.50}

class _ColdDiffDataset(Dataset):
    """On-the-fly paired (degraded@t, clean) sampler over the pristine set.

    V2 patch: kind sampling uses KIND_SAMPLE_WEIGHTS so noise dominates training.
    """
    def __init__(self, df, n_per=4):
        self.df = df.reset_index(drop=True); self.n_per = n_per
        self.tfm = T.Compose([T.Resize((COLD_DIFF_SIZE, COLD_DIFF_SIZE)),
                              T.ToTensor()])
        self._kinds = list(KIND_SAMPLE_WEIGHTS.keys())
        self._probs = list(KIND_SAMPLE_WEIGHTS.values())
    def __len__(self):
        return len(self.df) * self.n_per
    def __getitem__(self, i):
        row = self.df.iloc[i // self.n_per]
        try:
            img = Image.open(resolve_image(row['id_code'])).convert('RGB')
        except FileNotFoundError:
            img = Image.new('RGB', (COLD_DIFF_SIZE, COLD_DIFF_SIZE))
        img  = img.resize((COLD_DIFF_SIZE, COLD_DIFF_SIZE), Image.BILINEAR)
        kind = random.choices(self._kinds, weights=self._probs, k=1)[0]
        t    = random.randint(1, T_STEPS)
        deg  = _degrade_t(img, kind, t)
        return (self.tfm(deg), KIND_TO_ID[kind],
                torch.tensor(t / T_STEPS, dtype=torch.float32),
                self.tfm(img))


def train_cold_diffusion(epochs=15, batch_size=16, lr=2e-4):
    """V2 patch: 15 epochs, CosineAnnealingLR, noise-biased sampling.

    Estimated time: ~5 hours on A100 with T_STEPS=8, COLD_DIFF_SIZE=256.
    """
    if COLD_DIFF_CKPT.exists():
        print(f"[skip] {COLD_DIFF_CKPT.name} already on Drive — not retraining.")
        return
    print(f"Training Cold Diffusion v6 ({epochs} epochs, cosine LR, noise-biased)...")
    df  = pd.read_csv(PRISTINE_CSV)
    ds  = _ColdDiffDataset(df, n_per=4)
    dl  = DataLoader(ds, batch_size=batch_size, shuffle=True,
                     num_workers=2, pin_memory=True)
    net = CondUNet().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = GradScaler(enabled=True)
    crit   = nn.L1Loss()
    history = []
    for ep in range(epochs):
        net.train()
        running, n = 0.0, 0
        pbar = tqdm(dl, desc=f"cold-diff v6 ep {ep+1}/{epochs}", leave=False)
        for deg, kind_id, tnorm, clean in pbar:
            deg   = deg.to(DEVICE, non_blocking=True)
            clean = clean.to(DEVICE, non_blocking=True)
            kind_id = kind_id.to(DEVICE).long()
            tnorm   = tnorm.to(DEVICE).float()
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                pred = net(deg, kind_id, tnorm)
                loss = crit(pred, clean)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            running += loss.item() * deg.size(0); n += deg.size(0)
            pbar.set_postfix(L1=f"{running/n:.4f}",
                             lr=f"{opt.param_groups[0]['lr']:.2e}")
        sched.step()
        epoch_l1 = running / max(n, 1)
        history.append({'epoch': ep+1, 'l1': epoch_l1, 'lr': opt.param_groups[0]['lr']})
        print(f"  ep {ep+1}: L1={epoch_l1:.4f}  lr={opt.param_groups[0]['lr']:.2e}")
    torch.save({'state_dict': net.state_dict(),
                'T_STEPS': T_STEPS,
                'kind_weights': KIND_SAMPLE_WEIGHTS,
                'history': history}, COLD_DIFF_CKPT)
    print(f"Saved -> {COLD_DIFF_CKPT}")


train_cold_diffusion(epochs=15, batch_size=16, lr=2e-4)


# ---- Inference: Cold Diffusion Algorithm 2 sampling ----
_cold_net = None
def _load_cold_net():
    global _cold_net
    if _cold_net is None:
        _cold_net = CondUNet().to(DEVICE).eval()
        sd = torch.load(COLD_DIFF_CKPT, map_location=DEVICE, weights_only=False)
        _cold_net.load_state_dict(sd['state_dict'])
    return _cold_net


@torch.no_grad()
def enhance_cold_diffusion(img_pil, kind='blur', t_start=T_STEPS):
    """Iterative restoration via Cold Diffusion Algorithm 2."""
    net = _load_cold_net()
    orig_size = img_pil.size
    src = img_pil.convert('RGB').resize((COLD_DIFF_SIZE, COLD_DIFF_SIZE), Image.BILINEAR)
    x = T.ToTensor()(src).unsqueeze(0).to(DEVICE)
    kind_id = torch.tensor([KIND_TO_ID[kind]], device=DEVICE, dtype=torch.long)
    to_pil = T.ToPILImage()
    to_t   = T.ToTensor()
    for s in range(t_start, 0, -1):
        tnorm = torch.tensor([s / T_STEPS], device=DEVICE)
        x_hat = net(x, kind_id, tnorm).clamp(0, 1)
        if s > 1:
            xh_pil = to_pil(x_hat.squeeze().cpu())
            d_s   = _degrade_t(xh_pil, kind, s)
            d_sm1 = _degrade_t(xh_pil, kind, s - 1)
            d_s   = to_t(d_s.resize((COLD_DIFF_SIZE, COLD_DIFF_SIZE))).unsqueeze(0).to(DEVICE)
            d_sm1 = to_t(d_sm1.resize((COLD_DIFF_SIZE, COLD_DIFF_SIZE))).unsqueeze(0).to(DEVICE)
            x = (x - d_s + d_sm1).clamp(0, 1)
        else:
            x = x_hat
    arr = (x.squeeze().permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(arr).resize(orig_size)


COLD_T_FOR_LEVEL = {'low': 3, 'mid': 5, 'high': T_STEPS}
print("Cold Diffusion v6 restorer ready.")
'''


CELL_70_NEW = r'''# === V5 / V2 PATCH: SwinIR + GAN restorer ===
# Supervised paired restoration with adversarial term.
# Forward operator: the SAME Phase-1 degradation primitives, like Cold Diffusion.
#
# V2 patch (Step 1e) fixes the adversarial-drift documented in OBSERVATIONS:
#   - 8 epochs (was 3) so the generator has time to keep pace
#   - adv_w lowered 0.01 -> 0.005 so reconstruction loss dominates early
#   - Discriminator frozen after epoch 4 (freeze_d_after) to stop D-drift
#   - New checkpoint name swinir_gan_v6.pt so the resume guard fires fresh

SWINIR_CKPT = CHECKPOINTS_DIR / 'swinir_gan_v6.pt'   # V2 patch: bumped from v5
SWINIR_SIZE = 256
SWINIR_WIN  = 8

# ---- Generator: try basicsr SwinIR, fall back to slim Swin-UNet ----
def _make_swinir():
    try:
        from basicsr.archs.swinir_arch import SwinIR
        net = SwinIR(
            upscale=1, in_chans=3, img_size=SWINIR_SIZE, window_size=SWINIR_WIN,
            depths=[2, 2, 2, 2], embed_dim=60, num_heads=[2, 2, 2, 2],
            mlp_ratio=2.0, upsampler='', resi_connection='1conv',
        )
        print("  generator: basicsr SwinIR (slim)")
        return net
    except Exception as e:
        print(f"  [fallback] basicsr SwinIR unavailable ({e}); using slim Swin-UNet")
    class _Block(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1), nn.GELU(),
                nn.Conv2d(ch, ch, 3, padding=1),
            )
        def forward(self, x): return x + self.conv(x)
    class _SwinUNetSlim(nn.Module):
        def __init__(self, ch=48):
            super().__init__()
            self.in_proj = nn.Conv2d(3, ch, 3, padding=1)
            self.e1 = nn.Sequential(_Block(ch), _Block(ch))
            self.d1 = nn.Conv2d(ch, ch*2, 4, stride=2, padding=1)
            self.e2 = nn.Sequential(_Block(ch*2), _Block(ch*2))
            self.d2 = nn.Conv2d(ch*2, ch*4, 4, stride=2, padding=1)
            self.mid = nn.Sequential(_Block(ch*4), _Block(ch*4))
            self.u2  = nn.ConvTranspose2d(ch*4, ch*2, 2, stride=2)
            self.dec2 = nn.Sequential(_Block(ch*2), _Block(ch*2))
            self.u1  = nn.ConvTranspose2d(ch*2, ch, 2, stride=2)
            self.dec1 = nn.Sequential(_Block(ch), _Block(ch))
            self.out  = nn.Conv2d(ch, 3, 3, padding=1)
        def forward(self, x):
            h0 = self.in_proj(x)
            e1 = self.e1(h0)
            e2 = self.e2(self.d1(e1))
            m  = self.mid(self.d2(e2))
            u2 = self.dec2(self.u2(m) + e2)
            u1 = self.dec1(self.u1(u2) + e1)
            return torch.sigmoid(self.out(u1) + x)
    return _SwinUNetSlim()


# ---- Conditional PatchGAN discriminator ----
class _PatchGAN(nn.Module):
    def __init__(self, in_ch=6, base=48):
        super().__init__()
        def blk(i, o, s, norm=True):
            l = [nn.Conv2d(i, o, 4, stride=s, padding=1)]
            if norm:
                l.append(nn.GroupNorm(8, o))
            l.append(nn.LeakyReLU(0.2, True))
            return l
        self.net = nn.Sequential(
            *blk(in_ch, base,   2, norm=False),
            *blk(base,  base*2, 2),
            *blk(base*2,base*4, 2),
            *blk(base*4,base*8, 1),
            nn.Conv2d(base*8, 1, 4, padding=1),
        )
    def forward(self, x): return self.net(x)


# ---- Training dataset: same Phase-1 operators ----
class _SwinIRDataset(Dataset):
    def __init__(self, df, n_per=4):
        self.df = df.reset_index(drop=True); self.n_per = n_per
        self.tfm = T.Compose([T.Resize((SWINIR_SIZE, SWINIR_SIZE)), T.ToTensor()])
    def __len__(self):
        return len(self.df) * self.n_per
    def __getitem__(self, i):
        row = self.df.iloc[i // self.n_per]
        try:
            img = Image.open(resolve_image(row['id_code'])).convert('RGB')
        except FileNotFoundError:
            img = Image.new('RGB', (SWINIR_SIZE, SWINIR_SIZE))
        img = img.resize((SWINIR_SIZE, SWINIR_SIZE), Image.BILINEAR)
        kind  = random.choice(['blur', 'exposure', 'noise'])
        level = random.choice(['low', 'mid', 'high'])
        deg   = DEGRADERS[kind](img, DEGRADATION_PARAMS[kind][level])
        return self.tfm(deg), self.tfm(img)


def train_swinir_gan(epochs=8, batch_size=4, lr_g=1e-4, lr_d=1e-4,
                     adv_w=0.005, freeze_d_after=4):
    """V2 patch (Step 1e): 8 epochs, adv_w=0.005, freeze D after epoch 4."""
    if SWINIR_CKPT.exists():
        print(f"[skip] {SWINIR_CKPT.name} already on Drive — not retraining.")
        return
    print(f"Training SwinIR + GAN v6 ({epochs} epochs, adv_w={adv_w}, freeze D after ep {freeze_d_after})...")
    df = pd.read_csv(PRISTINE_CSV)
    ds = _SwinIRDataset(df, n_per=4)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    num_workers=2, pin_memory=True)
    G = _make_swinir().to(DEVICE)
    D = _PatchGAN(in_ch=6).to(DEVICE)
    opt_g = torch.optim.AdamW(G.parameters(), lr=lr_g, betas=(0.9, 0.99), weight_decay=1e-5)
    opt_d = torch.optim.AdamW(D.parameters(), lr=lr_d, betas=(0.9, 0.99), weight_decay=1e-5)
    scaler_g = GradScaler(enabled=True)
    scaler_d = GradScaler(enabled=True)
    l1  = nn.L1Loss()
    bce = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        G.train()
        train_d_this_epoch = ep < freeze_d_after
        if train_d_this_epoch:
            D.train()
            for p in D.parameters(): p.requires_grad = True
        else:
            D.eval()
            for p in D.parameters(): p.requires_grad = False
        run_g, run_d, n = 0.0, 0.0, 0
        pbar = tqdm(dl, desc=f"swinir-gan v6 ep {ep+1}/{epochs}", leave=False)
        for deg, clean in pbar:
            deg   = deg.to(DEVICE, non_blocking=True)
            clean = clean.to(DEVICE, non_blocking=True)
            # ----- D step (only while not frozen) -----
            d_loss_v = 0.0
            if train_d_this_epoch:
                opt_d.zero_grad(set_to_none=True)
                with autocast(enabled=True):
                    with torch.no_grad():
                        fake = G(deg).clamp(0, 1)
                    d_real = D(torch.cat([deg, clean], 1))
                    d_fake = D(torch.cat([deg, fake], 1))
                    d_loss = 0.5 * (bce(d_real, torch.ones_like(d_real)) +
                                    bce(d_fake, torch.zeros_like(d_fake)))
                scaler_d.scale(d_loss).backward()
                scaler_d.step(opt_d); scaler_d.update()
                d_loss_v = d_loss.item()
            # ----- G step -----
            opt_g.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                fake = G(deg).clamp(0, 1)
                d_out = D(torch.cat([deg, fake], 1))
                adv  = bce(d_out, torch.ones_like(d_out))
                rec  = l1(fake, clean)
                g_loss = rec + adv_w * adv
            scaler_g.scale(g_loss).backward()
            scaler_g.step(opt_g); scaler_g.update()
            run_g += g_loss.item() * deg.size(0)
            run_d += d_loss_v * deg.size(0)
            n     += deg.size(0)
            pbar.set_postfix(G=f"{run_g/n:.4f}",
                             D=f"{run_d/n:.4f}",
                             d_state='train' if train_d_this_epoch else 'frozen')
        print(f"  ep {ep+1}: G={run_g/n:.4f}  D={run_d/n:.4f}  d_state={'train' if train_d_this_epoch else 'frozen'}")
    torch.save({'state_dict': G.state_dict(),
                'adv_w': adv_w,
                'freeze_d_after': freeze_d_after}, SWINIR_CKPT)
    print(f"Saved -> {SWINIR_CKPT}")


train_swinir_gan(epochs=8, batch_size=4, lr_g=1e-4, lr_d=1e-4,
                 adv_w=0.005, freeze_d_after=4)


# ---- Inference ----
_swin_net = None
def _load_swinir():
    global _swin_net
    if _swin_net is None:
        _swin_net = _make_swinir().to(DEVICE).eval()
        sd = torch.load(SWINIR_CKPT, map_location=DEVICE, weights_only=False)
        _swin_net.load_state_dict(sd['state_dict'])
    return _swin_net


@torch.no_grad()
def enhance_swinir_gan(img_pil):
    """One-shot SwinIR-GAN restoration (no iteration, no kind conditioning)."""
    net = _load_swinir()
    orig = img_pil.size
    src = img_pil.convert('RGB').resize((SWINIR_SIZE, SWINIR_SIZE), Image.BILINEAR)
    x = T.ToTensor()(src).unsqueeze(0).to(DEVICE)
    with autocast(enabled=True):
        y = net(x).clamp(0, 1)
    arr = (y.squeeze().permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype('uint8')
    return Image.fromarray(arr).resize(orig)


print("SwinIR + GAN v6 restorer ready.")
'''


CELL_71_NEW = r'''# 4.3 Build enhanced versions of every degraded image (test ids only)
# V2 patch (Step 3 + Step 4): now builds CycleGAN and DDPM enhanced sets too.
test_id_set = set(pd.read_csv(P2 / 'metrics' / 'test_ids.csv')['id_code'].astype(str))

def build_enhanced(method_name, fn):
    """fn signature: fn(img_pil, kind, level) -> PIL.Image"""
    for k in DEGRADATION_TYPES:
        for l in DEGRADATION_LEVELS:
            src_dir = DEGRADED_DIR / k / l
            out_dir = ENHANCED_DIR / method_name / k / l
            out_dir.mkdir(parents=True, exist_ok=True)
            mani = pd.read_csv(src_dir / 'manifest.csv')
            mani = mani[mani['id_code'].astype(str).isin(test_id_set)]
            for _, row in tqdm(mani.iterrows(), total=len(mani),
                                desc=f'{method_name}/{k}/{l}', leave=False):
                out = out_dir / row['rel_path']
                if not out.exists():
                    src = Image.open(src_dir / row['rel_path']).convert('RGB')
                    fn(src, k, l).save(out)
            mani.assign(method=method_name).to_csv(out_dir / 'manifest.csv', index=False)

build_enhanced('clahe',     lambda im, k, l: enhance_clahe(im))
build_enhanced('genai',     lambda im, k, l: enhance_genai(im))
build_enhanced('cold_diff',  lambda im, k, l: enhance_cold_diffusion(im, kind=k,
                                                                      t_start=COLD_T_FOR_LEVEL[l]))
build_enhanced('swinir_gan', lambda im, k, l: enhance_swinir_gan(im))
build_enhanced('cyclegan',   lambda im, k, l: enhance_cyclegan(im))   # V2 NEW (Step 3)
build_enhanced('ddpm',       lambda im, k, l: enhance_ddpm(im))       # V2 NEW (Step 4)
print('Enhanced sets ready (incl. CycleGAN-CBAM + DDPM).')

# Cache enhanced images to Drive (mirrors cell 24's cache_degraded tar pattern).
# Eliminates ~15 min rebuild on session reload (PROJECT_OVERVIEW §11.5).
_ENH_CACHE = Path('/content/drive/MyDrive/Thesis/cache_enhanced.tar.gz')
if not _ENH_CACHE.exists():
    print(f'Caching enhanced tree to Drive -> {_ENH_CACHE} ...')
    import subprocess
    subprocess.run(['tar', '-czf', str(_ENH_CACHE),
                    '-C', str(ENHANCED_DIR.parent),
                    ENHANCED_DIR.name], check=False)
    print('  done.')
else:
    print(f'[skip] enhanced cache already exists on Drive -> {_ENH_CACHE}')
'''


CELL_72_NEW = r'''# 4.4 Re-evaluate the three classifiers on raw degraded vs every enhancer
# V2 patch: ENHANCERS now includes 'cyclegan' (GAN #2) and 'ddpm' (vanilla diffusion #2).
ENHANCERS = ('clahe', 'genai', 'cold_diff', 'swinir_gan', 'cyclegan', 'ddpm')
VARIANTS  = ('raw', *ENHANCERS)

# Display-friendly labels for plots and tables (cf. master prompt §6 notes).
ENHANCER_LABELS = {
    'raw':         'raw (no restoration)',
    'clahe':       'CLAHE',
    'genai':       'A-ESRGAN',
    'cold_diff':   'Cold Diffusion',
    'swinir_gan':  'SwinIR-GAN',
    'cyclegan':    'CycleGAN-CBAM',
    'ddpm':        'DDPM (Vanilla)',
}

rows = []
for name in MODEL_NAMES:
    model = models[name]
    print(f'\n=== {name} ===')
    for k in DEGRADATION_TYPES:
        for l in DEGRADATION_LEVELS:
            for variant in VARIANTS:
                root = DEGRADED_DIR / k / l if variant == 'raw' else ENHANCED_DIR / variant / k / l
                if not (root / 'manifest.csv').exists():
                    continue
                ds = FolderDataset(root, transform=TFM_EVAL)
                ds.df = ds.df[ds.df['id_code'].astype(str).isin(test_id_set)].reset_index(drop=True)
                m = evaluate(model, DataLoader(ds, batch_size=32, num_workers=2, pin_memory=True))
                rows.append({'model': name, 'degradation': k, 'level': l, 'variant': variant, **m})
                print(f'  {k}/{l}/{variant}: acc={m["accuracy"]:.4f}')
    del model; torch.cuda.empty_cache()

rec_df = pd.DataFrame(rows)
rec_df.to_csv(P4 / 'metrics' / 'recovery_accuracy.csv', index=False)
rec_df.head()
'''


# ---------------------------------------------------------------------------
# New cell sources
# ---------------------------------------------------------------------------

# ===== STEP 2: SHAP =====

SHAP_MD_HEADER = '''## V2 — Step 2 — SHAP via GradientExplainer across CNNs + ViT

Professor deliverable #1. The original V4 pipeline replaced KernelSHAP with Integrated Gradients
because KernelSHAP took 30+ seconds per image at 384 px. `shap.GradientExplainer` is
GPU-accelerated and runs in ~3–5 s per image at 384 px — fast enough to benchmark on all three
models. The cells below:

1. Implement `shap_heatmap_grad()` and a cached background fetcher (~50 samples in GPU memory).
2. Rebind the registry: `gradcam` (CNNs), `shap` (real SHAP, all models), `ig` (renamed from
   the old `shap` IG entry), `attention` (ViT).
3. Run the Phase 3 benchmark loop with the expanded registry.
4. Add SHAP-specific quality metrics — faithfulness, sufficiency, cross-model consistency,
   SHAP-vs-Grad-CAM agreement.
5. Render the side-by-side comparison grid (clean + 3 degradation-high conditions per model).
'''


SHAP_IMPL = r'''# === V2 PATCH (Step 2): real SHAP via GradientExplainer ===
import shap
import numpy as np
import torch
import torch.nn.functional as F

_SHAP_BG_CACHE = None

def get_shap_background(n_samples=50, device=None):
    """Return a small cached background tensor [n, 3, H, W] for GradientExplainer.

    Sampled deterministically from the training split so the background stays
    stable across SHAP calls (avoids variance from re-sampling).
    """
    global _SHAP_BG_CACHE
    if _SHAP_BG_CACHE is not None:
        return _SHAP_BG_CACHE
    device = device or DEVICE
    # Use the V3 training transform via TFM_EVAL_V3 so background statistics match input statistics.
    tfm = TFM_EVAL_V3 if 'TFM_EVAL_V3' in globals() else TFM_EVAL
    df  = pd.read_csv(PRISTINE_CSV)
    rng = np.random.RandomState(SEED if 'SEED' in globals() else 0)
    idx = rng.choice(len(df), size=min(n_samples, len(df)), replace=False)
    imgs = []
    for i in idx:
        try:
            im = Image.open(resolve_image(df.iloc[i]['id_code'])).convert('RGB')
        except FileNotFoundError:
            continue
        imgs.append(tfm(im))
    _SHAP_BG_CACHE = torch.stack(imgs).to(device)
    print(f'  SHAP background tensor: {_SHAP_BG_CACHE.shape}, mean={_SHAP_BG_CACHE.mean():.3f}')
    return _SHAP_BG_CACHE


def shap_heatmap_grad(model, x, target_class=None, n_samples=50):
    """Real SHAP heatmap via GradientExplainer. Returns (H, W) in [0, 1].

    Works for both CNN and ViT architectures at 384 px because GradientExplainer
    only needs grad w.r.t. inputs.
    """
    model.eval()
    device = x.device
    background = get_shap_background(n_samples=n_samples, device=device)
    if target_class is None:
        with torch.no_grad():
            target_class = int(model(x).argmax(1).item())
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(x)
    # shap returns either a list-of-arrays (one per class) or a stacked array
    if isinstance(shap_values, list):
        sv = shap_values[target_class]            # (1, 3, H, W)
    else:
        sv = np.asarray(shap_values)
        # Some shap versions return (1, 3, H, W, C) — squeeze the class axis if present
        if sv.ndim == 5:
            sv = sv[..., target_class]
    sv = np.asarray(sv).squeeze()                 # (3, H, W) or (H, W)
    if sv.ndim == 3:
        heat = np.abs(sv).mean(axis=0)            # (H, W)
    else:
        heat = np.abs(sv)
    if heat.max() > heat.min():
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    torch.cuda.empty_cache()
    return heat


# Rebind the XAI registry: ig=former shap entry, shap=real SHAP, plus existing gradcam + attention.
if 'METHOD_REGISTRY' not in globals():
    METHOD_REGISTRY = {}

# Save off the old IG-bound entry under a new key 'ig' before overwriting 'shap'.
if 'shap' in METHOD_REGISTRY and METHOD_REGISTRY['shap']['fn'].__name__ == 'ig_heatmap':
    METHOD_REGISTRY['ig'] = METHOD_REGISTRY['shap']
METHOD_REGISTRY['shap'] = {'fn': shap_heatmap_grad, 'models': MODEL_NAMES}
# Ensure gradcam / attention are present even on cold start:
if 'gradcam' not in METHOD_REGISTRY:
    METHOD_REGISTRY['gradcam'] = {'fn': gradcam_heatmap, 'models': ('resnet50', 'efficientnet_b3')}
if 'attention' not in METHOD_REGISTRY:
    METHOD_REGISTRY['attention'] = {'fn': attention_rollout, 'models': ('vit_base',)}

print('XAI registry after SHAP patch:')
for k, v in METHOD_REGISTRY.items():
    print(f'  {k:9s} -> {v["fn"].__name__:25s} models={v["models"]}')
'''


SHAP_QUALITY_METRICS = r'''# === V2 PATCH (Step 2c): SHAP-specific quality metrics ===
# These are the metrics the professor asked for. They are computed in addition
# to stability + insertion/deletion AUC which already exist in the benchmark.
#
#   faithfulness  : mask top-k% SHAP-important pixels and measure prediction drop.
#                   Higher area under (drop vs k) curve = more faithful.
#   sufficiency   : keep ONLY the top-k% SHAP pixels, mask the rest, measure
#                   how much of the prediction is retained.
#   consistency   : Spearman rho between SHAP heatmaps from DIFFERENT models on
#                   the same image — high means clinically meaningful, not
#                   architecture-specific.
#   shap_vs_gradcam : Spearman rho between SHAP and Grad-CAM heatmaps for the
#                     same (model, image). High = both methods agree.

import numpy as np
from scipy.stats import spearmanr

K_VALUES = [10, 20, 30, 50]   # percentages

def _top_k_mask(heat: np.ndarray, k_pct: int) -> np.ndarray:
    """Boolean mask of the top-k% pixels in `heat` (1 == keep)."""
    thresh = np.percentile(heat, 100 - k_pct)
    return heat >= thresh


@torch.no_grad()
def faithfulness_curve(model, x, heat, target_class, k_values=K_VALUES):
    """For each k in k_values: mask the top-k% SHAP pixels and report
    softmax probability of `target_class`. Lower probs at higher k =>
    those pixels really mattered. Returns list of (k, prob_drop)."""
    model.eval()
    p0 = torch.softmax(model(x), 1)[0, target_class].item()
    drops = []
    H, W  = heat.shape
    x_np  = x.detach().cpu().numpy()[0]            # (3, H, W)
    for k in k_values:
        m = _top_k_mask(heat, k)                   # (H, W) bool
        m3 = np.broadcast_to(m[None, :, :], x_np.shape)
        masked = np.where(m3, 0.0, x_np)           # zero out important pixels
        xm = torch.from_numpy(masked).unsqueeze(0).float().to(x.device)
        pk = torch.softmax(model(xm), 1)[0, target_class].item()
        drops.append((k, p0 - pk))
    auc = sum(d for _, d in drops) / max(len(drops), 1)
    return {'p0': p0, 'curve': drops, 'mean_drop': auc}


@torch.no_grad()
def sufficiency_curve(model, x, heat, target_class, k_values=K_VALUES):
    """For each k: KEEP only the top-k% pixels, mask the rest. Return how
    much of p0 is retained — higher is better."""
    model.eval()
    p0 = torch.softmax(model(x), 1)[0, target_class].item()
    retained = []
    x_np = x.detach().cpu().numpy()[0]
    for k in k_values:
        m = _top_k_mask(heat, k)
        m3 = np.broadcast_to(m[None, :, :], x_np.shape)
        kept = np.where(m3, x_np, 0.0)
        xk = torch.from_numpy(kept).unsqueeze(0).float().to(x.device)
        pk = torch.softmax(model(xk), 1)[0, target_class].item()
        retained.append((k, pk / max(p0, 1e-6)))
    return {'p0': p0, 'curve': retained,
            'mean_retained': sum(r for _, r in retained) / max(len(retained), 1)}


def heatmap_consistency(h1: np.ndarray, h2: np.ndarray) -> float:
    """Spearman rank correlation between two heatmaps (same shape)."""
    if h1.shape != h2.shape:
        side = min(h1.shape[0], h2.shape[0])
        h1 = h1[:side, :side]; h2 = h2[:side, :side]
    rho, _ = spearmanr(h1.flatten(), h2.flatten())
    return float(rho) if np.isfinite(rho) else float('nan')


# Run the SHAP-quality benchmark over the EXPLAIN_IDS already chosen for Phase 3.
P3_METRICS = P3 / 'metrics'
P3_METRICS.mkdir(parents=True, exist_ok=True)
shap_quality_rows = []
shap_cache_per_image = {}    # id_code -> {model: heatmap} for cross-model consistency

print('\n=== SHAP-quality benchmark (faithfulness / sufficiency / consistency) ===')
shap_fn = METHOD_REGISTRY['shap']['fn']
gradcam_fn = METHOD_REGISTRY['gradcam']['fn']
for id_code in tqdm(EXPLAIN_IDS, desc='shap-quality'):
    if id_code not in labels_df.index:
        continue
    label = int(labels_df[id_code])
    try:
        x = load_tensor(resolve_image(id_code))
    except FileNotFoundError:
        continue
    shap_cache_per_image[id_code] = {}
    for name in MODEL_NAMES:
        try:
            heat_shap = shap_fn(models[name], x, target_class=label)
        except Exception as e:
            print(f'  skip {id_code}/{name}: {e}')
            continue
        shap_cache_per_image[id_code][name] = heat_shap
        f = faithfulness_curve(models[name], x, heat_shap, label)
        s = sufficiency_curve(models[name], x, heat_shap, label)
        # SHAP vs Grad-CAM agreement (only valid for CNN models)
        sv_gc = float('nan')
        if name in METHOD_REGISTRY['gradcam']['models']:
            try:
                heat_gc = gradcam_fn(models[name], x, target_class=label)
                sv_gc   = heatmap_consistency(heat_shap, heat_gc)
            except Exception:
                pass
        shap_quality_rows.append({
            'id_code': id_code, 'model': name, 'label': label,
            'faithfulness_mean_drop': f['mean_drop'],
            'sufficiency_mean_retained': s['mean_retained'],
            'shap_vs_gradcam_rho': sv_gc,
        })

shap_quality_df = pd.DataFrame(shap_quality_rows)
shap_quality_df.to_csv(P3_METRICS / 'shap_quality_metrics.csv', index=False)
print('\nSHAP-quality summary (averaged over images, per model):')
print(shap_quality_df.groupby('model')[
    ['faithfulness_mean_drop', 'sufficiency_mean_retained', 'shap_vs_gradcam_rho']
].mean().round(4).to_string())

# Cross-model consistency: for each image, mean pairwise Spearman across the 3 models.
xmodel_rows = []
for id_code, per_model in shap_cache_per_image.items():
    names = sorted(per_model.keys())
    pairs = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            pairs.append((names[i], names[j],
                          heatmap_consistency(per_model[names[i]], per_model[names[j]])))
    xmodel_rows.append({
        'id_code': id_code,
        **{f'rho_{a}_vs_{b}': r for a, b, r in pairs},
        'mean_pairwise_rho': float(np.nanmean([r for _, _, r in pairs])) if pairs else float('nan'),
    })
shap_xmodel_df = pd.DataFrame(xmodel_rows)
shap_xmodel_df.to_csv(P3_METRICS / 'shap_cross_model_consistency.csv', index=False)
print('\nMean cross-model SHAP consistency (Spearman rho):',
      float(shap_xmodel_df['mean_pairwise_rho'].mean()))
'''


SHAP_VIZ = r'''# === V2 PATCH (Step 2d): SHAP comparison visualisation ===
# 10 demo images x clean + 3 degradation-high conditions; each panel shows
# original | gradcam | shap | ig/attention for one model. Saved to
# Drive/Thesis/results/phase3_xai_benchmark/samples/.
import matplotlib.pyplot as plt
from PIL import Image

P3_SAMPLES = P3 / 'samples' / 'shap_compare'
P3_SAMPLES.mkdir(parents=True, exist_ok=True)

DEMO_IDS = list(EXPLAIN_IDS)[:10]
CONDITIONS = [('clean', None, None),
              ('blur-high',     'blur',     'high'),
              ('exposure-high', 'exposure', 'high'),
              ('noise-high',    'noise',    'high')]


def overlay(img_pil, heat, size=384, alpha=0.45):
    arr = np.array(img_pil.resize((size, size))).astype(np.float32) / 255
    heat_resized = cv2.resize(heat.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
    out = (1 - alpha) * arr + alpha * plt.get_cmap('jet')(heat_resized)[..., :3]
    return np.clip(out, 0, 1)


def _get_image_tensor(id_code, kind, level):
    """Load (PIL, tensor) for clean (kind=None) or for a specific degradation slice."""
    if kind is None:
        path = resolve_image(id_code)
    else:
        path = DEGRADED_DIR / kind / level / f'{id_code}{DEG_SAVE_EXT}'
        if not path.exists():
            return None, None
    pil = Image.open(path).convert('RGB')
    return pil, load_tensor(path)


for id_code in DEMO_IDS:
    if id_code not in labels_df.index:
        continue
    label = int(labels_df[id_code])
    fig, axes = plt.subplots(len(MODEL_NAMES) * len(CONDITIONS), 4,
                              figsize=(13, 3.2 * len(MODEL_NAMES) * len(CONDITIONS)))
    if axes.ndim == 1:
        axes = axes[None, :]
    row = 0
    for cond_tag, kind, level in CONDITIONS:
        img_pil, x = _get_image_tensor(id_code, kind, level)
        if img_pil is None:
            continue
        for name in MODEL_NAMES:
            # Grad-CAM (only for CNNs) or attention (for ViT)
            method_alt = 'gradcam' if name in METHOD_REGISTRY['gradcam']['models'] else 'attention'
            fn_alt = METHOD_REGISTRY[method_alt]['fn']
            try:
                h_alt = fn_alt(models[name], x, target_class=label)
            except Exception:
                h_alt = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
            # SHAP (all models)
            try:
                h_shap = METHOD_REGISTRY['shap']['fn'](models[name], x, target_class=label)
            except Exception:
                h_shap = np.zeros_like(h_alt)
            # IG (all models, only if registered)
            ig_fn = METHOD_REGISTRY.get('ig', {}).get('fn')
            try:
                h_ig = ig_fn(models[name], x, target_class=label) if ig_fn is not None else np.zeros_like(h_alt)
            except Exception:
                h_ig = np.zeros_like(h_alt)
            axes[row, 0].imshow(img_pil.resize((384, 384)))
            axes[row, 0].set_title(f'{name}\n{cond_tag}', fontsize=8)
            axes[row, 1].imshow(overlay(img_pil, h_alt))
            axes[row, 1].set_title(f'{method_alt}', fontsize=8)
            axes[row, 2].imshow(overlay(img_pil, h_shap))
            axes[row, 2].set_title('SHAP', fontsize=8)
            axes[row, 3].imshow(overlay(img_pil, h_ig))
            axes[row, 3].set_title('IG', fontsize=8)
            for ax in axes[row]:
                ax.axis('off')
            row += 1
    fig.suptitle(f'SHAP comparison — id={id_code}, label={label}', fontsize=12, y=1.0)
    plt.tight_layout()
    out = P3_SAMPLES / f'shap_compare_{id_code}.png'
    plt.savefig(out, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'  Saved: {out}')

# Summary table: model x method x condition stability/insertion_auc/faithfulness
# (relies on xai_df produced by cell 53 + shap_quality_df above).
print('\n=== Phase 3 final summary table ===')
xai_summary = (xai_df.groupby(['model', 'method', 'degradation'])
                      [['stability', 'insertion_auc']].mean().round(4)
                      if 'xai_df' in globals() else None)
if xai_summary is not None:
    summary_csv = P3 / 'metrics' / 'phase3_final_summary.csv'
    xai_summary.to_csv(summary_csv)
    print(f'Saved: {summary_csv}')
    print(xai_summary.head(20).to_string())
'''


# ===== STEP 3: CYCLEGAN =====

CYCLEGAN_MD = '''## V2 — Step 3 — CycleGAN-CBAM (Phase 4 GAN #2)

Professor deliverable #2a. The original Phase 4 had only one GAN-based restorer (A-ESRGAN).
This adds CycleGAN with CBAM attention as the second GAN-based approach. CycleGAN is well
suited because it works with the unpaired degraded↔clean translation framing — we can use
arbitrary degraded fundus images as domain A and pristine APTOS as domain B, even when
specific (clean, degraded) pairs are not perfectly aligned.
'''

CYCLEGAN_IMPL = r'''# === V2 PATCH (Step 3): CycleGAN-CBAM ===
import itertools
import torch.nn as nn

CYCLEGAN_CKPT  = CHECKPOINTS_DIR / 'cyclegan_v1.pt'
CYCLEGAN_SIZE  = 256


# ---- Residual block ----
class _ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.InstanceNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.InstanceNorm2d(ch),
        )
    def forward(self, x):
        return x + self.block(x)


# ---- CBAM ----
class _CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al. 2018). Helps preserve
    fine retinal texture by attending to both channel and spatial dimensions.
    """
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, max(ch // reduction, 1)),
            nn.ReLU(inplace=True),
            nn.Linear(max(ch // reduction, 1), ch),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )
    def forward(self, x):
        B, C, H, W = x.shape
        avg_out = self.fc(self.avg_pool(x).view(B, C)).view(B, C, 1, 1)
        max_out = self.fc(self.max_pool(x).view(B, C)).view(B, C, 1, 1)
        x = x * torch.sigmoid(avg_out + max_out)
        sa_in = torch.cat([x.mean(dim=1, keepdim=True),
                           x.max(dim=1, keepdim=True)[0]], dim=1)
        return x * self.spatial(sa_in)


# ---- Generator ----
class _CycleGenerator(nn.Module):
    """Encoder + 9 residual blocks (each followed by CBAM) + decoder."""
    def __init__(self, in_ch=3, out_ch=3, ngf=64, n_blocks=9):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, ngf, 7, padding=3),
            nn.InstanceNorm2d(ngf), nn.ReLU(True),
            nn.Conv2d(ngf, ngf*2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf*2), nn.ReLU(True),
            nn.Conv2d(ngf*2, ngf*4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf*4), nn.ReLU(True),
        )
        blocks = []
        for _ in range(n_blocks):
            blocks.append(_ResidualBlock(ngf*4))
            blocks.append(_CBAM(ngf*4))
        self.res_blocks = nn.Sequential(*blocks)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ngf*4, ngf*2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf), nn.ReLU(True),
            nn.Conv2d(ngf, out_ch, 7, padding=3),
            nn.Tanh(),
        )
    def forward(self, x):
        return self.decoder(self.res_blocks(self.encoder(x)))


# ---- PatchGAN discriminator (70x70) ----
class _CyclePatchDisc(nn.Module):
    def __init__(self, in_ch=3, ndf=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf*2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf*2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*2, ndf*4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf*4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*4, 1, 4, padding=1),
        )
    def forward(self, x):
        return self.model(x)


# ---- Paired dataset: domain A = degraded, domain B = clean APTOS ----
class _CycleGANDataset(Dataset):
    """Returns (A, B) where A is a degraded image (random kind+level) and B is
    a (different) clean image. Random pairing per epoch — CycleGAN doesn't
    require A and B to be the same identity image.
    """
    def __init__(self, df, n_per=4):
        self.df = df.reset_index(drop=True); self.n_per = n_per
        self.tfm = T.Compose([
            T.Resize((CYCLEGAN_SIZE, CYCLEGAN_SIZE)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),     # to [-1, 1] for Tanh
        ])
    def __len__(self): return len(self.df) * self.n_per
    def __getitem__(self, i):
        row_a = self.df.iloc[i // self.n_per]
        row_b = self.df.iloc[(i // self.n_per + len(self.df) // 2) % len(self.df)]
        def _load(row):
            try:
                return Image.open(resolve_image(row['id_code'])).convert('RGB')
            except FileNotFoundError:
                return Image.new('RGB', (CYCLEGAN_SIZE, CYCLEGAN_SIZE))
        img_a_clean = _load(row_a).resize((CYCLEGAN_SIZE, CYCLEGAN_SIZE), Image.BILINEAR)
        img_b_clean = _load(row_b).resize((CYCLEGAN_SIZE, CYCLEGAN_SIZE), Image.BILINEAR)
        kind  = random.choice(['blur', 'exposure', 'noise'])
        level = random.choice(['low', 'mid', 'high'])
        img_a_deg = DEGRADERS[kind](img_a_clean, DEGRADATION_PARAMS[kind][level])
        return self.tfm(img_a_deg), self.tfm(img_b_clean)


def _cyclegan_lr_lambda(epoch, total_epochs, decay_start):
    """Linear LR decay from epoch=decay_start down to 0 by total_epochs."""
    if epoch < decay_start:
        return 1.0
    return max(0.0, 1.0 - (epoch - decay_start) / max(total_epochs - decay_start, 1))


def train_cyclegan(epochs=50, batch_size=4, lr=2e-4,
                    lam_cycle=10.0, lam_identity=5.0,
                    decay_start=25):
    """Resume-guarded CycleGAN-CBAM training (~6-10 hr on A100)."""
    if CYCLEGAN_CKPT.exists():
        print(f"[skip] {CYCLEGAN_CKPT.name} already on Drive — not retraining.")
        return
    print(f"Training CycleGAN-CBAM ({epochs} epochs, batch={batch_size}, lr={lr})...")
    df = pd.read_csv(PRISTINE_CSV)
    ds = _CycleGANDataset(df, n_per=4)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    num_workers=2, pin_memory=True)
    G_AB = _CycleGenerator().to(DEVICE)   # degraded -> clean
    G_BA = _CycleGenerator().to(DEVICE)   # clean -> degraded
    D_A  = _CyclePatchDisc().to(DEVICE)
    D_B  = _CyclePatchDisc().to(DEVICE)
    opt_g = torch.optim.AdamW(itertools.chain(G_AB.parameters(), G_BA.parameters()),
                               lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.AdamW(itertools.chain(D_A.parameters(),  D_B.parameters()),
                               lr=lr, betas=(0.5, 0.999))
    sched_g = torch.optim.lr_scheduler.LambdaLR(opt_g,
        lr_lambda=lambda e: _cyclegan_lr_lambda(e, epochs, decay_start))
    sched_d = torch.optim.lr_scheduler.LambdaLR(opt_d,
        lr_lambda=lambda e: _cyclegan_lr_lambda(e, epochs, decay_start))
    mse, l1 = nn.MSELoss(), nn.L1Loss()
    for ep in range(epochs):
        G_AB.train(); G_BA.train(); D_A.train(); D_B.train()
        run_g, run_da, run_db, n = 0.0, 0.0, 0.0, 0
        pbar = tqdm(dl, desc=f"cyclegan ep {ep+1}/{epochs}", leave=False)
        for real_a, real_b in pbar:
            real_a = real_a.to(DEVICE, non_blocking=True)
            real_b = real_b.to(DEVICE, non_blocking=True)
            # ----- Generator pass -----
            opt_g.zero_grad(set_to_none=True)
            fake_b = G_AB(real_a)
            fake_a = G_BA(real_b)
            rec_a  = G_BA(fake_b)
            rec_b  = G_AB(fake_a)
            idt_a  = G_BA(real_a)
            idt_b  = G_AB(real_b)
            loss_g_adv = (mse(D_B(fake_b), torch.ones_like(D_B(fake_b))) +
                          mse(D_A(fake_a), torch.ones_like(D_A(fake_a))))
            loss_cycle = lam_cycle * (l1(rec_a, real_a) + l1(rec_b, real_b))
            loss_idt   = lam_identity * (l1(idt_a, real_a) + l1(idt_b, real_b))
            loss_g     = loss_g_adv + loss_cycle + loss_idt
            loss_g.backward()
            opt_g.step()
            # ----- Discriminator pass -----
            opt_d.zero_grad(set_to_none=True)
            d_a_real = D_A(real_a); d_a_fake = D_A(fake_a.detach())
            d_b_real = D_B(real_b); d_b_fake = D_B(fake_b.detach())
            loss_da = 0.5 * (mse(d_a_real, torch.ones_like(d_a_real)) +
                             mse(d_a_fake, torch.zeros_like(d_a_fake)))
            loss_db = 0.5 * (mse(d_b_real, torch.ones_like(d_b_real)) +
                             mse(d_b_fake, torch.zeros_like(d_b_fake)))
            (loss_da + loss_db).backward()
            opt_d.step()
            run_g  += loss_g.item()  * real_a.size(0)
            run_da += loss_da.item() * real_a.size(0)
            run_db += loss_db.item() * real_a.size(0)
            n      += real_a.size(0)
            pbar.set_postfix(G=f"{run_g/n:.4f}",
                             DA=f"{run_da/n:.4f}",
                             DB=f"{run_db/n:.4f}")
        sched_g.step(); sched_d.step()
        print(f"  ep {ep+1}: G={run_g/n:.4f}  D_A={run_da/n:.4f}  D_B={run_db/n:.4f}  lr={opt_g.param_groups[0]['lr']:.2e}")
    torch.save({'G_AB': G_AB.state_dict(),
                'G_BA': G_BA.state_dict()}, CYCLEGAN_CKPT)
    print(f"Saved -> {CYCLEGAN_CKPT}")


train_cyclegan(epochs=50, batch_size=4, lr=2e-4, decay_start=25)


# ---- Inference ----
_cyclegan_G_AB = None
def _load_cyclegan():
    global _cyclegan_G_AB
    if _cyclegan_G_AB is None:
        _cyclegan_G_AB = _CycleGenerator().to(DEVICE).eval()
        sd = torch.load(CYCLEGAN_CKPT, map_location=DEVICE, weights_only=False)
        _cyclegan_G_AB.load_state_dict(sd['G_AB'])
    return _cyclegan_G_AB


@torch.no_grad()
def enhance_cyclegan(img_pil):
    """Restore a degraded fundus image using G_AB (degraded -> clean)."""
    G = _load_cyclegan()
    orig = img_pil.size
    src  = img_pil.convert('RGB').resize((CYCLEGAN_SIZE, CYCLEGAN_SIZE), Image.BILINEAR)
    tfm  = T.Compose([T.ToTensor(),
                      T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    x = tfm(src).unsqueeze(0).to(DEVICE)
    with autocast(enabled=True):
        y = G(x).clamp(-1, 1)
    arr = (y.squeeze().permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
    arr = (arr * 255).clip(0, 255).astype('uint8')
    return Image.fromarray(arr).resize(orig)


print("CycleGAN-CBAM restorer ready.")
'''


# ===== STEP 4: DDPM =====

DDPM_MD = '''## V2 — Step 4 — Conditional DDPM (vanilla diffusion, Phase 4 #2)

Professor deliverable #2b. The existing diffusion restorer (Cold Diffusion) uses a
non-standard forward process — the Phase 1 degradation operators as the "noising"
operator. The professor specifically asked for a vanilla DDPM-style model (Gaussian
noise forward process). This implements a conditional DDPM where the denoising
U-Net is conditioned on the degraded image (concatenated as extra channels) so the
network learns to recover the clean image from pure noise *given* the degraded view.
'''

DDPM_IMPL = r'''# === V2 PATCH (Step 4): conditional vanilla DDPM ===
import math
import torch.nn as nn
import torch.nn.functional as F

DDPM_CKPT = CHECKPOINTS_DIR / 'ddpm_fundus_v1.pt'
DDPM_SIZE = 256


class _SinusoidalPE(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000.0) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class _DDPMUNet(nn.Module):
    """U-Net conditioned on (timestep, degraded image).
    Input  : concat(noisy_clean, degraded) -> 6 channels.
    Output : predicted noise -> 3 channels.
    """
    def __init__(self, in_ch=6, out_ch=3, base_ch=64, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            _SinusoidalPE(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        def _block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.GroupNorm(8, out_c),
                nn.GELU(),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.GroupNorm(8, out_c),
                nn.GELU(),
            )
        self.enc1 = _block(in_ch, base_ch)
        self.enc2 = _block(base_ch, base_ch * 2)
        self.enc3 = _block(base_ch * 2, base_ch * 4)
        self.mid  = _block(base_ch * 4, base_ch * 4)
        self.film = nn.Linear(time_dim, base_ch * 4 * 2)
        self.dec3 = _block(base_ch * 8, base_ch * 2)
        self.dec2 = _block(base_ch * 4, base_ch)
        self.dec1 = _block(base_ch * 2, base_ch)
        self.final = nn.Conv2d(base_ch, out_ch, 1)
        self.pool = nn.MaxPool2d(2)
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        m  = self.mid(self.pool(e3))
        scale, shift = self.film(t_emb).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        m = m * (1 + scale) + shift
        d3 = self.dec3(torch.cat([self.up(m),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        return self.final(d1)


class FundusDDPM:
    """Standard DDPM (Ho et al. 2020) conditioned on the degraded image."""
    def __init__(self, model, T=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.model = model
        self.T = T
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_acp   = self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_omacp = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_acp * x0 + sqrt_omacp * noise, noise

    def train_step(self, clean, degraded, optimizer, scaler=None):
        B = clean.size(0)
        t = torch.randint(0, self.T, (B,), device=self.device)
        noise = torch.randn_like(clean)
        noisy_clean, _ = self.q_sample(clean, t, noise)
        inp = torch.cat([noisy_clean, degraded], dim=1)   # (B, 6, H, W)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with autocast(enabled=True):
                pred = self.model(inp, t)
                loss = F.mse_loss(pred, noise)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
        else:
            pred = self.model(inp, t)
            loss = F.mse_loss(pred, noise)
            loss.backward(); optimizer.step()
        return loss.item()

    @torch.no_grad()
    def restore(self, degraded, n_steps=50):
        B = degraded.size(0)
        x = torch.randn(B, 3, degraded.size(2), degraded.size(3), device=self.device)
        step_size = max(self.T // n_steps, 1)
        timesteps = list(range(self.T - 1, 0, -step_size))
        for t_val in timesteps:
            t = torch.full((B,), t_val, device=self.device, dtype=torch.long)
            inp = torch.cat([x, degraded], dim=1)
            pred_noise = self.model(inp, t)
            alpha = self.alphas[t_val]
            alpha_cumprod = self.alpha_cumprod[t_val]
            beta = self.betas[t_val]
            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_cumprod)) * pred_noise
            )
            if t_val > 1:
                x = x + torch.sqrt(beta) * torch.randn_like(x)
        return x.clamp(-1, 1)


class _DDPMDataset(Dataset):
    def __init__(self, df, n_per=4):
        self.df = df.reset_index(drop=True); self.n_per = n_per
        self.tfm = T.Compose([T.Resize((DDPM_SIZE, DDPM_SIZE)),
                              T.ToTensor(),
                              T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    def __len__(self): return len(self.df) * self.n_per
    def __getitem__(self, i):
        row = self.df.iloc[i // self.n_per]
        try:
            img = Image.open(resolve_image(row['id_code'])).convert('RGB')
        except FileNotFoundError:
            img = Image.new('RGB', (DDPM_SIZE, DDPM_SIZE))
        img = img.resize((DDPM_SIZE, DDPM_SIZE), Image.BILINEAR)
        kind  = random.choice(['blur', 'exposure', 'noise'])
        level = random.choice(['low', 'mid', 'high'])
        deg   = DEGRADERS[kind](img, DEGRADATION_PARAMS[kind][level])
        return self.tfm(deg), self.tfm(img)


def train_ddpm(epochs=15, batch_size=8, lr=1e-4, T=1000):
    """Resume-guarded conditional DDPM training (~3-6 hr on A100)."""
    if DDPM_CKPT.exists():
        print(f"[skip] {DDPM_CKPT.name} already on Drive — not retraining.")
        return
    print(f"Training conditional DDPM ({epochs} epochs, T={T}, batch={batch_size}, lr={lr})...")
    df  = pd.read_csv(PRISTINE_CSV)
    ds  = _DDPMDataset(df, n_per=4)
    dl  = DataLoader(ds, batch_size=batch_size, shuffle=True,
                     num_workers=2, pin_memory=True)
    net = _DDPMUNet().to(DEVICE)
    ddpm = FundusDDPM(net, T=T, device=DEVICE)
    opt  = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    scaler = GradScaler(enabled=True)
    for ep in range(epochs):
        net.train()
        running, n = 0.0, 0
        pbar = tqdm(dl, desc=f"ddpm ep {ep+1}/{epochs}", leave=False)
        for deg, clean in pbar:
            deg   = deg.to(DEVICE, non_blocking=True)
            clean = clean.to(DEVICE, non_blocking=True)
            loss = ddpm.train_step(clean, deg, opt, scaler=scaler)
            running += loss * deg.size(0); n += deg.size(0)
            pbar.set_postfix(MSE=f"{running/n:.4f}")
        print(f"  ep {ep+1}: MSE={running/n:.4f}")
    torch.save({'state_dict': net.state_dict(), 'T': T}, DDPM_CKPT)
    print(f"Saved -> {DDPM_CKPT}")


train_ddpm(epochs=15, batch_size=8, lr=1e-4, T=1000)


# ---- Inference ----
_ddpm_net = None
_ddpm     = None
def _load_ddpm(T=1000):
    global _ddpm_net, _ddpm
    if _ddpm is not None:
        return _ddpm
    _ddpm_net = _DDPMUNet().to(DEVICE).eval()
    sd = torch.load(DDPM_CKPT, map_location=DEVICE, weights_only=False)
    _ddpm_net.load_state_dict(sd['state_dict'])
    _ddpm = FundusDDPM(_ddpm_net, T=sd.get('T', T), device=DEVICE)
    return _ddpm


@torch.no_grad()
def enhance_ddpm(img_pil, n_steps=50):
    """Restore via conditional DDPM with DDIM-style fast sampling."""
    ddpm = _load_ddpm()
    orig = img_pil.size
    src  = img_pil.convert('RGB').resize((DDPM_SIZE, DDPM_SIZE), Image.BILINEAR)
    tfm  = T.Compose([T.ToTensor(),
                      T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    x = tfm(src).unsqueeze(0).to(DEVICE)
    y = ddpm.restore(x, n_steps=n_steps)
    arr = (y.squeeze().permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
    arr = (arr * 255).clip(0, 255).astype('uint8')
    return Image.fromarray(arr).resize(orig)


print("Conditional DDPM restorer ready.")
'''


# ===== STEP 1D: QUALITY CLASSIFIER CALIBRATION =====

Q_CALIBRATION_MD = '''## V2 — Step 1d — Quality classifier threshold calibration

The Phase 5 quality classifier (`Q_CKPT`) over-rejects on clean APTOS — Counter from cell 87
showed almost every clean test image labelled `reject`. Argmax routing on a class-imbalanced
output is the cause. Below we sweep thresholds on the softmax outputs over clean APTOS and pick
the operating point where the routing distribution is closest to the target ~60:25:15 mix of
good/usable/reject. Then we replace `predict_quality()` to use thresholds instead of argmax.
'''


Q_CALIBRATION_IMPL = r'''# === V2 PATCH (Step 1d): threshold calibration for the quality classifier ===
# Sweep softmax probability thresholds on clean APTOS until we get ~60:25:15.

from collections import Counter
import numpy as np

P5 = PHASE_DIRS['phase5_quality_ensemble']
P5_METRICS = P5 / 'metrics'
P5_METRICS.mkdir(parents=True, exist_ok=True)

# 1) Score clean APTOS test images
test_ids_q = pd.read_csv(P2 / 'metrics' / 'test_ids.csv')['id_code'].astype(str).tolist()
probs_rows = []
for sid in tqdm(test_ids_q, desc='quality probs'):
    try:
        img = Image.open(resolve_image(sid)).convert('RGB')
    except FileNotFoundError:
        continue
    x = TFM_EVAL(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        p = torch.softmax(qmodel(x), 1)[0].cpu().numpy()
    probs_rows.append({'id_code': sid, 'p_good': p[0], 'p_usable': p[1], 'p_reject': p[2]})

probs_df = pd.DataFrame(probs_rows)
probs_df.to_csv(P5_METRICS / 'quality_clean_probs.csv', index=False)
print(f'Scored {len(probs_df)} clean APTOS test images.')

# 2) Search thresholds: minimise |observed - target| across the routing distribution.
TARGET = {'good': 0.60, 'usable': 0.25, 'reject': 0.15}

def _route_under(p_good_thr, p_usable_thr, df):
    """If p_good > p_good_thr -> good. Else if p_usable > p_usable_thr -> usable. Else reject."""
    labels = []
    for _, r in df.iterrows():
        if r['p_good']   >= p_good_thr:                              labels.append('good')
        elif r['p_usable'] >= p_usable_thr:                           labels.append('usable')
        else:                                                          labels.append('reject')
    return labels


def _route_distance(labels):
    cnt = Counter(labels)
    total = sum(cnt.values())
    obs = {k: cnt.get(k, 0) / total for k in TARGET}
    return sum((obs[k] - TARGET[k])**2 for k in TARGET), obs


grid_good   = np.linspace(0.20, 0.80, 31)
grid_usable = np.linspace(0.10, 0.60, 26)
best = (float('inf'), None, None, None)
for tg in grid_good:
    for tu in grid_usable:
        lbls = _route_under(tg, tu, probs_df)
        d, obs = _route_distance(lbls)
        if d < best[0]:
            best = (d, tg, tu, obs)

print(f'Best thresholds: p_good>={best[1]:.3f}, p_usable>={best[2]:.3f}  (sq-dist={best[0]:.4f})')
print(f'Resulting distribution: {best[3]}')

QUALITY_THRESHOLDS = {'good': float(best[1]), 'usable': float(best[2])}
with open(P5_METRICS / 'quality_thresholds.json', 'w') as f:
    json.dump(QUALITY_THRESHOLDS, f, indent=2)

# 3) Replace predict_quality() to use thresholds.
def predict_quality(img):
    """V2 patch: threshold-based routing instead of argmax.

    Returns one of 'good' | 'usable' | 'reject'. Tuned on clean APTOS so the
    routing distribution sits near the 60:25:15 target documented in §1d
    of the master prompt.
    """
    if qmodel is None: return 'good'
    x = TFM_EVAL(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        p = torch.softmax(qmodel(x), 1)[0].cpu().numpy()
    if p[0] >= QUALITY_THRESHOLDS['good']:
        return 'good'
    if p[1] >= QUALITY_THRESHOLDS['usable']:
        return 'usable'
    return 'reject'

# Sanity check on clean APTOS
sanity = Counter(predict_quality(Image.open(resolve_image(sid)).convert('RGB'))
                 for sid in test_ids_q[:100] if Path(resolve_image(sid)).exists())
print(f'Sanity check (100 clean APTOS images): {dict(sanity)}')
print('Phase 5 quality classifier calibrated.')
'''


# ===== STEP 5+6: DIAGNOSTICS =====

DIAGNOSTICS_MD = '''## V2 — Step 5 + 6 — Per-class diagnostics + failure-mode analysis

Concluding diagnostics for the dissertation chapter:

1. Per-class precision / recall / F1 for the V4 ensemble on the clean test set and on each
   high-severity degradation slice. Plus a confusion matrix figure per slice.
2. Failure-mode table: per-image `raw_pred` vs `enhanced_pred` grouped by `true_class`, showing
   which classes benefit most from restoration and which (if any) are harmed.
'''


DIAGNOSTICS_IMPL = r'''# === V2 PATCH (Step 5+6): per-class diagnostics + failure-mode analysis ===
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

TARGET_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']

# Use the V4 ensemble predictions cached during Phase 2 (cell 41).
ens_csv = P2 / 'metrics' / 'v4' / 'v4_ensemble_predictions.csv'
if not ens_csv.exists():
    # Fallback: re-evaluate just the clean test set so this cell is still useful.
    print(f'[note] {ens_csv} not found — running clean V4 ensemble eval inline')
    _tl = globals().get('test_loader_v3') or globals().get('test_loader')
    if _tl is None:
        print('[skip] no test loader in memory — run Phase 2 cells first')
        ens_df_v4 = pd.DataFrame()
    else:
        rows = []
        for batch in tqdm(_tl, desc='V4 clean recompute'):
            if len(batch) == 3:
                x, y, ids = batch
            else:
                x, y = batch[0], batch[1]
                ids = [f'idx_{i}' for i in range(len(y))]
            x = x.to(DEVICE, non_blocking=True)
            probs = None
            for name in MODEL_NAMES:
                mdl = load_classifier_v3(name)
                with torch.no_grad():
                    p = torch.softmax(mdl(x), 1)
                probs = p if probs is None else probs + p
                del mdl; torch.cuda.empty_cache()
            probs = probs / len(MODEL_NAMES)
            pred = probs.argmax(1).cpu().numpy()
            for yi, pi, ci in zip(y.numpy(), pred, ids):
                rows.append({'id_code': ci, 'true_label': int(yi), 'pred': int(pi), 'condition': 'clean'})
        ens_df_v4 = pd.DataFrame(rows)
else:
    ens_df_v4 = pd.read_csv(ens_csv)

print('V4 ensemble predictions:', ens_df_v4.shape)

# --- 1. Per-class classification report (clean) ---
clean = ens_df_v4[ens_df_v4.get('condition', 'clean') == 'clean']
if len(clean) > 0:
    print('\n=== V4 ensemble — clean test set ===')
    print(classification_report(clean['true_label'], clean['pred'],
                                target_names=TARGET_NAMES, zero_division=0))
    cm = confusion_matrix(clean['true_label'], clean['pred'], labels=list(range(5)))
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES, cmap='Blues', ax=ax)
    ax.set_title('V4 Ensemble — Clean Test Set')
    ax.set_ylabel('True Label'); ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    cm_out = P2 / 'plots' / 'v4_confusion_clean.png'
    plt.savefig(cm_out, dpi=160, bbox_inches='tight'); plt.show()
    print(f'Saved: {cm_out}')

# --- 2. Failure-mode analysis: raw vs enhanced per true class (high severity only) ---
if 'rec_df' in globals():
    print('\n=== Failure-mode: per-class accuracy delta with each enhancer (high severity) ===')
    failure_rows = []
    for variant in VARIANTS:
        for k in DEGRADATION_TYPES:
            # Only high severity to keep runtime reasonable.
            l = 'high'
            root = DEGRADED_DIR / k / l if variant == 'raw' else ENHANCED_DIR / variant / k / l
            if not (root / 'manifest.csv').exists():
                continue
            ds = FolderDataset(root, transform=TFM_EVAL)
            ds.df = ds.df[ds.df['id_code'].astype(str).isin(test_id_set)].reset_index(drop=True)
            dl = DataLoader(ds, batch_size=32, num_workers=2, pin_memory=True)
            preds_acc = {n: [] for n in MODEL_NAMES}
            ys_all, ids_all = [], []
            for n in MODEL_NAMES:
                mdl = load_classifier_v3(n) if 'load_classifier_v3' in globals() else load_classifier(n)
                for x, y, ids in dl:
                    with torch.no_grad():
                        p = torch.softmax(mdl(x.to(DEVICE)), 1)
                    preds_acc[n].append(p.cpu().numpy())
                    if n == MODEL_NAMES[0]:
                        ys_all.append(y.numpy()); ids_all.extend(ids)
                del mdl; torch.cuda.empty_cache()
            ys_all = np.concatenate(ys_all)
            avgp   = sum(np.concatenate(preds_acc[n]) for n in MODEL_NAMES) / len(MODEL_NAMES)
            pred   = avgp.argmax(1)
            for ci, yi, pi in zip(ids_all, ys_all, pred):
                failure_rows.append({'id_code': ci, 'true_label': int(yi),
                                     'pred': int(pi), 'variant': variant,
                                     'degradation': k, 'level': l,
                                     'correct': int(pi == yi)})
    failure_df = pd.DataFrame(failure_rows)
    failure_df.to_csv(P4 / 'metrics' / 'failure_mode_table.csv', index=False)
    print('\nPer-(true_label, variant) accuracy at high severity (avg across kinds):')
    print(failure_df.groupby(['true_label', 'variant'])['correct'].mean().unstack().round(3).to_string())
    print('\nPer-(degradation, variant) accuracy:')
    print(failure_df.groupby(['degradation', 'variant'])['correct'].mean().unstack().round(3).to_string())
else:
    print('[skip] rec_df not in memory — run cell 72 first')
'''


# ---------------------------------------------------------------------------
# Build the v2 notebook
# ---------------------------------------------------------------------------

def main():
    print('Loading v1 notebook...')
    with SRC.open('r', encoding='utf-8') as f:
        nb = json.load(f)
    cells = nb['cells']
    print(f'  cells in v1: {len(cells)}')

    # Strip outputs from EVERY existing code cell so the v2 starts clean.
    # (Outputs from v1 are stale anyway once we change upstream cells.)
    for c in cells:
        if c.get('cell_type') == 'code':
            c['outputs'] = []
            c['execution_count'] = None

    # ----- In-place replacements -----
    print('Applying in-place replacements...')
    replace_cell(cells, 17, CELL_17_NEW)
    replace_cell(cells, 39, CELL_39_NEW)
    replace_cell(cells, 53, CELL_53_NEW)
    # Cell 61 used to rename 'shap' -> 'IG' in the saved CSVs because the V4
    # patch had repurposed the 'shap' key for IG. In V2 we have the real SHAP
    # under 'shap' and IG under its own 'ig' key, so renaming would corrupt
    # the data. Neutralise the cell.
    replace_cell(cells, 61, CELL_61_NEW)
    replace_cell(cells, 67, CELL_67_NEW)
    replace_cell(cells, 70, CELL_70_NEW)
    replace_cell(cells, 71, CELL_71_NEW)
    replace_cell(cells, 72, CELL_72_NEW)

    # ----- Insertions -----
    # Build new cells. We insert from highest index downward so earlier insert positions stay valid.

    # SHAP block — insert AFTER cell 52 (the smoke-check). Position 53 onward shifts down.
    shap_block = [md_cell(SHAP_MD_HEADER), code_cell(SHAP_IMPL),
                  code_cell(SHAP_QUALITY_METRICS), code_cell(SHAP_VIZ)]
    # CycleGAN — insert AFTER cell 70 (now patched).
    cyclegan_block = [md_cell(CYCLEGAN_MD), code_cell(CYCLEGAN_IMPL)]
    # DDPM — insert AFTER CycleGAN.
    ddpm_block = [md_cell(DDPM_MD), code_cell(DDPM_IMPL)]
    # Quality calibration — insert AFTER cell 79 (where qmodel is loaded).
    qcal_block = [md_cell(Q_CALIBRATION_MD), code_cell(Q_CALIBRATION_IMPL)]
    # Diagnostics — APPEND at the very end.
    diag_block = [md_cell(DIAGNOSTICS_MD), code_cell(DIAGNOSTICS_IMPL)]

    # IMPORTANT: insert in REVERSE order of position so earlier positions remain stable.
    # Order of inserts (lowest -> highest original-index position):
    #   after 52  -> SHAP block
    #   after 70  -> CycleGAN
    #   then     -> DDPM (right after CycleGAN)
    #   after 79 (original) -> quality cal
    #   end       -> diagnostics

    # We process from end -> start to keep earlier insertion points stable.
    print('Applying inserts (high -> low to keep indices stable)...')
    # 1) Append diagnostics at end
    cells.extend(diag_block)
    # 2) Insert quality calibration AFTER original index 81 (the predict_quality + pipeline cell).
    #    Inserting before cell 81 would let it overwrite our threshold-based predict_quality.
    insert_at_q = 82
    cells[insert_at_q:insert_at_q] = qcal_block
    # 3) Insert DDPM after original index 70 (= insert at 71). CycleGAN comes first, then DDPM.
    insert_at_p4 = 71
    cells[insert_at_p4:insert_at_p4] = cyclegan_block + ddpm_block
    # 4) Insert SHAP block after original index 52 (= insert at 53)
    insert_at_shap = 53
    cells[insert_at_shap:insert_at_shap] = shap_block

    print(f'  cells in v2: {len(cells)}')

    # Update the top-level V4 PATCH MARKER markdown to reflect v2.
    if cells[0].get('cell_type') == 'markdown':
        cells[0]['source'] = [
            '# V2 PATCH MARKER (do not delete)\n',
            '\n',
            'This is the V2-extended notebook built by `build_v2.py`. It applies:\n',
            '\n',
            '- Step 1a–1e: engineering fixes (distilled ckpts, EyeQ phantom, Cold Diffusion noise, Q_CKPT calibration, SwinIR-GAN adv drift)\n',
            '- Step 2: real SHAP (`shap.GradientExplainer`) across CNNs + ViT, plus faithfulness / sufficiency / cross-model consistency metrics\n',
            '- Step 3: CycleGAN-CBAM (Phase 4 GAN #2)\n',
            '- Step 4: vanilla DDPM conditional on degraded input (Phase 4 Diffusion #2)\n',
            '- Step 5+6: per-class precision/recall + confusion matrices, failure-mode analysis\n',
            '\n',
            'All new training cells include resume guards. Drive checkpoints with bumped names:\n',
            '  - `cold_diffusion_v6.pt`\n',
            '  - `swinir_gan_v6.pt`\n',
            '  - `cyclegan_v1.pt`\n',
            '  - `ddpm_fundus_v1.pt`\n',
        ]

    nb['cells'] = cells

    # Validate JSON serialises before writing the real file.
    test_json = json.dumps(nb)
    print(f'  serialised size: {len(test_json) / 1024:.1f} KB')

    print(f'Writing v2 -> {DST}')
    with DST.open('w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print('Done.')


if __name__ == '__main__':
    main()
