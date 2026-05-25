"""Apply V4 bug-fixes + Cold Diffusion patch to Thesis_optimized_final (1).ipynb.

Edits (in place — preserves all other cells, outputs, metadata):
  - Cell 39: load_classifier_v3 prefers _v3_distilled.pt if present
  - Cell 41: V4 ensemble weights softmax by per-model val kappa
  - Cell 66: markdown header switched from SD img2img -> Cold Diffusion
  - Cell 67: SD img2img code replaced with Cold Diffusion implementation
  - Cell 68: build_enhanced takes (img, kind, level); cold_diff added
  - Cell 69: ENHANCERS includes 'cold_diff'; variants loop is dynamic
  - Cell 70: recovery plot iterates ENHANCERS dynamically
  - Cell 72: XAI recovery plot iterates ENHANCERS; qualitative grid widens
"""
import json
import shutil
from pathlib import Path

NB_PATH = Path("C:/Dissertation/Thesis_optimized_final (1).ipynb")
BACKUP  = NB_PATH.with_suffix(".ipynb.bak")


# ---------------- replacement source code for each target cell ----------------

CELL_39 = '''def load_classifier_v3(name):
    """Load a v3 multi-scale classifier from disk.

    Prefers the distilled checkpoint (`{name}_v3_distilled.pt`) when it
    exists so the V4 KD pass actually feeds the downstream ensemble.
    Falls back to `{name}_v3_best.pt`.
    """
    m = build_model_v3(name, pretrained=False).to(DEVICE)
    distilled = CKPT_DIR_V3 / f"{name}_v3_distilled.pt"
    best      = CKPT_DIR_V3 / f"{name}_v3_best.pt"
    src = distilled if distilled.exists() else best
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
        print(f"\\n--- {name} v3 (TTA) ---")
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


CELL_41 = '''# === V4 PATCH: 8-view TTA + 3-model soft-vote ensemble (val-kappa weighted) ===
# Equal-weight voting let the weakest backbone drag down the ensemble — now each
# model's softmax is weighted by its checkpoint's val-kappa, so a strong model
# dominates on the slices it's best at.

@torch.no_grad()
def evaluate_v3_tta8(model, loader):
    """8-view test-time augmentation: identity, 3 flips, 3 rotations, flip+rot."""
    model.eval(); ys, probs = [], []
    for x, y, _ in loader:
        x = x.to(DEVICE, non_blocking=True)
        views = [
            x,
            torch.flip(x, [3]),
            torch.flip(x, [2]),
            torch.flip(x, [2, 3]),
            torch.rot90(x, 1, [2, 3]),
            torch.rot90(x, 2, [2, 3]),
            torch.rot90(x, 3, [2, 3]),
            torch.flip(torch.rot90(x, 1, [2, 3]), [3]),
        ]
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


def _read_val_kappa(name):
    """Return the val kappa saved in the model's checkpoint (distilled preferred)."""
    distilled = CKPT_DIR_V3 / f"{name}_v3_distilled.pt"
    best      = CKPT_DIR_V3 / f"{name}_v3_best.pt"
    src = distilled if distilled.exists() else best
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    return float(ckpt.get("val", {}).get("kappa_qw", 0.5))


@torch.no_grad()
def evaluate_v3_ensemble(loader, use_tta8=True):
    """Val-kappa-weighted soft-vote across all v3 backbones."""
    models  = {n: load_classifier_v3(n) for n in MODEL_NAMES}
    kappas  = {n: _read_val_kappa(n) for n in MODEL_NAMES}
    # Clamp negative/zero kappas, then renormalise
    weights = {n: max(0.05, kappas[n]) for n in MODEL_NAMES}
    total_w = sum(weights.values())
    weights = {n: weights[n] / total_w for n in MODEL_NAMES}
    print(f"  ensemble weights: " + ", ".join(f"{n}={weights[n]:.2f}" for n in MODEL_NAMES))
    for m in models.values():
        m.eval()
    ys, probs = [], []
    for x, y, _ in loader:
        x = x.to(DEVICE, non_blocking=True)
        avg = None
        for name, mdl in models.items():
            if use_tta8:
                views = [
                    x, torch.flip(x, [3]), torch.flip(x, [2]),
                    torch.flip(x, [2, 3]),
                    torch.rot90(x, 1, [2, 3]), torch.rot90(x, 2, [2, 3]),
                    torch.rot90(x, 3, [2, 3]),
                ]
                m_pred = sum(torch.softmax(mdl(v), 1) for v in views) / len(views)
            else:
                m_pred = torch.softmax(mdl(x), 1)
            contrib = m_pred * weights[name]
            avg = contrib if avg is None else avg + contrib
        probs.append(avg.cpu().numpy()); ys.append(y.numpy())
    for m in models.values(): del m
    torch.cuda.empty_cache()
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


# ----- Run the ensemble on clean + every degraded condition -----
rows_ens = []

print("=== Ensemble (8-TTA, val-kappa weighted) on clean ===")
m = evaluate_v3_ensemble(test_loader_v3, use_tta8=True)
rows_ens.append({"degradation": "clean", "level": "none", **m})
print(f"  acc={m['accuracy']:.4f}  f1={m['f1_macro']:.4f}  kappa={m['kappa_qw']:.4f}")

for k in DEGRADATION_TYPES:
    for l in DEGRADATION_LEVELS:
        m = evaluate_v3_ensemble(deg_loader_v3(k, l), use_tta8=True)
        rows_ens.append({"degradation": k, "level": l, **m})
        print(f"  {k}/{l}: acc={m['accuracy']:.4f}  f1={m['f1_macro']:.4f}  kappa={m['kappa_qw']:.4f}")

ens_df_v4 = pd.DataFrame(rows_ens)
ens_csv = P2 / "metrics" / "v3" / "ensemble_v4_results.csv"
ens_df_v4.to_csv(ens_csv, index=False)
print("Saved:", ens_csv)
ens_df_v4
'''


CELL_66_MD = '''## V5 - Cold Diffusion for fundus restoration (replaces SD img2img)

The earlier Stable-Diffusion-2.1 img2img variant was **removed**: SD was trained on LAION, not fundus, and even at `strength=0.30` it hallucinates anatomically-plausible-but-wrong vessels and can fabricate lesions the classifier then "detects".

This cell replaces it with **Cold Diffusion** (Bansal et al., 2022). Cold Diffusion swaps the Gaussian-noise forward process for an arbitrary deterministic degradation operator — and you happen to already have those operators in Phase 1 (`gaussian_blur`, `exposure_shift`, `gaussian_noise`). So the model is trained to invert your *exact* synthetic pipeline. Lesion-preserving by construction, fast (~90 min train, 8-step inference), and a clean novel-contribution story for the dissertation.
'''


CELL_67 = '''# === V5 PATCH: Cold Diffusion for fundus restoration ===
# Bansal et al. 2022 (https://arxiv.org/abs/2208.09392): the diffusion forward
# process can be any deterministic degradation D(x, t). Train R_theta so that
# R(D(x, t), t) ≈ x for all t, then sample via Algorithm 2:
#
#       x_{s-1} = x_s - D(R(x_s, s), s) + D(R(x_s, s), s-1)
#
# We plug in the EXACT Phase-1 degradation operators (blur / exposure / noise)
# as D, conditioning the U-Net on (kind, t/T) via FiLM.

import math, random

COLD_DIFF_CKPT = CHECKPOINTS_DIR / 'cold_diffusion_v5.pt'
COLD_DIFF_SIZE = 256        # work at 256 px to fit budget; resize back at end
T_STEPS        = 8          # train + inference steps

# Severity schedule: t in [0, T_STEPS] -> degradation parameter
SEV_HIGH = {'blur': 9.0, 'exposure_gain_low': 0.2, 'noise': 0.12}

def _severity(t, kind):
    f = t / T_STEPS  # in [0, 1]
    if kind == 'blur':
        return max(0.01, f * SEV_HIGH['blur'])
    if kind == 'exposure':
        # interpolate gain from 1.0 (clean) to 0.2 (high under-exposure)
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
        self.film = nn.Linear(t_emb*2, ch*4*2)   # gamma+beta over mid features
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
        m  = m * (1 + g[:, :, None, None]) + b[:, :, None, None]   # FiLM
        d2 = self.d2(torch.cat([self.up2(m),  e2], 1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
        return torch.sigmoid(self.out(d1))


KIND_TO_ID = {'blur': 0, 'exposure': 1, 'noise': 2}

class _ColdDiffDataset(Dataset):
    """On-the-fly paired (degraded@t, clean) sampler over the pristine set."""
    def __init__(self, df, n_per=4):
        self.df = df.reset_index(drop=True); self.n_per = n_per
        self.tfm = T.Compose([T.Resize((COLD_DIFF_SIZE, COLD_DIFF_SIZE)),
                              T.ToTensor()])
    def __len__(self):
        return len(self.df) * self.n_per
    def __getitem__(self, i):
        row = self.df.iloc[i // self.n_per]
        try:
            img = Image.open(resolve_image(row['id_code'])).convert('RGB')
        except FileNotFoundError:
            img = Image.new('RGB', (COLD_DIFF_SIZE, COLD_DIFF_SIZE))
        img  = img.resize((COLD_DIFF_SIZE, COLD_DIFF_SIZE), Image.BILINEAR)
        kind = random.choice(list(KIND_TO_ID.keys()))
        t    = random.randint(1, T_STEPS)
        deg  = _degrade_t(img, kind, t)
        return (self.tfm(deg), KIND_TO_ID[kind],
                torch.tensor(t / T_STEPS, dtype=torch.float32),
                self.tfm(img))


def train_cold_diffusion(epochs=4, batch_size=16, lr=2e-4):
    if COLD_DIFF_CKPT.exists():
        print(f"[skip] {COLD_DIFF_CKPT.name} already on Drive — not retraining.")
        return
    print("Training Cold Diffusion (~90 min on A100)...")
    df  = pd.read_csv(PRISTINE_CSV)
    ds  = _ColdDiffDataset(df, n_per=4)
    dl  = DataLoader(ds, batch_size=batch_size, shuffle=True,
                     num_workers=2, pin_memory=True)
    net = CondUNet().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    scaler = GradScaler(enabled=True)
    crit   = nn.L1Loss()
    for ep in range(epochs):
        net.train()
        running, n = 0.0, 0
        pbar = tqdm(dl, desc=f"cold-diff ep {ep+1}/{epochs}", leave=False)
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
            pbar.set_postfix(L1=f"{running/n:.4f}")
        print(f"  ep {ep+1}: L1={running/n:.4f}")
    torch.save({'state_dict': net.state_dict(), 'T_STEPS': T_STEPS}, COLD_DIFF_CKPT)
    print(f"Saved -> {COLD_DIFF_CKPT}")


train_cold_diffusion(epochs=4, batch_size=16, lr=2e-4)


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
    """Iterative restoration via Cold Diffusion Algorithm 2.

    Args:
        img_pil  : the degraded input PIL image
        kind     : 'blur' | 'exposure' | 'noise' — which D was applied
        t_start  : assumed severity of the input (1..T_STEPS)
    """
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


# Map degradation level -> assumed t at inference. 'high' uses the full chain.
COLD_T_FOR_LEVEL = {'low': 3, 'mid': 5, 'high': T_STEPS}

print("Cold Diffusion restorer ready.")
'''


CELL_68 = '''# 4.3 Build enhanced versions of every degraded image (test ids only)
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
build_enhanced('cold_diff', lambda im, k, l: enhance_cold_diffusion(im, kind=k,
                                                                     t_start=COLD_T_FOR_LEVEL[l]))
print('Enhanced sets ready.')
'''


CELL_69 = '''# 4.4 Re-evaluate the three classifiers on raw degraded vs every enhancer
ENHANCERS = ('clahe', 'genai', 'cold_diff')
VARIANTS  = ('raw', *ENHANCERS)
rows = []
for name in MODEL_NAMES:
    model = models[name]
    print(f'\\n=== {name} ===')
    for k in DEGRADATION_TYPES:
        for l in DEGRADATION_LEVELS:
            for variant in VARIANTS:
                root = DEGRADED_DIR / k / l if variant == 'raw' else ENHANCED_DIR / variant / k / l
                if not (root / 'manifest.csv').exists():
                    continue
                ds = FolderDataset(root, transform=TFM_EVAL)  # <-- force V3 transform
                ds.df = ds.df[ds.df['id_code'].astype(str).isin(test_id_set)].reset_index(drop=True)
                m = evaluate(model, DataLoader(ds, batch_size=32, num_workers=2, pin_memory=True))
                rows.append({'model': name, 'degradation': k, 'level': l, 'variant': variant, **m})
                print(f'  {k}/{l}/{variant}: acc={m["accuracy"]:.4f}')
    del model; torch.cuda.empty_cache()

rec_df = pd.DataFrame(rows)
rec_df.to_csv(P4 / 'metrics' / 'recovery_accuracy.csv', index=False)
rec_df.head()
'''


CELL_70 = '''# 4.5 Recovery plots — accuracy
level_order_no_none = ['low', 'mid', 'high']
for kind in DEGRADATION_TYPES:
    fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(4.2*len(MODEL_NAMES), 4), sharey=True)
    for ax, name in zip(axes, MODEL_NAMES):
        sub = rec_df[(rec_df['degradation'] == kind) & (rec_df['model'] == name)].copy()
        sub['level'] = pd.Categorical(sub['level'], categories=level_order_no_none, ordered=True)
        for variant in VARIANTS:
            d = sub[sub['variant'] == variant].sort_values('level')
            ax.plot(d['level'], d['accuracy'], marker='o', label=variant)
        ax.set_title(name); ax.set_xlabel('level'); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
    axes[0].set_ylabel('accuracy'); axes[-1].legend()
    fig.suptitle(f'Accuracy recovery — {kind}')
    plt.tight_layout()
    out = P4 / 'plots' / f'recovery_accuracy_{kind}.png'
    plt.savefig(out, dpi=150); plt.show(); print('Saved:', out)
'''


CELL_72 = '''# 4.7 Recovery plots — XAI  (dynamic over ENHANCERS; qualitative grid widens)
for metric in ('stability', 'insertion_auc'):
    for kind in DEGRADATION_TYPES:
        sub = xai_rec[xai_rec['degradation'] == kind].copy()
        if sub.empty: continue
        sub['level'] = pd.Categorical(sub['level'], categories=level_order_no_none, ordered=True)
        g = sub.groupby(['model', 'variant', 'level'])[metric].mean().reset_index()
        fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(4.2*len(MODEL_NAMES), 4), sharey=True)
        for ax, name in zip(axes, MODEL_NAMES):
            for variant in VARIANTS:
                d = g[(g['model'] == name) & (g['variant'] == variant)].sort_values('level')
                ax.plot(d['level'], d[metric], marker='o', label=variant)
            ax.set_title(name); ax.grid(alpha=0.3); ax.set_xlabel('level')
        axes[0].set_ylabel(metric); axes[-1].legend()
        fig.suptitle(f'XAI {metric} recovery — {kind}'); plt.tight_layout()
        out = P4 / 'plots' / f'recovery_xai_{metric}_{kind}.png'
        plt.savefig(out, dpi=150); plt.show()

# Side-by-side qualitative — one row per degradation, one column per variant
demo_id = EXPLAIN_IDS_P4[0]
n_cols  = 2 + len(ENHANCERS)   # clean, degraded, + each enhancer
fig, axes = plt.subplots(len(DEGRADATION_TYPES), n_cols,
                         figsize=(2.8*n_cols, 3.2*len(DEGRADATION_TYPES)))
for r, kind in enumerate(DEGRADATION_TYPES):
    clean = Image.open(resolve_image(demo_id)).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
    deg   = Image.open(find_in_folder(DEGRADED_DIR / kind / 'high', demo_id)).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
    panels = [(clean, 'clean'), (deg, f'{kind}-high')]
    for variant in ENHANCERS:
        try:
            img = Image.open(find_in_folder(ENHANCED_DIR / variant / kind / 'high', demo_id)).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
            panels.append((img, variant))
        except FileNotFoundError:
            panels.append((Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE)), f'{variant} (missing)'))
    for ax, (im, ttl) in zip(axes[r], panels):
        ax.imshow(im); ax.set_title(ttl); ax.axis('off')
plt.tight_layout()
out = P4 / 'samples' / f'recovery_{demo_id}.png'
plt.savefig(out, dpi=150, bbox_inches='tight'); plt.show()
print('Saved:', out)
'''


# ---------------- patch ----------------
def to_source_list(src: str):
    """Jupyter stores source as list-of-strings with explicit \\n endings."""
    lines = src.splitlines(keepends=True)
    return lines


def main():
    if not BACKUP.exists():
        shutil.copy2(NB_PATH, BACKUP)
        print(f"Backup written -> {BACKUP}")
    else:
        print(f"Backup already exists at {BACKUP} (left untouched)")

    with open(NB_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    patches = {
        39: ("code",     CELL_39),
        41: ("code",     CELL_41),
        66: ("markdown", CELL_66_MD),
        67: ("code",     CELL_67),
        68: ("code",     CELL_68),
        69: ("code",     CELL_69),
        70: ("code",     CELL_70),
        72: ("code",     CELL_72),
    }

    for idx, (expected_type, new_src) in patches.items():
        cell = nb["cells"][idx]
        if cell["cell_type"] != expected_type:
            raise RuntimeError(f"Cell {idx} type mismatch: expected {expected_type}, got {cell['cell_type']}")
        cell["source"] = to_source_list(new_src)
        # Clear stale outputs for code cells (kernel will re-execute)
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        print(f"  patched cell {idx} ({expected_type})")

    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"\nWrote {NB_PATH}")


if __name__ == "__main__":
    main()
