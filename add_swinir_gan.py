"""Add SwinIR + GAN enhancer to Thesis_optimized_final (1).ipynb.

Inserts a markdown header + code cell right after the Cold Diffusion code cell,
then updates build_enhanced and ENHANCERS to include 'swinir_gan'. Cell
location is by content match (function names), not hard-coded indices, so it
is robust to prior insertions.
"""
import json
import shutil
from pathlib import Path

NB_PATH = Path("C:/Dissertation/Thesis_optimized_final (1).ipynb")
BACKUP  = NB_PATH.with_suffix(".ipynb.bak2")


SWINIR_MD = '''## V5 - SwinIR + GAN restorer (4th enhancer)

A second supervised restorer alongside Cold Diffusion. SwinIR (Liang et al., 2021) is a Swin-Transformer backbone purpose-built for image restoration; pairing it with a PatchGAN discriminator gives the perceptual sharpness that pure L1 lacks while still being supervised on paired (clean, degraded) data.

- **Architecture**: slim SwinIR (embed_dim=60, depths=[2,2,2,2], window=8, ~1M params). Tries `basicsr.archs.swinir_arch.SwinIR` first; falls back to a built-in slim Swin-UNet if basicsr's version is unavailable.
- **Loss**: `L1(R(deg), clean) + 0.01 * BCE(D(deg, R(deg)), 1)` with conditional PatchGAN.
- **Training**: 3 epochs, batch 4, on-the-fly Phase-1 degradation. ~90 min on A100, ~3 hr on T4. Resume-guarded by `swinir_gan_v5.pt` on Drive.
- **Inference**: single forward pass, ~0.1 s/img — much faster than Cold Diffusion's 8-step iteration.
'''


SWINIR_CODE = '''# === V5 PATCH: SwinIR + GAN restorer ===
# Supervised paired restoration with adversarial term.
# Forward operator: the SAME Phase-1 degradation primitives, like Cold Diffusion.

SWINIR_CKPT = CHECKPOINTS_DIR / 'swinir_gan_v5.pt'
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
    # ---- slim Swin-UNet fallback ----
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
            return torch.sigmoid(self.out(u1) + x)   # learn the residual
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


def train_swinir_gan(epochs=3, batch_size=4, lr_g=1e-4, lr_d=1e-4, adv_w=0.01):
    if SWINIR_CKPT.exists():
        print(f"[skip] {SWINIR_CKPT.name} already on Drive — not retraining.")
        return
    print("Training SwinIR + GAN (~90 min on A100, ~3 hr on T4)...")
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
        G.train(); D.train()
        run_g, run_d, n = 0.0, 0.0, 0
        pbar = tqdm(dl, desc=f"swinir-gan ep {ep+1}/{epochs}", leave=False)
        for deg, clean in pbar:
            deg   = deg.to(DEVICE, non_blocking=True)
            clean = clean.to(DEVICE, non_blocking=True)
            # ----- D step -----
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
            # ----- G step -----
            opt_g.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                fake = G(deg).clamp(0, 1)
                adv  = bce(D(torch.cat([deg, fake], 1)), torch.ones_like(d_real))
                rec  = l1(fake, clean)
                g_loss = rec + adv_w * adv
            scaler_g.scale(g_loss).backward()
            scaler_g.step(opt_g); scaler_g.update()
            run_g += g_loss.item() * deg.size(0)
            run_d += d_loss.item() * deg.size(0)
            n     += deg.size(0)
            pbar.set_postfix(G=f"{run_g/n:.4f}", D=f"{run_d/n:.4f}")
        print(f"  ep {ep+1}: G={run_g/n:.4f}  D={run_d/n:.4f}")
    torch.save({'state_dict': G.state_dict()}, SWINIR_CKPT)
    print(f"Saved -> {SWINIR_CKPT}")


train_swinir_gan(epochs=3, batch_size=4, lr_g=1e-4, lr_d=1e-4, adv_w=0.01)


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


print("SwinIR + GAN restorer ready.")
'''


def find_cell(nb, predicate):
    for i, c in enumerate(nb["cells"]):
        src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        if predicate(c["cell_type"], src):
            return i, src
    raise RuntimeError("Cell not found")


def to_source_list(src: str):
    return src.splitlines(keepends=True)


def make_code_cell(src):
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": to_source_list(src),
    }


def make_md_cell(src):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": to_source_list(src),
    }


def main():
    if not BACKUP.exists():
        shutil.copy2(NB_PATH, BACKUP)
        print(f"Backup written -> {BACKUP}")
    else:
        print(f"Backup already exists at {BACKUP} (left untouched)")

    with open(NB_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # 1) Find the Cold Diffusion code cell (contains 'train_cold_diffusion(')
    cold_idx, _ = find_cell(
        nb,
        lambda t, s: t == "code" and "train_cold_diffusion(" in s
                     and "def train_cold_diffusion" in s,
    )
    print(f"  cold-diff code cell at index {cold_idx}")

    # 2) Insert SwinIR markdown + code right after it
    nb["cells"].insert(cold_idx + 1, make_md_cell(SWINIR_MD))
    nb["cells"].insert(cold_idx + 2, make_code_cell(SWINIR_CODE))
    print(f"  inserted SwinIR markdown @ {cold_idx + 1}, code @ {cold_idx + 2}")

    # 3) Update build_enhanced cell to add swinir_gan call
    bld_idx, bld_src = find_cell(
        nb,
        lambda t, s: t == "code" and "def build_enhanced" in s
                     and "build_enhanced('cold_diff'" in s,
    )
    if "build_enhanced('swinir_gan'" not in bld_src:
        new_bld = bld_src.replace(
            "build_enhanced('cold_diff', lambda im, k, l: enhance_cold_diffusion(im, kind=k,\n                                                                     t_start=COLD_T_FOR_LEVEL[l]))",
            "build_enhanced('cold_diff',  lambda im, k, l: enhance_cold_diffusion(im, kind=k,\n                                                                      t_start=COLD_T_FOR_LEVEL[l]))\nbuild_enhanced('swinir_gan', lambda im, k, l: enhance_swinir_gan(im))",
        )
        if new_bld == bld_src:
            raise RuntimeError("build_enhanced replace did not match")
        nb["cells"][bld_idx]["source"] = to_source_list(new_bld)
        nb["cells"][bld_idx]["outputs"] = []
        nb["cells"][bld_idx]["execution_count"] = None
        print(f"  patched build_enhanced cell at index {bld_idx}")

    # 4) Update ENHANCERS cell
    enh_idx, enh_src = find_cell(
        nb,
        lambda t, s: t == "code" and "ENHANCERS = ('clahe', 'genai', 'cold_diff')" in s,
    )
    new_enh = enh_src.replace(
        "ENHANCERS = ('clahe', 'genai', 'cold_diff')",
        "ENHANCERS = ('clahe', 'genai', 'cold_diff', 'swinir_gan')",
    )
    nb["cells"][enh_idx]["source"] = to_source_list(new_enh)
    nb["cells"][enh_idx]["outputs"] = []
    nb["cells"][enh_idx]["execution_count"] = None
    print(f"  patched ENHANCERS at index {enh_idx}")

    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"\nWrote {NB_PATH}  (total cells now: {len(nb['cells'])})")


if __name__ == "__main__":
    main()
