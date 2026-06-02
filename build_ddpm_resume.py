"""
Builds a standalone 'resume' notebook that loads the saved checkpoints from
Drive and runs ONLY the V2-Step 4 Conditional DDPM (vanilla diffusion) step,
skipping the slow SHAP / IG / ensemble / training cells and their re-uploads.

Source: Thesis_optimized_final_version3.ipynb
Output: Thesis_DDPM_resume.ipynb
"""
import json, copy
from pathlib import Path

SRC = Path("Thesis_optimized_final_version3.ipynb")
OUT = Path("Thesis_DDPM_resume.ipynb")

nb = json.load(open(SRC, encoding="utf-8"))

# --- cells the DDPM step (cell 78) actually depends on, in original order ---
#  1  drive mount            2  pip install            3  DRIVE_ROOT / CHECKPOINTS_DIR / RESULTS_ROOT
#  4  LOCAL paths / PHASE     5  mkdir dirs             6  smart_extract (APTOS/EYEQ from Drive zips)
#  8  torch + all constants   9  imports + transforms  10 APTOSDataset / resolve helpers
# 11  sampler/loss/EMA utils  17 filter_pristine -> PRISTINE_CSV
# 18  APTOS layout probe      19 IMAGE_INDEX + resolve_image
# 21  DEGRADERS / apply_degradation
# 78  === V2 Step 4: conditional vanilla DDPM ===  (train is resume-guarded -> skips)
BOOTSTRAP = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 17, 18, 19, 21]
DDPM_CELL = 78


def strip(cell):
    c = copy.deepcopy(cell)
    if c.get("cell_type") == "code":
        c["outputs"] = []
        c["execution_count"] = None
    # drop bulky/colab-specific cell metadata (widget state, collapsed outputs, etc.)
    meta = c.get("metadata", {})
    c["metadata"] = {k: meta[k] for k in ("id",) if k in meta}
    return c


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


cells = []

cells.append(md(
    "# Conditional DDPM (V2 - Step 4) - RESUME notebook\n"
    "\n"
    "Standalone re-run of the **vanilla conditional DDPM** step using the checkpoints "
    "already saved on Drive. Skips SHAP / IG / ensemble / classifier-training cells so "
    "nothing heavy is recomputed or re-uploaded.\n"
    "\n"
    "**Run order:** all cells top-to-bottom. The DDPM training cell is resume-guarded "
    "(`if DDPM_CKPT.exists(): skip`), so with `ddpm_fundus_v1.pt` on Drive it loads "
    "instead of retraining.\n"
))

# ---- pre-flight checkpoint check (runs after paths are set, so it's appended later) ----
PREFLIGHT = (
    "# --- pre-flight: confirm the DDPM checkpoint is on Drive before we rely on it ---\n"
    "import random\n"
    "import torchvision.transforms as T\n"
    "_ddpm_ckpt = CHECKPOINTS_DIR / 'ddpm_fundus_v1.pt'\n"
    "assert CHECKPOINTS_DIR.exists(), f'Drive checkpoints dir missing: {CHECKPOINTS_DIR}'\n"
    "print('checkpoints dir :', CHECKPOINTS_DIR)\n"
    "print('ddpm checkpoint :', _ddpm_ckpt, '->', 'FOUND' if _ddpm_ckpt.exists() else 'MISSING')\n"
    "print('PRISTINE_CSV    :', PRISTINE_CSV, '->', 'FOUND' if PRISTINE_CSV.exists() else 'MISSING')\n"
    "if not _ddpm_ckpt.exists():\n"
    "    print('\\n[warning] checkpoint missing -> the next cell WILL train from scratch (~hours).')\n"
)

# ---- post-DDPM smoke test so you SEE results immediately ----
DEMO = (
    "# === DDPM smoke test: restore a few degraded test images (uses the loaded checkpoint) ===\n"
    "import matplotlib.pyplot as plt\n"
    "from PIL import Image\n"
    "import pandas as pd\n"
    "\n"
    "_demo = pd.read_csv(PRISTINE_CSV).sample(3, random_state=SEED)['id_code'].tolist()\n"
    "_out  = RESULTS_ROOT / 'ddpm' / 'resume_smoke'\n"
    "_out.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "for _kind, _level in [('blur', 'high'), ('noise', 'high'), ('exposure', 'high')]:\n"
    "    fig, axes = plt.subplots(len(_demo), 3, figsize=(9, 3 * len(_demo)))\n"
    "    if len(_demo) == 1: axes = axes[None, :]\n"
    "    for r, _id in enumerate(_demo):\n"
    "        clean    = Image.open(resolve_image(_id)).convert('RGB').resize((DDPM_SIZE, DDPM_SIZE))\n"
    "        degraded = apply_degradation(clean, _kind, _level)\n"
    "        restored = enhance_ddpm(degraded, n_steps=50)\n"
    "        for c, (im, ttl) in enumerate(zip([clean, degraded, restored],\n"
    "                                          ['clean', f'{_kind}/{_level}', 'DDPM restored'])):\n"
    "            axes[r, c].imshow(im); axes[r, c].axis('off')\n"
    "            if r == 0: axes[r, c].set_title(ttl, fontsize=10)\n"
    "    fig.suptitle(f'Conditional DDPM restore - {_kind}/{_level}', fontsize=12)\n"
    "    fig.tight_layout(rect=[0, 0, 1, 0.96])\n"
    "    fig.savefig(_out / f'ddpm_resume_{_kind}_{_level}.png', dpi=120, bbox_inches='tight')\n"
    "    plt.show()\n"
    "print('Smoke-test figures saved ->', _out)\n"
)

# assemble: bootstrap cells, then pre-flight, then the DDPM cell, then demo
for idx in BOOTSTRAP:
    cells.append(strip(nb["cells"][idx]))

cells.append(md("## Pre-flight check + DDPM step"))
cells.append(code(PREFLIGHT))
cells.append(strip(nb["cells"][DDPM_CELL]))
cells.append(md("## Verify the restorer works on the loaded checkpoint"))
cells.append(code(DEMO))

# clean notebook-level metadata: keep kernelspec/language_info, DROP colab widget-state blob
src_meta = nb.get("metadata", {})
clean_meta = {k: src_meta[k] for k in ("kernelspec", "language_info") if k in src_meta}
clean_meta.setdefault("kernelspec", {"name": "python3", "display_name": "Python 3"})
clean_meta["accelerator"] = "GPU"
clean_meta["colab"] = {"provenance": []}

out_nb = {
    "cells": cells,
    "metadata": clean_meta,
    "nbformat": nb.get("nbformat", 4),
    "nbformat_minor": nb.get("nbformat_minor", 5),
}
json.dump(out_nb, open(OUT, "w", encoding="utf-8"), indent=1)
print(f"Wrote {OUT} with {len(cells)} cells "
      f"(bootstrap={len(BOOTSTRAP)} + DDPM + preflight + demo).")
