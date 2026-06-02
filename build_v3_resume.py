"""
Build a RESUME notebook from Thesis_optimized_final_version3.ipynb that runs the
WHOLE pipeline through to the end of Phase 5, but:
  * keeps every definition/loader cell (so nothing breaks),
  * relies on the notebook's existing resume-guards for training + caches,
  * STUBS the slow, already-completed compute cells the user doesn't want to
    repeat (Optuna HPO, knowledge distillation, the SHAP/IG XAI benchmark),
  * DISABLES CycleGAN (removed from the enhancer list + build call),
  * strips all outputs and the bulky Colab widget metadata.

So a fresh Colab runtime can be 'Run all' -> it bootstraps + loads checkpoints,
fast-forwards past the done work, and continues DDPM -> Phase 5 producing results.

Output: Thesis_v3_resume_DDPM_to_Phase5.ipynb
"""
import json, copy, re
from pathlib import Path

SRC = Path("Thesis_optimized_final_version3.ipynb")
OUT = Path("Thesis_v3_resume_DDPM_to_Phase5.ipynb")
nb = json.load(open(SRC, encoding="utf-8"))

# ---- cells to replace wholesale (slow + already done, not needed by the tail) ----
STUBS = {
    32: "# [RESUME] Optuna hyper-parameter search already done -- skipped.\n"
        "print('[resume] skipped Optuna HPO (using existing checkpoints/HPARAMS).')\n",
    45: "# [RESUME] Knowledge distillation already done -- skipped.\n"
        "print('[resume] skipped distillation selection.')\n",
    47: "# [RESUME] Knowledge distillation runner already done -- skipped.\n"
        "print('[resume] skipped distillation training.')\n",
    54: "# [RESUME] SHAP (GradientExplainer) setup already done -- skipped.\n"
        "print('[resume] skipped SHAP setup.')\n",
    55: "# [RESUME] SHAP quality metrics already done -- skipped.\n"
        "print('[resume] skipped SHAP metrics.')\n",
    56: "# [RESUME] SHAP comparison visualisation already done -- skipped.\n"
        "print('[resume] skipped SHAP comparison plots.')\n",
    57: "# [RESUME] Phase-3 XAI benchmark already done -- reload CSV instead of recomputing.\n"
        "import pandas as pd\n"
        "_xai_csv = P3 / 'metrics' / 'xai_results.csv'\n"
        "if _xai_csv.exists():\n"
        "    xai_df = pd.read_csv(_xai_csv)\n"
        "    print(f'[resume] loaded {len(xai_df)} XAI rows from {_xai_csv} (benchmark skipped).')\n"
        "else:\n"
        "    print('[resume][warn] xai_results.csv not found on Drive; benchmark skipped anyway.')\n",
    58: "# [RESUME] Phase-3 XAI aggregated tables/plots already done -- skipped.\n"
        "print('[resume] skipped XAI aggregation plots.')\n",
    66: "# [RESUME] SHAP/IG degradation-progression figures already generated -- skipped.\n"
        "# Preserve the only thing the rest of the notebook reads from this cell:\n"
        "N_FIGURES = 3\n"
        "SAMPLE_IDS = EXPLAIN_IDS[:N_FIGURES]\n"
        "print(f'[resume] skipped IG progression; SAMPLE_IDS={list(SAMPLE_IDS)}')\n",
    76: "# [RESUME] CycleGAN disabled by request (not training/using it).\n"
        "print('[resume] CycleGAN disabled -- removed from enhancers.')\n",
}


def strip(cell):
    c = copy.deepcopy(cell)
    if c.get("cell_type") == "code":
        c["outputs"] = []
        c["execution_count"] = None
    meta = c.get("metadata", {})
    c["metadata"] = {k: meta[k] for k in ("id",) if k in meta}
    return c


def set_src(cell, text):
    cell["source"] = text.splitlines(keepends=True)
    return cell


def drop_cyclegan(src, is_enhancers_tuple_cell):
    """Remove cyclegan from build_enhanced() call cell (81) and ENHANCERS cell (82)."""
    out = []
    for line in src.splitlines(keepends=True):
        low = line.lower()
        if "enhancers" in low and "=" in line and "cyclegan" in low:
            # ENHANCERS tuple line: surgically drop the cyclegan element, keep the line
            line = re.sub(r"['\"]cyclegan['\"]\s*,\s*", "", line)
            out.append(line)
        elif "cyclegan" in low:
            # standalone build_enhanced('cyclegan',...) line or display-name dict entry -> drop
            continue
        else:
            out.append(line)
    return "".join(out)


cells = []
banner = (
    "# V3 pipeline - RESUME notebook (Conditional DDPM -> end of Phase 5)\n\n"
    "Re-runs the **full** pipeline on a fresh Colab runtime using the checkpoints "
    "already on Drive, but **fast-forwards past work that was already completed** so "
    "you don't pay for it twice:\n\n"
    "- Training (v2/v3 classifiers, restorers) loads from checkpoints via the existing resume-guards.\n"
    "- **Stubbed** (already done, slow): Optuna HPO, knowledge distillation, and the Phase-3 "
    "SHAP/IG XAI benchmark + aggregation. The XAI results are reloaded from `xai_results.csv`.\n"
    "- **CycleGAN is disabled** (removed from the enhancer list and build step).\n"
    "- Everything from **Conditional DDPM through Phase 5** runs fresh and produces results.\n\n"
    "**Usage:** GPU runtime -> Run all. First run still pays one-time data extraction + pip install.\n"
)
cells.append({"cell_type": "markdown", "metadata": {},
              "source": banner.splitlines(keepends=True)})

for i, c in enumerate(nb["cells"]):
    cell = strip(c)
    if i in STUBS:
        set_src(cell, STUBS[i])
    elif i == 81:
        set_src(cell, drop_cyclegan("".join(cell["source"]), False))
    elif i == 82:
        set_src(cell, drop_cyclegan("".join(cell["source"]), True))
    cells.append(cell)

src_meta = nb.get("metadata", {})
clean_meta = {k: src_meta[k] for k in ("kernelspec", "language_info") if k in src_meta}
clean_meta.setdefault("kernelspec", {"name": "python3", "display_name": "Python 3"})
clean_meta["accelerator"] = "GPU"
clean_meta["colab"] = {"provenance": []}

out_nb = {"cells": cells, "metadata": clean_meta,
          "nbformat": nb.get("nbformat", 4), "nbformat_minor": nb.get("nbformat_minor", 5)}
json.dump(out_nb, open(OUT, "w", encoding="utf-8"), indent=1)

print(f"Wrote {OUT}: {len(cells)} cells")
print(f"Stubbed cells: {sorted(STUBS)}")
print("CycleGAN removed from cells 81 (build) and 82 (ENHANCERS + display name).")
