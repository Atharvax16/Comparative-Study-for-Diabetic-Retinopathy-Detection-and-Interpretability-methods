# ============================================================================
# PASTE THIS AS A NEW CELL AT THE BOTTOM OF YOUR RUNNING COLAB NOTEBOOK.
# It uses what's already in memory (rec_df, xai_rec, enhance_cold_diffusion,
# the loaded models, the cached JPEGs on Drive). No re-upload, no re-run
# of Phase 1-3 needed.
#
# What it does:
#   1. Defines RESTORE_LEVELS = ('low', 'mid')
#   2. Re-renders Phase 4 accuracy plots: raw line spans all 3 levels,
#      enhancer lines stop at mid (visual scoping statement)
#   3. Re-renders Phase 4 XAI plots with the same scoping
#   4. Builds ONE supplementary figure: "Why we don't restore at high" —
#      a noise-high cold_diff failure case
#   5. Patches QUALITY_POLICY['reject'] -> no enhancement + flag for
#      re-acquisition, and re-runs the Phase 5 routed pipeline
# ============================================================================

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image

# ---- 1. Scope definition ----
RESTORE_LEVELS = ('low', 'mid')                              # NEW: restoration scope
ALL_LEVELS     = ('low', 'mid', 'high')                       # raw still shown at high

P4_PLOTS_V2 = P4 / 'plots' / 'low_mid_scope'
P4_SAMPS_V2 = P4 / 'samples' / 'low_mid_scope'
P4_PLOTS_V2.mkdir(parents=True, exist_ok=True)
P4_SAMPS_V2.mkdir(parents=True, exist_ok=True)

# ---- 2. Re-render Phase 4 accuracy plots with scoped enhancer lines ----
print("=== Phase 4 accuracy plots (low+mid restoration scope) ===")
for kind in DEGRADATION_TYPES:
    fig, axes = plt.subplots(1, len(MODEL_NAMES),
                             figsize=(4.4*len(MODEL_NAMES), 4), sharey=True)
    for ax, name in zip(axes, MODEL_NAMES):
        sub = rec_df[(rec_df['degradation'] == kind) & (rec_df['model'] == name)].copy()
        sub['level'] = pd.Categorical(sub['level'], categories=list(ALL_LEVELS), ordered=True)
        for variant in VARIANTS:
            d = sub[sub['variant'] == variant].sort_values('level')
            if variant == 'raw':
                ax.plot(d['level'], d['accuracy'], marker='o', linewidth=2.2,
                        label='raw (no restoration)', color='black')
            else:
                d_scope = d[d['level'].isin(RESTORE_LEVELS)]
                ax.plot(d_scope['level'], d_scope['accuracy'],
                        marker='o', linestyle='--', label=variant)
        ax.axvspan(2 - 0.5, 2 + 0.5, color='lightcoral', alpha=0.15)   # shade high
        ax.text(2, 0.05, 'ungradable\n(no restoration)',
                ha='center', va='bottom', fontsize=8, style='italic', color='darkred')
        ax.set_title(name); ax.set_xlabel('severity'); ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('accuracy'); axes[-1].legend(loc='lower left', fontsize=8)
    fig.suptitle(f'Accuracy under {kind} — restoration scoped to low+mid')
    plt.tight_layout()
    out = P4_PLOTS_V2 / f'scoped_accuracy_{kind}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.show()
    print(f"  Saved: {out}")

# ---- 3. Re-render XAI plots with same scoping ----
print("\n=== Phase 4 XAI plots (low+mid restoration scope) ===")
for metric in ('stability', 'insertion_auc'):
    for kind in DEGRADATION_TYPES:
        sub = xai_rec[xai_rec['degradation'] == kind].copy()
        if sub.empty:
            continue
        sub['level'] = pd.Categorical(sub['level'], categories=list(ALL_LEVELS), ordered=True)
        g = (sub.groupby(['model', 'variant', 'level'], observed=True)[metric]
             .mean().reset_index())
        fig, axes = plt.subplots(1, len(MODEL_NAMES),
                                 figsize=(4.4*len(MODEL_NAMES), 4), sharey=True)
        for ax, name in zip(axes, MODEL_NAMES):
            for variant in VARIANTS:
                d = g[(g['model'] == name) & (g['variant'] == variant)].sort_values('level')
                if variant == 'raw':
                    ax.plot(d['level'], d[metric], marker='o', linewidth=2.2,
                            color='black', label='raw')
                else:
                    d_scope = d[d['level'].isin(RESTORE_LEVELS)]
                    ax.plot(d_scope['level'], d_scope[metric],
                            marker='o', linestyle='--', label=variant)
            ax.axvspan(2 - 0.5, 2 + 0.5, color='lightcoral', alpha=0.15)
            ax.set_title(name); ax.grid(alpha=0.3); ax.set_xlabel('severity')
        axes[0].set_ylabel(metric); axes[-1].legend(loc='best', fontsize=8)
        fig.suptitle(f'XAI {metric} — {kind} (restoration scoped to low+mid)')
        plt.tight_layout()
        out = P4_PLOTS_V2 / f'scoped_xai_{metric}_{kind}.png'
        plt.savefig(out, dpi=150, bbox_inches='tight'); plt.show()
        print(f"  Saved: {out}")

# ---- 4. "Why we don't restore at high" supplementary figure ----
print("\n=== Supplementary: restoration failure at noise-high ===")

# Pick a representative test id we already have noise-high data for
demo_id = None
for cand in list(test_id_set)[:25]:
    try:
        _ = find_in_folder(DEGRADED_DIR / 'noise' / 'high', cand)
        _ = find_in_folder(ENHANCED_DIR / 'cold_diff' / 'noise' / 'high', cand)
        demo_id = cand
        break
    except FileNotFoundError:
        continue

if demo_id is None:
    print("  [skip] no test id with both noise-high degraded + cold_diff variant on disk")
else:
    clean_pil = Image.open(resolve_image(demo_id)).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
    deg_pil   = Image.open(find_in_folder(DEGRADED_DIR / 'noise' / 'high', demo_id)).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
    panels = [(clean_pil, 'clean (reference)'), (deg_pil, 'noise-high input\n"ungradable"')]
    for variant in ENHANCERS:
        try:
            v_pil = Image.open(find_in_folder(ENHANCED_DIR / variant / 'noise' / 'high', demo_id)).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
            # Look up classifier accuracy for this slice from rec_df (averaged across models)
            sub_acc = rec_df[(rec_df['degradation'] == 'noise') &
                             (rec_df['level'] == 'high') &
                             (rec_df['variant'] == variant)]['accuracy'].mean()
            panels.append((v_pil, f'{variant}\nacc={sub_acc:.3f}'))
        except FileNotFoundError:
            panels.append((Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE)),
                           f'{variant}\n(missing)'))

    # Also append the raw baseline accuracy for context
    raw_acc = rec_df[(rec_df['degradation'] == 'noise') &
                     (rec_df['level'] == 'high') &
                     (rec_df['variant'] == 'raw')]['accuracy'].mean()
    panels[1] = (deg_pil, f'noise-high input\nraw acc={raw_acc:.3f}\n(no restoration)')

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3*n, 3.6))
    for ax, (im, ttl) in zip(axes, panels):
        ax.imshow(im); ax.set_title(ttl, fontsize=10); ax.axis('off')
    fig.suptitle(f'Why we stop at mid — restoration at noise-high (id={demo_id})',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = P4_SAMPS_V2 / f'failure_noise_high_{demo_id}.png'
    plt.savefig(out, dpi=160, bbox_inches='tight'); plt.show()
    print(f"  Saved: {out}")
    print(f"\n  Headline numbers from rec_df at noise-high:")
    nh = rec_df[(rec_df['degradation'] == 'noise') & (rec_df['level'] == 'high')]
    print(nh.groupby('variant')['accuracy'].mean().round(3).to_string())

# ---- 5. Patch Phase 5 routing policy ----
print("\n=== Phase 5 routing policy update ===")
try:
    OLD_POLICY = dict(QUALITY_POLICY)
    QUALITY_POLICY['reject'] = {
        'enhancement': 'none',
        'model': 'resnet50',
        'xai': 'gradcam',
        'flag': 'reacquire',          # NEW: explicit re-acquisition flag
    }
    print("  Old reject path:", OLD_POLICY['reject'])
    print("  New reject path:", QUALITY_POLICY['reject'])
    print("\n  Re-run cells 82 (route pipeline) and 83 (aggregate) to see the")
    print("  updated headline numbers with the corrected policy.")
except NameError:
    print("  [skip] QUALITY_POLICY not in memory — run Phase 5 cells first.")

print("\nDone. New artifacts under:")
print(f"  Plots:   {P4_PLOTS_V2}")
print(f"  Samples: {P4_SAMPS_V2}")
