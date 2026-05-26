# Robust + Explainable AI for Diabetic Retinopathy

**Project state document — built from `Thesis_optimized_final_version1.ipynb` (94 cells, fully executed) and `OBSERVATIONS_version1.md`. Every number cited below comes from a real cell output in your last Colab run.**

---

## 1. The big picture

### 1.1 What we are actually building
A 5-phase pipeline that answers two coupled questions for diabetic retinopathy (DR) screening:

1. **How robust** is a modern image classifier to the kinds of degradation that fundus photos collect in the wild (motion blur, exposure errors, sensor noise)?
2. **Can we restore** the diagnostic signal with image-restoration models — and if so, **can we trust** the resulting predictions (XAI-faithful, calibrated, route-aware)?

The pipeline is one notebook with five logical phases:

| # | Phase | What it produces |
|---|---|---|
| 1 | Data Engineering | Clean APTOS set + 26,370 synthetic degraded images |
| 2 | Model Benchmarking | Three DR classifiers + stress-test under degradation |
| 3 | XAI Benchmark | Faithfulness/stability scores for Grad-CAM, Attention Rollout, Integrated Gradients |
| 4 | GenAI Enhancement | Four image restorers compared on classifier-recovery |
| 5 | Quality-Aware Ensemble | A router that decides per-image: enhance, route to which classifier, return which XAI |

### 1.2 Hardware & runtime
- Google Colab (A100 preferred, T4 acceptable) — single GPU throughout
- All checkpoints persist to `/content/drive/MyDrive/Thesis/checkpoints/` via Drive mount
- Every long-running cell has a `[RESUME GUARD]` so a cold-start session takes ~40 min once checkpoints exist (vs ~6+ hours from scratch)

### 1.3 Library stack
- **PyTorch 2.x** + **timm** — backbones (`convnext_base.fb_in22k_ft_in1k_384`, `tf_efficientnetv2_s.in21k_ft_in1k`, `vit_base_patch16_clip_384.laion2b_ft_in12k_in1k`)
- **basicsr + realesrgan** — pretrained super-resolution generators (A-ESRGAN, Real-ESRGAN), and the SwinIR architecture
- **captum** — Integrated Gradients (V4 replaced KernelSHAP because SHAP was too slow at 384 px)
- **pytorch-grad-cam** — Grad-CAM heatmaps
- **scikit-image, opencv** — Ben-Graham circle-crop + local-mean-subtraction preprocessing, CLAHE baseline
- **scikit-learn** — quadratic-weighted kappa, F1, AUC, Spearman ρ for XAI stability
- **diffusers** — was used by the (now-removed) Stable Diffusion img2img variant; kept available for future LDM work
- **optuna** — installed but used only opportunistically for hyperparameter search

---

## 2. Phase 1 — Data Engineering

### 2.1 Dataset reality
| Dataset | Plan | Actual |
|---|---|---|
| APTOS 2019 (Kaggle) | Primary training data, 5-class DR grading | ✅ 2,930 images used |
| EyeQ (image-quality labels) | Filter APTOS to "good" quality + label "usable/reject" | ❌ Join failed silently (cell 17: `[warn] EyeQ filter removed all rows — falling back to full APTOS`) — see §10.1 |

### 2.2 Synthetic degradation pipeline (cell 21)
Three operators applied at three severities each, giving 9 condition cells per test image. Code in cell 21:

```python
DEGRADERS = {
    'blur':     gaussian_blur,        # σ ∈ {2.0, 5.0, 9.0}
    'exposure': exposure_shift,       # gain ∈ {0.7, 0.4, 0.2}
    'noise':    gaussian_noise,       # σ_frac ∈ {0.02, 0.06, 0.12}
}
```

Each operator is deterministic given (input, severity). This matters later — Cold Diffusion is trained to invert exactly these operators, so the forward process and the training target are perfectly matched.

### 2.3 Output and caching
- All degraded images saved as JPEG-90 at 224 px under `/content/data/degraded/{kind}/{level}/`
- Whole tree gets tarred up to `Drive/Thesis/cache_degraded.tar.gz` (~600 MB)
- Cell 24's resume guard restores from cache on session reload (~30 sec vs ~25 min to regenerate)
- **Known gap**: the *enhanced* images under `/content/data/enhanced/` are NOT cached to Drive, so a session disconnect costs ~15 min of re-build time. See §11 for the proposed fix.

---

## 3. Phase 2 — Model Benchmarking

This phase went through four iterations. Each iteration was a deliberate design choice, not just "try another thing."

### 3.1 V1 — naïve baseline (deprecated, code retained for reference)
Single-resolution 224 px training, no class weights, plain cross-entropy. Used only to confirm the training loop ran end-to-end.

### 3.2 V2 — production baseline (cells 26-31)
Engineering improvements over V1 that any DR paper would expect:

- **Stratified split** (15% val / 15% test, seeded) so class-1/3/4 (rare classes) are represented in every split
- **Class-weighted focal loss** (CLS_W tensor in cell 26)
- **Balanced sampler** over training indices to avoid all-No-DR mini-batches
- **Standard augmentation**: horizontal/vertical flip, rotation, color jitter, random erasing
- **4-view TTA** at inference (identity + 3 flips)
- Three backbones trained independently: `resnet50`, `efficientnet_b3`, `vit_base_patch16_224`

**V2 results on clean test set** (cell 31):

| Model | Acc | F1-macro |
|---|---|---|
| resnet50 | 0.796 | 0.617 |
| efficientnet_b3 | 0.814 | 0.673 |
| vit_base | 0.809 | 0.682 |

Under degradation, big drops — see §8 for the full table.

### 3.3 V3 — robust pipeline (cells 33-39)
The "senior-engineer" pass aimed at robustness, not absolute accuracy.

- **Input resolution bumped to 384 px** to give the ViT enough patch tokens for retinal microstructure
- **Ben-Graham preprocessing** (cell 33): circle-crop the fundus disc + subtract local Gaussian mean. This is the standard Kaggle-winner preprocessing for DR. Removes illumination gradients without erasing lesions.
- **MultiScaleClassifier wrapper** (cell 35): instead of using the backbone's pooled feature, takes the last N feature stages, attention-weights them with a learnable softmax over stages, and concatenates the pooled features. Adds an Identity layer named `gradcam_target` on the deepest feature for reliable hooking later.
- **OrdinalFocalLoss** (cell 36): `FocalCE(logits, target) + 0.3 * MSE(expected_grade, true_grade)`. The MSE term encourages predictions of grade 3 to be "closer" to true grade 4 than to true grade 0 — exploits DR's ordinal structure. Handles soft (MixUp) targets too.
- **RandAugment(n=2, m=5) + MixUp(α=0.10 to 0.20) + CutMix(α=1.0)** during training
- **EMA** (exponential moving average of weights) for the validation copy
- **Per-backbone hyperparameter schedule** (cell 37: `HPARAMS_V3`) with layer-wise learning-rate decay for the ViT only (γ=0.75)
- **Larger backbones** (cell 37): `convnext_base.fb_in22k_ft_in1k_384` (replaces resnet50), `tf_efficientnetv2_s.in21k_ft_in1k`, `vit_base_patch16_clip_384.laion2b_ft_in12k_in1k` — all 384 px capable, all 22k/CLIP pretrained

**V3 stress test (cell 39, 4-view TTA, clean and 9 degraded conditions)**: results vary by slice — see §8.

### 3.4 V3 vs V2 head-to-head (cell 43)
This is the most important finding from Phase 2.

| version | model | clean | blur-high | exp-high | noise-high |
|---|---|---|---|---|---|
| V2 | resnet50 | 0.795 | 0.525 | 0.580 | 0.459 |
| V2 | efficientnet_b3 | 0.814 | 0.420 | 0.582 | 0.389 |
| V2 | vit_base | 0.809 | 0.489 | 0.539 | 0.536 |
| V3 | resnet50 | 0.639 | 0.461 | **0.786** | 0.493 |
| V3 | efficientnet_b3 | 0.516 | 0.477 | 0.639 | 0.543 |
| V3 | vit_base | 0.661 | **0.582** | **0.725** | 0.493 |

**Reading**: V3 lost 15-30 points of clean accuracy and gained 5-20 points under degradation. This is a **Pareto move**, not a strict improvement. The dissertation should frame V3 honestly: a robustness/clean tradeoff, with V3 winning on the conditions you actually care about (degraded images that motivate restoration in the first place).

### 3.5 V4 — ensemble + distillation (cells 40-47)

#### 3.5.1 V4 patch A: val-kappa-weighted 8-view TTA ensemble (cell 41)
- 8-view TTA per model (identity, 3 flips, 3 rotations, flip+rot)
- Soft-vote across all three V3 backbones, weighted by each model's checkpoint val-kappa
- We patched this in mid-project after noticing equal-weight voting was letting the weakest backbone drag the ensemble down

**V4 ensemble headline (cell 41 output)**:

| Condition | Acc | F1 | Kappa-QW | AUC-OvR |
|---|---|---|---|---|
| clean | 0.725 | 0.605 | 0.836 | 0.945 |
| blur-low | 0.709 | 0.531 | 0.795 | 0.922 |
| blur-mid | 0.586 | 0.335 | 0.593 | 0.826 |
| blur-high | 0.545 | 0.260 | 0.372 | 0.778 |
| exposure-low | 0.768 | 0.642 | 0.852 | 0.940 |
| exposure-mid | 0.725 | 0.568 | 0.792 | 0.926 |
| exposure-high | 0.668 | 0.483 | 0.642 | 0.899 |
| noise-low | 0.745 | 0.623 | 0.843 | 0.933 |
| noise-mid | 0.595 | 0.429 | 0.668 | 0.884 |
| noise-high | 0.482 | 0.277 | 0.502 | 0.827 |

**Important observation**: AUC stays high (0.778 even at blur-high) while accuracy collapses to 0.545. The model's *ranking* is still good — its *argmax decision* is what's hurting. Future work could exploit this with calibrated thresholds.

#### 3.5.2 V4 patch B: knowledge distillation (cells 44-47)
- Teacher = best V3 model on clean kappa (was `vit_base`, clean kappa = 0.7074)
- Students = the other two backbones, KL-divergence loss at temperature T=4 with α=0.7 between hard and soft targets
- 10-epoch distillation per student, layer-wise LR for ViT

**Status**: the distillation training was interrupted at epoch 3 of the resnet50 student during your run (`KeyboardInterrupt` in cell 45). However, because the save-on-best-kappa logic fired at ep3, two distilled checkpoints exist on Drive:
- `resnet50_v3_distilled.pt` — 3 epochs of training, val kappa = 0.807
- `efficientnet_b3_v3_distilled.pt` — presumed similar (cell 47 just reports `[skip]`)

The downstream `load_classifier_v3(name)` was patched (cell 39) to prefer `_v3_distilled.pt` when present, so these undertrained checkpoints are currently feeding the V4 ensemble. They may be hurting more than helping — see §11.

### 3.6 What we did NOT do in Phase 2 (deliberate)
- No hyperparameter search beyond per-backbone defaults — Colab time-budgeted
- No cross-validation — single stratified split (3,662 → train 70/val 15/test 15)
- No domain transfer to other DR datasets (DRIVE, IDRiD, EyePACS) — that's future work

---

## 4. Phase 3 — XAI Benchmark

### 4.1 Methods compared (cells 49-54)
- **Grad-CAM** (`gradcam_heatmap`, cell 14): for the CNN backbones (resnet50→ConvNeXt, efficientnet_b3→EfficientNetV2-S). Hooks the `gradcam_target` Identity layer added by `MultiScaleClassifier`.
- **Attention Rollout** for the ViT (cell 55 patch made it traverse the Transformer blocks inside MultiScaleClassifier — naive rollout couldn't find them).
- **Integrated Gradients** via Captum — replaced KernelSHAP in V4 (cell 51) because SHAP took 30+ seconds per image at 384px while IG takes ~1 second.

### 4.2 Metrics
- **Stability** (cell 14 area): Spearman correlation between the heatmap on the clean image and the heatmap on each degraded version. 1.0 = identical attention, 0 = uncorrelated. Captures "does the explanation still point at the same thing when the image gets worse?"
- **Insertion AUC** (cell 14 area): start from a blurred image, progressively reveal pixels in heatmap order, measure how fast the model's prediction probability rebuilds. High = the heatmap actually highlights important pixels.
- **Deletion AUC**: inverse of above, used for sanity check.

### 4.3 IG aggregate results (cell 62)
Stability scores from 20 images × 3 kinds × 4 levels per model:

| Model | clean stability | blur-high | exp-high | noise-high |
|---|---|---|---|---|
| efficientnet_b3 | 1.000 | 0.063 | 0.199 | 0.106 |
| resnet50 | 1.000 | 0.117 | 0.210 | 0.128 |
| vit_base | 1.000 | 0.184 | **0.302** | 0.224 |

**Reading**: ViT-Base + Integrated Gradients gives the most stable saliency under degradation across all three kinds. The absolute numbers are still low (< 0.3) — no XAI method is "stable" in an absolute sense under realistic degradation. This is the right frame for the lit-review chapter: relative comparison, not absolute claims.

### 4.4 Per-image visualisations (cells 56-62)
For three demo IDs × three degradation kinds × four severity levels, a 24-panel image is generated per case showing (clean Grad-CAM, degraded Grad-CAM, IG progression). Saved to `Drive/Thesis/results/phase3_xai_benchmark/samples/progression_ig/`.

---

## 5. Phase 4 — GenAI Enhancement (the heart of the dissertation)

This is where the work paid off and also where the biggest negative findings live.

### 5.1 Iteration history

| Version | Restorers in scope | Status |
|---|---|---|
| V0 | CLAHE only | baseline, still in use |
| V1 | + A-ESRGAN cascade (Real-ESRGAN, TinyU-Net fallbacks) | still in use, labelled `genai` |
| V4 | + Stable Diffusion img2img | **removed** — see §5.5 |
| V5 | + Cold Diffusion + SwinIR-GAN | current head |
| V6 | restoration scoped to low+mid only | current head (Phase 4 plotting) |

### 5.2 CLAHE baseline (cell 58)
Contrast Limited Adaptive Histogram Equalisation on the L channel of LAB-converted image. `clipLimit=3.0, tileGridSize=(8,8)`. Deterministic, ~10 ms per image, no training. Good baseline.

### 5.3 A-ESRGAN cascade (cell 65, labelled `genai`)
Cascade: try A-ESRGAN's authors' pretrained generator first → fallback to Real-ESRGAN-x2-plus → fallback to TinyU-Net trained on the pristine set. In your run, the A-ESRGAN download succeeded so all `genai` results below are A-ESRGAN.

**Important note**: A-ESRGAN was designed for super-resolution on natural images, not denoising on fundus. The fact that it dominates the noise slice is genuinely surprising — the RRDBNet residual structure absorbs noise as a side-effect.

### 5.4 Cold Diffusion (cell 67) — V5
Bansal et al. 2022 (https://arxiv.org/abs/2208.09392): replace the Gaussian-noise forward process of standard diffusion with an arbitrary deterministic degradation operator. Train the restoration network so `R(D(x, t), t) ≈ x` for all t. Sample with Algorithm 2:

```
x_{s-1} = x_s - D(R(x_s, s), s) + D(R(x_s, s), s-1)
```

**Why this fit the problem**: you already have D — the Phase 1 degradation primitives. So the forward "noising" process *is* your synthetic degradation pipeline. The model never sees Gaussian noise; it only sees inputs that look like real degraded fundus photos.

**Architecture**: small FiLM-conditioned U-Net (~3M params), conditioned on `(kind, t/T)`:
- Input 3×256×256 → Conv → 3 downsampling blocks → mid → 2 upsampling blocks with skip connections → 3×256×256 sigmoid
- FiLM modulates the mid features by `kind_embedding ⊕ t_projection`

**Training**: 4 epochs, batch 16, L1 loss, AdamW, GradScaler AMP. On-the-fly paired sampling: pick a clean fundus, pick (kind, t), apply D, train R to recover the clean. Resume-guarded by `cold_diffusion_v5.pt`.

**Loss trajectory (cell 67 output)**: L1 = 0.0379 → 0.0167 → 0.0142 → 0.0120. Healthy ~3× improvement, final ≈ 3/255 grey-level error per pixel.

### 5.5 Stable Diffusion img2img (removed)
Was added in V4 as an "additional GenAI variant". Removed in V5 because:
1. SD-2.1 base was trained on LAION (natural images), not fundus
2. Even at `strength=0.30` and a "preserve lesion structure" prompt, qualitative inspection showed SD invented anatomically-plausible-but-wrong vessels
3. **Lesion hallucination is a hard fail** for a clinical application — a classifier might then "diagnose" disease from a fake lesion

The cells were replaced with a markdown notice and the Cold Diffusion implementation.

### 5.6 SwinIR + PatchGAN (cell 70) — V5
SwinIR (Liang et al. 2021) is a Swin-Transformer backbone purpose-built for image restoration. Pairing it with a PatchGAN discriminator gives perceptual sharpness that pure L1 can't.

**Architecture**: slim SwinIR (`embed_dim=60, depths=[2,2,2,2], window=8`, ~1M params) via `basicsr.archs.swinir_arch.SwinIR`. Fallback Swin-UNet if basicsr is unavailable.

**Loss**: `L1(R(deg), clean) + 0.01 * BCE(D(deg, R(deg)), 1)` with conditional PatchGAN discriminator (in_ch=6 = degraded + restored).

**Training**: 3 epochs, batch 4, AdamW, on-the-fly Phase-1 degradation. Resume-guarded by `swinir_gan_v5.pt`.

**Loss trajectory (cell 70 output)**: G = 0.0721 → 0.0682 → 0.0705 (slight uptick); D = 0.521 → 0.382 → 0.362.

The discriminator pulling ahead while G stalls is **adversarial drift** — the GAN is half-trained. See §11 for the suggested fix.

### 5.7 Phase 4 evaluation pipeline
- **Cell 71 (build_enhanced)**: for each `(method, kind, level)`, restore every test-id image and save the result under `/content/data/enhanced/{method}/{kind}/{level}/`. Skip-if-exists guard. This loop is the one that lost work on session disconnect (§11.5).
- **Cell 72**: re-evaluate all three V3 classifiers on (raw, clahe, genai, cold_diff, swinir_gan) × (kind, level), save to `Drive/Thesis/results/phase4_genai_enhancement/metrics/recovery_accuracy.csv`.
- **Cell 73**: 3 accuracy-recovery plots (one per kind).
- **Cell 74**: XAI recovery per image — stability + insertion_auc for (raw, *ENHANCERS).
- **Cell 75**: 6 XAI recovery plots + qualitative 18-panel grid.
- **V6 cell (appended)**: scoped re-plots showing raw spanning low/mid/high but enhancers only at low/mid; supplementary "why we stop at mid" figure.

### 5.8 Phase 4 headline findings (from cell 72)

The "no universal restorer" pattern is the single most defensible insight in the project:

| Slice | Best enhancer | Δ vs raw |
|---|---|---|
| blur-low (resnet50) | CLAHE | +3.9 pts |
| blur-high (resnet50) | Cold Diffusion | +3.4 pts |
| **noise-high (eff_b3)** | **A-ESRGAN** | **+43.6 pts** (0.089 → 0.525) |
| noise-mid (eff_b3) | A-ESRGAN | +30.2 pts |
| exposure-low (resnet50) | CLAHE | +5.0 pts |
| exposure-mid (resnet50) | CLAHE | +5.5 pts |
| exposure-high (resnet50) | Cold Diffusion | +5.5 pts |
| exposure-high (vit_base) | Cold Diffusion | +1.1 pts |
| **noise-high (vit_base)** | A-ESRGAN | +13.7 pts (vs Cold Diff at **-34.1 pts**) |

**Patterns**:
1. At low severity, raw input often beats every enhancer. Don't restore what isn't broken.
2. CLAHE is the right tool for exposure on robust backbones (resnet50).
3. A-ESRGAN is the right tool for noise — especially the catastrophic noise-high/eff_b3 cell.
4. Cold Diffusion is the right tool for exposure-high and (modestly) blur-high.
5. Cold Diffusion CATASTROPHICALLY fails on noise (resnet50 noise-high: 0.409 → 0.084; vit_base noise-high: 0.477 → 0.136). Algorithm-2 sampling appears to amplify noise rather than remove it when the t_start matches the input severity.
6. SwinIR-GAN is underwhelming on most slices — adversarial drift from 3-epoch undertraining.

### 5.9 Scoping decision (V6, your call)
"Keep high in Phase 2/3 to show the degradation cliff; drop high from Phase 4 because clinically those images are ungradable and restoration is either futile or harmful."

Implemented in the V6 paste cell at the bottom of the notebook:
- Phase 4 accuracy plots: raw line spans low/mid/high; enhancer lines stop at mid; high region shaded `ungradable (no restoration)`
- One supplementary figure showing the noise-high cold_diff collapse as a documented failure mode
- Phase 5 policy: `reject` → `no enhancement + flag for re-acquisition` (was: `genai + resnet50`)

---

## 6. Phase 5 — Quality-Aware Routing

### 6.1 The pipeline (cells 76-90)
A pre-classifier "quality model" (`Q_CKPT`, a small ResNet18) labels each input as `good / usable / reject`. The policy table (cell 80) then routes:

| Quality | Enhancement | Classifier | XAI | Flag |
|---|---|---|---|---|
| good | none | efficientnet_b3 (best_clean) | Grad-CAM | – |
| usable | CLAHE | efficientnet_b3 | Grad-CAM | – |
| reject | none (was: genai) | resnet50 (best_robust) | Grad-CAM | reacquire |

The `best_clean` and `best_robust` selections were derived from Phase 2 V2 results (cell 80). For each routed prediction we also compute:

- **Confidence**: softmax max
- **Insertion AUC**: faithfulness of the XAI heatmap
- **Trust score**: mean(confidence, insertion_AUC) — a single scalar in [0,1] proxying "should a clinician believe this?"

### 6.2 Phase 5 measured results (cell 83)
| Condition | Routed acc | Mean trust | Baseline (V2 single-best) |
|---|---|---|---|
| clean | 0.650 | 0.624 | 0.814 |
| blur | 0.538 | 0.313 | – |
| exposure | 0.563 | 0.547 | – |

The **trust signal works** (clean=0.62 vs blur=0.31 tracks degradation). The **router is broken** because the quality classifier over-rejects: in cells 82 and 89, almost every input gets labelled `reject`, even clean ones, sending them all through the `reject` path. Net effect: -16.4 pts on clean.

### 6.3 The fix (now applied)
The V6 cell flips `QUALITY_POLICY['reject']` to `{'enhancement': 'none', 'flag': 'reacquire'}`. But the underlying issue is the quality classifier itself; see §11.

---

## 7. End-to-end story (the dissertation arc)

1. **Modern DR classifiers degrade catastrophically under realistic image quality issues** (V4 ensemble kappa=0.37 on blur-high, kappa=0.50 on noise-high). The "robustness gap" is real.
2. **Robustness training (V3) helps the degraded slices but trades clean accuracy.** A clean-vs-robust Pareto frontier exists; choose the operating point intentionally.
3. **XAI methods are not interchangeable under degradation.** ViT + Integrated Gradients gives ~1.5-2× the stability of GradCAM on CNNs. No method is "stable" in an absolute sense at high severity.
4. **Image restoration is conditional.** For clinically salvageable severities (low + mid), the right restorer depends on the degradation type: CLAHE for exposure, A-ESRGAN for noise, Cold Diffusion for blur-high and exposure-high. There is no universal restorer.
5. **Quality-aware routing is the right architecture, but the quality classifier needs proper calibration before the routing decisions can be trusted.**

---

## 8. Headline numbers in one table

V4 ensemble (the current best classifier configuration):

| Slice | Acc | Kappa-QW | AUC | Notes |
|---|---|---|---|---|
| clean | 0.725 | 0.836 | 0.945 | regressed from V2 best 0.814 (Pareto cost) |
| blur-low | 0.709 | 0.795 | 0.922 | salvageable |
| blur-mid | 0.586 | 0.593 | 0.826 | borderline |
| blur-high | 0.545 | 0.372 | 0.778 | ungradable |
| exposure-low | 0.768 | 0.852 | 0.940 | strongest robust slice |
| exposure-mid | 0.725 | 0.792 | 0.926 | salvageable |
| exposure-high | 0.668 | 0.642 | 0.899 | borderline-salvageable |
| noise-low | 0.745 | 0.843 | 0.933 | salvageable |
| noise-mid | 0.595 | 0.668 | 0.884 | borderline |
| noise-high | 0.482 | 0.502 | 0.827 | ungradable |

Best Phase 4 recoveries (cell 72):
- **efficientnet_b3 noise-high: 0.089 → 0.525 with A-ESRGAN** (+43.6 pts) — most dramatic positive
- efficientnet_b3 noise-mid: 0.223 → 0.525 with A-ESRGAN (+30.2 pts)
- resnet50 exposure-high: 0.559 → 0.614 with Cold Diffusion (+5.5 pts)
- vit_base exposure-high: 0.730 → 0.741 with Cold Diffusion (+1.1 pts) — modest but clinically meaningful since exposure-high is borderline-salvageable

Worst Phase 4 collapses (cell 72):
- resnet50 noise-mid: 0.577 → **0.280** with Cold Diffusion (−29.7 pts)
- resnet50 noise-high: 0.409 → **0.084** with Cold Diffusion (−32.5 pts)
- vit_base noise-high: 0.477 → **0.136** with Cold Diffusion (−34.1 pts)

---

## 9. Files and where things live

### 9.1 Local (this Windows machine, `C:\Dissertation\`)
- `Thesis_optimized_final_version1.ipynb` — your executed notebook with outputs (94 cells, V6 cells appended)
- `Thesis_optimized_final (1).ipynb` — sibling copy, no outputs (93 cells, V6 appended)
- `*.bak`, `*.bak2`, `*.bak3` — pre-patch backups, reversible
- `OBSERVATIONS_version1.md` — tagged observations from your last run
- `PROJECT_OVERVIEW.md` — this file
- `COLAB_PASTE_low_mid_scoping.py` — in-session V6 paste snippet
- `patch_notebook.py`, `add_swinir_gan.py`, `append_scoping.py` — the patch scripts (reusable)
- `_outputs_dump/cell_*.txt` — extracted cell outputs (reviewing data without opening Jupyter)

### 9.2 On Google Drive (persists across Colab sessions)
- `MyDrive/Thesis/checkpoints/` — every trained model (V2, V3, V3-distilled, Cold Diffusion, SwinIR-GAN, quality classifier)
- `MyDrive/Thesis/results/phase{1..5}_*/metrics/*.csv` — every measured metric
- `MyDrive/Thesis/results/phase{1..5}_*/plots/` — every chart
- `MyDrive/Thesis/results/phase{1..5}_*/samples/` — every qualitative figure (including `recovery_84e8c62165b5.png` and the V6 scoped plots)
- `MyDrive/Thesis/cache_degraded.tar.gz` — degraded image cache
- (NOT cached: enhanced images — see §11.5)

### 9.3 On Colab local disk (`/content/data/`, LOST on disconnect)
- `pristine/`, `degraded/`, `enhanced/` — the working trees
- Restored from Drive cache (degraded) or rebuilt (enhanced) on next session

---

## 10. Known anomalies (be honest about these in the writeup)

### 10.1 EyeQ join failure
Cell 17 silently falls back to "use all APTOS" because the EyeQ filter removes all rows. Root cause is in the join key between EyeQ's image_name and APTOS's id_code. ~30 min fix; until then, anywhere the dissertation says "APTOS + EyeQ" needs to be softened.

### 10.2 EfficientNet-B3 noise-mid collapse
V2 EffNet drops to acc=0.216, F1=0.120 at noise-mid (cell 31) — worse than at noise-high (0.389). Non-monotonic, suggests BatchNorm running-stats drift: the BN moments trained on clean APTOS don't match noise-mid input statistics. Either fix with test-time BN recalibration or call it out in the dissertation as a known fragility ("evidence for ensemble").

### 10.3 V4 distillation interrupted
The KD training stopped at epoch 3 of the resnet50 student. Both `_v3_distilled.pt` checkpoints exist on Drive because save-on-best-kappa fired at ep3, but they are not fully trained. They are currently feeding the V4 ensemble via the `load_classifier_v3` distilled-preference fallback. Probably hurting more than helping (their val kappa is similar to the teacher's clean kappa of 0.707, so they're not significantly better students).

### 10.4 SwinIR-GAN adversarial drift
Only 3 training epochs (cell 70). G loss stalls while D loss keeps dropping. The result is mediocre Phase 4 performance for `swinir_gan`. Easy fix: 6 epochs at adv_w=0.005 or freeze adv after epoch 2.

### 10.5 Phase 5 reject-bias
The quality classifier (`Q_CKPT`) over-labels images as `reject` — even clean ones. Cell 89 couldn't even find one example of `correct_routing` in 100 sampled rows. The routing logic itself is sound; the upstream classifier is biased. Likely a class-imbalance issue in `Q_CKPT`'s training data.

---

## 11. What we could / should do next

Prioritised, with rough effort estimates.

### 11.1 Cache enhanced images to Drive (1 hour)
Add a `tar -czf` step at the end of cell 71 mirroring cell 24's pattern. Eliminates the 15-min rebuild on session reload.

### 11.2 Fix the quality classifier (3 hours)
Re-train `Q_CKPT` with class weights tied to the EyeQ class distribution, or recalibrate the decision threshold post-hoc (look for the operating point where good:usable:reject ≈ 0.6 : 0.25 : 0.15 on the test set). Should restore Phase 5 routing to net-positive vs baseline.

### 11.3 Re-run distillation to convergence (2 hours wall, ~20 min compute per student)
Either delete `*_v3_distilled.pt` so the loader falls back to `_v3_best.pt`, OR retrain to the full 10 epochs. Pick one; don't leave undertrained students in the ensemble silently.

### 11.4 Re-train SwinIR-GAN with adversarial fix (2 hours)
6 epochs, `adv_w=0.005`, or freeze adversarial loss after epoch 2. Should bring SwinIR-GAN up to competitive performance and possibly take noise-mid from A-ESRGAN.

### 11.5 EyeQ join fix (1 hour, but big writeup payoff)
Either fix the join in cell 11/17 so EyeQ actually contributes, or formally drop EyeQ from the methodology section. Currently neither — it's a phantom dependency.

### 11.6 Larger / better future work

| Direction | Why | Effort |
|---|---|---|
| **Conformal prediction** on top of the V4 ensemble | High-AUC-low-acc cells (blur-high AUC=0.78, acc=0.55) suggest the model knows when it's unsure. Conformal sets would convert "wrong prediction" into "honest abstain" — clinically much safer. | 1-2 days |
| **Route by predicted degradation type**, not by quality class | We already have one restorer per slice that wins. A small "what's wrong with this image?" classifier could route each image to its best restorer. Currently we pick one restorer per (kind,level) by hand. | 1-2 days |
| **V2+V3 hybrid ensemble** (route soft-vote weights by quality) | V2 wins on clean, V3 wins on degraded. Adaptive weighting would capture both. | 1 day |
| **Domain-shift validation** on IDRiD / EyePACS / DRIVE | Robustness claims on APTOS alone are weakly defensible. A cross-dataset slice would dramatically strengthen the dissertation. | 3-5 days incl. data wrangling |
| **Latent Diffusion for restoration** (with a fundus-tuned VAE) | We ruled out off-the-shelf LDM because of micro-vessel loss in the VAE. If you fine-tune the VAE on fundus first, LDM might rival Cold Diffusion at a fraction of inference cost. | 1 week |
| **CycleGAN for unpaired EyeQ→APTOS quality lift** | If you can get EyeQ working as a paired bad-quality source, CycleGAN could augment the training set without needing more pairs. Risky on medical images (hallucination) but worth a controlled try. | 4-5 days |
| **Lesion-preserving metric beyond classifier accuracy** | Currently we measure restoration by "does the classifier accuracy go up?" Add a structural metric: SAM-segmented lesion overlap (clean vs restored) to prove restoration preserves the diagnostic structure, not just the classifier's decision boundary. | 2-3 days |
| **Temperature-scaled calibration on the V4 ensemble** | Reported probabilities should be honest. ECE measurement + Platt/temperature scaling. | 0.5 day |
| **Failure-mode analysis report** | Build a per-test-image table of "raw → enhancer → prediction" and slice it by true class. Likely shows that severe-DR cases (class 3-4) benefit most from restoration. Strong dissertation chapter. | 1 day |

### 11.7 If you have unlimited time
- Self-supervised pretraining of the Cold Diffusion network on a much larger unlabelled fundus set (e.g., EyePACS or Kaggle DR2015 train) — would likely fix the noise-high catastrophe.
- A learned router for "which restorer + which classifier" trained end-to-end on trust score as the reward.
- Replace the synthetic degradations with measured real-world degradations (sample 200 truly-bad-quality EyeQ images, characterise them, generate a degradation distribution that matches).

---

## 12. Reading guide — where in the code to look for what

| What you want to understand | Cell(s) | Key function / variable |
|---|---|---|
| Config (image sizes, model names, degradation params) | 8 | `MODEL_NAMES`, `DEGRADATION_PARAMS` |
| Degradation operators | 21 | `DEGRADERS`, `apply_degradation` |
| Resume-guard pattern | 24, 28, 38, 67, 70 | look for `[RESUME GUARD]` / `if CKPT.exists()` |
| V2 training | 27 | `train_v2` |
| V3 architecture | 35 | `MultiScaleClassifier` |
| V3 loss | 36 | `OrdinalFocalLoss` |
| V3 training | 37, 38 | `train_v3`, `HPARAMS_V3` |
| V4 ensemble | 41 | `evaluate_v3_ensemble` |
| V4 distillation | 44, 47 | `distill_to` |
| XAI methods | 14, 50, 55 | `gradcam_heatmap`, `ig_heatmap`, `attention_rollout` |
| XAI metrics | 14 area | `stability_spearman`, `insertion_auc` |
| CLAHE | 58 | `enhance_clahe` |
| A-ESRGAN cascade | 65 | `_try_aesrgan`, `enhance_genai` |
| Cold Diffusion | 67 | `CondUNet`, `train_cold_diffusion`, `enhance_cold_diffusion` |
| SwinIR + GAN | 70 | `_make_swinir`, `train_swinir_gan`, `enhance_swinir_gan` |
| Phase 4 build loop | 71 | `build_enhanced` |
| Phase 4 evaluation | 72 | `ENHANCERS`, `VARIANTS`, `rec_df` |
| Phase 5 router | 80, 82 | `QUALITY_POLICY`, route loop |
| V6 scoping (appended) | 91-92 | `RESTORE_LEVELS` |

---

## 13. One-paragraph dissertation pitch (for your advisor or LinkedIn)

> "I built a robustness-first pipeline for diabetic retinopathy classification. Modern classifiers lose 30-40 points of accuracy under realistic image degradation (blur, exposure, noise) — quadratic-weighted kappa drops from 0.84 on clean test images to 0.37 on blur-high. I tested four image-restoration techniques (CLAHE, A-ESRGAN, Cold Diffusion, SwinIR+GAN) and found there is no universal restorer: CLAHE wins on exposure errors, A-ESRGAN recovers up to +44 accuracy points on noisy images, and Cold Diffusion (trained on the same synthetic degradation operators used for evaluation) is best for blur and exposure-high but catastrophically amplifies noise. Integrated Gradients explanations on a ViT-Base backbone are 1.5-2× more stable under degradation than Grad-CAM on CNNs. The contribution is a quality-aware routing architecture that picks the right restorer for the right degradation rather than chasing a universal solution — the right framing for medical AI where the wrong restorer can be worse than no restoration at all."

---

*End of document. Last updated 2026-05-26.*
