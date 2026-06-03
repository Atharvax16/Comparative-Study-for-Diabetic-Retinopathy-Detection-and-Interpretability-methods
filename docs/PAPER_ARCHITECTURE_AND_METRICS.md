# Dissertation — Architectures & Supporting Metrics (paper reference)

**Project:** Robust + Explainable AI for Diabetic Retinopathy (APTOS 2019 + EyeQ).
**Source notebook:** `Thesis_optimized_final_version3.ipynb` (Google Colab, A100).
**Artifact store (figures + CSVs):** Google Drive `MyDrive/Thesis/` → see the shared
folder link. Per-phase outputs live under `phase{1..5}_*/metrics/` and `.../samples|plots/`.

> Purpose of this file: a single, accurate reference of *what architecture was used*
> and *which metric supports each claim*, so the paper text never overstates what the
> code actually does. Where a number is needed, the CSV that holds it is named.

---

## 0. Pipeline at a glance

| Phase | What it does | Key output CSV (on Drive) |
|---|---|---|
| 1 — Data engineering | Pristine set + synthetic degradations (blur/exposure/noise × low/mid/high) | `phase1_*/metrics/manifest_*.csv` |
| 2 — Classification | Train DR graders (v2 224px, v3 384px), stress-test under degradation | `phase2_*/metrics/accuracy_pivot.csv` |
| 3 — XAI benchmark | IG / SHAP / attention-rollout / Grad-CAM; faithfulness + stability | `phase3_*/metrics/xai_results.csv` |
| 4 — GenAI restoration | CLAHE, ESRGAN/U-Net, Cold Diffusion, SwinIR+GAN, DDPM, pathology-DDPM | `phase4_*/metrics/recovery_accuracy.csv`, `recovery_xai.csv`, `ddpm_pathology_hallucination.csv` |
| 5 — Quality ensemble | Quality classifier routes each image to best clean/robust grader | `phase5_*/metrics/quality_clean_probs.csv` |

Global config: `NUM_CLASSES = 5` (`No DR, Mild, Moderate, Severe, Proliferative`),
`SEED = 42`, stratified split `70/15/15` (train/val/test).

---

## 1. Degradation model (Phase 1)

Three physically-motivated corruptions, three severities each, applied to clean fundus images:

| Corruption | Operator | low / mid / high |
|---|---|---|
| **Blur** | Gaussian blur (`cv2.GaussianBlur`, σ) | σ = 2.0 / 5.0 / 9.0 |
| **Exposure** | Multiplicative gain | gain = 0.7 / 0.4 / 0.2 (under-exposure) |
| **Noise** | Additive Gaussian (std as fraction of 255) | std = 0.02 / 0.06 / 0.12 |

These same operators are the **forward/degradation operator** reused by Cold Diffusion
and as the paired-training corruption for SwinIR+GAN and the DDPMs — so restoration is
evaluated on exactly the corruptions the classifier is stressed with.

---

## 2. Classification architectures (Phase 2)

Three backbones (one CNN-residual, one efficient-CNN, one transformer) via `timm`:
`MODEL_NAMES = ('resnet50', 'efficientnet_b3', 'vit_base')`.

### 2.1 v2 baseline graders
- Plain `timm.create_model(..., num_classes=5)`, input **224 px**.
- Training infra: class-balanced sampler, **Focal loss**, EMA weights, warmup→cosine LR.

### 2.2 v3 graders — `MultiScaleClassifier` (the main model)
- Wraps a `timm` backbone in **`features_only=True`** mode and fuses multi-stage feature
  maps with a learnable **stage-attention** weight (`stage_attn`).
- **Ordinal head**: `forward(x, return_ordinal=True)` returns logits **and** an ordinal
  score — DR grading is ordinal, not nominal.
- **`OrdinalFocalLoss`** (focal + ordinal penalty + label smoothing).
- Input **384 px** with **Ben Graham preprocessing** (circle-crop + local-mean subtraction)
  — the standard APTOS retinal-fundus normalisation.

### 2.3 v4 ensemble + distillation
- **8-view TTA** (flips/rotations), then **soft-vote across the 3 backbones weighted by
  each model's validation Quadratic-Weighted-Kappa** (`evaluate_v3_ensemble`). Weighting
  by val-QWK stops the weakest backbone dragging the vote down.
- **Knowledge distillation**: best v3 model (by clean QWK) → the other two
  (KL on softened logits + hard CE). *[Distillation cell present; can be skipped on resume.]*

### 2.4 Supporting metrics (classification)
- **Accuracy**, **macro-F1**, **macro AUC (OvR)** — `evaluate()` / `evaluate_tta()`.
- **Quadratic Weighted Kappa (QWK / `kappa_qw`)** — `cohen_kappa_score(..., weights="quadratic")`.
  This is the **headline metric** (APTOS standard; penalises distant grade errors more).
- **Robustness curve**: accuracy per `(degradation, level)` vs clean → `accuracy_pivot.csv`.

---

## 3. Explainability methods (Phase 3)

| Method | Implementation | Applies to |
|---|---|---|
| **Integrated Gradients** | Captum, `n_steps` path integral | all backbones (primary in v4) |
| **SHAP** | `shap.GradientExplainer` w/ cached background | CNN + ViT |
| **Attention rollout** | recursive attention multiplication | ViT / transformer blocks |
| **Grad-CAM** | hooks on last conv stage | CNN backbones |

### 3.1 Supporting metrics (XAI) — defined in code, written to `xai_results.csv`
- **Insertion AUC** (`insertion_auc`): progressively *insert* most-salient pixels, area
  under the recovered-probability curve. **Higher = more faithful** (the highlighted
  pixels really drive the prediction).
- **Deletion AUC** (`deletion_auc`): progressively *remove* salient pixels. Lower = better.
- **Stability** (`stability_spearman`): Spearman ρ between the clean-image heatmap and the
  degraded-image heatmap for the same model. **Higher = explanation is robust to degradation.**
- **Localization IoU** (`localization_iou`): overlap of top-percentile heatmap with the
  fundus mask (sanity that attribution lands on retina, not background).

> Claim you can safely make: explanations are evaluated **quantitatively** (faithfulness +
> robustness), not just shown qualitatively.

---

## 4. Restoration / GenAI enhancers (Phase 4)

Seven enhancers were implemented; the **enhancer key → method** map and current run status:

| Key | Method | Type | Paper note |
|---|---|---|---|
| `clahe` | CLAHE | classical contrast equalisation | non-learned baseline |
| `genai` | A-ESRGAN → Real-ESRGAN → TinyU-Net | GAN super-res (with fallback) | off-the-shelf generative baseline |
| `cold_diff` | **Cold Diffusion** | deterministic degradation-diffusion | see §4.1 |
| `swinir_gan` | **SwinIR + PatchGAN** | supervised transformer restorer | see §4.2 |
| `cyclegan` | CycleGAN-CBAM | unpaired GAN | **excluded from current run** |
| `ddpm` | **Conditional DDPM** | stochastic diffusion (Ho et al. 2020) | see §4.3 |
| `ddpm_path` | **Pathology-preserving DDPM** | constrained diffusion | see §4.4 — main contribution |

### 4.1 Cold Diffusion (`cold_diff`)
- 256 px, `T_STEPS = 8`. Forward operator = the Phase-1 degraders on a severity schedule;
  reverse = Cold-Diffusion **Algorithm 2** (`x ← x − D(x̂,s) + D(x̂,s−1)`).
- A small conditional U-Net (`CondUNet`) conditioned on `(kind, t)`. Noise samples
  up-weighted in training (`KIND_SAMPLE_WEIGHTS = {blur .25, exposure .25, noise .50}`).
- Level→steps: `{low:3, mid:5, high:8}`. **Loss: pure L1** (no adversarial term).

### 4.2 SwinIR + PatchGAN (`swinir_gan`)
- Slim SwinIR (embed 60, depths [2,2,2,2], window 8, ~1M params), 256 px, **single forward pass**.
- **Loss = L1 + 0.005·BCE(PatchGAN)** — reconstruction-dominated (adv weight lowered
  0.01→0.005 to stop adversarial drift). **Discriminator frozen after epoch 4** (fixed
  schedule, not threshold-triggered) to prevent D-drift.

### 4.3 Conditional DDPM (`ddpm`) — vanilla diffusion baseline
- Standard DDPM (Ho et al. 2020), `T = 1000`, linear β-schedule `1e-4→0.02`, 256 px.
- U-Net takes `concat(noisy_x0, degraded)` (6 ch) → predicts noise ε.
- **Loss = MSE(ε_pred, ε)** only. No notion of pathology → **can hallucinate lesions.**

### 4.4 Pathology-preserving DDPM (`ddpm_path`) — main contribution
Same diffusion backbone as §4.3, plus three guardrails so cleaning does **not** alter disease findings:
1. **Classifier-feature perceptual loss** (`λ_perc = 0.1`): L1 between restored and clean
   in the **feature space of our trained DR grader** (EfficientNetV2-S), not raw pixels.
2. **Input-fidelity loss** (`λ_fid = 0.05`): L1 anchoring the restored `x0` to the degraded
   input → can't invent unsupported structure.
3. **Sampling-time anti-hallucination clamp** (`clamp_strength`): every reverse step blends
   the predicted `x0` toward the degraded input, capping drift from the evidence.

`Loss = MSE(ε) + 0.1·perceptual(x0, clean) + 0.05·L1(x0, degraded)`.

### 4.5 Supporting metrics (restoration)
- **Downstream recovery (primary):** classifier accuracy/QWK on *restored* images vs raw
  degraded → `recovery_accuracy.csv`. This is the metric that matters for a diagnostic
  pipeline (does restoration recover the *decision*, not just pixels).
- **XAI recovery:** insertion-AUC / stability on restored images → `recovery_xai.csv`.
- **Image-quality (reference-based):** **PSNR**, **SSIM** (skimage), **LPIPS** (AlexNet,
  perceptual), **FID** (Fréchet Inception Distance — needs a large sample to be stable).
- **Hallucination detection (Step 8, key for `ddpm_path`):** `hallucination_check()` →
  `ddpm_pathology_hallucination.csv`:
  - `grade_before` / `grade_after` (DR grade from our grader on degraded vs restored),
  - `grade_changed` (did restoration flip the diagnosis?),
  - `pixel_deviation` (mean |restored − degraded|),
  - `risk_score` → `hallucination_risk ∈ {low, medium, high}`.

> Safe claim: the pathology-preserving variant is validated by a **dedicated
> hallucination metric**, not just by looking nicer — `grade_changed` rate directly
> measures clinically-unsafe edits.

---

## 5. Quality-aware ensemble routing (Phase 5)

- **Quality classifier:** `resnet18` (3-way: `good / usable / reject`), trained on EyeQ-style
  quality labels.
- **Threshold calibration:** softmax thresholds swept so routing distribution ≈ target
  `{good 0.60, usable 0.25, reject 0.15}` (`_route_under`, `_route_distance`).
- **Routing policy:** good-quality → `best_clean` grader; degraded-but-usable → `best_robust`
  grader (selected from Phase-2 stress results). Reject → flag for re-acquisition.
- **Supporting metrics:** routing distribution vs target; end-to-end accuracy/QWK of the
  routed ensemble vs a single-best baseline; false-reject rate on known-clean images.

---

## 6. Claim → evidence map (use this when writing)

| Paper claim | Backed by | Artifact |
|---|---|---|
| Models degrade under realistic corruption | Phase-2 stress test | `accuracy_pivot.csv` |
| Transformer + multi-scale fusion improves grading | v2 vs v3 QWK comparison | phase2 metrics |
| Explanations are faithful & robust | insertion-AUC, stability ρ | `xai_results.csv` |
| Restoration recovers the *diagnosis*, not just pixels | recovery accuracy/QWK | `recovery_accuracy.csv` |
| Vanilla DDPM hallucinates; pathology-DDPM does not | `grade_changed`, `risk` rates | `ddpm_pathology_hallucination.csv` |
| Reference image-quality of restorers | PSNR/SSIM/LPIPS(/FID) | restorer metric CSVs |
| Quality routing beats single model | routed vs single-best | phase5 metrics |

---

## 7. Limitations to state explicitly (keeps the paper honest)

- **Perceptual loss bias:** `ddpm_path` only preserves features *our own grader* finds
  salient; a biased grader → a restorer that inherits that bias. State this.
- **Synthetic degradations** approximate but are not real acquisition artefacts; external
  validation on genuinely low-quality clinical images would strengthen claims.
- **FID** on small validation subsets is statistically unreliable (needs ~1–2k images);
  report it only on the full test set or rely on LPIPS for small-sample perceptual quality.
- **CycleGAN-CBAM** is implemented but **excluded from the reported run** — don't claim
  results for it unless re-enabled and re-evaluated.
- Cold Diffusion / DDPM use **fast sampling** (8 / ~50 steps); quality numbers will shift
  with more steps — report the step count alongside every restoration metric.

---

## 8. Suggested citations (verify exact refs)
- Diffusion: Ho, Jain, Abbeel, *Denoising Diffusion Probabilistic Models*, NeurIPS 2020.
- Cold Diffusion: Bansal et al., *Cold Diffusion: Inverting Arbitrary Image Transforms
  Without Noise*, 2022.
- SwinIR: Liang et al., *SwinIR: Image Restoration Using Swin Transformer*, ICCV-W 2021.
- (A-)ESRGAN: Wang et al., ESRGAN 2018 / Real-ESRGAN 2021.
- XAI: Sundararajan et al. (Integrated Gradients, 2017); Lundberg & Lee (SHAP, 2017);
  Selvaraju et al. (Grad-CAM, 2017); Abnar & Zuidema (Attention Rollout, 2020).
- Fundus preprocessing: Ben Graham, Kaggle DR winning solution, 2015.
- Metrics: QWK (Cohen 1968); insertion/deletion (Petsiuk et al., RISE, 2018); FID
  (Heusel et al., 2017); LPIPS (Zhang et al., 2018).
