# Research Journey — Robust & Explainable AI for Diabetic Retinopathy

*A complete, plain-language account of how this project started, what each
experiment returned, the obstacles we hit, how we overcame them, what we
observed, the literature behind each decision, and exactly where we stand today.*

Last updated: 2026-06-07

---

## 0. Executive summary (read this first)

**The question.** Deep-learning models grade diabetic retinopathy (DR) at ~85% on
clean images, but 15–30% of real screening images are blurry, mis-exposed, or
noisy. We asked: *what happens to the diagnosis (and its explanation) as image
quality drops, and can generative-AI restoration bring the accuracy back?*

**The headline finding.** Restoration improves how the image *looks*
(PSNR/SSIM ↑) but **does not** recover the *diagnosis*. Diffusion-based restorers
actively **harm** classification; GAN-based super-resolution (A-ESRGAN) is the
only restorer that is "distribution-safe." The reason is **distribution shift** —
a restorer's output is out-of-distribution for the classifier. The practical
answer is therefore **not** "restore everything" but **quality-aware triage**:
abstain on / re-acquire un-trustworthy images, and restore only where it
measurably helps (severe noise).

**Where we stand.** Scientifically successful: we have a coherent, honest,
publishable result (a clean negative result on restoration + a working triage
alternative). It is **not** yet submission-ready: two confounds and one
validity risk (a possibly under-trained baseline classifier) must be closed
first. Realistic venue: **AICS 2026** primary, MICCAI workshop a stretch.

---

## 1. How it started — motivation & research questions

Clinical DR screening produces large volumes of fundus photographs of uneven
quality. Benchmarks report 84–89% grading accuracy on *curated* images, but that
number is measured on clean data. The under-reported risk is what happens to
**both the prediction and the explanation** when quality degrades — and whether
the now-popular generative "restoration" tools actually help a *diagnostic*
model, as opposed to merely making images look nicer.

We framed four research questions:

| RQ | Question |
|----|----------|
| **RQ1** | How do Vision Transformers compare to CNNs (ResNet-50/ConvNeXt, EfficientNet) as degradation increases? |
| **RQ2** | Do explanation methods (Grad-CAM, SHAP/IG, Attention Rollout) stay faithful and stable as quality drops? |
| **RQ3** | Can GenAI restoration recover **diagnostic accuracy** (not just pixels) vs a CLAHE baseline? |
| **RQ4** | Can a quality-aware system route each image to the best pipeline and output a trust score? |

**Datasets.** APTOS 2019 Blindness Detection (5-grade DR labels) [APTOS-2019];
EyeQ for per-image quality labels good/usable/reject [Fu et al., 2019].

---

## 2. The pipeline at a glance

```
APTOS 2019 + EyeQ
 ├─ Phase 1  Data engineering — pristine subset → 9 synthetic degradations (blur/exposure/noise × low/mid/high)
 ├─ Phase 2  Train ResNet-50 / EfficientNet / ViT → stress-test on every degradation            → RQ1
 ├─ Phase 3  Grad-CAM / SHAP→IG / Attention Rollout → stability (SSIM), insertion/deletion AUC   → RQ2
 ├─ Phase 4  CLAHE · A-ESRGAN · SwinIR+GAN · Cold Diffusion · vanilla DDPM · pathology-DDPM
 │           → re-evaluate accuracy + explanation recovery                                        → RQ3
 │  Phase 4b Restoration-proof control: push CLEAN images through each restorer, re-classify
 ├─ Phase 5  EyeQ quality classifier (good/usable/reject) → routing → clinical trust score        → RQ4
 └─ Phase 6  "Overnight booster": QWK + CIs, calibration, selective prediction, augmentation
             baseline, restored-image adaptation                                                  → rigor pass
```

---

## 3. Step-by-step journey

For each phase: **Goal → What we did → What we received → Obstacle → How we
overcame it → Citations → What we observed.**

### Phase 1 — Data engineering
- **Goal.** Create a controlled testbed: take only high-quality ("pristine")
  APTOS images, then synthesise realistic, *graded* degradations so every effect
  is isolated and reproducible.
- **What we did.** Built three degradation primitives — Gaussian blur, exposure
  (gain) shift, and additive Gaussian noise — each at low/mid/high severity,
  giving 9 degradation conditions. Stratified train/val/test split with per-class
  weights and a balanced sampler to handle APTOS's heavy class imbalance.
- **Obstacle.** APTOS is dominated by grade-0 (no-DR); naïve training/eval would
  be misleadingly "accurate" by predicting the majority class.
- **How we overcame it.** Stratified splitting, class-weighted focal loss
  [Lin et al., 2017], a balanced sampler, and (later) reporting **Quadratic
  Weighted Kappa** instead of raw accuracy.
- **Observed.** A clean, reproducible degradation grid that every later phase
  reuses (test-ids are frozen so Phases 3–6 explain/evaluate the *same* images).

### Phase 2 — Model benchmarking (RQ1)
- **Goal.** Compare CNN vs Transformer robustness under degradation.
- **What we did.** Trained ResNet-50, EfficientNet, and ViT-Base; later upgraded
  ("V3") to stronger backbones — **ConvNeXt-Base** [Liu et al., 2022],
  **EfficientNetV2-S** [Tan & Le, 2021], and a **CLIP-ViT-B/16 @384**
  [Dosovitskiy et al., 2021; Radford et al., 2021] — with Ben Graham fundus
  preprocessing [Graham, 2015], a multi-scale feature-fusion head, ordinal+focal
  loss, MixUp/CutMix [Zhang et al., 2018; Yun et al., 2019], RandAugment
  [Cubuk et al., 2020], EMA, layer-wise LR decay, and 8-view TTA.
- **What we received.** Stress-test grid (`stress_test_results_v3.csv`). Example
  (QWK, with TTA): under **severe noise**, ViT retains QWK ≈ 0.52 while the
  ConvNeXt drops to ≈ 0.42 and EfficientNet collapses.
- **Obstacle.** **EfficientNet behaves as a collapsed/degenerate model** (≈0.05
  accuracy in several contexts — see Phases 4b/6). It is not "fragile," it is
  broken, and cannot be presented as a robustness finding.
- **How we overcame it (partial).** Flagged it for fix-or-drop; the headline
  comparison stands on ResNet/ConvNeXt vs ViT.
- **Citations.** [He et al., 2016], [Liu et al., 2022], [Tan & Le, 2021],
  [Dosovitskiy et al., 2021], [Graham, 2015].
- **Observed.** ViTs degrade more gracefully than CNNs — a gap invisible on clean
  benchmarks. **RQ1 answered.**

### Phase 3 — Explainability benchmark (RQ2)
- **Goal.** Test whether explanations stay faithful/stable as quality drops.
- **What we did.** Grad-CAM [Selvaraju et al., 2017], SHAP [Lundberg & Lee, 2017]
  (later replaced by **Integrated Gradients** [Sundararajan et al., 2017] for
  speed/stability), and Attention Rollout [Abnar & Zuidema, 2020]. Measured
  explanation **stability** (SSIM of heatmaps under degradation) and faithfulness
  (insertion/deletion AUC).
- **Obstacle.** KernelSHAP was prohibitively slow and unstable at scale.
- **How we overcame it.** Swapped to Integrated Gradients via Captum; added a
  dedicated Grad-CAM hook layer so attribution fires on every backbone
  (ConvNeXt/EffNetV2/ViT).
- **Observed.** **Explanations drift faster than accuracy** — heatmap stability
  falls toward (even below) zero while accuracy is still ~50%. A model can be
  "right for the wrong reasons." **RQ2 answered.**

### Phase 4 — GenAI restoration (RQ3)
- **Goal.** Does restoration recover the *diagnosis*?
- **What we did.** Six restorers on the degraded test set: CLAHE [Pizer et al.,
  1987] (baseline), **A-ESRGAN** [Wei et al., 2021] (GAN super-resolution),
  **SwinIR+GAN** [Liang et al., 2021], **Cold Diffusion** [Bansal et al., 2022],
  a **conditional vanilla DDPM** [Ho et al., 2020], and a **pathology-preserving
  DDPM** (our variant). Re-evaluated all classifiers on raw-degraded vs each
  restorer.
- **What we received.** Restorers raise PSNR/SSIM but **do not** beat "do
  nothing" on accuracy. Cold Diffusion gives the largest fidelity gain (+12 dB
  PSNR on exposure) yet still *loses* accuracy.
- **Obstacle #1.** The conditional DDPM was under-trained — its reverse process
  collapsed to noise rather than reconstructing a fundus image.
- **Obstacle #2.** Why does higher fidelity not help diagnosis? Correlation alone
  could not isolate cause.
- **How we overcame it → Phase 4b control.**
- **Citations.** [Pizer et al., 1987], [Wei et al., 2021], [Liang et al., 2021],
  [Bansal et al., 2022], [Ho et al., 2020].
- **Observed.** Pixel fidelity ↑ does **not** imply diagnosis ↑.

### Phase 4b — Restoration-proof control (the key experiment)
- **Goal.** Isolate *why* restoration fails, independent of any real degradation.
- **What we did.** Took **clean, un-degraded** images, pushed them through each
  restorer, and re-classified (`distribution_shift.csv`, ResNet-50, small fixed
  sample). If a restorer hurts even clean images, the damage is the restorer's
  own **distribution shift**, not a failure to "fix" degradation.

  | Restorer (clean → restore → classify) | Accuracy | vs clean 0.85 |
  |---|---|---|
  | **A-ESRGAN (GAN)** | **0.875** | **+0.025 — only safe restorer** |
  | DDPM-pathology | 0.725 | −0.125 |
  | SwinIR+GAN | 0.675 | −0.175 |
  | CLAHE | 0.775 | −0.075 |
  | Cold Diffusion | 0.300 | −0.55 |
  | DDPM (vanilla) | 0.100 | −0.75 (catastrophic) |

- **Observed.** **A-ESRGAN is the only distribution-safe restorer; every
  diffusion variant introduces large distribution shift** that the classifier
  cannot tolerate. This is the mechanistic core of RQ3.

### Phase 5 — Quality-aware triage (RQ4)
- **Goal.** Use image-quality awareness to route images instead of restoring all.
- **What we did.** Trained an EyeQ quality classifier (good/usable/reject) with
  threshold calibration; routed reject→re-acquire and good/usable→best
  classifier, restoring only where it helps.
- **Obstacle.** First pass was anecdotal (a handful of scenarios), not a full,
  fair, metric-backed comparison.
- **How we overcame it.** Reframed as **selective prediction** and built the full
  comparison in Phase 6.
- **Citations.** [Fu et al., 2019] (EyeQ), [Geifman & El-Yaniv, 2017] (selective
  classification).

### ⭐ Supervisor checkpoint — option (a) vs (b)
We reported the Phase-5 puzzle to our supervisor and offered two paths:
- **(a)** invest compute to fully train the conditional generative DDPM, or
- **(b)** reframe Phase 5 as triage and use restoration only where it helps.

**She chose (b)**, reasoning that diffusion models are strong for *text-based*
conditioning but weak for restoration (a broad image property), and citing
[arXiv:2412.09324] as evidence this holds for medical imaging. She noted that a
proper diffusion-restoration fix (e.g. adapting [arXiv:2308.09388]) would require
**paired degraded/clean training data** — unrealistic here — so it should be
**future work**. Publication: MICCAI workshop *if* we show a substantial
accuracy gain vs the original dataset; otherwise **AICS 2026** (UCD, Oct 15–16).

### Phase 6 — Overnight "booster" (the rigor pass)
Appended (append-only, fault-isolated) to the resume notebook; results in
`results/phase6_overnight_boost/`. All six stages completed (`stage_status.csv`:
all `ok`).

- **A — Enriched metrics.** Re-computed the recovery grid with **QWK**, balanced
  accuracy, per-class recall, and **bootstrap 95% CIs** [Cohen, 1968].
- **B — Calibration.** Temperature scaling + ECE [Guo et al., 2017]:
  ConvNeXt ECE 0.070 (already calibrated), **ViT 0.071 → 0.051**, EfficientNet
  ECE 0.228 (poorly calibrated — another sign it is the weak model).
- **C — Selective prediction + triage.** Confidence-thresholded
  accuracy/QWK-vs-coverage for *do-nothing*, *restore-all*, and *triage*.
- **D — Augmentation baseline.** Trained ConvNeXt with on-the-fly degradation
  augmentation (the standard "train-time robustness" alternative to test-time
  restoration).
- **E — Restored-image adaptation.** Trained a classifier on A-ESRGAN-restored
  images to test whether adapting the classifier closes the OOD gap.
- **Citations.** [Guo et al., 2017], [Geifman & El-Yaniv, 2017], [Cohen, 1968],
  [Cubuk et al., 2020].

---

## 4. Key results (current numbers)

### 4.1 Restoration under QWK (Phase 6A, ResNet-50/ConvNeXt) — the sharpened RQ3
QWK by restorer vs `raw` (do-nothing):

| Degradation/level | raw | clahe | genai(A-ESRGAN) | swinir | cold_diff | ddpm | ddpm_path |
|---|---|---|---|---|---|---|---|
| blur low | **0.650** | 0.591 | 0.602 | 0.539 | 0.581 | 0.031 | 0.313 |
| blur mid | **0.449** | 0.445 | 0.238 | 0.408 | 0.301 | −0.004 | 0.206 |
| blur high | 0.239 | **0.251** | 0.172 | 0.200 | 0.091 | 0.052 | 0.061 |
| exposure low | **0.807** | 0.700 | 0.633 | 0.482 | 0.614 | −0.025 | 0.472 |
| exposure mid | **0.744** | 0.629 | 0.496 | 0.651 | 0.620 | 0.000 | 0.455 |
| exposure high | 0.509 | **0.590** | 0.359 | 0.466 | 0.503 | 0.011 | 0.367 |
| noise low | **0.813** | 0.746 | 0.625 | 0.560 | 0.536 | 0.021 | 0.408 |
| noise mid | **0.674** | 0.483 | 0.519 | 0.564 | −0.005 | −0.030 | 0.295 |
| **noise high** | −0.007 | −0.035 | 0.336 | 0.065 | 0.001 | 0.023 | **0.433** |

**Reading it.** `raw` wins everywhere on blur/exposure (CLAHE ties only at high
severity). Restoration *substantially helps* **only at severe noise**, where raw
QWK collapses to ≈0 and ddpm-pathology (0.433) / A-ESRGAN (0.336) recover it.
Vanilla DDPM is ≈0/negative everywhere (catastrophic).

### 4.2 Selective prediction (Phase 6C) — the positive result, at full coverage

| Model | Pipeline | Accuracy | **QWK** |
|---|---|---|---|
| ConvNeXt | do-nothing | 0.619 | 0.465 |
| ConvNeXt | restore-all | 0.602 | 0.442 |
| ConvNeXt | **triage** | **0.656** | **0.582** |
| ViT | do-nothing | 0.638 | 0.621 |
| ViT | restore-all | 0.561 | 0.388 |
| ViT | triage | 0.636 | 0.596 |
| EfficientNet | (all) | ~0.054 | ~0.008 |

At **80% coverage** (reject worst 20%), ConvNeXt **triage = 0.739 acc / 0.665
QWK**, clearly beating do-nothing (0.691/0.525) and restore-all (0.674/0.464);
accuracy keeps rising as coverage drops (selective prediction works).

**Two crucial reads:**
1. **`restore-all` keeps acceptable *accuracy* but its QWK craters (down to
   0.15–0.44).** Restoring everything destroys the ordinal/minority-class
   structure — *accuracy hid the harm; QWK exposed it.* Switching the headline
   metric to QWK was decisive.
2. For the already-robust **ViT**, triage ≈ do-nothing — abstention, not
   restoration, is the lever. For the weaker ConvNeXt, triage adds the most.

### 4.3 Calibration (Phase 6B)
ConvNeXt ECE 0.070 (T≈1.0); **ViT 0.071 → 0.051** after temperature scaling;
EfficientNet ECE 0.228 (mis-calibrated). Gives the "clinical trust score" a
principled basis.

---

## 5. What we observed (synthesis)

1. **ViTs are more robust than CNNs** under degradation (RQ1).
2. **Explanations degrade faster than accuracy** (RQ2).
3. **Pixel fidelity ↑ ≠ diagnosis ↑** (RQ3). The cause is **distribution shift**,
   proven by the clean-image control (Phase 4b).
4. **GAN-safe, DM-harmful:** A-ESRGAN is the only distribution-safe restorer;
   every diffusion variant harms diagnosis (vanilla DDPM catastrophically). This
   matches the literature our supervisor cited [arXiv:2412.09324].
5. **QWK reveals harm that accuracy hides** — especially for restore-everything.
6. **Triage > do-nothing > restore-all** on the degraded stream; abstaining on
   low-confidence images monotonically improves served-set quality (RQ4).
7. **Restoration helps only at severe noise** — the natural routing rule.

---

## 6. Where we stand — honest assessment

**Successful as research?** Yes. We have a coherent, mechanistically-explained,
literature-aligned story: *generative restoration does not recover DR diagnosis
(and diffusion harms it); quality-aware triage with a calibrated trust score is
the better answer.* A rigorous negative result plus a working alternative is a
legitimate contribution, and it is exactly the option-(b) framing the supervisor
endorsed.

**Submission-ready?** Not yet. Three issues stand between "solid result" and
"reviewer-proof paper":

1. **Possible under-trained baseline (validity risk — highest priority).** Our
   main ConvNeXt shows clean QWK ≈ 0.625, but a fresh 12-epoch model reached
   QWK ≈ 0.942. If the headline classifier is under-trained, it weakens *every*
   baseline in the paper. Must verify / retrain to convergence.
2. **Confounded augmentation comparison (Phase 6D).** The original stress grid
   used TTA; the augmentation baseline did not — and (1) above further muddies
   it. Cannot claim "augmentation improves robustness" until re-run apples-to-
   apples. (Stage 6E "train-on-restored" similarly inconclusive: QWK 0.17–0.38,
   not better than raw → adapting the classifier did **not** close the OOD gap.)
3. **EfficientNet is collapsed** (~0.05). Fix or drop it; do not present it as a
   robustness finding.

**No "home run."** We do **not** have the "pipeline substantially beats the clean
original dataset" result the supervisor named as the MICCAI bar — restoration
doesn't beat raw broadly; the triage win is over do-nothing/restore-all, not over
clean originals.

---

## 7. What's left to be submission-ready (priority order)

1. **Resolve the under-trained baseline** — retrain/verify the main classifiers
   to convergence; re-run the stress + recovery grids.
2. **Fair re-runs with matched evaluation** (same TTA setting both sides) and
   **error bars** (bootstrap CIs already implemented in Phase 6A; report them).
3. **Fix or drop EfficientNet.**
4. **Finalise the selective-prediction figure** as the paper's centrepiece
   (accuracy/QWK vs coverage; triage vs do-nothing vs restore-all).
5. **Write up future work** (cite [arXiv:2412.09324]; point to
   [arXiv:2308.09388] + the paired-data requirement; note the under-trained
   conditional DDPM).
6. *(Stretch, attended run)* RETFound [Zhou et al., 2023] backbone and test-time
   adaptation [Wang et al., 2021] for a stronger robustness lever.

---

## 8. Publication plan

- **Primary: AICS 2026** (UCD, Oct 15–16; CFP ~late July/Aug). Timeline and bar
  fit what we have; good polish time.
- **Stretch: MICCAI workshop** — only if, after fixing the baseline, we can show
  a stronger net gain. Even a rejection yields useful feedback.

---

## 9. References

> Verify every entry against the original source before submission. The two
> arXiv IDs supplied by the supervisor are listed as given; confirm their exact
> titles/authors.

- **[APTOS-2019]** APTOS 2019 Blindness Detection. Kaggle competition, Asia
  Pacific Tele-Ophthalmology Society.
- **[Fu et al., 2019]** Fu, H. et al. *Evaluation of Retinal Image Quality
  Assessment Networks in Different Color Spaces* (EyeQ). MICCAI 2019.
- **[Graham, 2015]** Graham, B. *Kaggle Diabetic Retinopathy Detection —
  winning solution* (Ben Graham fundus preprocessing).
- **[He et al., 2016]** He, K. et al. *Deep Residual Learning for Image
  Recognition* (ResNet). CVPR 2016.
- **[Liu et al., 2022]** Liu, Z. et al. *A ConvNet for the 2020s* (ConvNeXt).
  CVPR 2022. arXiv:2201.03545.
- **[Tan & Le, 2021]** Tan, M., Le, Q. *EfficientNetV2: Smaller Models and
  Faster Training.* ICML 2021. arXiv:2104.00298.
- **[Dosovitskiy et al., 2021]** Dosovitskiy, A. et al. *An Image is Worth
  16×16 Words* (ViT). ICLR 2021. arXiv:2010.11929.
- **[Radford et al., 2021]** Radford, A. et al. *Learning Transferable Visual
  Models From Natural Language Supervision* (CLIP). ICML 2021.
- **[Selvaraju et al., 2017]** *Grad-CAM.* ICCV 2017.
- **[Lundberg & Lee, 2017]** *A Unified Approach to Interpreting Model
  Predictions* (SHAP). NeurIPS 2017.
- **[Sundararajan et al., 2017]** *Axiomatic Attribution for Deep Networks*
  (Integrated Gradients). ICML 2017.
- **[Abnar & Zuidema, 2020]** *Quantifying Attention Flow in Transformers*
  (Attention Rollout). ACL 2020.
- **[Pizer et al., 1987]** *Adaptive Histogram Equalization and Its Variations*
  (CLAHE). CVGIP 1987.
- **[Wei et al., 2021]** Wei, Z. et al. *A-ESRGAN: Training Real-World Blind
  Super-Resolution with Attention U-Net Discriminators.* arXiv:2112.10046.
  (See also Wang, X. et al. *Real-ESRGAN.* ICCVW 2021.)
- **[Liang et al., 2021]** Liang, J. et al. *SwinIR: Image Restoration Using
  Swin Transformer.* ICCVW 2021. arXiv:2108.10257.
- **[Bansal et al., 2022]** Bansal, A. et al. *Cold Diffusion: Inverting
  Arbitrary Image Transforms Without Noise.* arXiv:2208.09392.
- **[Ho et al., 2020]** Ho, J., Jain, A., Abbeel, P. *Denoising Diffusion
  Probabilistic Models.* NeurIPS 2020. arXiv:2006.11239.
- **[Lin et al., 2017]** Lin, T.-Y. et al. *Focal Loss for Dense Object
  Detection.* ICCV 2017.
- **[Zhang et al., 2018]** Zhang, H. et al. *mixup: Beyond Empirical Risk
  Minimization.* ICLR 2018.
- **[Yun et al., 2019]** Yun, S. et al. *CutMix.* ICCV 2019.
- **[Cubuk et al., 2020]** Cubuk, E.D. et al. *RandAugment.* CVPRW 2020.
- **[Guo et al., 2017]** Guo, C. et al. *On Calibration of Modern Neural
  Networks* (temperature scaling). ICML 2017.
- **[Geifman & El-Yaniv, 2017]** *Selective Classification for Deep Neural
  Networks.* NeurIPS 2017.
- **[Cohen, 1968]** Cohen, J. *Weighted kappa.* Psychological Bulletin, 1968.
- **[Zhou et al., 2023]** Zhou, Y. et al. *A foundation model for generalizable
  disease detection from retinal images* (RETFound). Nature 2023.
- **[Wang et al., 2021]** Wang, D. et al. *Tent: Fully Test-Time Adaptation by
  Entropy Minimization.* ICLR 2021.
- **[arXiv:2412.09324]** *(Supervisor-provided)* Evidence that diffusion models
  underperform at image restoration, including for medical imaging. **Confirm
  title/authors.**
- **[arXiv:2308.09388]** *(Supervisor-provided)* A diffusion-based image-
  restoration approach proposed as future work; requires paired degraded/clean
  training data. **Confirm title/authors.**

---

*Artifacts: `results/phase2_model_benchmarking/`, `results/phase3_xai_benchmark/`,
`results/phase4_genai_enhancement/`, `results/phase4b_restoration_proof/`,
`results/phase5_quality_ensemble/`, `results/phase6_overnight_boost/`. Pipeline:
`notebooks/Thesis_v3_resume_DDPM_to_Phase5.ipynb` (Phase-6 booster appended at end).*
