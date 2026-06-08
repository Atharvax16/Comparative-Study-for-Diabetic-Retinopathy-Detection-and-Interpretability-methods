# Paper-Writing Brief — for Manus (or any LLM writing assistant)

Paste this whole file as context. It contains the task, the exact data/results to
use, the citations, and **integrity guardrails**. A deeper narrative companion is
`docs/RESEARCH_JOURNEY.md` — use it for background, but the numbers in **this**
file are the ones to put in the paper.

---

## 1. Your task

Write a complete, submission-quality research paper from the material below.
Produce: Title, Abstract, Keywords, and full sections (Introduction, Related
Work, Methods, Experiments, Results, Discussion, Limitations, Future Work,
Conclusion), plus correctly-formatted references. Use **only** the facts,
numbers, and citations provided here. **Do not invent results, datasets, baselines,
or numbers.** If something is needed but not provided, insert a clearly-marked
`[TODO: ...]` placeholder instead of fabricating.

---

## 2. Paper identity

- **Working title (pick/refine one):**
  - *"Fidelity Is Not Diagnosis: Why Generative Restoration Fails — and Quality-Aware Triage Succeeds — for Diabetic Retinopathy Under Image Degradation"*
  - *"Restore, Reject, or Route? A Quality-Aware Pipeline for Robust Diabetic Retinopathy Grading"*
- **Primary venue:** AICS 2026 (national conference, UCD). **Stretch:** a MICCAI workshop.
- **Length/format:** standard conference paper (~8–10 pages, double-column).
  Write venue-neutral; we will reformat to the template.
- **Audience:** medical-imaging + ML researchers. Assume familiarity with CNNs,
  ViTs, diffusion models, and DR grading; explain domain specifics briefly.

---

## 3. One-sentence thesis + contributions

**Thesis.** On degraded retinal screening images, generative restoration improves
pixel fidelity but **does not** recover diagnostic accuracy — diffusion-based
restorers actively harm it — because the restorer's output is *out-of-distribution*
for the classifier; a **quality-aware triage** that abstains on low-confidence
images (and restores only where it measurably helps) is the better, deployable
answer.

**Contributions (state these explicitly):**
1. A controlled robustness benchmark for DR grading across 9 graded degradations
   (blur/exposure/noise × low/mid/high) covering CNNs and ViTs.
2. Evidence that **explanations degrade faster than accuracy** (XAI stability).
3. A **causal isolation** of why restoration fails — a *clean-image control* that
   measures each restorer's own distribution shift, independent of degradation.
4. The **"GAN-safe, DM-harmful"** result, quantified with **Quadratic Weighted
   Kappa (QWK)**, showing accuracy *hides* the harm that QWK reveals.
5. A **quality-aware triage / selective-prediction** system with a calibrated
   trust score that beats both "do-nothing" and "restore-everything."

---

## 4. Integrity guardrails (read carefully — do not violate)

- **DO NOT claim** the pipeline beats the *clean original* dataset broadly. It
  does not. The triage win is **relative to do-nothing and restore-all** on the
  *degraded* stream, and restoration helps **only at severe noise**.
- **Frame restoration as a negative result** (honest and intentional), not a
  failure to be hidden. The positive result is triage + selective prediction.
- **Present absolute accuracy modestly.** A baseline-convergence check is ongoing;
  the **robust contributions are the *relative/comparative* findings** (restorer
  vs raw; triage vs do-nothing vs restore-all; QWK vs accuracy discrepancy).
- **EfficientNet is a collapsed model** (~0.05). Either omit it or report it
  explicitly as a degenerate/failed run — do **not** present it as a
  "fragile CNN" robustness finding.
- **The augmentation baseline and the "train-on-restored" experiment are
  preliminary/confounded** (TTA mismatch + baseline-training caveat). Put them in
  *Future Work* or a clearly-labelled *preliminary* paragraph — do **not** build
  claims on them.
- **The conditional DDPM was under-trained** (collapsed to noise) — mention as a
  limitation / future work, not as a tested method.
- Use **QWK as the headline metric**; report accuracy alongside but emphasise QWK,
  because the dataset is imbalanced (ordinal 5-grade DR).

---

## 5. Target structure (write each section)

1. **Abstract** (~200 words): problem → method → the negative result on
   restoration → the triage positive result → numbers (QWK triage vs baselines).
2. **Introduction:** real-world DR screening has 15–30% low-quality images;
   benchmarks measure clean accuracy; the gap; our questions (RQ1–RQ4); the
   counter-intuitive finding; contributions.
3. **Related Work:** DR grading; robustness/corruptions; image restoration (GAN
   super-resolution vs diffusion); diffusion limits for restoration
   (cite arXiv:2412.09324); XAI faithfulness; selective prediction; calibration.
4. **Methods:** datasets; degradation model; classifiers; restorers; XAI methods;
   metrics (incl. QWK + bootstrap CIs); the **clean-image distribution-shift
   control**; the quality-aware triage / selective-prediction protocol;
   temperature-scaling calibration.
5. **Experiments & Results:** RQ1 robustness → RQ2 XAI → RQ3 restoration (+ the
   distribution-shift control) → RQ4 triage / selective prediction → calibration.
   Use the tables in §7.
6. **Discussion:** why fidelity ≠ diagnosis (OOD); why GAN safe / DM harmful; why
   QWK matters; clinical implication (re-acquire, don't fabricate).
7. **Limitations:** synthetic degradations; single dataset; under-trained
   conditional DDPM; baseline-convergence check ongoing; EfficientNet failure.
8. **Future Work:** paired-data diffusion restoration (cite arXiv:2308.09388,
   note paired-data requirement); retinal foundation models (RETFound);
   test-time adaptation; real-world (non-synthetic) degradation.
9. **Conclusion.**
10. **References** (§9).

---

## 6. Methods — "what we used"

- **Datasets.** APTOS 2019 Blindness Detection (5-grade DR: No DR / Mild /
  Moderate / Severe / Proliferative; heavily class-imbalanced). EyeQ for per-image
  quality labels (good / usable / reject).
- **Degradations (synthetic, graded).** Gaussian blur, exposure (gain) shift,
  additive Gaussian noise; each at low/mid/high → 9 conditions. Frozen test-ids
  reused across all phases.
- **Classifiers ("V3").** ConvNeXt-Base @384 (labelled "resnet50" key),
  EfficientNetV2-S (labelled "efficientnet_b3"; **collapsed**), CLIP-ViT-B/16 @384
  (labelled "vit_base"). Ben Graham fundus preprocessing; multi-scale feature
  fusion head; ordinal + focal loss; MixUp/CutMix; RandAugment; EMA; layer-wise
  LR decay; 8-view TTA.
- **Restorers.** CLAHE (baseline); **A-ESRGAN** (GAN super-resolution);
  SwinIR+GAN; Cold Diffusion; conditional vanilla DDPM (under-trained);
  pathology-preserving DDPM (our variant).
- **XAI.** Grad-CAM; Integrated Gradients (replacing KernelSHAP for
  speed/stability); Attention Rollout. Stability via SSIM of heatmaps under
  degradation; faithfulness via insertion/deletion AUC.
- **Metrics.** Quadratic Weighted Kappa (QWK, headline), accuracy, macro-F1,
  balanced accuracy, per-class recall, macro AUC; **bootstrap 95% CIs**.
  Calibration via temperature scaling + Expected Calibration Error (ECE).
- **Triage / selective prediction.** EyeQ quality classifier (good/usable/reject)
  with threshold calibration; confidence-thresholded selective prediction
  (accuracy/QWK vs coverage); fixed prior triage rule = restore with A-ESRGAN
  only at severe noise, else pass raw. (No test-label leakage: the selective
  threshold is validation-safe.)

---

## 7. Results — "the scores / what we observed" (use these exact numbers)

### Table 1 — Distribution-shift control (clean → restore → re-classify, ConvNeXt; small fixed sample)
Baseline clean (no restore) accuracy = **0.85**.

| Restorer | Accuracy | Δ vs clean |
|---|---|---|
| **A-ESRGAN (GAN)** | **0.875** | **+0.025 (only safe restorer)** |
| CLAHE | 0.775 | −0.075 |
| DDPM-pathology | 0.725 | −0.125 |
| SwinIR+GAN | 0.675 | −0.175 |
| Cold Diffusion | 0.300 | −0.55 |
| DDPM (vanilla) | 0.100 | −0.75 (catastrophic) |

*Takeaway:* the damage exists even on clean images → it is the restorer's own
**distribution shift**, not a failure to fix real degradation.

### Table 2 — QWK by restorer vs raw (degraded test set, ConvNeXt)
| Degradation / level | raw | CLAHE | A-ESRGAN | SwinIR | Cold-Diff | DDPM | DDPM-path |
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

*Takeaway:* `raw` wins on blur/exposure; restoration substantially helps **only at
severe noise**; vanilla DDPM is ≈0/negative everywhere.

### Table 3 — Selective prediction / triage at full coverage (degraded stream)
| Model | Pipeline | Accuracy | **QWK** |
|---|---|---|---|
| ConvNeXt | do-nothing | 0.619 | 0.465 |
| ConvNeXt | restore-all | 0.602 | 0.442 |
| ConvNeXt | **triage** | **0.656** | **0.582** |
| ViT | do-nothing | 0.638 | 0.621 |
| ViT | restore-all | 0.561 | 0.388 |
| ViT | triage | 0.636 | 0.596 |

At **80% coverage** (reject worst 20% by confidence), ConvNeXt **triage = 0.739
accuracy / 0.665 QWK**, vs do-nothing 0.691 / 0.525 and restore-all 0.674 / 0.464.
Accuracy/QWK rise monotonically as coverage drops.

*Two key reads:* (a) `restore-all` keeps acceptable accuracy but its **QWK craters
(0.15–0.44)** — restoring everything destroys ordinal/minority-class structure;
accuracy hid this, QWK exposed it. (b) For the already-robust ViT, abstention (not
restoration) is the lever; triage helps the weaker ConvNeXt most.

### Table 4 — Calibration (ECE, temperature scaling)
| Model | Temperature | ECE before | ECE after |
|---|---|---|---|
| ConvNeXt | 1.00 | 0.070 | 0.070 |
| ViT | 0.857 | 0.071 | **0.051** |
| EfficientNet | 0.936 | 0.228 | 0.220 (mis-calibrated) |

### RQ1 qualitative result (state in prose)
ViTs degrade more gracefully than CNNs under increasing degradation — a gap
invisible on clean benchmarks (e.g. under severe noise ViT retains markedly higher
QWK than the ConvNeXt; EfficientNet collapses).

### RQ2 qualitative result (state in prose)
Explanation stability (SSIM of heatmaps) falls toward — and can go below — zero
under noise while accuracy is still ~50%: explanations drift faster than accuracy
("right for the wrong reasons").

---

## 8. Figures available (reference these; paths in repo)
- `results/phase4b_restoration_proof/plots/distribution_shift.png` — distribution shift (Table 1).
- `results/phase4b_restoration_proof/plots/fidelity_vs_accuracy.png` — fidelity↑ vs accuracy↓.
- `results/phase4b_restoration_proof/plots/ddpm_forward_backward.png`, `cold_diffusion_forward_backward.png` — diffusion process figures.
- `results/phase6_overnight_boost/plots/selective_resnet50.png` (+ `_vit_base.png`) — selective-prediction curves (Table 3). **Hero figure.**
- `results/phase2_model_benchmarking/plots/accuracy_vs_degradation_noise.png` — RQ1.
- `results/phase3_xai_benchmark/plots/stability_vs_noise.png` — RQ2.

---

## 9. References (use these; verify before final submission)

DR/quality: APTOS 2019 (Kaggle); Fu et al., 2019 (EyeQ, MICCAI); Graham, 2015
(Ben Graham preprocessing). Backbones: He et al., 2016 (ResNet); Liu et al., 2022
(ConvNeXt); Tan & Le, 2021 (EfficientNetV2); Dosovitskiy et al., 2021 (ViT);
Radford et al., 2021 (CLIP). XAI: Selvaraju et al., 2017 (Grad-CAM); Lundberg &
Lee, 2017 (SHAP); Sundararajan et al., 2017 (Integrated Gradients); Abnar &
Zuidema, 2020 (Attention Rollout). Restoration: Pizer et al., 1987 (CLAHE);
Wei et al., 2021 / Wang et al., 2021 (A-ESRGAN / Real-ESRGAN); Liang et al., 2021
(SwinIR); Bansal et al., 2022 (Cold Diffusion, arXiv:2208.09392); Ho et al., 2020
(DDPM). Training: Lin et al., 2017 (Focal Loss); Zhang et al., 2018 (MixUp);
Yun et al., 2019 (CutMix); Cubuk et al., 2020 (RandAugment). Evaluation/robustness:
Guo et al., 2017 (calibration/temperature scaling); Geifman & El-Yaniv, 2017
(selective classification); Cohen, 1968 (weighted kappa); Zhou et al., 2023
(RETFound, Nature); Wang et al., 2021 (TENT). Supervisor-provided (confirm
titles/authors): arXiv:2412.09324 (diffusion underperforms at restoration,
incl. medical); arXiv:2308.09388 (diffusion restoration approach needing paired
data — cite as future work).

---

## 10. Style guide
- Precise, sober, evidence-first; no marketing language.
- Lead results with **QWK**, report accuracy alongside.
- Always pair a claim with its number and (where available) its CI.
- Use "out-of-distribution / distribution shift" as the explanatory mechanism.
- Be explicit that the restoration result is a **deliberate, useful negative
  result**, and triage is the positive contribution.
- Insert `[TODO: ...]` rather than guessing any missing detail (author names,
  exact dataset sizes, p-values, template-specific items).
