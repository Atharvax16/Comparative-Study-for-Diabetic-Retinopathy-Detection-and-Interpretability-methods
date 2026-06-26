# Key Findings — `Thesis_v3_restoration_proof.ipynb`

Paper-ready results extract from the final end-to-end re-run (Google Colab, **A100**) of the
restoration-proof notebook. Every number below is pulled directly from a saved CSV/JSON under
`result/results/` (path named per claim) or a notebook cell output (`cell NN`). This file is the
factual backbone for the write-up: it fills the `[PLACEHOLDER]` numbers in
`MASTER_PROMPT_for_paper.md` and anchors each sentence to the CLAIM→EVIDENCE map.

Tag legend: `[OBSERVED]` directly in output · `[INSIGHT]` what it means for the paper ·
`[ANOMALY]` something broken/off to report honestly · `[ACTION]` follow-up before submission.

---

## 0. Provenance & setup

- **[OBSERVED]** Dataset is **APTOS-only**. The EyeQ quality join fell back to the full APTOS set:
  **2,930** pristine images (`cell 18`), split **2,050 / 440 / 440** train/val/test (`cell 27`),
  5 DR classes (No DR, Mild, Moderate, Severe, Proliferative), seed 42, stratified.
  Class counts (train): `{0:1003, 2:565, 1:210, 4:164, 3:108}` — strong imbalance (~49% No-DR),
  handled with class weights `[0.41, 1.95, 0.73, 3.80, 2.50]` and a balanced sampler.
- **[ACTION]** Anywhere the paper says "APTOS + EyeQ", soften to "APTOS, with EyeQ as a
  planned-but-unused quality filter." EyeQ contributed **0** images.
- **Degradations**: 3 corruptions × 3 severities (blur σ=2/5/9, exposure gain 0.7/0.4/0.2,
  noise std 0.02/0.06/0.12), applied to all test images; the *same* operators are the forward
  process for the restorers.
- **Restorers reported (6)**: CLAHE, A-ESRGAN→Real-ESRGAN (off-the-shelf GAN; A-ESRGAN import
  failed → ran as `real_esrgan_x2`, `cell 70`), Cold Diffusion (8-step), SwinIR+PatchGAN,
  vanilla conditional DDPM, **pathology-preserving DDPM (main contribution)**.
- **[ANOMALY]** **CycleGAN-CBAM is excluded everywhere** (`cell 77`: "CycleGAN disabled"). Report
  it only as implemented-but-unevaluated future work — never with results (HARD RULE 2).

---

## 1. Headline findings (TL;DR)

1. **Grading collapses under corruption.** The v4 ensemble falls from **acc 0.877 / QWK 0.949**
   (clean) to **acc 0.330 / QWK 0.157** at noise-high — a near-total loss of diagnostic signal
   at the highest severity. (RQ1)
2. **Explanations degrade with the image.** Explanation stability (Spearman ρ vs clean heatmap)
   drops from 1.0 to as low as **−0.05** (EfficientNet Grad-CAM, blur-high); ViT attention is the
   most stable family but still falls to ~0.12–0.18 at high severity. (RQ2)
3. **Restoration improves the picture but not the diagnosis — the central negative result.**
   In **7 of 18** low/mid restoration cells the restorer raised **both PSNR and SSIM yet lowered
   downstream accuracy** (`fidelity_up_accuracy_down.csv`). Headline: Cold Diffusion on exposure
   **+12.2 dB PSNR, +0.112 SSIM, but −0.041 accuracy**. No *learned* restorer produced a positive
   accuracy delta on any corruption (the only non-negative cell is classical CLAHE on blur,
   +0.002 — within noise). (RQ3)
4. **The mechanism is distribution shift, not (only) lesion loss.** Pushing *clean* images through
   the restorers and re-classifying shows large accuracy drops with no degradation present
   (e.g. Cold Diffusion drops ResNet-50 from 0.85→0.25 on clean images) — the restorer output is
   out-of-distribution for the grader. (RQ3)
5. **Vanilla DDPM hallucinates / destroys content; the pathology-DDPM is far safer.** Vanilla DDPM
   collapses to ~8–9% accuracy and PSNR ~8.5 dB (it does not reconstruct the fundus); the
   pathology-DDPM keeps PSNR/SSIM near the input and has a **3.3% grade-change (hallucination)
   rate** with risk distribution **174 low / 6 medium / 0 high**. (RQ3)
6. **One genuine restoration win exists, at the extreme.** At **noise-high** (outside the main
   low/mid scope), the pathology-DDPM lifts ResNet-50 from **raw 0.218 → 0.575** and the GAN
   super-res from 0.218 → 0.557 — the only regime where restoration clearly helps (`cell 83`).
7. **Quality-aware routing did not beat the best single grader in this run.** Routed clean
   accuracy **0.7625** < single-best ViT **0.8477** < ensemble **0.877**. The routing *story*
   (calibrated thresholds, re-acquisition policy) holds, but as a headline accuracy claim it is a
   negative/nuanced result. (RQ4)

---

## 2. Model benchmarks (Phase 2)

### Table A — v3 single-model stress test (multi-scale, 384 px, ordinal head)
Source: `phase2_model_benchmarking/metrics/v3/stress_test_results_v3.csv`. Accuracy / **QWK**.

| Condition | ResNet-50 | EfficientNet-B3 | ViT-Base |
|---|---|---|---|
| clean | 0.639 / 0.625 | 0.516 / 0.173 | 0.661 / 0.707 |
| blur-low | 0.802 / 0.868 | 0.650 / 0.686 | 0.700 / 0.807 |
| blur-mid | 0.520 / 0.515 | 0.495 / 0.520 | 0.573 / 0.648 |
| blur-high | 0.461 / 0.429 | 0.477 / 0.486 | 0.582 / 0.550 |
| exposure-low | 0.859 / 0.929 | 0.791 / 0.884 | 0.809 / 0.899 |
| exposure-mid | 0.843 / 0.911 | 0.752 / 0.849 | 0.795 / 0.879 |
| exposure-high | 0.786 / 0.838 | 0.639 / 0.712 | 0.725 / 0.823 |
| noise-low | 0.848 / 0.889 | 0.734 / 0.827 | 0.777 / 0.876 |
| noise-mid | 0.680 / 0.651 | 0.634 / 0.450 | 0.605 / 0.684 |
| noise-high | 0.493 / 0.415 | 0.543 / 0.139 | 0.493 / 0.522 |

- **[ANOMALY]** v3 single-model *clean* accuracy looks low (0.52–0.66) and EfficientNet clean QWK
  is only 0.173 — the multi-scale ordinal models are individually weak/under-calibrated on clean
  data; the **ensemble is what delivers the strong clean number** (Table B). Report the ensemble
  as the headline grader, not the single models.
- **[INSIGHT]** v3 is much more robust at **low/mid** severity than v2 (e.g. ResNet-50 blur-low
  0.598→0.802; exposure-mid 0.584→0.843), which is the real benefit of the 384 px multi-scale +
  Ben-Graham pipeline.

### Table B — v4 ensemble (8-view TTA, val-QWK-weighted soft-vote) — **headline grader**
Source: `phase2_model_benchmarking/metrics/v3/ensemble_v4_results.csv`.

| Condition | Acc | macro-F1 | **QWK** | macro-AUC |
|---|---|---|---|---|
| **clean** | **0.877** | 0.771 | **0.949** | 0.960 |
| blur-low | 0.659 | 0.450 | 0.639 | 0.863 |
| blur-mid | 0.584 | 0.308 | 0.462 | 0.824 |
| blur-high | 0.518 | 0.211 | 0.265 | 0.772 |
| exposure-low | 0.734 | 0.516 | 0.777 | 0.915 |
| exposure-mid | 0.698 | 0.490 | 0.713 | 0.899 |
| exposure-high | 0.607 | 0.397 | 0.492 | 0.870 |
| noise-low | 0.736 | 0.501 | 0.790 | 0.899 |
| noise-mid | 0.689 | 0.391 | 0.636 | 0.843 |
| **noise-high** | **0.330** | 0.219 | **0.157** | 0.770 |

- **[INSIGHT]** Clean QWK **0.949** is competitive with published APTOS work. The **degradation
  cliff** is the story: exposure is the most survivable corruption (QWK stays ≥0.49 even at high),
  **noise-high is catastrophic** (QWK 0.157, near chance), blur sits in between.

### Table C — v2 vs v3 honest head-to-head (accuracy)
Source: `phase2_model_benchmarking/metrics/v2_vs_v3_summary.csv`.

| Model | clean v2→v3 | blur-high v2→v3 | exposure-high v2→v3 | noise-high v2→v3 |
|---|---|---|---|---|
| ResNet-50 | 0.757 → 0.639 | 0.473 → 0.461 | 0.525 → **0.786** | 0.500 → 0.493 |
| EfficientNet-B3 | 0.766 → 0.516 | 0.080 → **0.477** | 0.566 → **0.639** | 0.205 → **0.543** |
| ViT-Base | 0.848 → 0.661 | 0.568 → **0.582** | 0.545 → **0.725** | 0.602 → 0.493 |

- **[INSIGHT]** v3 *single-model clean* accuracy regressed, but v3 **dramatically rescued the
  failure cases** — most strikingly EfficientNet-B3 at blur-high (0.080→0.477) and noise-high
  (0.205→0.543), where v2 had effectively collapsed. The paper framing: v3 trades a little clean
  peak for large robustness gains, then the ensemble recovers the clean peak.

---

## 3. Explainability benchmark (Phase 3 / RQ2)

Sources: `phase3_xai_benchmark/metrics/summary_{insertion_auc,deletion_auc,stability}.csv`
(mean over 20 images per cell). Methods: Grad-CAM (CNNs), attention-rollout (ViT), IG, SHAP
(GradientExplainer). Higher insertion = more faithful; lower deletion = better; stability =
Spearman ρ between clean and degraded heatmaps (1.0 = unchanged).

### Faithfulness (insertion AUC, clean) and robustness (stability)
| Model / method | insertion (clean) | stability blur-high | stability noise-high | stability exposure-high |
|---|---|---|---|---|
| ResNet-50 / Grad-CAM | 0.706 | 0.157 | 0.336 | 0.424 |
| ResNet-50 / IG | 0.704 | 0.087 | 0.099 | 0.104 |
| ResNet-50 / SHAP | 0.771 | 0.107 | 0.129 | 0.223 |
| EfficientNet-B3 / Grad-CAM | 0.235 | **−0.050** | 0.043 | 0.100 |
| EfficientNet-B3 / IG | 0.210 | 0.078 | 0.112 | 0.199 |
| ViT-Base / IG | 0.621 | 0.165 | 0.177 | 0.264 |
| ViT-Base / SHAP | 0.677 | 0.123 | 0.138 | 0.256 |

- **[INSIGHT]** **No explanation method is robust at high severity** — all stabilities fall toward
  0. The best-behaved is **ResNet-50 Grad-CAM under exposure/noise** (ρ≈0.34–0.42) and ViT under
  exposure (IG ρ≈0.26). The clearest faithfulness ranking on clean images is SHAP ≥ Grad-CAM ≈ IG
  for ResNet-50/ViT.
- **[ANOMALY]** **EfficientNet-B3 explanations are unreliable**: clean insertion AUC ~0.21–0.24
  (barely above its deletion AUC) and **negative stability** (−0.05 at blur-high). This matches its
  weak/uncalibrated classifier (Table A) — its saliency is close to noise. Use ResNet-50 and ViT
  for the qualitative XAI figures.
- **[OBSERVED]** SHAP quality (`shap_quality_metrics.csv`): cross-model SHAP consistency is modest,
  mean pairwise ρ ≈ **0.34–0.51** across the 20 probe images (`shap_cross_model_consistency.csv`) —
  the three architectures attend to overlapping but not identical evidence.
- **[ANOMALY]** Localization IoU was **not computed** (`summary_iou.csv` all empty, count=0): no
  fundus-lesion masks available. Report IoU as not-measured, not zero.

---

## 4. Restoration recovery (Phase 4)

Source: `phase4_genai_enhancement/metrics/recovery_accuracy.csv` and `cell 83` per-model dumps.

- **[OBSERVED] Vanilla DDPM is broken as a restorer.** Across every (model, corruption, level) its
  restored accuracy sits at **~0.05–0.12** — it does not reconstruct a usable fundus image
  (confirmed by PSNR ~8.5 dB in Phase 4b).
- **[ANOMALY] EfficientNet-B3 is pathological on degraded input**: restored accuracy pinned near
  **0.05** (single-class prediction) regardless of restorer — it cannot be rescued. Drop it from
  recovery claims or report it explicitly as a failure case.
- **[OBSERVED] The one clear rescue is at noise-high** (`cell 105` headline row, mean over models):
  | variant @ noise-high | mean acc |
  |---|---|
  | raw (degraded) | 0.278 |
  | **DDPM (pathology)** | **0.400** |
  | A-ESRGAN (genai) | 0.378 |
  | CLAHE | 0.210 |
  | SwinIR+GAN | 0.269 |
  | Cold Diffusion | 0.073 |
  | DDPM (vanilla) | 0.086 |
  Per-model (`cell 83`): ResNet-50 noise-high raw 0.218 → **pathology-DDPM 0.575**, A-ESRGAN 0.557.
- **[INSIGHT]** At low/mid severity restoration is at best neutral and usually slightly harmful
  (next section). The honest story: restoration only pays off when the image is *severely* noisy
  and the raw signal is already near chance.

---

## 5. Restoration proof (Phase 4b) — **the centerpiece (RQ3)**

### Table D — Two-sided proof: pixel fidelity vs diagnostic accuracy
Source: `phase4b_restoration_proof/metrics/restoration_proof_master.csv` (low+mid, mean over the
3 classifiers). PSNR/SSIM gain are **vs the degraded input**; acc_delta is **vs raw degraded**.

| Corruption | Restorer | ΔPSNR (dB) | ΔSSIM | Acc (restored) | Acc (raw) | **Δacc** |
|---|---|---|---|---|---|---|
| blur | CLAHE | −10.01 | −0.249 | 0.436 | 0.435 | +0.002 |
| blur | Cold Diffusion | +3.62 | +0.011 | 0.402 | 0.435 | −0.032 |
| blur | vanilla DDPM | −21.52 | −0.820 | 0.085 | 0.435 | −0.350 |
| blur | pathology DDPM | +2.09 | +0.004 | 0.364 | 0.435 | −0.071 |
| blur | A-ESRGAN | +1.74 | −0.008 | 0.405 | 0.435 | −0.029 |
| blur | SwinIR+GAN | −0.98 | −0.021 | 0.400 | 0.435 | −0.035 |
| exposure | CLAHE | +2.90 | −0.108 | 0.450 | 0.488 | −0.038 |
| exposure | Cold Diffusion | **+12.20** | **+0.112** | 0.448 | 0.488 | −0.041 |
| exposure | vanilla DDPM | −8.22 | −0.735 | 0.091 | 0.488 | −0.398 |
| exposure | pathology DDPM | +0.00 | −0.035 | 0.401 | 0.488 | −0.087 |
| exposure | A-ESRGAN | −0.08 | −0.017 | 0.421 | 0.488 | −0.068 |
| exposure | SwinIR+GAN | +3.61 | +0.002 | 0.396 | 0.488 | −0.092 |
| noise | CLAHE | −13.00 | −0.377 | 0.475 | 0.489 | −0.014 |
| noise | Cold Diffusion | −0.98 | +0.062 | 0.311 | 0.489 | −0.178 |
| noise | vanilla DDPM | −23.69 | −0.626 | 0.077 | 0.489 | −0.412 |
| noise | pathology DDPM | +1.16 | +0.199 | 0.394 | 0.489 | −0.095 |
| noise | A-ESRGAN | +2.51 | +0.215 | 0.427 | 0.489 | −0.062 |
| noise | SwinIR+GAN | +0.14 | +0.162 | 0.425 | 0.489 | −0.064 |

- **[INSIGHT] Every learned restorer has Δacc ≤ 0** at low/mid severity. The best image-quality
  result (Cold Diffusion exposure, +12.2 dB) still *loses* accuracy (−0.041). This is the
  empirical core of the paper.

### Table E — "Fidelity up, diagnosis down" (the paradox subset)
Source: `phase4b_restoration_proof/metrics/fidelity_up_accuracy_down.csv`. **7 of 18** cells where
the restorer improved **both** PSNR and SSIM yet accuracy fell.

| Corruption | Restorer | ΔPSNR | ΔSSIM | Δacc |
|---|---|---|---|---|
| exposure | Cold Diffusion | +12.20 | +0.112 | −0.041 |
| blur | Cold Diffusion | +3.62 | +0.011 | −0.032 |
| exposure | SwinIR+GAN | +3.61 | +0.002 | −0.092 |
| noise | A-ESRGAN | +2.51 | +0.215 | −0.062 |
| blur | pathology DDPM | +2.09 | +0.004 | −0.071 |
| noise | pathology DDPM | +1.16 | +0.199 | −0.095 |
| noise | SwinIR+GAN | +0.14 | +0.162 | −0.064 |

- **HEADLINE (use verbatim):** *Cold Diffusion on under-exposed images raised PSNR by 12.2 dB and
  SSIM by 0.112, yet downstream accuracy fell by 0.041 — reference image-quality metrics are not a
  safe proxy for diagnostic faithfulness.*

### Table F — Distribution-shift probe (clean images, no degradation)
Source: `phase4b_restoration_proof/metrics/distribution_shift.csv` (40 clean images). If a restorer
drops accuracy on *clean* images, the harm is OOD distribution shift, not lesion loss.

| Variant | ResNet-50 | EfficientNet-B3 | ViT-Base |
|---|---|---|---|
| clean (no restore) | 0.850 | 0.050 | 0.775 |
| CLAHE | 0.775 | 0.000 | 0.600 |
| A-ESRGAN | **0.875** | 0.150 | **0.775** |
| Cold Diffusion | 0.250 | 0.200 | 0.500 |
| SwinIR+GAN | 0.675 | 0.025 | 0.650 |
| DDPM (vanilla) | 0.150 | 0.000 | 0.200 |
| DDPM (pathology) | 0.650 | 0.025 | 0.700 |

- **[INSIGHT]** **A-ESRGAN is the only restorer that preserves the clean distribution**
  (ResNet 0.85→0.875, ViT 0.775→0.775). **Cold Diffusion and vanilla DDPM are heavily OOD** even
  with no degradation (ResNet 0.85→0.25 and →0.15). The pathology-DDPM is intermediate
  (0.85→0.65). This cleanly attributes most of the accuracy loss in Table D to distribution shift
  rather than erased lesions.

### Hallucination report (pathology-DDPM)
Source: `phase4_genai_enhancement/metrics/ddpm_pathology_hallucination.csv` (`cell 110`), 180
high-severity images.

- **[OBSERVED]** Grade-change (hallucination) rate **3.3%**; risk distribution **174 low / 6 medium
  / 0 high**; mean pixel deviation ~0.01–0.02 (L1 in [0,1]). The flagged changes are mostly
  Mild→Severe flips at blur-high.
- **[INSIGHT]** The three guardrails (classifier-feature perceptual loss, input-fidelity L1,
  sampling-time anti-hallucination clamp) keep the pathology-DDPM near the input and rarely flip the
  grade — the contrast with vanilla DDPM (PSNR ~8.5 dB, content destroyed) is the ablation that
  justifies the design.

### Diffusion process figures
`cell 117` saved forward/backward trajectory figures for both the conditional DDPM and Cold
Diffusion to `phase4b_restoration_proof/plots/{ddpm,cold_diffusion}_forward_backward.png` — use as
the methods/qualitative diffusion illustration.

---

## 6. Quality-aware routing (Phase 5 / RQ4)

Sources: `phase5_quality_ensemble/metrics/{ensemble_summary.csv, quality_thresholds.json,
quality_classifier_metrics.json, five_scenarios.csv}`.

- **[OBSERVED]** Quality classifier was **re-calibrated by threshold sweep** (`cell 94`):
  `p_good ≥ 0.20`, `p_usable ≥ 0.30`, giving a routing mix **good 0.31 / usable 0.41 / reject 0.28**
  on clean APTOS (target was ~60/25/15). Argmax routing had been over-rejecting almost everything.
- **[ANOMALY]** The underlying quality classifier is weak: overall accuracy **0.509**; the *usable*
  class is effectively broken (precision 0.074, recall 0.015, F1 0.025), while *good* (P 0.976,
  R 0.483) and *reject* (R 0.994) dominate. In practice it is a good-vs-reject detector; report this
  honestly.
- **[OBSERVED]** Routed accuracy (`ensemble_summary.csv`): clean **0.7625** (mean trust 0.680),
  exposure 0.6375, blur 0.575. Baseline single-best ViT clean acc is **0.8477**.
- **[INSIGHT / ANOMALY]** **Routing did not beat the best single grader or the ensemble** on clean
  data (0.7625 < 0.848 < 0.877). The contribution of Phase 5 is the *system design* — calibrated
  quality gating and a `reject → flag for re-acquisition` policy (changed from "enhance with GenAI",
  `cell 104`) consistent with the Phase 4 finding that restoration doesn't help — **not** a headline
  accuracy gain. Frame RQ4 accordingly.
- **[OBSERVED]** Five operational scenarios are covered with real examples
  (`five_scenarios.csv`): correct_routing, restoration_save, mis_routing_lucky, pipeline_failure,
  low_trust_correct — good material for a qualitative routing figure / case study.

---

## 7. Answers to the research questions

### RQ1 — How much does DR grading degrade under realistic corruption?
**Answer: severely, and non-uniformly by corruption type.** The v4 ensemble drops from
**QWK 0.949 / acc 0.877 (clean)** to **QWK 0.157 / acc 0.330 (noise-high)** (Table B). Ordering of
survivability: **exposure > blur > noise**; exposure-high still holds QWK 0.49 while noise-high is
near chance. *Evidence:* ensemble & per-model stress tests (Tables A–B).

### RQ2 — Do explanations stay faithful and robust under degradation?
**Answer: faithful on clean images, but not robust under degradation.** Clean insertion AUC is
strong for ResNet-50/ViT (0.62–0.77) but **stability collapses toward 0** for every method at high
severity (down to −0.05 for EfficientNet Grad-CAM). ViT-attention and ResNet Grad-CAM are the most
stable; EfficientNet explanations are unreliable throughout. *Evidence:* Phase 3 summary tables;
SHAP cross-model ρ≈0.34–0.51. *Caveat:* localization IoU not measured (no masks).

### RQ3 — Can generative restoration recover the diagnosis without hallucinating?
**Answer: no — restoration recovers pixels, not the diagnosis (main negative result).** At low/mid
severity **every learned restorer has Δaccuracy ≤ 0**, and in **7/18 cells PSNR & SSIM both rise
while accuracy falls** (Tables D–E; headline Cold-Diffusion exposure +12.2 dB / −0.041 acc). The
distribution-shift probe (Table F) shows the harm is largely because restorer outputs are OOD for
the grader (only A-ESRGAN preserves clean accuracy). The pathology-DDPM **does control
hallucination** (3.3% grade-change, 0 high-risk) and is the **only restorer that rescues accuracy at
noise-high** (ResNet 0.218→0.575). *Evidence:* Phase 4b proof tables + hallucination CSV.
*Net:* image-quality metrics are not a safe proxy for diagnostic faithfulness — the paper's thesis.

### RQ4 — Does quality-aware routing beat a single grader?
**Answer: not on accuracy in this run; its value is system design and safety.** Routed clean
accuracy **0.7625** < single-best ViT **0.8477** < ensemble **0.877**. The quality classifier is a
de-facto good/reject detector (usable class broken). The defensible contribution is the calibrated
gating + `reject → re-acquire` policy that is consistent with RQ3 (don't waste compute restoring
ungradable images). *Evidence:* `ensemble_summary.csv`, `quality_classifier_metrics.json`.

---

## 8. Limitations & honesty notes (write the Limitations section from these)
- Degradations are **synthetic**; restorers were trained to invert exactly those operators —
  external validation on genuinely low-quality clinical images is still needed.
- **APTOS-only**; EyeQ never contributed (join failed) — quality labels are weak, and the Phase 5
  quality classifier is correspondingly unreliable (acc 0.509, usable class collapsed).
- Diffusion restorers use **fast sampling** (Cold Diffusion 8 steps, vanilla/pathology DDPM
  truncated) — quality scales with steps; always state the step count beside any restoration claim.
- The pathology perceptual loss preserves only features **our own grader** finds salient → inherits
  any grader bias.
- **EfficientNet-B3** is weak/uncalibrated and pathological under degradation; treat it as a
  reported failure case, not a contributor to headline numbers.
- **LPIPS and FID were not computed** in this run (only PSNR/SSIM) — report reference quality as
  PSNR/SSIM, or add LPIPS before submission.
- **CycleGAN-CBAM** is implemented but excluded from all experiments — future work only.

---

## 9. Paper-mapping appendix (finding → where it goes)

| Finding (this doc) | Fills MASTER_PROMPT / paper slot |
|---|---|
| Table B ensemble clean QWK 0.949, noise-high 0.157 | `[QWK]` placeholders; Results §5, Table "Robustness" |
| Tables A & C (v2 vs v3) | "Multi-scale fusion improves grading" claim; Results §5 |
| Phase 3 stability/insertion tables | "Explanations faithful & robust" claim; Results §5 XAI table |
| Table D two-sided proof | "Restoration recovers diagnosis, not pixels" claim; Results §5 restoration table |
| Table E fidelity-up/accuracy-down (7/18) | Discussion §6 centerpiece; Abstract headline |
| Table F distribution-shift | Discussion §6 mechanism; supplementary |
| Hallucination 3.3% / 0 high-risk | "Vanilla DDPM hallucinates; pathology-DDPM reduces it" claim |
| Phase 5 routed 0.7625 vs 0.848/0.877 | "Quality-aware routing" claim (nuanced); Results §5 + Discussion |
| §8 list | Limitations §7 |

*All numbers verifiable in `result/results/`; cross-checked against notebook cell outputs of
`notebooks/Thesis_v3_restoration_proof.ipynb`.*
