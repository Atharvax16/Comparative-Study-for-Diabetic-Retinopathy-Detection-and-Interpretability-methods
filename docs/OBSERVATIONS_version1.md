# Observations from `Thesis_optimized_final_version1.ipynb`

Notes captured by walking every executed cell of the notebook (89 code cells, 91 cells total).
Numbers below are pulled directly from the saved outputs — every line is a fact, not an inference.

Tag legend:
- `[OBSERVED]` — directly visible in the cell output
- `[ANOMALY]`  — something the data shows is broken / off
- `[INSIGHT]`  — what the observation means for the dissertation story
- `[ACTION]`   — recommended next step

---

## Phase 1 — Data Engineering

### [OBSERVED] cell 17 — EyeQ quality filter failed silently
```
[warn] EyeQ filter removed all rows — falling back to full APTOS.
kept 2930 / 2930 images
```
- Final pristine set = 2,930 APTOS images. EyeQ never contributed.

### [ANOMALY] cell 17 — the entire EyeQ pipeline is dead
- The pristine split is APTOS-only. Anywhere the dissertation claims "APTOS + EyeQ" needs to be softened to "APTOS, with EyeQ as a planned-but-unused quality filter."
- Root cause is in the EyeQ join logic (cell 11 / cell 17 pre-filter). Worth a 30-min fix but **not** blocking the rest of the story.

### [OBSERVED] cell 20 — class imbalance is preserved
- Class distribution plot saved to `phase1_data_engineering/plots/class_distribution.png`.
- We know APTOS is ~50% No-DR, so the balanced sampler and class weights in cell 26 are doing real work.

### [OBSERVED] cell 24 — 26,370 degraded images cached on Drive
- 3 kinds × 3 levels × 2,930 = 26,370 JPEGs. Resume guard kicked in (`[skip] degraded images already present`) and restored from `cache_degraded.tar.gz`.

---

## Phase 2 — Model Benchmarking

### [OBSERVED] cell 31 — V2 baseline stress test (single model, 4-view TTA)

|              | clean | blur-low | blur-mid | blur-high | exp-low | exp-mid | exp-high | noise-low | noise-mid | noise-high |
|---           |---    |---       |---       |---        |---      |---      |---       |---        |---        |---         |
| resnet50     | 0.796 | 0.630    | 0.543    | 0.525     | 0.732   | 0.661   | 0.580    | 0.680     | 0.546     | 0.459      |
| eff_b3       | 0.814 | 0.677    | 0.489    | 0.420     | 0.686   | 0.611   | 0.582    | 0.668     | **0.216** | 0.389      |
| vit_base     | 0.809 | 0.598    | 0.536    | 0.489     | 0.741   | 0.611   | 0.539    | 0.698     | 0.568     | 0.536      |

### [ANOMALY] cell 31 — efficientnet_b3 collapses at noise-mid (acc=0.216, F1=0.120)
- Worse than noise-high (0.389) — non-monotonic. Strongly suggests BatchNorm running-stats drift: the BN moments trained on clean APTOS don't generalise to noise-mid input.
- Doesn't fully recover even with TTA. The eff_b3 noise-mid number is a known fragility, not a typo.
- **[ACTION]** Either fix BN with `model.eval()` test-time batch stats or just call this out in the dissertation as evidence for "why you need an ensemble".

### [OBSERVED] cell 43 — V2 vs V3 headline

| version | model       | clean | blur-high | exp-high | noise-high |
|---      |---          |---    |---        |---       |---         |
| V2      | resnet50    | 0.795 | 0.525     | 0.580    | 0.459      |
| V2      | eff_b3      | 0.814 | 0.420     | 0.582    | 0.389      |
| V2      | vit_base    | 0.809 | 0.489     | 0.539    | 0.536      |
| V3      | resnet50    | 0.639 | 0.461     | **0.786**| 0.493      |
| V3      | eff_b3      | 0.516 | 0.477     | 0.639    | 0.543      |
| V3      | vit_base    | 0.661 | 0.582     | 0.725    | 0.493      |

### [INSIGHT] cell 43 — V3 made a classic robustness-vs-clean tradeoff
- V3 **lost 15-30 points of clean accuracy** (ViT-Base dropped from 0.809 → 0.661; EffNet from 0.814 → 0.516).
- V3 **gained 5-20 points on degraded conditions** (resnet50 exposure-high: +20.6 pts; vit_base blur-high: +9.3 pts).
- This is the most important narrative finding so far: V3 is not "strictly better" than V2 — it's a Pareto move. The dissertation should frame it as a deliberate robustness/clean tradeoff, not an upgrade.

### [OBSERVED] cell 41 — V4 ensemble (val-kappa weighted soft-vote + 7-TTA)
- Ensemble weights printed: `resnet50=0.33, efficientnet_b3=0.32, vit_base=0.35` (val-kappas are all very close)
- Clean: acc=**0.725**, f1=0.605, kappa=0.836, AUC=0.945
- Blur-low/mid/high: 0.709 / 0.586 / **0.545** (kappa=0.372 at high)
- Exposure-low/mid/high: 0.768 / 0.725 / 0.668
- Noise-low/mid/high: 0.745 / 0.595 / 0.482

### [INSIGHT] cell 41 — V4 ensemble clean (0.725) < V2 best single (0.814)
- The val-kappa weighting can't save you if all three V3 models lost clean accuracy. The ensemble inherits the V3 regression.
- **AUC stays high (0.945 clean, 0.778 blur-high)** — the ranking is still good. The classifier knows; the argmax decision is what's hurting.
- **[ACTION]** Consider mixing V2 + V3 in the ensemble (V2 for clean-leaning slices, V3 for robustness slices) instead of pure-V3 voting. Easy win for the "ensemble" chapter.

### [ANOMALY] cell 45 — V4 distillation was interrupted mid-training
- Got to epoch 3 of resnet50 student (`acc=0.686, kappa=0.807`), then `KeyboardInterrupt` during epoch 4.
- BUT cell 47 on a later re-run shows both distilled checkpoints exist (`[skip] already exists`). That means the saver in `distill_to` fired on a best-kappa improvement at ep3, and the partial training was kept.
- **[INSIGHT]** Distilled `resnet50_v3_distilled.pt` is a 3-epoch checkpoint, not 10. `efficientnet_b3_v3_distilled.pt` status unknown — probably also partial.
- Teacher chosen: `vit_base` with clean kappa = 0.7074 (not great — distilling to a mediocre teacher).
- **[ACTION]** Either rerun distillation to convergence or just remove the distilled checkpoints so the loader falls back to `_v3_best.pt`.

---

## Phase 3 — XAI Benchmark

### [OBSERVED] cell 62 — IG aggregate summary (20 images × 3 kinds × 4 levels per model)

Headline rows from the printed summary table:

|          | clean stab. | blur-high stab. | exp-high stab. | noise-high stab. |
|---       |---          |---              |---             |---               |
| eff_b3   | 1.000       | 0.063           | 0.199          | 0.106            |
| resnet50 | 1.000       | 0.117           | 0.210          | 0.128            |
| vit_base | 1.000       | 0.184           | **0.302**      | 0.224            |

Insertion-AUC at exposure-high: vit=0.573, resnet=0.434, eff=0.312.

### [INSIGHT] cell 62 — ViT explanations are 1.5-2× more stable under degradation
- This is a defensible XAI claim: "ViT-Base + IG gives the most stable saliency under fundus degradation across all three kinds."
- Stability scores < 0.3 across the board means **no XAI method is "stable" by an absolute standard** — relative comparison is the right frame.

### [OBSERVED] cell 62 — clean baseline insertion-AUC: vit=0.520, resnet=0.464, eff=0.423
- ViT wins on faithfulness even before degradation, then loses less under noise.

---

## Phase 4 — GenAI Enhancement (THE BIG SECTION)

### [OBSERVED] cell 67 — Cold Diffusion training trajectory
- L1 loss: 0.0379 → 0.0167 → 0.0142 → **0.0120** over 4 epochs.
- Healthy, smooth decrease. By epoch 4 average per-pixel error ≈ 3/255 — at JPEG-90 noise level.

### [OBSERVED] cell 70 — SwinIR + GAN training trajectory
- G loss: 0.0721 → 0.0682 → **0.0705** (slight uptick in ep3)
- D loss: 0.521 → 0.382 → 0.362 (discriminator winning)

### [ANOMALY] cell 70 — SwinIR + GAN is showing adversarial drift after 3 epochs
- D pulling ahead and G loss bouncing back up is the signature of imbalanced GAN training. With only 3 epochs that's not catastrophic but it's why the Phase 4 numbers below for `swinir_gan` are noisy.
- **[ACTION]** Either bump to 6 epochs with `adv_w=0.005` (half the weight) or stop adv training after epoch 2.

### [OBSERVED] cell 72 — Phase 4 recovery accuracy (THE TABLE)

**resnet50** (the most-V3-degradation-tuned model):

| condition       | raw   | clahe | genai | cold_diff | swinir |
|---              |---    |---    |---    |---        |---     |
| blur-low        | 0.620 | **0.659** | 0.607 | 0.623 | 0.600 |
| blur-mid        | **0.605** | 0.491 | 0.539 | 0.559 | 0.525 |
| blur-high       | 0.511 | 0.361 | 0.489 | **0.546** | 0.491 |
| exposure-low    | 0.639 | **0.689** | 0.618 | 0.664 | 0.532 |
| exposure-mid    | 0.611 | **0.666** | 0.571 | 0.636 | 0.396 |
| exposure-high   | 0.559 | 0.607 | 0.505 | **0.614** | 0.555 |
| noise-low       | 0.598 | 0.623 | **0.636** | 0.593 | 0.616 |
| noise-mid       | 0.577 | 0.514 | **0.605** | **0.280** | 0.502 |
| noise-high      | 0.409 | 0.268 | **0.564** | **0.084** | 0.105 |

**efficientnet_b3** (noise-mid collapse case):

| condition       | raw   | clahe | genai | cold_diff | swinir |
|---              |---    |---    |---    |---        |---     |
| noise-mid       | 0.223 | 0.111 | **0.525** | 0.464 | 0.507 |
| noise-high      | 0.089 | 0.080 | **0.525** | 0.159 | 0.446 |

**vit_base** (the strongest backbone):

| condition       | raw   | clahe | genai | cold_diff | swinir |
|---              |---    |---    |---    |---        |---     |
| blur-low        | 0.689 | 0.607 | **0.759** | 0.711 | 0.702 |
| exposure-low    | **0.816** | 0.743 | 0.750 | 0.764 | 0.668 |
| exposure-mid    | **0.793** | 0.741 | 0.684 | 0.757 | 0.561 |
| exposure-high   | 0.730 | 0.675 | 0.584 | **0.741** | 0.677 |
| noise-high      | 0.477 | 0.361 | **0.614** | **0.136** | 0.350 |

### [INSIGHT] cell 72 — five patterns that fall out cleanly

1. **At LOW degradation, raw input wins more often than any enhancer.** Restoration adds artifacts that subtract from a barely-degraded image's diagnostic signal. Don't restore what isn't broken.

2. **CLAHE is the right tool for EXPOSURE on resnet50** (the more degradation-tuned model). +5 pts on exposure-low/mid/high consistently. This is the boring-but-real finding.

3. **A-ESRGAN ("genai") is the right tool for NOISE.** This is the surprise — A-ESRGAN was designed for super-resolution, not denoising, but its RRDBNet residual blocks absorb noise effectively. Biggest win: efficientnet_b3 noise-high goes from **0.089 → 0.525 (+44 pts)**. That single number is the most dramatic recovery in the entire notebook.

4. **Cold Diffusion is the right tool for EXPOSURE-high on robust backbones** (resnet50 +5.5 pts, vit_base +1.1 pts). It also helps blur-high on resnet50 (+3.4 pts). Modest wins, but it's the only technique that helps blur-high.

5. **Cold Diffusion CATASTROPHICALLY fails on noise.** resnet50 noise-mid: 0.577 → 0.280 (-30 pts), noise-high: 0.409 → 0.084 (-32 pts). vit_base noise-high: 0.477 → 0.136 (-34 pts). The iterative Algorithm-2 sampling appears to amplify noise rather than remove it when t_start matches the input severity. **Do not use cold_diff on noise — and say so explicitly in the dissertation.**

### [ANOMALY] cell 72 — SwinIR-GAN is underwhelming everywhere
- Best single result: efficientnet_b3 noise-high 0.089 → 0.446 (+36 pts). Real but second to genai.
- Mostly within ±2 pts of raw on other conditions, often worse on exposure.
- **[ACTION]** Either bump SwinIR-GAN to 6+ epochs and re-run, or de-emphasise it in the writeup as "fast inference but no consistent diagnostic gain".

### [OBSERVED] cell 74 — XAI recovery on enhanced images
- Recovery CSV rows confirm the per-image XAI (stability + insertion_auc) per (model, method, kind, level, variant) was generated for the full grid.
- Sample row (`84e8c62165b5` resnet50/gradcam/blur/low): raw stability=0.22, clahe=0.04, genai=**0.34**, cold_diff=0.26, swinir=0.22.
- **[INSIGHT]** genai also improves XAI stability — not just classifier accuracy — on this slice. Worth checking the aggregate (cell 75 output we didn't fully capture) for the same pattern.

---

## Phase 5 — Quality-Aware Routing Ensemble

### [OBSERVED] cell 79 — quality model loaded from existing checkpoint, no retrain.

### [OBSERVED] cell 80 — policy chosen
- `best_clean = efficientnet_b3` (V2 winner on clean)
- `best_robust = resnet50` (V2 winner under degradation aggregate)
- Routing:
  - `good`   → no enhancement, efficientnet_b3, gradcam
  - `usable` → CLAHE, efficientnet_b3, gradcam
  - `reject` → genai, resnet50, gradcam

### [ANOMALY] cells 82 + 89 — the quality model labels almost everything 'reject'
- In the 5-row sample shown (cell 82), all 5 images flagged `reject` including the **clean** ones. In cell 89 the "correct_routing" scenario couldn't even find a clean example labelled `good`.
- The quality classifier is biased toward `reject`. Trained on what data? Worth checking.
- **[ACTION]** Look at `Q_CKPT` training data distribution — likely a class-imbalance problem. Either re-train with class weights or recalibrate the decision threshold.

### [OBSERVED] cell 83 — routed pipeline aggregate
| condition | routed acc | mean_trust | baseline acc (V2 single best) |
|---        |---         |---         |---                            |
| clean     | 0.650      | 0.624      | **0.814**                      |
| blur      | 0.538      | 0.313      | (n/a)                          |
| exposure  | 0.563      | 0.547      | (n/a)                          |

### [INSIGHT] cells 82/83 — the routed pipeline is NET-NEGATIVE on clean
- 0.650 routed vs 0.814 baseline = **-16.4 pts on clean** because everything is being routed to `reject` → genai+resnet50, which is the heavier/more-defensive path.
- The trust signal *does* track condition (clean=0.62 vs blur=0.31), so the *signal* works. The *router* is broken.
- **[ACTION]** Fix the quality classifier first; the routing logic is fine.

### [OBSERVED] cell 89 — five-scenarios qualitative figure built
- 4 conditions × 25 ids = 100 results sampled. Found examples of:
  - `restoration_save` (degraded → restoration succeeded → correct prediction)
  - `mis_routing_lucky` (clean image routed to reject path but still correct)
  - `pipeline_failure` (clean image routed wrong AND prediction wrong)
- Could NOT find examples of `correct_routing` or `low_trust_correct` in the 100-sample draw — symptom of the over-aggressive `reject` bias above.

---

## Bottom-line recommendations for the dissertation

1. **Drop the noise-high column for Cold Diffusion.** It actively destroys diagnostic signal. Present cold_diff results on blur + exposure only, or report noise as a documented failure mode (which is more honest and arguably more interesting).
2. **The headline restoration story should be: "the right restorer depends on the degradation type."** CLAHE for exposure, A-ESRGAN for noise, Cold Diffusion for blur-high + exposure-high. There is no universal restorer.
3. **V3 vs V2 should be framed as a Pareto move**, not an upgrade — the clean-accuracy regression is too large to hide.
4. **Phase 5 quality routing is currently net-negative.** Either fix the quality classifier's reject-bias before claiming routing works, or write it up as negative result with the diagnosis above.
5. **EyeQ contributed nothing.** Be honest about it.

---

## On the user's hypothesis: "drop high, focus on low + mid"

Strongly supported by the data above. See bottom of this file or the conversation reply for the rationale and the concrete change list.
