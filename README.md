# Robust and Explainable AI for Diabetic Retinopathy

A dissertation project investigating the robustness and interpretability of deep-learning models for Diabetic Retinopathy (DR) detection under varying image-quality conditions, with a focus on Generative-AI image restoration as a recovery mechanism.

> **Goal.** Build a DR-screening pipeline that maintains high diagnostic accuracy *and* trustworthy explanations even when input fundus photographs are blurred, under-exposed, or noisy — and that can transparently route low-quality images through a GenAI restoration step before classification.

---

## Research Questions

| RQ | Question |
| -- | -------- |
| **RQ1** | How do Vision Transformers (ViT-Base) compare to CNNs (ResNet-50, EfficientNet-B3) under varying levels of synthetic image degradation (blur, exposure, noise)? |
| **RQ2** | To what extent does the fidelity and stability of explainability methods (Grad-CAM, SHAP, Self-Attention) degrade under different image-quality conditions? |
| **RQ3** | Can GenAI-based quality enhancement (e.g. diffusion / Swin2SR) significantly recover both diagnostic accuracy and XAI localization precision compared to a CLAHE baseline? |
| **RQ4** | Can a quality-aware ensemble framework dynamically pick the optimal enhancement-model-XAI combination to maximize clinical trust in real-world DR screening? |

---

## Methodology — Five Phases

```
APTOS 2019 + EyeQ
        |
   [Phase 1] Quality filter -> pristine subset -> 9 synthetic degradations
        |
   [Phase 2] Train ResNet-50 / EfficientNet-B3 / ViT-Base, stress-test on degraded sets        -> RQ1
        |
   [Phase 3] Grad-CAM (CNNs) / SHAP (all) / Attention rollout (ViT)
            + Faithfulness (Deletion/Insertion AUC), Stability (Spearman), Localization (IoU)  -> RQ2
        |
   [Phase 4] CLAHE baseline + GenAI restoration (Swin2SR)
            -> re-evaluate accuracy + XAI fidelity                                              -> RQ3
        |
   [Phase 5] EyeQ-trained quality classifier -> dynamic routing -> Clinical Trust Score        -> RQ4
```

Each phase writes its outputs to a separate directory so results can be presented in isolation.

---

## Repository Structure

```
.
├── Thesis.ipynb                                          # consolidated single-notebook version (run end-to-end)
├── notebooks/
│   ├── 00_setup.ipynb                                    # Drive mount, dataset extraction, deps
│   ├── 01_phase1_data_engineering.ipynb                  # quality filter + synthetic degradation
│   ├── 02_phase2_model_benchmarking.ipynb                # train + stress-test all three models
│   ├── 03_phase3_xai_benchmark.ipynb                     # XAI methods + faithfulness/stability metrics
│   ├── 04_phase4_genai_enhancement.ipynb                 # CLAHE + GenAI restoration + recovery analysis
│   └── 05_phase5_quality_aware_ensemble.ipynb            # routed pipeline + clinical trust score
├── results/                                              # gitignored in practice — see "Outputs"
│   ├── phase1_data_engineering/{metrics,plots,samples}/
│   ├── phase2_model_benchmarking/{metrics,plots}/
│   ├── phase3_xai_benchmark/{metrics,plots,samples}/
│   ├── phase4_genai_enhancement/{metrics,plots,samples}/
│   └── phase5_quality_ensemble/{metrics,plots,samples}/
├── Project_Master_Overview_Robust_and_Explainable_AI_for_Diabetic_Retinopathy.pdf
├── thesis.webp                                           # pipeline diagram (also shown above)
└── README.md
```

---

## Datasets

This repository does **not** ship the data. Download from the original sources and zip them as `Thesis.zip` containing `APTOS.zip` and `EYEQ.zip` inside.

| Dataset | Used for | Source |
| ------- | -------- | ------ |
| APTOS 2019 Blindness Detection | Primary classification task (5 grades of DR severity) | Kaggle competition |
| EyeQ | Per-image quality labels (good / usable / reject) for Phase 1 filtering and Phase 5 routing | Public release on GitHub |

Expected on Drive:
```
MyDrive/Thesis/Thesis.zip          # outer container
   APTOS.zip                       # inner — APTOS images + train.csv
   EYEQ.zip                        # inner — EyeQ images + quality CSV
```
The setup notebook auto-detects nested zips and extracts everything to `/content/data/` on Colab local disk.

---

## Setup (Google Colab)

1. **Upload the data**: place `Thesis.zip` in `MyDrive/Thesis/` on Google Drive.
2. **Open** `Thesis.ipynb` (or, if you prefer per-phase notebooks, the `notebooks/` folder).
3. **Choose runtime** with GPU enabled — A100 recommended; T4 works but Phase 2 training takes longer.
4. Run the **setup section** once. It mounts Drive, installs dependencies, extracts the zips, and creates the per-phase results directories.

The notebook installs the following at runtime:
```
torch, torchvision, timm, captum, shap, grad-cam,
scikit-image, opencv-python-headless, tabulate,
diffusers, transformers, accelerate, optuna
```

No local Python environment is required — everything runs in Colab.

---

## Usage

### Single-notebook workflow (recommended)
Open `Thesis.ipynb` and run cells top to bottom. Each phase section writes its own results and is independent within a session.

### Per-phase workflow
Open the matching notebook in `notebooks/` and run it. Phases must run in order on a fresh session because each phase consumes the previous phase's outputs.

### Resuming an interrupted session
After Phase 1 / Phase 4 finish, the notebook offers to back up `degraded/` and `enhanced/` to Drive as gzip caches. On the next session a resume cell pulls them back to local disk so Phase 1 / Phase 4 don't re-run.

---

## Outputs

Per-phase artefacts under `Drive/Thesis/results/phase<N>_<name>/`:

| Folder | Contents |
| ------ | -------- |
| `metrics/` | CSVs and JSONs (training history, stress-test results, XAI summaries, ensemble predictions, etc.) |
| `plots/` | Headline figures: accuracy-vs-degradation curves, XAI faithfulness/stability curves, recovery plots, confusion matrices. |
| `samples/` | Qualitative figures: degradation grids, heatmap overlays, side-by-side restorations. |
| `logs/` | Optional run logs. |

Trained models live in `Drive/Thesis/checkpoints/` (`<arch>_best.pt` for each classifier, plus the EyeQ quality classifier).

---

## Current Status

| Phase | State | Notes |
| ----- | ----- | ----- |
| 1 — Data engineering | Complete | 2,930 pristine images × 9 degraded variants. Disk-optimised by writing 224×224 JPEGs. |
| 2 — Model benchmarking | First pass complete | ResNet-50 0.838 acc, EfficientNet-B3 0.806 acc, ViT-Base 0.686 acc. ViT being re-trained with proper hyperparameters (lower LR, layer-wise LR decay, longer warmup, MixUp). |
| 3 — XAI benchmark | First pass complete | Attention rollout most stable (Spearman > 0.9 across degradations). Grad-CAM degrades sharply under noise. SHAP currently undersampled (`n=60`) — bumping for next pass. |
| 4 — GenAI enhancement | First pass complete | Swin2SR + CLAHE evaluated. Negative result on most conditions in current pass — switching to a stronger restorer (Real-ESRGAN / fundus-specific CycleGAN). |
| 5 — Quality-aware ensemble | First pass complete | EyeQ classifier 90% accuracy. End-to-end routing implemented; trust-score formula being calibrated against per-image correctness. |

---

## Roadmap

- Class-imbalance fixes: stratified split, focal loss with class weights, balanced sampler, MixUp/CutMix.
- Per-architecture training recipe with Optuna hyperparameter search.
- Replace TinyU-Net fallback with Real-ESRGAN / fundus-tuned CycleGAN for Phase 4.
- Add IoU localization against IDRiD lesion masks.
- Calibrated trust score (temperature scaling on confidence + normalized insertion AUC).
- Bootstrap 95% confidence intervals on all reported metrics.

---

## Tech Stack

- **Framework**: PyTorch + timm
- **XAI**: Captum, SHAP, custom Grad-CAM and attention-rollout implementations
- **Restoration**: Swin2SR (HuggingFace `transformers`), TinyU-Net fallback
- **Ops**: scikit-learn, scikit-image, OpenCV, Pillow, tqdm, matplotlib
- **Hyperparameter search**: Optuna
- **Runtime**: Google Colab (A100 / T4)

---

## Citation

If you use this code or its results, please cite:

```
@misc{kocharekar_dr_robust_xai_2026,
  author       = {Kocharekar, Atharva},
  title        = {Robust and Explainable AI for Diabetic Retinopathy:
                  Quality-Aware Ensemble with GenAI Restoration},
  year         = {2026},
  howpublished = {Master's dissertation},
  url          = {https://github.com/<your-username>/<this-repo>}
}
```

---

## License

Released under the MIT License. See `LICENSE` for details. Dataset usage is governed by the original APTOS 2019 and EyeQ licences.

---

## Acknowledgements

- APTOS 2019 Blindness Detection — Asia Pacific Tele-Ophthalmology Society.
- EyeQ dataset — for the quality-grading labels that drive Phase 1 filtering and Phase 5 routing.
- Open-source community for `timm`, `captum`, `shap`, `grad-cam`, `diffusers`, and `transformers`.
