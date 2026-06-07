"""Append-only 'V7 overnight booster' cells to the resume notebook.

Touches no existing cell. Adds fault-isolated stages that save incrementally.
Run:  python scripts/append_booster.py
"""
import json, sys, io
from pathlib import Path

NB = Path("notebooks/Thesis_v3_resume_DDPM_to_Phase5.ipynb")

# ----------------------------------------------------------------------------
MD0 = r'''## V7 — Overnight booster pack (append-only, fault-tolerant)

These cells were **appended** below the original pipeline. They do **not** modify
any existing cell. Each stage is wrapped so a failure is logged and skipped —
the remaining stages still run and results are saved incrementally to
`results/phase6_overnight_boost/`.

Order (safest first, so a crash late at night never loses the important results):

- **A** Enriched recovery metrics — QWK, balanced acc, per-class recall, bootstrap 95% CIs (inference only)
- **B** Temperature-scaling calibration + ECE before/after (inference only)
- **C** Confidence-based selective prediction + triage vs do-nothing vs restore-all (inference only)
- **D** Train-time degradation-augmentation robust baseline (training — best effort)
- **E** Classifier adapted to A-ESRGAN-restored images (training — best effort)
- **F** Summary report

Re-running is safe: training stages skip if their checkpoint already exists.
'''

SETUP = r'''# === V7 OVERNIGHT BOOSTER PACK — setup & helpers ============================
# Append-only. Reuses functions defined earlier: load_classifier_v3, evaluate_v3,
# build_model_v3, train_v3, FolderDatasetV3, deg_loader_v3, apply_degradation, etc.
import os, time, traceback
import numpy as np, pandas as pd, torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             balanced_accuracy_score, recall_score, roc_auc_score)

P_BOOST = RESULTS_ROOT / 'phase6_overnight_boost'
for _sub in ('metrics', 'plots', 'logs', 'checkpoints'):
    (P_BOOST / _sub).mkdir(parents=True, exist_ok=True)
_LOG = P_BOOST / 'logs' / 'overnight.log'

def blog(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    try:
        with open(_LOG, 'a') as f:
            f.write(line + '\n')
    except Exception:
        pass

_STAGE_STATUS = []
def run_stage(name, fn):
    blog(f"==== STAGE START: {name} ====")
    t0 = time.time()
    try:
        fn()
        status = 'ok'
        blog(f"==== STAGE OK: {name} ({time.time()-t0:.0f}s) ====")
    except Exception as e:
        status = 'FAILED'
        blog(f"==== STAGE FAILED: {name}: {e!r} ====")
        blog(traceback.format_exc())
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    _STAGE_STATUS.append({'stage': name, 'status': status,
                          'seconds': round(time.time() - t0)})
    pd.DataFrame(_STAGE_STATUS).to_csv(P_BOOST / 'metrics' / 'stage_status.csv', index=False)

# --- fallbacks so this section is robust even if some globals are absent ------
try:
    VARIANTS
except NameError:
    VARIANTS = ('raw', 'clahe', 'genai', 'cold_diff', 'swinir_gan', 'ddpm', 'ddpm_path')
try:
    test_id_set
except NameError:
    test_id_set = set(pd.read_csv(P2 / 'metrics' / 'test_ids.csv')['id_code'].astype(str))

# --- enriched-metric helpers -------------------------------------------------
def per_class_recall(y, p, n=NUM_CLASSES):
    r = recall_score(y, p, labels=list(range(n)), average=None, zero_division=0)
    return {f'recall_c{i}': float(r[i]) for i in range(n)}

def _qwk(a, b):
    return cohen_kappa_score(a, b, weights='quadratic')

def bootstrap_ci(y, p, fn, n_boot=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); p = np.asarray(p); N = len(y); stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        try:
            stats.append(fn(y[idx], p[idx]))
        except Exception:
            pass
    if not stats:
        return (float('nan'), float('nan'))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)

@torch.no_grad()
def collect_preds(model, loader):
    model.eval(); ys = []; prs = []
    for x, y, _ in loader:
        x = x.to(DEVICE, non_blocking=True)
        prs.append(torch.softmax(model(x), 1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(prs)

def variant_root(variant, k, l):
    return (DEGRADED_DIR / k / l) if variant == 'raw' else (ENHANCED_DIR / variant / k / l)

def make_loader(root, bs=24):
    ds = FolderDatasetV3(root)
    ds.df = ds.df[ds.df['id_code'].astype(str).isin(test_id_set)].reset_index(drop=True)
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)

blog('Booster setup ready. Output dir: ' + str(P_BOOST))
'''

STAGE_A = r'''# === STAGE A — enriched recovery metrics (QWK + per-class + CIs) ============
def stage_enriched_metrics():
    out_csv = P_BOOST / 'metrics' / 'recovery_metrics_enriched.csv'
    rows = []
    for name in MODEL_NAMES:
        model = load_classifier_v3(name)
        for k in DEGRADATION_TYPES:
            for l in DEGRADATION_LEVELS:
                for variant in VARIANTS:
                    root = variant_root(variant, k, l)
                    if not (root / 'manifest.csv').exists():
                        continue
                    y, pr = collect_preds(model, make_loader(root))
                    if len(y) == 0:
                        continue
                    p = pr.argmax(1)
                    rec = {'model': name, 'degradation': k, 'level': l,
                           'variant': variant, 'n': int(len(y)),
                           'accuracy': accuracy_score(y, p),
                           'balanced_acc': balanced_accuracy_score(y, p),
                           'f1_macro': f1_score(y, p, average='macro', zero_division=0),
                           'kappa_qw': _qwk(y, p)}
                    lo, hi = bootstrap_ci(y, p, accuracy_score)
                    rec['acc_ci_lo'], rec['acc_ci_hi'] = lo, hi
                    lo, hi = bootstrap_ci(y, p, _qwk)
                    rec['kappa_ci_lo'], rec['kappa_ci_hi'] = lo, hi
                    rec.update(per_class_recall(y, p))
                    rows.append(rec)
            blog(f"  [A] {name}/{k} done")
        pd.DataFrame(rows).to_csv(out_csv, index=False)   # incremental save per model
        del model
        torch.cuda.empty_cache()
    blog(f"[A] saved {out_csv} ({len(rows)} rows)")

run_stage('A_enriched_metrics', stage_enriched_metrics)
'''

STAGE_B = r'''# === STAGE B — temperature-scaling calibration + ECE ========================
def stage_calibration():
    def _logits(model, loader):
        model.eval(); ys = []; lg = []
        with torch.no_grad():
            for x, y, _ in loader:
                x = x.to(DEVICE, non_blocking=True)
                lg.append(model(x).float().cpu())
                ys.append(y.clone())
        return torch.cat(lg), torch.cat(ys)

    def _ece(probs, y, n_bins=15):
        conf = probs.max(1); pred = probs.argmax(1); correct = (pred == y).astype(float)
        bins = np.linspace(0, 1, n_bins + 1); e = 0.0
        for i in range(n_bins):
            m = (conf > bins[i]) & (conf <= bins[i + 1])
            if m.sum() > 0:
                e += abs(correct[m].mean() - conf[m].mean()) * m.mean()
        return float(e)

    rows = []
    for name in MODEL_NAMES:
        model = load_classifier_v3(name)
        lv, yv = _logits(model, val_loader_v3)
        lt, yt = _logits(model, test_loader_v3)
        T = torch.ones(1, requires_grad=True)
        opt = torch.optim.LBFGS([T], lr=0.01, max_iter=100)
        nll = torch.nn.CrossEntropyLoss()
        def closure():
            opt.zero_grad(); loss = nll(lv / T.clamp(min=0.05), yv); loss.backward(); return loss
        opt.step(closure)
        Tval = float(T.detach().clamp(min=0.05))
        yt_np = yt.numpy()
        pr_pre = torch.softmax(lt, 1).numpy()
        pr_post = torch.softmax(lt / Tval, 1).numpy()
        rows.append({'model': name, 'temperature': Tval,
                     'ece_pre': _ece(pr_pre, yt_np), 'ece_post': _ece(pr_post, yt_np),
                     'acc': float((pr_post.argmax(1) == yt_np).mean())})
        blog(f"  [B] {name}: T={Tval:.3f}  ECE {rows[-1]['ece_pre']:.3f} -> {rows[-1]['ece_post']:.3f}")
        del model
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(P_BOOST / 'metrics' / 'calibration_ece.csv', index=False)
    blog("[B] calibration_ece.csv saved")

run_stage('B_calibration', stage_calibration)
'''

STAGE_C = r'''# === STAGE C — selective prediction + triage vs do-nothing vs restore-all ===
def stage_selective_triage():
    import matplotlib.pyplot as plt
    # Fixed prior policy (NOT chosen by peeking at test labels): the only cell
    # where restoration substantially helped in earlier runs was severe noise.
    def triage_use_genai(k, l):
        return (k == 'noise' and l == 'high')

    cov_grid = np.linspace(0.1, 1.0, 19)
    for name in MODEL_NAMES:
        model = load_classifier_v3(name)
        store = {}
        for k in DEGRADATION_TYPES:
            for l in DEGRADATION_LEVELS:
                for variant in ('raw', 'genai'):
                    root = variant_root(variant, k, l)
                    if (root / 'manifest.csv').exists():
                        store[(k, l, variant)] = collect_preds(model, make_loader(root))

        def build_stream(selector):
            ys, prs = [], []
            for k in DEGRADATION_TYPES:
                for l in DEGRADATION_LEVELS:
                    variant = 'genai' if selector(k, l) else 'raw'
                    if (k, l, variant) not in store:
                        variant = 'raw'
                    if (k, l, variant) not in store:
                        continue
                    y, pr = store[(k, l, variant)]
                    ys.append(y); prs.append(pr)
            if not ys:
                return np.array([]), np.zeros((0, NUM_CLASSES))
            return np.concatenate(ys), np.concatenate(prs)

        streams = {'do_nothing': build_stream(lambda k, l: False),
                   'restore_all': build_stream(lambda k, l: True),
                   'triage': build_stream(triage_use_genai)}

        rows = []
        for sname, (y, pr) in streams.items():
            if len(y) == 0:
                continue
            conf = pr.max(1); pred = pr.argmax(1); order = np.argsort(-conf)
            y_o = y[order]; pred_o = pred[order]
            for cov in cov_grid:
                kc = max(1, int(round(cov * len(y_o))))
                yk, pk = y_o[:kc], pred_o[:kc]
                rows.append({'model': name, 'pipeline': sname, 'coverage': round(float(cov), 3),
                             'accuracy': accuracy_score(yk, pk),
                             'kappa_qw': _qwk(yk, pk) if len(set(yk.tolist())) > 1 else float('nan')})
        df = pd.DataFrame(rows)
        df.to_csv(P_BOOST / 'metrics' / f'selective_{name}.csv', index=False)

        try:
            fig, ax = plt.subplots(1, 2, figsize=(11, 4))
            for sname in streams:
                d = df[df['pipeline'] == sname]
                if len(d):
                    ax[0].plot(d['coverage'], d['accuracy'], marker='o', label=sname)
                    ax[1].plot(d['coverage'], d['kappa_qw'], marker='o', label=sname)
            for a, t in zip(ax, ('accuracy', 'QWK')):
                a.set_xlabel('coverage (fraction served)'); a.set_ylabel(t)
                a.grid(alpha=0.3); a.legend()
            fig.suptitle(f'Selective prediction on degraded stream — {name}')
            plt.tight_layout()
            plt.savefig(P_BOOST / 'plots' / f'selective_{name}.png', dpi=150)
            plt.close()
        except Exception as e:
            blog(f"  [C] plot skipped for {name}: {e!r}")
        blog(f"  [C] {name} selective curves saved")
        del model
        torch.cuda.empty_cache()

run_stage('C_selective_triage', stage_selective_triage)
'''

STAGE_D = r'''# === STAGE D — train-time degradation-augmentation robust baseline ==========
# Best-effort training stage. Skips automatically if checkpoint already exists.
def stage_aug_baseline():
    import torchvision.transforms as T
    AUG_NAME = 'resnet50'   # ConvNeXt-base backbone via BACKBONE_V3 (strongest working model)
    ckpt = P_BOOST / 'checkpoints' / f'{AUG_NAME}_augrobust_best.pt'
    hist = P_BOOST / 'metrics' / f'aug_baseline_history_{AUG_NAME}.csv'

    if ckpt.exists():
        blog("[D] aug-robust checkpoint already exists — skipping training")
    else:
        class RandomDegrade:
            def __init__(self, p=0.7): self.p = p
            def __call__(self, img):
                if np.random.rand() < self.p:
                    k = str(np.random.choice(DEGRADATION_TYPES))
                    l = str(np.random.choice(DEGRADATION_LEVELS))
                    try:
                        img = apply_degradation(img, k, l)
                    except Exception:
                        pass
                return img
        tfm_aug = T.Compose([RandomDegrade(0.7)] + list(TFM_TRAIN_V3.transforms))
        ds_aug = APTOSDataset(PRISTINE_CSV, APTOS_IMAGES, tfm_aug)
        tr_loader = DataLoader(Subset(ds_aug, tr_idx), batch_size=16, sampler=sampler,
                               num_workers=2, pin_memory=True)
        hp = dict(HPARAMS_V3[AUG_NAME]); hp['epochs'] = 12
        blog(f"[D] training aug-robust {AUG_NAME} for {hp['epochs']} epochs...")
        train_v3(AUG_NAME, tr_loader, val_loader_v3, ckpt_path=ckpt,
                 history_csv=hist, hp=hp, class_weights=CLS_W)

    m = build_model_v3(AUG_NAME, pretrained=False).to(DEVICE)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False)['state_dict'])
    m.eval()
    rows = [{'degradation': 'clean', 'level': 'none', **evaluate_v3(m, test_loader_v3)}]
    for k in DEGRADATION_TYPES:
        for l in DEGRADATION_LEVELS:
            rows.append({'degradation': k, 'level': l, **evaluate_v3(m, deg_loader_v3(k, l))})
    pd.DataFrame(rows).to_csv(P_BOOST / 'metrics' / f'aug_baseline_stress_{AUG_NAME}.csv', index=False)
    blog("[D] aug-robust stress eval saved")
    del m
    torch.cuda.empty_cache()

run_stage('D_aug_baseline', stage_aug_baseline)
'''

STAGE_E = r'''# === STAGE E — classifier adapted to A-ESRGAN-restored images ===============
# Tests whether the 'restorer output is OOD' effect closes when the classifier
# is trained on restored images. Best-effort; restoration is applied on the fly.
def stage_finetune_on_restored():
    import torchvision.transforms as T
    FT_NAME = 'resnet50'
    ckpt = P_BOOST / 'checkpoints' / f'{FT_NAME}_ftrestored_best.pt'
    hist = P_BOOST / 'metrics' / f'ft_restored_history_{FT_NAME}.csv'

    if 'enhance_genai' not in globals():
        blog("[E] enhance_genai not defined — skipping restored-adaptation stage")
        return

    if ckpt.exists():
        blog("[E] ft-restored checkpoint already exists — skipping training")
    else:
        class RestoreThenAug:
            def __call__(self, img):
                try:
                    img = enhance_genai(img)
                except Exception:
                    pass
                return img
        tfm = T.Compose([RestoreThenAug()] + list(TFM_TRAIN_V3.transforms))
        ds = APTOSDataset(PRISTINE_CSV, APTOS_IMAGES, tfm)
        tr_loader = DataLoader(Subset(ds, tr_idx), batch_size=16, sampler=sampler,
                               num_workers=2, pin_memory=True)
        hp = dict(HPARAMS_V3[FT_NAME]); hp['epochs'] = 8; hp['lr'] = hp['lr'] * 0.5
        blog(f"[E] training {FT_NAME} on A-ESRGAN-restored images for {hp['epochs']} epochs "
             f"(restoration applied on the fly — may be slow)...")
        train_v3(FT_NAME, tr_loader, val_loader_v3, ckpt_path=ckpt,
                 history_csv=hist, hp=hp, class_weights=CLS_W)

    m = build_model_v3(FT_NAME, pretrained=False).to(DEVICE)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False)['state_dict'])
    m.eval()
    rows = []
    for k in DEGRADATION_TYPES:
        for l in DEGRADATION_LEVELS:
            root = ENHANCED_DIR / 'genai' / k / l
            if (root / 'manifest.csv').exists():
                rows.append({'degradation': k, 'level': l, 'variant': 'genai_ftadapted',
                             **evaluate_v3(m, make_loader(root))})
    pd.DataFrame(rows).to_csv(P_BOOST / 'metrics' / f'ft_restored_stress_{FT_NAME}.csv', index=False)
    blog("[E] ft-restored eval saved")
    del m
    torch.cuda.empty_cache()

run_stage('E_finetune_restored', stage_finetune_on_restored)
'''

STAGE_F = r'''# === STAGE F — summary report ===============================================
def stage_summary():
    lines = ["# Overnight booster summary", ""]
    sc = P_BOOST / 'metrics' / 'stage_status.csv'
    if sc.exists():
        try:
            lines += ["## Stage status", pd.read_csv(sc).to_markdown(index=False), ""]
        except Exception:
            lines += ["## Stage status", pd.read_csv(sc).to_string(index=False), ""]
    em = P_BOOST / 'metrics' / 'recovery_metrics_enriched.csv'
    if em.exists():
        df = pd.read_csv(em)
        sub = df[df['model'] == 'resnet50']
        if len(sub):
            best = sub.loc[sub.groupby(['degradation', 'level'])['kappa_qw'].idxmax()]
            cols = ['degradation', 'level', 'variant', 'accuracy', 'kappa_qw']
            try:
                tbl = best[cols].to_markdown(index=False)
            except Exception:
                tbl = best[cols].to_string(index=False)
            lines += ["## Best restorer per cell (resnet50, by QWK)", tbl, "",
                      "_Descriptive only — chosen on the test split. A deployable policy "
                      "must select on validation; see Stage C for the validation-safe "
                      "confidence-based triage._", ""]
    cal = P_BOOST / 'metrics' / 'calibration_ece.csv'
    if cal.exists():
        try:
            lines += ["## Calibration (ECE)", pd.read_csv(cal).to_markdown(index=False), ""]
        except Exception:
            pass
    (P_BOOST / 'OVERNIGHT_SUMMARY.md').write_text("\n".join(lines))
    blog("Summary written to " + str(P_BOOST / 'OVERNIGHT_SUMMARY.md'))

run_stage('F_summary', stage_summary)
blog("ALL STAGES COMPLETE — see results/phase6_overnight_boost/")
'''

CELLS = [
    ('markdown', MD0),
    ('code', SETUP),
    ('code', STAGE_A),
    ('code', STAGE_B),
    ('code', STAGE_C),
    ('code', STAGE_D),
    ('code', STAGE_E),
    ('code', STAGE_F),
]

def make_cell(ctype, src):
    cell = {'cell_type': ctype, 'metadata': {}, 'source': src.splitlines(keepends=True)}
    if ctype == 'code':
        cell['execution_count'] = None
        cell['outputs'] = []
    return cell

def main():
    nb = json.loads(NB.read_text(encoding='utf-8'))
    # idempotent: drop any previously-appended booster cells
    marker = 'V7 OVERNIGHT BOOSTER PACK'
    marker_md = 'V7 — Overnight booster pack'
    nb['cells'] = [c for c in nb['cells']
                   if marker not in ''.join(c.get('source', []))
                   and marker_md not in ''.join(c.get('source', []))]
    for ctype, src in CELLS:
        nb['cells'].append(make_cell(ctype, src))
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    # sanity: re-parse
    json.loads(NB.read_text(encoding='utf-8'))
    print(f"Appended {len(CELLS)} cells. Total cells now: {len(nb['cells'])}")

if __name__ == '__main__':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    main()
