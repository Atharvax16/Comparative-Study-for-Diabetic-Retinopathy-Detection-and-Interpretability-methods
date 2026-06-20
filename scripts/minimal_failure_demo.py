"""Minimal failure demo: visual fidelity up, diagnostic accuracy down.

The paper argues "visual fidelity != diagnostic accuracy" but currently only in
prose (Discussion). This script surfaces the explicit, honest instance of that
claim from the already-logged metrics: restoration cells where the restorer
*raised* both PSNR and SSIM yet *lowered* downstream classification accuracy.

Why aggregate (per-cell) and not a single image: the per-image PSNR/SSIM logs
(fidelity_per_image.csv) cover low+mid severity only, while the per-image
grade-flip log (ddpm_pathology_hallucination.csv) covers high severity only --
disjoint, so no honest per-image PSNR+flip pairing exists. The per-cell paradox
is computed on the full test split and is the defensible evidence.

Source : results/phase4b_restoration_proof/metrics/restoration_proof_master.csv
Output : results/phase4b_restoration_proof/metrics/fidelity_up_accuracy_down.csv
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "results/phase4b_restoration_proof/metrics/restoration_proof_master.csv"
OUT = ROOT / "results/phase4b_restoration_proof/metrics/fidelity_up_accuracy_down.csv"


def main() -> None:
    m = pd.read_csv(MASTER)

    # The paradox set: pixels improved on BOTH reference metrics, yet the
    # downstream DR classifier got WORSE.
    par = m[(m["psnr_gain"] > 0) & (m["ssim_gain"] > 0) & (m["acc_delta"] < 0)].copy()
    par = par.sort_values("psnr_gain", ascending=False)

    cols = ["degradation", "variant", "psnr_raw", "psnr_restored", "psnr_gain",
            "ssim_raw", "ssim_restored", "ssim_gain",
            "acc_raw", "acc_restored", "acc_delta"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    par[cols].to_csv(OUT, index=False)

    print(f"'Fidelity up, accuracy down' cells (PSNR up AND SSIM up AND acc down): "
          f"{len(par)} of {len(m)}")
    print(f"Written to {OUT}\n")

    if par.empty:
        print("No paradox cell found.")
        return

    p = par.iloc[0]
    print("=== HEADLINE CELL ===")
    print(f"  restorer    : {p['variant']} on {p['degradation']}")
    print(f"  PSNR        : {p['psnr_raw']:.2f} -> {p['psnr_restored']:.2f} dB "
          f"(+{p['psnr_gain']:.2f})")
    print(f"  SSIM        : {p['ssim_raw']:.4f} -> {p['ssim_restored']:.4f} "
          f"(+{p['ssim_gain']:.4f})")
    print(f"  accuracy    : {p['acc_raw']:.4f} -> {p['acc_restored']:.4f} "
          f"({p['acc_delta']:+.4f})")
    print("  reading     : a large pixel-fidelity gain that *hurts* the diagnosis.\n")

    print("All paradox cells (sorted by PSNR gain):")
    show = par[["degradation", "variant", "psnr_gain", "ssim_gain", "acc_delta"]]
    print(show.to_string(index=False,
                         formatters={"psnr_gain": "{:+.2f}".format,
                                     "ssim_gain": "{:+.3f}".format,
                                     "acc_delta": "{:+.4f}".format}))


if __name__ == "__main__":
    main()
