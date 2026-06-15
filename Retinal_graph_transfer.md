# Retinal Graph Transfer — Project Ideas

Applying directional / geometric message passing (DimeNet-style GNNs) from molecular property prediction to retinal vessel graphs, in service of the existing GenAI restoration + DR-grading pipeline.

---

## Why this transfer works

The retina is already a geometric graph, and its angles are already clinical.

Segment the vessels → skeletonize → you get a graph:

- **Nodes** = bifurcation points. Node attribute: **branching angle**.
- **Edges** = vessel segments. Edge attributes: caliber, length, tortuosity, artery-vs-vein.

Vessel branching angle, tortuosity, and arteriolar–venular ratio are established biomarkers that shift in diabetic retinopathy (alongside venous beading, IRMA, neovascularization). So a model whose entire reason for existing is "exploit the angle at a junction" maps onto retinal bifurcations almost too cleanly. That is the transfer.

**Simplification in our favor:** DimeNet's heavy machinery (spherical harmonics) exists for *3D* directions. The retinal graph is *2D*, so the angular basis collapses to something much simpler to implement.

---

## The ideas (weakest → strongest)

### Idea 1 — Geometric vessel-graph GNN for DR grading (the obvious one)
Extract the vessel graph, run directional message passing with bifurcation angles as first-class inputs, predict the DR grade. Fuse with a CNN that handles lesion-local features (microaneurysms, exudates) the graph can't see.

- **Verdict:** Solid and publishable; a great *learning vehicle*. But "GNN for DR grading" is a crowded shelf. The geometric/directional twist is the only thing keeping it fresh. A reviewer will ask "did angles actually help?" → needs a clean ablation showing the angle term moves the metric.
- **Role:** Good backbone, not the headline.

### Idea 2 — Vessel-graph consistency as a restoration quality metric (attacks the central paradox)
The pipeline's central paradox: PSNR goes *up* while diagnostic accuracy goes *down* — because PSNR is a pixel metric blind to whether *structure* survived. So replace it.

Extract the vessel graph from the clean image and from the restored image, and measure how much the **geometry** changed — branching angles, topology, segment connectivity. A restorer that preserves vascular geometry is clinically faithful; one that scrambles it isn't, even if its pixels look sharp.

- **Verdict:** A quality metric aligned with diagnosis instead of fighting it. A direct, named answer to the exact problem the OMIA paper is built around.

### Idea 3 — The vessel graph as a hallucination detector (the gold)
Most original existing thread: detecting fabricated pathology from GenAI restorers (e.g. SwinIR+GAN hallucinating lesions on exposure-degraded images), validated against **diagnostic decision flips**.

Geometry is a near-perfect hallucination signal. When a restorer invents a vessel, splits a real one, or bends a junction to a non-physiological angle, that shows up as a **geometric anomaly in the graph** that no pixel metric will flag.

So: run the geometric GNN on the pre- and post-restoration vessel graphs, and treat geometry-inconsistent changes — especially ones that flip the grade — as the hallucination alarm.

- **Verdict:** Fuses three things already owned (restoration pipeline + hallucination direction + decision-flip validation) and adds the missing capability: a structural, clinically-grounded notion of "this change is fake." Potentially the spine of a thesis chapter.

### Idea 4 — Geometry-aware routing (lightweight bolt-on)
The current routing strategy (Cold Diffusion for blur/exposure, DDPM for noise) picks by degradation type. Instead, pick — or *audit* the pick — by which restorer best preserves vascular geometry for that image.

- **Verdict:** Low effort, nice ablation, not a paper on its own.

---

## The coupling insight (ties Ideas 2 & 3 together)

There is a **chicken-and-egg**: you need a decent image to extract a decent vessel graph, but degraded images give garbage segmentations — which is the whole reason restoration exists.

This is not a bug; it's the **thesis-level insight**. Restoration and graph-extraction are *coupled*, and the graph can close the loop by certifying whether a restoration is structurally trustworthy. Frame the dependency as the contribution.

---

## Recommendation

- Don't lead with Idea 1 (crowded lane).
- Build the vessel-graph extraction once (it serves all ideas).
- Aim the spear at **Idea 3**, using **Idea 2's metric** as the measurement instrument.

This stays out of the crowded "another DR-grading GNN" lane and squarely in the lane already half-built, where the unfair advantage lives.

---

## Before committing a single GPU-hour

1. **Novelty check.** Someone may already have done "geometric GNN on retinal vessel graphs" or "graph-consistency restoration metric." Search the literature *before* building, not after.
2. **The minimal failure demo.** DimeNet-style move: construct the smallest example that proves the gap — a case where a GAN restorer sharpens pixels (PSNR up) while measurably wrecking vessel geometry (graph metric down, grade flips). If even one such image pair can be produced from existing results, the project is real.

---

## Next-step options

- Run the novelty search on Idea 3 + Idea 2 first, **or**
- Sketch the vessel-graph extraction pipeline (segmentation → skeleton → PyG graph) to see how much engineering it actually is.
