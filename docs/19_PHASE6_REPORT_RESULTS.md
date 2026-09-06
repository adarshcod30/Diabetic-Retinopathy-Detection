# Phase 6 Results — Lesion Overlays, ICDR Evidence, and the Redesigned Report

The last three buildable Phase 6 items, delivered together since each feeds the next: lesion
detections -> an overlay render + a structured evidence table -> a templated rationale sentence ->
all three added to the PDF report alongside what Phase 2 already produced.

## What this covers, and what it doesn't

Built: `src/drdetect/explain/evidence.py` (`extract_lesion_evidence`, `render_lesion_overlay`,
`build_icdr_rationale`), `PredictionResult.add_lesion_evidence` (a new, separate pipeline step --
see below), and an extended `build_report_pdf` that renders a third image panel, a colour legend,
and the rationale text when lesion evidence is present.

Not built: the 30-second clinician timing study. This needs a real human reviewer; this project
has no data to substitute for one, and has flagged this limitation consistently since docs/08.

## Method

- **Reuses Phase 5's model-running code, not a second implementation.** `extract_lesion_evidence`
  calls the same tiled-segmentation and OD/fovea-localisation internals `drdetect.fusion.features`
  already built and evaluated for the fusion head -- it only adds the per-instance/per-quadrant
  structure a human-readable report needs (which lesion, which quadrant, how big) on top of the
  aggregate scalars the fusion head consumes.
- **Outlines, not filled blobs** (`cv2.findContours` + `cv2.drawContours`), per the roadmap's own
  stated reasoning: a filled overlay hides the tissue underneath it; an outline points at it
  without obscuring it. Microaneurysms are marked with small circles instead -- they're candidate
  points from a classifier (docs/15), not a segmented region, so there's no contour to draw.
- **A separate pipeline step, not folded into `run_pipeline`.** `add_lesion_evidence` needs 5 extra
  loaded models; the Phase 2 vertical slice (`scripts/predict.py`, the Gradio demo) has no reason
  to always pay that cost. `drdetect.fusion.features` is imported lazily (a `TYPE_CHECKING`-only
  import at module level) so loading `drdetect.serve.pipeline` itself stays cheap for callers that
  never touch lesion evidence.
- **The rationale is templated from actual detections, not a restatement of the grade.** It reports
  what was found (microaneurysm count, haemorrhage quadrant spread, exudate area percentages), not
  "the model predicted grade N." For grade 4 specifically, it says outright that proliferative DR's
  defining feature -- neovascularisation -- has no segmentation model in this project (no public
  pixel masks exist to train one, per docs/01's own scope-honesty note), so a grade-4 rationale is
  never allowed to imply NV evidence that was never actually detected.

## Result — verified end to end on a real image, not just unit-tested in isolation

Run against IDRiD_20 (a real segmentation-subset image) through the full chain -- grading, all 5
lesion models, overlay rendering, rationale, PDF assembly:

```
predicted grade: 3 (Severe NPDR), confidence 52.0%
detected: 147 microaneurysm candidates; 77 haemorrhage regions across 4 quadrants;
          hard exudates covering 2.09% of the image; soft exudates covering 0.11%
```

The rationale text produced: *"ICDR grade 3 (Severe NPDR): 147 microaneurysm candidate(s); 77
haemorrhage region(s) across 4 quadrant(s); hard exudates covering 2.09% of the image; soft
exudates (cotton-wool spots) covering 0.11% of the image."* Worth noting plainly: haemorrhages
detected in **all 4 quadrants** is one of ICDR's own defining criteria for severe NPDR (the
"4-2-1 rule") -- the detected evidence pattern for this specific image is independently consistent
with the grade the classifier predicted, not just narrating it after the fact. This is one image,
not a systematic check across many (that would need ground-truth ICDR criteria annotations this
project doesn't have), but it's a genuine, unprompted alignment worth recording.

The rendered PDF was visually inspected (not just executed without error): all three image panels,
the colour legend, the evidence sentence, confidence, per-grade probabilities, and the quality
metrics all fit comfortably on one A4 page with room to spare -- no overflow, no cramped text.

## What's still open

- **The 30-second clinician timing study** -- the roadmap's own exit criterion for this phase,
  and the one item this project cannot complete without a human reviewer. Recruiting one (even
  n=1, per the roadmap's own suggestion) remains squarely the user's decision, not something
  automatable.
- **The rationale template is not exhaustive of ICDR's own criteria** -- venous beading and IRMA
  (the other two legs of severe NPDR's 4-2-1 rule) have no detector in this project and are not
  mentioned in the rationale at all, rather than being silently assumed absent. Worth stating if
  this rationale is ever shown to a real clinician: it reports what was measured, not a complete
  ICDR checklist.
- **Only tested on one image.** A systematic check of rationale/evidence coherence across many
  images, or against real clinician-written rationale, was not attempted this pass.
