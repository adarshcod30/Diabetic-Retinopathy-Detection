# Annotated Literature Review — Evidence Base

Every design decision in [`01_PROJECT_ANALYSIS.md`](01_PROJECT_ANALYSIS.md) traces to something here.
Papers are grouped by the pipeline stage they inform. **Read the ★ ones first.**

---

## A. Landmark clinical validations — the performance bar

### ★ Gulshan V. et al. (2016) — *Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs*, **JAMA** 316(22):2402-2410
<https://jamanetwork.com/journals/jama/fullarticle/2588763> · DOI: 10.1001/jama.2016.17216

- 128,175 images graded 3–7× by 54 US-licensed ophthalmologists; validated on EyePACS-1 and Messidor-2.
- **Referable DR: sensitivity 97.5 %, specificity 93.4 % (EyePACS-1); 96.1 % / 93.9 % (Messidor-2).**
- **Why it matters here:** this is the paper that established the field. Note the *grading protocol* —
  multiple graders per image is why their labels are cleaner than APTOS's single-grader labels.
- **What to steal:** the multi-grader adjudication idea; the two-dataset validation design.

### ★ Ting D.S.W. et al. (2017) — *Development and Validation of a Deep Learning System for Diabetic Retinopathy and Related Eye Diseases Using Retinal Images From Multiethnic Populations With Diabetes*, **JAMA** 318(22):2211-2223
<https://pubmed.ncbi.nlm.nih.gov/29234807/> · [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC5820739/)

- SELENA, VGGNet-based, ~500,000 images, 10 multiethnic cohorts.
- **Referable DR: AUC 0.936, sens 90.5 %, spec 91.6 %. Vision-threatening DR: AUC 0.958, sens 100 %, spec 91.1 %.**
- **Why it matters here:** the multiethnic validation is the honest comparator for our targets — and
  its numbers sit right at the brief's >90 %/>85 % bar. Achievable, not trivial.

### ★ Abràmoff M.D. et al. (2018) — *Pivotal trial of an autonomous AI-based diagnostic system for detection of diabetic retinopathy in primary care offices*, **npj Digital Medicine** 1:39
<https://www.nature.com/articles/s41746-018-0040-6>

- 900 prospective patients, primary care, compared against Wisconsin FPRC widefield stereo + OCT.
- **Sensitivity 87.2 %, specificity 90.7 %, imageability 96.1 %.** First FDA-authorised autonomous AI
  diagnostic in any field of medicine.
- **Why it matters here:** the *prospective* numbers are lower than Gulshan's retrospective ones. This
  is the single best argument for our "lock an external test set" discipline. Also note **imageability
  is reported as a primary endpoint** — quality assessment is not an afterthought in a real system.

### ★ Gulshan V. et al. (2019) — *Performance of a Deep-Learning Algorithm vs Manual Grading for Detecting Diabetic Retinopathy in India*, **JAMA Ophthalmology** 137(9):987-993
<https://research.google/pubs/performance-of-a-deep-learning-algorithm-vs-manual-grading-for-detecting-diabetic-retinopathy-in-india/>

- Prospective, 3,049 patients at **Aravind Eye Hospital** and **Sankara Nethralaya**.
- **Aravind: sens 88.9 %, spec 92.2 %, AUC 0.963. Sankara Nethralaya: sens 92.1 %, spec 95.2 %, AUC 0.980.**
- **Why it matters here:** this is the direct Indian-population benchmark. APTOS images come from
  Aravind, so this is the closest published comparator to our own training distribution.

### Reproduction / replication studies
- *Replication study of Gulshan et al.* — <https://arxiv.org/pdf/1803.04337>
- Reproduction using public data — [PMC6553744](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6553744/)
- **Finding to internalise: AUC 0.951 on EyePACS but only 0.853 on Messidor-2** from a faithful
  reimplementation. Expect and plan for this external-validation gap.

---

## B. Real-world deployment — why field conditions break models

### ★ Beede E. et al. (2020) — *A Human-Centered Evaluation of a Deep Learning System Deployed in Clinics for the Detection of Diabetic Retinopathy*, **CHI 2020**
<https://dl.acm.org/doi/10.1145/3313831.3376718> · [full HTML](https://dl.acm.org/doi/fullHtml/10.1145/3313831.3376718) · [Google Research](https://research.google/pubs/pub48768/)

- Interviews + observation across **11 clinics in Thailand** running a deployed DR model.
- **Key finding:** socio-environmental factors dominate. Images captured in non-darkened rooms were
  degraded; the model's data-quality threshold clashed with what a resource-constrained clinic can
  actually produce; ungradable rejections wasted patient trips and eroded nurse trust.
- **Why it matters here:** this paper is the *entire justification* for Stage 1 (quality assessment
  with actionable recapture guidance) and for reporting metrics stratified by quality tier. If you
  read one paper on deployment, read this one.

### Ruamviboonsuk P. et al. (2022) — *Real-time diabetic retinopathy screening by deep learning in a multisite national screening programme: a prospective interventional cohort study*, **Lancet Digital Health**
<https://www.sciencedirect.com/science/article/pii/S2589750022000176>

- Prospective, national-scale, real-time. The gold standard for "does it work in a programme, not a paper."

### AIDRSS multicentric validation in India (2025)
<https://arxiv.org/pdf/2501.05826> — AI-driven DR screening validated across multiple Indian centres.

### Abràmoff M.D. et al. (2025) — *Autonomous AI in DR Testing — Lessons Learned on Successful Health System Adoption*, **Ophthalmology Science**
<https://www.ophthalmologyscience.org/article/S2666-9145(25)00233-7/fulltext>

---

## C. Indian epidemiology — get the motivation numbers right

| Source | Finding |
|---|---|
| **ICMR-INDIAB**, *Lancet Diabetes & Endocrinology* 2023 — [PDF](https://www.thelancet.com/pdfs/journals/landia/PIIS2213-8587(23)00119-5.pdf) | 113,043 participants; **weighted diabetes prevalence 11.4 %** → ~101 M adults |
| **SMART India**, *Lancet Global Health* 2022 — [link](https://www.thelancet.com/journals/langlo/article/PIIS2214-109X(22)00411-9/fulltext) | DR prevalence stratified by known/undiagnosed diabetes and urban/rural |
| Systematic review & meta-analysis, 2022 — [PubMed](https://pubmed.ncbi.nlm.nih.gov/35647959/) | **DR: 17.44 % urban, 14.00 % rural** |
| National Survey 2015-19 — [PMC8725073](https://pmc.ncbi.nlm.nih.gov/articles/PMC8725073/) | DR prevalence ~12.5 % |
| DR screening in the public sector — [PubMed](https://pubmed.ncbi.nlm.nih.gov/35225509/) | **~1 retina specialist per 1.26 million population** — the real capacity bottleneck |
| Situational analysis of DR screening in India — [PMC8725067](https://pmc.ncbi.nlm.nih.gov/articles/PMC8725067/) | Programme-level infrastructure gaps |

> **Use these, not the brief's figures.** Cite ICMR-INDIAB for diabetes burden and the retina-specialist
> ratio for the capacity argument.

---

## D. Datasets

### ★ Porwal P. et al. (2020) — *IDRiD: Diabetic Retinopathy – Segmentation and Grading Challenge*, **Medical Image Analysis** 59:101561
<https://www.sciencedirect.com/science/article/abs/pii/S1361841519301033> · [challenge site](https://idrid.grand-challenge.org/) · [NSF PDF](https://par.nsf.gov/servlets/purl/10189648) · [IEEE DataPort](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid)

- First Indian-population dataset with **pixel-level** lesion annotation (MA, HE, SE, EX).
- ISBI-2018 challenge: 3 sub-tasks (lesion segmentation, grading, landmark localisation); **148
  submissions from 495 registrations**.
- **Why it matters here:** IDRiD's masks are the ground truth for our *quantified explainability*
  metric — without them, "is the Grad-CAM right?" is unanswerable.

### APTOS 2019 Blindness Detection
<https://www.kaggle.com/c/aptos2019-blindness-detection> — 3,662 training images from **Aravind Eye
Hospital**, rural India, 5-class ICDR.

### EyeQ — fundus quality grading
28,792 images from EyePACS labelled **Good / Usable / Reject** (12,543 train / 16,249 test).

### Messidor-2 — <https://www.adcis.net/en/third-party/messidor2/> · DRIVE — <https://drive.grand-challenge.org/>

---

## E. Image quality assessment

- **VISTA** (Neural Computing & Applications, 2024) — split-and-reconstruct network; **EyeQ accuracy
  0.9066, precision 0.8843, recall 0.8905, F1 0.8868.** ← *our target to match.*
  <https://link.springer.com/article/10.1007/s00521-024-10174-6>
- **FGR-Net** (Expert Systems with Applications, 2023) — autoencoder reconstruction + classifier; forces
  attention onto fovea, optic disc, vessels. <https://www.sciencedirect.com/science/article/pii/S0957417423021462>
- *Diagnostic Quality Assessment of Fundus Photographs: Hierarchical Deep Learning with Clinically
  Significant Explanations* — <https://arxiv.org/pdf/2302.09391> ← the "explain the rejection" idea
- **FundusQ-Net** — regression-based quality grading. <https://www.sciencedirect.com/science/article/abs/pii/S0169260723001876>
- *Deep learning for gradability classification of **handheld, non-mydriatic** retinal images* —
  directly relevant to portable field cameras. <https://www.researchgate.net/publication/351320521>

---

## F. Architectures & grading methods

### Foundation models
- ★ **RETFound** — Zhou Y. et al. (2023), *A foundation model for generalizable disease detection from
  retinal images*, **Nature** 622:156-163. <https://www.nature.com/articles/s41586-023-06555-x> ·
  [code](https://github.com/openmedlab/RETFound_MAE)
  - MAE self-supervised on **1.6 M unlabelled retinal images** (904,170 CFP + 736,442 OCT); 75 % patch
    masking. Benchmarked on IDRiD, APTOS-2019, Messidor-2.
  - **Use as:** initialisation for the grading backbone if compute allows.
- *Training a high-performance retinal foundation model with half the data and 400× less compute*,
  **Nature Communications** 2025 — <https://www.nature.com/articles/s41467-025-62123-z> ← the
  practical alternative on a budget.
- **DINORET** — adapting natural-domain foundation models without catastrophic forgetting.
  <https://arxiv.org/pdf/2409.17332>

### Ordinal regression for DR grading
- ★ **Dual-SwinOrd** (2025) — dual-head Swin Transformer with semantic prior injection.
  **APTOS-2019: QWK 0.9370, Acc 87.98 %; DDR: QWK 0.9040, Acc 86.54 %.** ← current strong SOTA reference.
  <https://www.mdpi.com/2306-5354/13/4/374> · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13112979/)
- **AOR-DR** — autoregressive ordinal regression with parameterised diffusion optimisation, **MICCAI 2025**.
  <https://arxiv.org/pdf/2507.04978>
- *Joint ordinal regression and multiclass classification with transformer–CNN fusion*, **Applied
  Intelligence** 2023 — <https://link.springer.com/article/10.1007/s10489-023-04949-y>
- **Takeaway:** ordinal formulation consistently beats plain cross-entropy on QWK. Adopt it.

### Lesion-driven grading
- *A deep learning model for classification of DR based on retinal lesion detection* —
  <https://arxiv.org/pdf/2110.07745> ← direct precedent for our lesion-feature fusion head.
- *Advanced Segmentation of DR Lesions Using DeepLabv3+* — <https://arxiv.org/pdf/2504.17306>

### Preprocessing
- **Ben Graham's preprocessing** — winner, Kaggle Diabetic Retinopathy 2015. Local-average colour
  subtraction; still the highest-value single preprocessing step. Used by most APTOS top solutions
  alongside `crop_from_gray` and `circle_crop`
  ([reference implementation](https://github.com/MamatShamshiev/Kaggle-APTOS-2019-Blindness-Detection)).
- *Enhancement of DR Prognostication Using Deep Learning, CLAHE, and ESRGAN* —
  [PMC10378524](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10378524/)

---

## G. Explainability — and its limits

### ★ Adebayo J. et al. (2018) — *Sanity Checks for Saliency Maps*, **NeurIPS 2018**
<https://arxiv.org/abs/1810.03292>

- Introduces the **model-randomisation** and **data-randomisation** tests. Several popular saliency
  methods are shown to be independent of both the model weights and the training labels — i.e. they
  behave like edge detectors and explain nothing.
- **Why it matters here:** running these tests is what separates a rigorous explainability claim from
  a pretty picture. Guided Grad-CAM in particular fails them
  ([accessible summary](https://glassboxmedicine.com/2019/10/12/guided-grad-cam-is-broken-sanity-checks-for-saliency-maps/)).

### Saliency reliability in medical imaging specifically
- *Assessing the validity of saliency maps for abnormality localization in medical imaging* —
  <https://arxiv.org/pdf/2006.00063>
- *Assessing the (Un)Trustworthiness of Saliency Maps for Localizing Abnormalities* —
  <https://arxiv.org/pdf/2008.02766>
- *Gradient-Based Saliency Maps Are Not Trustworthy Visual Explanations of Automated AI Musculoskeletal
  Diagnoses*, **J. Imaging Informatics in Medicine** 2024 —
  <https://link.springer.com/article/10.1007/s10278-024-01136-4>
- *Revisiting Sanity Checks for Saliency Maps* — <https://arxiv.org/pdf/2110.14297>

### Beyond heatmaps — concept- and lesion-grounded explanation
- **Concept-based Lesion Aware Transformer**, *IEEE TMI* 2024 —
  <https://www.researchgate.net/publication/382303558> ← lesions as concepts, grades as outcomes
- **AdaCBM** — adaptive concept bottleneck model, **MICCAI 2024** —
  <https://link.springer.com/chapter/10.1007/978-3-031-72117-5_4>
- *Explainable and Interpretable DR Classification Based on Neural-Symbolic Learning* —
  <https://arxiv.org/pdf/2204.00624>
- *TWLR: Text-Guided Weakly-Supervised Lesion Localization and Severity Regression* —
  <https://arxiv.org/pdf/2512.13008>

---

## H. Calibration & uncertainty

- **Guo C. et al. (2017)** — *On Calibration of Modern Neural Networks*, ICML. Temperature scaling;
  modern nets are systematically overconfident. The basis for our Stage 5.
- *Uncertainty-aware deep learning methods for robust diabetic retinopathy classification* —
  <https://arxiv.org/pdf/2201.09042> ← directly on-task; informs the human-escalation design.

---

## I. Surveys & context

- *From Retinal Pixels to Patients: Evolution of Deep Learning in DR* (2025) — <https://arxiv.org/pdf/2511.11065>
- *Artificial Intelligence and Diabetic Retinopathy: AI Framework, Prospective Studies, Head-to-head
  Validation, and Cost-effectiveness*, **Diabetes Care** 2023 —
  <https://diabetesjournals.org/care/article/46/10/1728/153626/>
- *The impact of artificial intelligence in screening for diabetic retinopathy in India*, **Eye** (Nature) —
  <https://www.nature.com/articles/s41433-019-0626-5>

---

## How to use this file

1. **Before writing a stage**, read the ★ paper for that stage.
2. **When you report a number**, find the row in [§3 of the analysis](01_PROJECT_ANALYSIS.md#3-published-performance--the-bar-we-must-clear)
   it should be compared against, and make sure the task definitions match.
3. **When a reviewer asks "why did you do X?"**, the answer should be a citation from this file, not
   an intuition.
