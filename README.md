# AMPLIFAI Challenge

**Automated Multi-Phase Liver Findings AI** — an open challenge for automated LI-RADS categorisation of hepatic lesions from multi-phase CT.

## Task

Given a multi-phase CT scan and a pre-defined target lesion segmentation, predict the LI-RADS category for that lesion.

**Valid labels:** `LR-1`, `LR-2`, `LR-3`, `LR-4`, `LR-5`, `LR-M`, `LR-TIV`

## Input data (per case)

```
<case_id>/
├── ct/
│   ├── <case_id>_ART.nii.gz   ← arterial phase
│   ├── <case_id>_VEN.nii.gz   ← portal venous phase
│   ├── <case_id>_DEL.nii.gz   ← delayed phase
│   └── <case_id>_DRY.nii.gz   ← non-contrast / native
└── annotations/
    └── lesion.nii.gz           ← binary lesion segmentation mask
```

## Evaluation metric

```
Final Score = 0.85 × Adjusted QWK + 0.15 × Special Category Recognition
```

- **Adjusted QWK** — Quadratic Weighted Kappa on the ordinal LR-1 to LR-5 scale
- **Special Category Recognition** — 3-class balanced accuracy (Ordinal / LR-M / LR-TIV)

See [`evaluate.py`](evaluate.py) for the full implementation. You can run it locally to test your predictions before submitting.

## Quick start

### 1. Test the metric locally

```bash
pip install -r requirements.txt
python evaluate.py --ground_truth path/to/gt.csv --predictions path/to/pred.csv --no-bootstrap
```

Ground truth CSV format: `case_id, label`  
Predictions CSV format: `case_id, prediction`

### 2. Build your submission

Start from [`example_submission/run.py`](example_submission/run.py) — fill in the `predict()` function with your model.

Your submission zip must contain:
```
your_submission.zip
├── run.py      ← your entry point (must be named run.py)
├── metadata    ← required, copy from example_submission/
└── ...         ← model weights, helper scripts, etc.
```

### 3. Submit

Upload your zip on the [AMPLIFAI Codabench page](#).

See [`SUBMISSION_GUIDE.md`](SUBMISSION_GUIDE.md) for full details on the submission format, Docker environment, and path conventions.

