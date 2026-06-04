# AMPLIFAI Challenge — Submission Guide

## Overview

Your submission is a **zip file containing your model inference code**.  
When submitted, it runs inside a Docker container on the challenge server, processes the held-out CT cases, and must write a `predictions.csv` file assigning a LI-RADS category to the pre-defined target lesion in each case.

---

## What your code will have access to

| Path inside container | Contents |
|---|---|
| `/app/ingested_program/` | Your submission code |
| `/app/data/cases/<case_id>/` | CT volumes + lesion annotation per case (read-only) |
| `/app/input_data/sample_cases.csv` | List of case IDs to process |
| `/app/output/` | **Write your `predictions.csv` here** |

### Per-case data structure

```
/app/data/cases/
└── <case_id>/
    ├── ct/
    │   ├── <case_id>_ART.nii.gz   ← arterial phase
    │   ├── <case_id>_VEN.nii.gz   ← portal venous phase
    │   ├── <case_id>_DEL.nii.gz   ← delayed phase
    │   └── <case_id>_DRY.nii.gz   ← non-contrast / native
    └── annotations/
        └── lesion.nii.gz           ← binary lesion segmentation mask
```

All volumes are 3D NIfTI files (`.nii.gz`). The lesion mask is registered to the CT volumes.

---

## Required output

Write a single file: **`/app/output/predictions.csv`**

```
case_id,prediction
CASE00001,LR-4
CASE00002,LR-M
CASE00003,LR-5
...
```

**Valid prediction labels:** `LR-1`, `LR-2`, `LR-3`, `LR-4`, `LR-5`, `LR-M`, `LR-TIV`

Every case in `sample_cases.csv` must have a prediction. Submissions with fewer than 95% case coverage may be excluded from ranking.

---

## Submission structure

```
your_submission.zip
├── run.py          ← required entry point (must be named run.py)
├── metadata        ← required (tells the system how to call your code)
└── ...             ← model weights, helper scripts, etc.
```

**`metadata` file contents** (include this exactly):
```
command: python3 /app/ingested_program/run.py $input $output
```

---

## Docker environment

**Image:** `codalab/codalab-legacy:gpu310`  
**Python:** 3.10  
**CUDA:** 12.4.1 (GPU available when drivers are enabled)

Install any additional packages at the top of `run.py`:
```python
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "nibabel", "SimpleITK", "your-package"])
```

---

## Minimal working example

```python
# run.py
import os, sys, subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "nibabel", "pandas", "numpy"])

import nibabel as nib
import pandas as pd

input_dir, output_dir = sys.argv[1], sys.argv[2]
os.makedirs(output_dir, exist_ok=True)

cases    = pd.read_csv("/app/input_data/sample_cases.csv")
DATA_ROOT = "/app/data/cases"
results  = []

for case_id in cases["case_id"]:
    # Load CT phases
    art    = nib.load(f"{DATA_ROOT}/{case_id}/ct/{case_id}_ART.nii.gz")
    ven    = nib.load(f"{DATA_ROOT}/{case_id}/ct/{case_id}_VEN.nii.gz")
    del_   = nib.load(f"{DATA_ROOT}/{case_id}/ct/{case_id}_DEL.nii.gz")
    dry    = nib.load(f"{DATA_ROOT}/{case_id}/ct/{case_id}_DRY.nii.gz")

    # Load lesion segmentation
    lesion = nib.load(f"{DATA_ROOT}/{case_id}/annotations/lesion.nii.gz")

    # --- Run your model ---
    prediction = your_model.predict(art, ven, del_, dry, lesion)
    # ----------------------

    results.append({"case_id": case_id, "prediction": prediction})

pd.DataFrame(results).to_csv(
    os.path.join(output_dir, "predictions.csv"), index=False
)
```

---

## Evaluation metric

```
Final Score = 0.85 × Adjusted QWK + 0.15 × Special Category Recognition
```

- **Adjusted QWK** — Quadratic Weighted Kappa on LR-1 through LR-5 cases
- **Special Category Recognition** — 3-class balanced accuracy (Ordinal / LR-M / LR-TIV)

The full scoring implementation is available in `evaluate.py`.

---
