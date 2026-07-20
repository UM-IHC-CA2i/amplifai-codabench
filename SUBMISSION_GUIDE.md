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

`sample_cases.csv` is a single column, one case ID per row:

```csv
case_id
CASE00017
CASE00043
CASE00044
CASE00065
CASE00066
...
```

Don't assume a specific ID format or count — just read the `case_id` column and iterate. The exact case list depends on which phase you're submitting to.

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
└── ...             ← model weights, helper scripts, bundled packages
```

**`metadata` file contents** (include this exactly):
```
command: python3 /app/ingested_program/run.py $input $output
```

---

## Docker environment

**Image:** `codalab/codalab-legacy:gpu310`  
**Python:** 3.10  
**GPU:** NVIDIA RTX 4090 (24 GB VRAM), CUDA 12.x available  
**CPU:** AMD Ryzen 9 7950X, 32 threads (16 cores) — no CPU limit is set on submission containers, and only one submission runs at a time, so your code has exclusive, unrestricted use of all 32 threads while it's running  
**Network:** **Disabled** — no outbound internet access inside the container

### Bundling your dependencies

Because network access is disabled, `pip install` will not work at runtime. You must include all required packages in your submission zip.

**Option A — bundle with pip:**

```bash
# From your local environment (Python 3.10):
pip install nibabel pandas numpy torch --target ./packages
zip -r submission.zip run.py metadata packages/
```

Then at the top of `run.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "packages"))
```

**Option B — wheel files:**

Download wheels for your target platform (`linux_x86_64`, Python 3.10, CUDA) and include them in the zip. Install them from a local path:

```python
import subprocess, sys, os
pkg_dir = os.path.join(os.path.dirname(__file__), "wheels")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-index",
                       "--find-links", pkg_dir, "nibabel", "torch"])
```

**Packages already in the base image** (do not bundle your own copies of these — see the ABI warning below):

| Package | Version |
|---|---|
| Python | 3.10.14 |
| pip | 24.1.1 |
| numpy | 1.26.4 |
| pandas | 2.2.2 |
| scipy | 1.14.0 |
| scikit-learn | 1.5.0 |
| torch | 2.3.1+cu121 (CUDA 12.4.1) |

**Not present** — bundle these yourself if your code needs them: `nibabel`, `opencv-python`, `SimpleITK`, and anything else not listed above.

### ⚠️ Compiled-package / NumPy ABI compatibility

If you bundle *any* package with compiled C/C++ extensions (`numpy`, `scipy`, `scikit-learn`, `torch`, `pyradiomics`, `opencv`, `SimpleITK`, etc.), it must be built against the same major NumPy version that actually gets imported at runtime. `sys.path.insert(0, "packages/")` puts your bundled packages *ahead* of the image's own — so if your bundle's compiled extension expects a different NumPy major version than the one that ends up loaded, you'll hit errors like:

```
AttributeError: _ARRAY_API not found
ImportError: numpy.core.multiarray failed to import
```

Safest options, in order of preference:
- Don't bundle `numpy`/`pandas`/`scipy`/`scikit-learn`/`torch` at all — the versions already in the image (table above) are mutually compatible.
- If you must bundle a compiled package on top of the base image's NumPy (e.g. `pyradiomics`, `opencv`, `SimpleITK`), make sure it was built/downloaded for **NumPy 1.26**, not whatever version your local dev environment defaults to (many local setups today default to NumPy 2.x).

### Test your submission locally before uploading

Network is disabled in the real environment, so the only reliable check is running your exact zip inside the same image, offline, before you submit:

```bash
mkdir -p /tmp/test_input /tmp/test_output
echo "case_id" > /tmp/test_input/sample_cases.csv
echo "CASE00001" >> /tmp/test_input/sample_cases.csv   # use a real case_id you have data access to

unzip -o submission.zip -d /tmp/test_submission

docker run --rm --network none \
  --gpus all \
  -v /tmp/test_submission:/app/ingested_program:ro \
  -v /path/to/data:/app/data:ro \
  -v /tmp/test_input:/app/input_data:ro \
  -v /tmp/test_output:/app/output \
  codalab/codalab-legacy:gpu310 \
  python3 /app/ingested_program/run.py /app/input_data /app/output

cat /tmp/test_output/predictions.csv
```

If this doesn't produce a valid `predictions.csv` locally with `--network none`, it won't on the real server either — this catches missing-dependency and ABI-mismatch bugs (like the NumPy issue above) before you ever use a submission slot.

---

## Minimal working example

```python
# run.py
import sys, os
# If you bundled packages:
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "packages"))

import nibabel as nib
import pandas as pd

input_dir, output_dir = sys.argv[1], sys.argv[2]
os.makedirs(output_dir, exist_ok=True)

cases     = pd.read_csv("/app/input_data/sample_cases.csv")
DATA_ROOT = "/app/data/cases"
results   = []

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

See [`example_submission/run.py`](example_submission/run.py) for a complete template. Run [`example_submission/build.sh`](example_submission/build.sh) once to generate its bundled `packages/` before zipping — it's gitignored, not checked in, so this step is required after cloning.

---

## Evaluation metric

```
Final Score = 0.85 × Adjusted QWK + 0.15 × Special Category Recognition
```

- **Adjusted QWK** — Quadratic Weighted Kappa on LR-1 through LR-5 cases
- **Special Category Recognition** — 3-class balanced accuracy (Ordinal / LR-M / LR-TIV)

The full scoring implementation is available in [`evaluate.py`](evaluate.py). Run it locally to check your predictions before submitting:

```bash
python evaluate.py --ground_truth path/to/gt.csv --predictions path/to/pred.csv --no-bootstrap
```

---
