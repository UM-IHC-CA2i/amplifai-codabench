#!/usr/bin/env python3
"""
AMPLIFAI Challenge — example submission.

This is a minimal working template. Replace the predict() function with
your own model inference. Everything else (file paths, output format) must
stay the same.
"""

import os
import sys

# Network is disabled during ingestion, so dependencies must be bundled rather
# than pip-installed at runtime. pandas/numpy already ship in the base image;
# only nibabel needs bundling here (see packages/, built with:
#   pip install --target=packages --no-deps nibabel packaging importlib-resources typing-extensions
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "packages"))

import nibabel as nib
import numpy as np
import pandas as pd

VALID_LABELS = ["LR-1", "LR-2", "LR-3", "LR-4", "LR-5", "LR-M", "LR-TIV"]
DATA_ROOT    = "/app/data/cases"


def load_case(case_id: str) -> dict:
    """Load all volumes for a case. Returns a dict of nibabel images."""
    base = os.path.join(DATA_ROOT, case_id)
    return {
        "ART":    nib.load(os.path.join(base, "ct", f"{case_id}_ART.nii.gz")),
        "VEN":    nib.load(os.path.join(base, "ct", f"{case_id}_VEN.nii.gz")),
        "DEL":    nib.load(os.path.join(base, "ct", f"{case_id}_DEL.nii.gz")),
        "DRY":    nib.load(os.path.join(base, "ct", f"{case_id}_DRY.nii.gz")),
        "lesion": nib.load(os.path.join(base, "annotations", "lesion.nii.gz")),
    }


def predict(volumes: dict) -> str:
    """
    Replace this with your model inference.

    Args:
        volumes: dict with keys ART, VEN, DEL, DRY, lesion (nibabel images)

    Returns:
        One of: LR-1, LR-2, LR-3, LR-4, LR-5, LR-M, LR-TIV
    """
    # --- YOUR MODEL HERE ---
    # e.g. art_array = volumes["ART"].get_fdata()
    #      mask      = volumes["lesion"].get_fdata().astype(bool)
    #      prediction = your_model.predict(art_array, mask)
    #      return prediction
    # -----------------------

    # Placeholder: returns a fixed label
    return "LR-3"


def main() -> None:
    input_dir  = sys.argv[1] if len(sys.argv) > 1 else "/app/input_data"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "/app/output"
    os.makedirs(output_dir, exist_ok=True)

    cases_path = os.path.join("/app/input_data", "sample_cases.csv")
    if not os.path.exists(cases_path):
        cases_path = os.path.join(input_dir, "sample_cases.csv")

    cases    = pd.read_csv(cases_path)
    case_ids = cases["case_id"].tolist()
    print(f"Processing {len(case_ids)} cases...")

    results = []
    for case_id in case_ids:
        try:
            volumes    = load_case(case_id)
            prediction = predict(volumes)
        except Exception as e:
            print(f"  WARNING: {case_id} failed ({e})", file=sys.stderr)
            prediction = "LR-3"   # fallback — replace with your error handling

        results.append({"case_id": case_id, "prediction": prediction})
        print(f"  {case_id}: {prediction}")

    out_path = os.path.join(output_dir, "predictions.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nDone. {len(results)} predictions written to {out_path}")


if __name__ == "__main__":
    main()
