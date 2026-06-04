#!/usr/bin/env python3
"""
LIRADS Challenge Evaluation Script

Final Score = 0.85 × Adjusted QWK + 0.15 × Special Category Recognition

Adjusted QWK:
  - Only cases where GT is LR-1 through LR-5 (ordinal) are included.
  - Cases where the model predicted LR-M or LR-TIV on an ordinal GT are excluded
    from QWK — they are already captured as false positives in the SCR component.
  - QWK is clipped to [0, 1] (negative kappa → 0).

Special Category Recognition:
  - 3-class balanced accuracy across ALL cases:
      Class 0 — Ordinal  (LR-1 to LR-5)
      Class 1 — LR-M
      Class 2 — LR-TIV
  - LR-M and LR-TIV are evaluated as separate classes; confusing one for the
    other is penalised. Balanced accuracy averages per-class recall so each
    class contributes equally regardless of prevalence.
  - This is the sole component that evaluates LR-M and LR-TIV performance.
    The two components are orthogonal and do not overlap.
"""

import sys
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score

# ── Label constants ────────────────────────────────────────────────────────────
ORDINAL_LABELS = ["LR-1", "LR-2", "LR-3", "LR-4", "LR-5"]
SPECIAL_LABELS = ["LR-M", "LR-TIV"]
VALID_LABELS   = ORDINAL_LABELS + SPECIAL_LABELS

ORDINAL_INT = {label: i for i, label in enumerate(ORDINAL_LABELS)}

QWK_WEIGHT = 0.85
SCR_WEIGHT = 0.15


# ── Core metrics ──────────────────────────────────────────────────────────────

def compute_adjusted_qwk(gt: pd.Series, pred: pd.Series) -> float:
    """QWK restricted to ordinal GT cases (LR-1 to LR-5) with ordinal predictions.

    Cases where GT is ordinal but the model predicted LR-M or LR-TIV are excluded —
    those errors are captured by the SCR component.
    """
    mask     = gt.isin(ORDINAL_LABELS) & pred.isin(ORDINAL_LABELS)
    gt_ord   = gt[mask].reset_index(drop=True)
    pred_ord = pred[mask].reset_index(drop=True)

    if len(gt_ord) == 0:
        return 0.0

    gt_int   = gt_ord.map(ORDINAL_INT)
    pred_int = pred_ord.map(ORDINAL_INT)

    if len(set(gt_int) | set(pred_int)) < 2:
        return 1.0 if (gt_int == pred_int).all() else 0.0

    kappa = cohen_kappa_score(gt_int, pred_int, weights="quadratic")
    return float(max(0.0, kappa))


def compute_special_category_recognition(gt: pd.Series, pred: pd.Series) -> float:
    """3-class balanced accuracy: Ordinal vs LR-M vs LR-TIV (all cases).

    LR-M and LR-TIV are distinct classes — confusing one for the other is
    penalised. Balanced accuracy averages per-class recall so prevalence
    differences between ordinal and special cases don't dominate the score.
    """
    def to_3class(s: pd.Series) -> pd.Series:
        return s.map(lambda x: "ordinal" if x in ORDINAL_LABELS else x)

    gt_3   = to_3class(gt)
    pred_3 = to_3class(pred)

    if len(gt_3.unique()) < 2:
        return float((gt_3 == pred_3).mean())

    return float(balanced_accuracy_score(gt_3, pred_3))


def compute_composite_score(qwk: float, scr: float) -> float:
    return QWK_WEIGHT * qwk + SCR_WEIGHT * scr


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_ci(
    gt: pd.Series,
    pred: pd.Series,
    n_iter: int = 10_000,
    alpha: float = 0.95,
    seed: int = 42,
) -> dict:
    """Percentile bootstrap CI for composite score, QWK, and SCR."""
    rng = np.random.default_rng(seed)
    n = len(gt)
    gt_arr   = gt.reset_index(drop=True)
    pred_arr = pred.reset_index(drop=True)

    qwk_boot, scr_boot, comp_boot = [], [], []

    for _ in range(n_iter):
        idx  = rng.integers(0, n, size=n)
        g    = gt_arr.iloc[idx].reset_index(drop=True)
        p    = pred_arr.iloc[idx].reset_index(drop=True)
        qwk  = compute_adjusted_qwk(g, p)
        scr  = compute_special_category_recognition(g, p)
        qwk_boot.append(qwk)
        scr_boot.append(scr)
        comp_boot.append(compute_composite_score(qwk, scr))

    lo, hi = (1 - alpha) / 2 * 100, (1 - (1 - alpha) / 2) * 100

    def _ci(values):
        arr = np.array(values)
        return {
            "mean":     float(np.mean(arr)),
            "ci_lower": float(np.percentile(arr, lo)),
            "ci_upper": float(np.percentile(arr, hi)),
        }

    return {
        "composite": _ci(comp_boot),
        "qwk":       _ci(qwk_boot),
        "scr":       _ci(scr_boot),
    }


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_and_merge(gt_path: str, pred_path: str) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Returns (merged_valid_df, full_gt_df, completion_rate).
    merged_valid_df has columns: case_id, label (GT), prediction.
    """
    gt_df   = pd.read_csv(gt_path)
    pred_df = pd.read_csv(pred_path)

    gt_df.columns   = gt_df.columns.str.strip().str.lower()
    pred_df.columns = pred_df.columns.str.strip().str.lower()

    # Accept 'label' or 'ground_truth' column names
    if "ground_truth" in gt_df.columns:
        gt_df = gt_df.rename(columns={"ground_truth": "label"})
    if "prediction" not in pred_df.columns and "label" in pred_df.columns:
        pred_df = pred_df.rename(columns={"label": "prediction"})

    merged = gt_df[["case_id", "label"]].merge(
        pred_df[["case_id", "prediction"]], on="case_id", how="left"
    )

    valid_mask      = merged["prediction"].isin(VALID_LABELS)
    completion_rate = float(valid_mask.mean())
    merged_valid    = merged[valid_mask].copy().reset_index(drop=True)

    return merged_valid, gt_df, completion_rate


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(
    gt_path: str,
    pred_path: str,
    bootstrap: bool = True,
    n_iter: int = 10_000,
) -> dict:
    merged, gt_df, completion_rate = load_and_merge(gt_path, pred_path)

    gt   = merged["label"]
    pred = merged["prediction"]

    qwk        = compute_adjusted_qwk(gt, pred)
    scr        = compute_special_category_recognition(gt, pred)
    final      = compute_composite_score(qwk, scr)

    result = {
        "final_score":                   round(final, 6),
        "adjusted_qwk":                  round(qwk, 6),
        "special_category_recognition":  round(scr, 6),
        "completion_rate":               round(completion_rate, 6),
        "n_cases_scored":                len(merged),
        "n_cases_total":                 len(gt_df),
        "compliant":                     completion_rate >= 0.95,
    }

    if bootstrap:
        result["bootstrap_ci"] = bootstrap_ci(gt, pred, n_iter=n_iter)

    return result


def print_results(results: dict, n_iter: int) -> None:
    width = 54
    print(f"\n{'=' * width}")
    print("  LIRADS Challenge — Evaluation Results")
    print(f"{'=' * width}")
    print(f"  Final Score                  : {results['final_score']:.4f}")
    print(f"    Adjusted QWK        (×0.85): {results['adjusted_qwk']:.4f}")
    print(f"    Special Cat. Recog  (×0.15): {results['special_category_recognition']:.4f}")
    print(f"  Completion Rate              : {results['completion_rate']:.1%}")
    print(f"  Cases Scored                 : {results['n_cases_scored']} / {results['n_cases_total']}")
    print(f"  Compliant (≥95% complete)    : {results['compliant']}")

    if "bootstrap_ci" in results:
        ci = results["bootstrap_ci"]
        print(f"\n  95% Bootstrap CI  (n = {n_iter:,} iterations)")
        for key, label in [("composite", "Final Score"), ("qwk", "QWK       "), ("scr", "SCR       ")]:
            c = ci[key]
            print(f"    {label}: {c['mean']:.4f}  [{c['ci_lower']:.4f}, {c['ci_upper']:.4f}]")

    print(f"{'=' * width}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LIRADS Challenge Evaluation — computes composite score"
    )
    parser.add_argument("--ground_truth", required=True,
                        help="CSV with columns: case_id, label")
    parser.add_argument("--predictions", required=True,
                        help="CSV with columns: case_id, prediction")
    parser.add_argument("--no-bootstrap", action="store_true",
                        help="Skip bootstrap CI (faster, for quick checks)")
    parser.add_argument("--n-iter", type=int, default=10_000,
                        help="Bootstrap iterations (default: 10 000)")
    parser.add_argument("--output", default=None,
                        help="Optional: save results as JSON to this path")
    args = parser.parse_args()

    results = evaluate(
        args.ground_truth,
        args.predictions,
        bootstrap=not args.no_bootstrap,
        n_iter=args.n_iter,
    )

    print_results(results, args.n_iter)

    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")

    # Non-compliant exits with code 1 so CI pipelines can catch it
    if not results["compliant"]:
        print("WARNING: completion rate below 95% — submission may be excluded from awards.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
