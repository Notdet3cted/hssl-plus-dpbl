"""
Aggregate per-fold JSON results into fold_results.csv
This module is now a thin wrapper around evaluate_models.py (Phase 6).

Usage:
    python -m src.aggregate_results
"""
import os
import json
import csv
import numpy as np
from src.logger import setup_logger

logger = setup_logger("AggregateResults")

MODELS = ["rf", "cnn", "hssl", "hssl+dpbl"]
METRICS = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]


def aggregate():
    results_dir = "results"
    folds_path = "reports/loso_folds.json"

    with open(folds_path, "r") as f:
        folds = json.load(f)
    subjects = list(folds.keys())

    rows = []
    missing = []

    for subj in subjects:
        for model in MODELS:
            path = os.path.join(results_dir, f"{model}_fold_{subj}.json")
            if not os.path.exists(path):
                missing.append(f"{model}_fold_{subj}")
                continue
            with open(path, "r") as f:
                data = json.load(f)
            row = {"model": model, "fold": subj}
            for m in METRICS:
                row[m] = data.get(m, None)
            # Store confusion matrix as string
            cm = data.get("confusion_matrix", None)
            row["confusion_matrix"] = json.dumps(cm) if cm else ""
            rows.append(row)

    if missing:
        logger.warning(f"Missing {len(missing)} result files: {missing[:10]}...")
    else:
        logger.info("All result files found for all folds × models.")

    # Write fold_results.csv
    out_path = os.path.join(results_dir, "fold_results.csv")
    fieldnames = ["model", "fold"] + METRICS + ["confusion_matrix"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved {len(rows)} rows to {out_path}")

    # Summary: mean/std/min/max metrics per model
    summary_path = os.path.join(results_dir, "model_summary.csv")
    summary_rows = []
    for model in MODELS:
        model_rows = [r for r in rows if r["model"] == model]
        if not model_rows:
            continue
        summary = {"model": model, "n_folds": len(model_rows)}
        for m in METRICS:
            vals = [r[m] for r in model_rows if r[m] is not None]
            if vals:
                arr = np.array(vals)
                summary[f"{m}_mean"] = round(float(np.mean(arr)), 4)
                summary[f"{m}_std"] = round(float(np.std(arr, ddof=1)), 4)
                summary[f"{m}_min"] = round(float(np.min(arr)), 4)
                summary[f"{m}_max"] = round(float(np.max(arr)), 4)
            else:
                for sfx in ("_mean", "_std", "_min", "_max"):
                    summary[f"{m}{sfx}"] = None
        summary_rows.append(summary)

    if summary_rows:
        s_fields = ["model", "n_folds"]
        for m in METRICS:
            s_fields += [f"{m}_mean", f"{m}_std", f"{m}_min", f"{m}_max"]
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=s_fields)
            writer.writeheader()
            writer.writerows(summary_rows)
        logger.info(f"Saved model summary to {summary_path}")

        # Print summary table
        logger.info("=" * 90)
        logger.info(
            f"{'Model':<12} {'N':>4} {'Acc':>10} {'Prec':>10} {'Rec':>10} "
            f"{'F1':>10} {'AUC':>10} {'PR-AUC':>10}"
        )
        logger.info("-" * 90)
        for s in summary_rows:
            def _fmt(key):
                v = s.get(key)
                if v is None:
                    return "     N/A  "
                return f"{v:>10.4f}"
            logger.info(
                f"{s['model']:<12} {s['n_folds']:>4} "
                f"{_fmt('accuracy_mean')} {_fmt('precision_mean')} {_fmt('recall_mean')} "
                f"{_fmt('f1_score_mean')} {_fmt('roc_auc_mean')} {_fmt('pr_auc_mean')}"
            )
        logger.info("=" * 90)

    # Validation report
    report = {
        "total_folds": len(subjects),
        "total_models": len(MODELS),
        "expected": len(subjects) * len(MODELS),
        "actual": len(rows),
        "missing": missing,
        "is_complete": len(missing) == 0,
    }
    report_path = os.path.join(results_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return out_path, summary_path


if __name__ == "__main__":
    aggregate()