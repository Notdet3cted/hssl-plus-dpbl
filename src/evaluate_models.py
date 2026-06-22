"""
FASE 6 - Model Evaluation
Evaluate all LOSO folds for all models, compute metrics, export fold_results.csv

Usage:
    python -m src.evaluate_models
    python -m src.evaluate_models --from_predictions   # re-evaluate from saved predictions
"""
import os
import json
import csv
import numpy as np
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from src.logger import setup_logger

logger = setup_logger("ModelEvaluator")

# DPBL standalone removed — only ensemble hssl+dpbl is evaluated
MODELS = ["rf", "cnn", "hssl", "hssl+dpbl"]
METRICS = ["accuracy", "precision", "recall", "f1_score", "weighted_precision", "weighted_recall", "weighted_f1", "roc_auc", "pr_auc"]


class ModelEvaluator:
    """
    Phase 6: LOSO Model Evaluation.
    - Collects per-fold results from JSON files OR recomputes from saved predictions.
    - Validates metric completeness across all folds × models.
    - Exports fold_results.csv and model_summary.csv.
    """

    def __init__(self, results_dir="results", reports_dir="reports"):
        self.results_dir = results_dir
        self.predictions_dir = os.path.join(results_dir, "predictions")
        self.reports_dir = reports_dir
        os.makedirs(self.results_dir, exist_ok=True)

        # Load fold config
        folds_path = os.path.join(self.reports_dir, "loso_folds.json")
        if not os.path.exists(folds_path):
            raise FileNotFoundError(f"LOSO folds config not found: {folds_path}")
        with open(folds_path, "r") as f:
            self.folds = json.load(f)
        self.subjects = list(self.folds.keys())
        logger.info(f"Loaded {len(self.subjects)} LOSO folds: {self.subjects}")

    # ------------------------------------------------------------------
    # Core metric computation
    # ------------------------------------------------------------------
    def compute_metrics(self, y_true, y_prob, num_classes=None):
        """Compute all 6 metrics from true labels and probability predictions."""
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        # Shape validation
        if y_prob.ndim != 2:
            raise ValueError(f"y_prob must be 2D (n_samples, n_classes), got shape {y_prob.shape}")
        if y_true.ndim != 1:
            raise ValueError(f"y_true must be 1D (n_samples,), got shape {y_true.shape}")
        if len(y_true) != len(y_prob):
            raise ValueError(f"y_true ({len(y_true)}) and y_prob ({len(y_prob)}) length mismatch")

        y_pred = np.argmax(y_prob, axis=1)

        # Use np.unique to determine actual classes present
        unique_classes = np.unique(y_true)
        if num_classes is None:
            num_classes = y_prob.shape[1]

        if not np.allclose(y_prob.sum(axis=1), 1, atol=1e-4):
            logger.warning("Probabilities do not sum to 1. This may affect AUC metrics.")

        result = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }

        # Confusion matrix — saved separately
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        result["confusion_matrix"] = cm.tolist()

        # AUC metrics — use np.unique(y_true) for binarization
        try:
            if num_classes > 2:
                present_classes = unique_classes.tolist()
                y_true_bin = label_binarize_safe(y_true, present_classes, num_classes)
                result["roc_auc"] = float(roc_auc_score(
                    y_true_bin, y_prob, multi_class="ovr", average="macro"
                ))
                result["pr_auc"] = float(average_precision_score(
                    y_true_bin, y_prob, average="macro"
                ))
            else:
                result["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
                result["pr_auc"] = float(average_precision_score(y_true, y_prob[:, 1]))
        except Exception as e:
            logger.warning(f"AUC computation failed: {e}")
            result["roc_auc"] = 0.0
            result["pr_auc"] = 0.0
            result["auc_error"] = str(e)

        return result

    # ------------------------------------------------------------------
    # Re-evaluate from saved predictions
    # ------------------------------------------------------------------
    def evaluate_from_predictions(self, model, fold):
        """Recompute metrics from saved prediction file."""
        pred_path = os.path.join(self.predictions_dir, f"{model}_fold_{fold}.json")
        if not os.path.exists(pred_path):
            return None

        with open(pred_path, "r") as f:
            pred_data = json.load(f)

        y_true = np.array(pred_data["y_true"])
        y_prob = np.array(pred_data["y_prob"])

        # Shape validation before evaluation
        if y_prob.ndim != 2 or y_true.ndim != 1 or len(y_true) != len(y_prob):
            logger.error(
                f"[{model}] Fold {fold}: invalid prediction shape "
                f"y_true={y_true.shape}, y_prob={y_prob.shape}. Skipping."
            )
            return None

        metrics = self.compute_metrics(y_true, y_prob)

        # Save updated result JSON
        result_path = os.path.join(self.results_dir, f"{model}_fold_{fold}.json")
        with open(result_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"[{model}] Fold {fold} re-evaluated -> F1={metrics['f1_score']:.4f}")
        return metrics

    # ------------------------------------------------------------------
    # Collect results (from JSON files)
    # ------------------------------------------------------------------
    def collect_fold_results(self):
        """Collect all per-fold result JSONs into a list of dicts."""
        rows = []
        missing = []

        for fold in self.subjects:
            for model in MODELS:
                path = os.path.join(self.results_dir, f"{model}_fold_{fold}.json")
                if not os.path.exists(path):
                    missing.append(f"{model}_fold_{fold}")
                    continue
                with open(path, "r") as f:
                    data = json.load(f)

                row = {"model": model, "fold": fold}
                for m in METRICS:
                    row[m] = data.get(m, None)
                # Confusion matrix stored as JSON string
                cm = data.get("confusion_matrix", None)
                row["confusion_matrix"] = json.dumps(cm) if cm else ""
                rows.append(row)

        if missing:
            logger.warning(f"Missing {len(missing)} result files: {missing}")
        else:
            logger.info("All fold × model result files present — evaluation complete.")

        return rows, missing

    # ------------------------------------------------------------------
    # Re-evaluate ALL from predictions
    # ------------------------------------------------------------------
    def reevaluate_all_from_predictions(self):
        """Recompute metrics for all models × folds from saved predictions."""
        reeval_count = 0
        for fold in self.subjects:
            for model in MODELS:
                result = self.evaluate_from_predictions(model, fold)
                if result is not None:
                    reeval_count += 1
        logger.info(f"Re-evaluated {reeval_count} prediction files.")
        return reeval_count

    # ------------------------------------------------------------------
    # Export fold_results.csv
    # ------------------------------------------------------------------
    def export_fold_results(self, rows):
        """Write per-fold results to fold_results.csv."""
        out_path = os.path.join(self.results_dir, "fold_results.csv")
        fieldnames = ["model", "fold"] + METRICS + ["confusion_matrix"]

        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Saved {len(rows)} rows -> {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # Export model_summary.csv (mean, std, min, max per model)
    # ------------------------------------------------------------------
    def export_model_summary(self, rows):
        """Compute and export mean/std/min/max per model."""
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
                    summary[f"{m}_std"] = round(float(np.std(arr, ddof=1)), 4) if len(arr) > 1 else 0.0
                    summary[f"{m}_min"] = round(float(np.min(arr)), 4)
                    summary[f"{m}_max"] = round(float(np.max(arr)), 4)
                else:
                    for sfx in ("_mean", "_std", "_min", "_max"):
                        summary[f"{m}{sfx}"] = None
            summary_rows.append(summary)

        if not summary_rows:
            logger.warning("No summary data to export.")
            return None

        # Build fieldnames
        s_fields = ["model", "n_folds"]
        for m in METRICS:
            s_fields += [f"{m}_mean", f"{m}_std", f"{m}_min", f"{m}_max"]

        summary_path = os.path.join(self.results_dir, "model_summary.csv")
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=s_fields)
            writer.writeheader()
            writer.writerows(summary_rows)
        logger.info(f"Saved model summary -> {summary_path}")

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
                    return "   N/A   "
                return f"{v:>10.4f}"
            logger.info(
                f"{s['model']:<12} {s['n_folds']:>4} "
                f"{_fmt('accuracy_mean')} {_fmt('precision_mean')} {_fmt('recall_mean')} "
                f"{_fmt('f1_score_mean')} {_fmt('roc_auc_mean')} {_fmt('pr_auc_mean')}"
            )
        logger.info("=" * 90)

        return summary_path

    # ------------------------------------------------------------------
    # Export per-model confusion matrices
    # ------------------------------------------------------------------
    def export_confusion_matrices(self, rows):
        """Save each model's aggregated confusion matrix to a separate JSON file."""
        cm_dir = os.path.join(self.results_dir, "confusion_matrices")
        os.makedirs(cm_dir, exist_ok=True)

        for model in MODELS:
            model_rows = [r for r in rows if r["model"] == model]
            if not model_rows:
                continue

            # Sum confusion matrices across folds
            cm_sum = None
            for r in model_rows:
                cm_str = r.get("confusion_matrix", "")
                if cm_str:
                    cm = json.loads(cm_str)
                    cm_arr = np.array(cm)
                    if cm_sum is None:
                        cm_sum = cm_arr
                    else:
                        # Pad to same size if needed
                        max_size = max(cm_sum.shape[0], cm_arr.shape[0])
                        padded = np.zeros((max_size, max_size), dtype=int)
                        padded[:cm_arr.shape[0], :cm_arr.shape[1]] = cm_arr
                        cm_sum_padded = np.zeros((max_size, max_size), dtype=int)
                        cm_sum_padded[:cm_sum.shape[0], :cm_sum.shape[1]] = cm_sum
                        cm_sum = cm_sum_padded + padded

            if cm_sum is not None:
                out_path = os.path.join(cm_dir, f"{model}_confusion_matrix.json")
                with open(out_path, "w") as f:
                    json.dump(cm_sum.tolist(), f, indent=2)
                
                cm_norm = cm_sum.astype('float') / cm_sum.sum(axis=1)[:, np.newaxis]
                cm_norm = np.nan_to_num(cm_norm) # handle division by zero
                out_path_norm = os.path.join(cm_dir, f"{model}_normalized_confusion_matrix.json")
                with open(out_path_norm, "w") as f:
                    json.dump(cm_norm.tolist(), f, indent=2)

                logger.info(f"Saved confusion matrix and normalized version for {model}")

    # ------------------------------------------------------------------
    # Validation: check completeness
    # ------------------------------------------------------------------
    def validate_completeness(self, rows, missing):
        """Validate that all folds have complete metrics for all models."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_folds": len(self.subjects),
            "total_models": len(MODELS),
            "expected_results": len(self.subjects) * len(MODELS),
            "actual_results": len(rows),
            "missing_results": missing,
            "is_complete": len(missing) == 0,
            "metrics_checked": METRICS,
        }

        # Check for None metrics in existing results
        incomplete_metrics = []
        for r in rows:
            for m in METRICS:
                if r.get(m) is None:
                    incomplete_metrics.append(f"{r['model']}_fold_{r['fold']}:{m}")
        report["incomplete_metrics"] = incomplete_metrics

        report_path = os.path.join(self.results_dir, "evaluation_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        if report["is_complete"] and not incomplete_metrics:
            logger.info("✓ EVALUATION COMPLETE: All folds × models have full metrics.")
        else:
            logger.warning(
                f"EVALUATION INCOMPLETE: {len(missing)} missing, "
                f"{len(incomplete_metrics)} incomplete metrics."
            )

        return report

    # ------------------------------------------------------------------
    # Save evaluation metadata
    # ------------------------------------------------------------------
    def save_metadata(self, report, fold_csv, summary_csv):
        """Save evaluation metadata for reproducibility."""
        metadata = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "models_evaluated": MODELS,
            "metrics_computed": METRICS,
            "total_folds": len(self.subjects),
            "subjects": self.subjects,
            "output_files": {
                "fold_results": fold_csv,
                "model_summary": summary_csv,
                "evaluation_report": os.path.join(self.results_dir, "evaluation_report.json"),
            },
            "completeness": report.get("is_complete", False),
            "missing_count": len(report.get("missing_results", [])),
        }
        meta_path = os.path.join(self.results_dir, "evaluation_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved evaluation metadata -> {meta_path}")
        return meta_path

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    def run(self, from_predictions=False):
        """
        Full Phase 6 evaluation pipeline:
        1. Optionally re-evaluate from saved predictions
        2. Collect per-fold results
        3. Export fold_results.csv
        4. Export model_summary.csv
        5. Export confusion matrices (separate files)
        6. Validate completeness
        7. Save evaluation metadata
        """
        logger.info("=" * 60)
        logger.info("PHASE 6: MODEL EVALUATION")
        logger.info("=" * 60)

        # Step 1: Optionally re-evaluate from predictions
        if from_predictions:
            logger.info("Re-evaluating from saved predictions...")
            self.reevaluate_all_from_predictions()

        # Step 2: Collect results
        rows, missing = self.collect_fold_results()

        if not rows:
            logger.error("No results found. Run training first (src/run_all_folds.py).")
            return None

        # Step 3: Export fold_results.csv
        fold_csv = self.export_fold_results(rows)

        # Step 4: Export model_summary.csv
        summary_csv = self.export_model_summary(rows)

        # Step 5: Export confusion matrices (separate per model)
        self.export_confusion_matrices(rows)

        # Step 6: Validate completeness
        report = self.validate_completeness(rows, missing)

        # Step 7: Save evaluation metadata
        self.save_metadata(report, fold_csv, summary_csv)

        logger.info("Phase 6 evaluation pipeline complete.")
        return {
            "fold_results_csv": fold_csv,
            "model_summary_csv": summary_csv,
            "evaluation_report": report,
        }


# ------------------------------------------------------------------
# Safe label_binarize using np.unique
# ------------------------------------------------------------------
def label_binarize_safe(y_true, present_classes, num_classes):
    """
    Binarize labels using present classes to handle missing classes in folds.
    """
    from sklearn.preprocessing import label_binarize
    return label_binarize(y_true, classes=present_classes)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 6: Model Evaluation")
    parser.add_argument(
        "--from_predictions", action="store_true",
        help="Re-evaluate metrics from saved prediction files"
    )
    args = parser.parse_args()

    evaluator = ModelEvaluator()
    evaluator.run(from_predictions=args.from_predictions)


if __name__ == "__main__":
    main()