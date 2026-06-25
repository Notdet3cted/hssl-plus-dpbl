"""
HSSL vs HSSL+DPBL Per-Subject Comparison Analysis

Generates a dedicated comparison between HSSL (without DPBL) and HSSL+DPBL
to quantify the contribution of the Dynamic Personalized Baseline Layer.

Usage:
    python -m src.compare_hssl_dpbl
"""

import os
import json
import csv
import numpy as np
from scipy import stats
from src.logger import setup_logger
from src.experiment_tracker import ExperimentTracker

logger = setup_logger("CompareHSSL_DPBL")

MODELS_TO_COMPARE = ["hssl", "hssl+dpbl"]
METRICS = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]


class HSSLvsDPBLAnalyzer:
    def __init__(self, results_dir=None):
        self.tracker = ExperimentTracker()
        self.results_dir = results_dir or self.tracker.config["paths"].get("results", "results")
        self.reports_dir = self.tracker.config["paths"].get("reports", "reports")
        self.output_dir = os.path.join(self.results_dir, "hssl_vs_dpbl")
        os.makedirs(self.output_dir, exist_ok=True)

    def load_fold_results(self):
        """Load per-fold results for both HSSL and HSSL+DPBL models."""
        folds_path = os.path.join(self.reports_dir, "loso_folds.json")
        with open(folds_path, 'r') as f:
            folds = json.load(f)

        subjects = sorted(folds.keys())
        results = {model: {} for model in MODELS_TO_COMPARE}

        for model in MODELS_TO_COMPARE:
            for subj in subjects:
                # Try multiple naming conventions
                candidates = [
                    os.path.join(self.results_dir, f"{model}_fold_{subj}.json"),
                    os.path.join(self.results_dir, f"{model.replace('+', '_plus_')}_fold_{subj}.json"),
                ]
                for path in candidates:
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            results[model][subj] = json.load(f)
                        break

        return results, subjects

    def per_subject_comparison(self, results, subjects):
        """Create per-subject metric comparison table."""
        rows = []
        for subj in subjects:
            row = {"subject": subj}
            for metric in METRICS:
                for model in MODELS_TO_COMPARE:
                    key = f"{model}_{metric}"
                    if subj in results[model] and metric in results[model][subj]:
                        row[key] = results[model][subj][metric]
                    else:
                        row[key] = None

                # Compute delta (HSSL+DPBL - HSSL)
                hssl_val = row.get(f"hssl_{metric}")
                dpbl_val = row.get(f"hssl+dpbl_{metric}")
                if hssl_val is not None and dpbl_val is not None:
                    row[f"delta_{metric}"] = dpbl_val - hssl_val
                else:
                    row[f"delta_{metric}"] = None

            rows.append(row)
        return rows

    def aggregate_statistics(self, per_subject_rows):
        """Compute aggregate stats: mean, std, paired t-test, effect size."""
        agg = {}
        for metric in METRICS:
            hssl_vals = []
            dpbl_vals = []
            for row in per_subject_rows:
                h = row.get(f"hssl_{metric}")
                d = row.get(f"hssl+dpbl_{metric}")
                if h is not None and d is not None:
                    hssl_vals.append(h)
                    dpbl_vals.append(d)

            if len(hssl_vals) >= 2:
                hssl_arr = np.array(hssl_vals)
                dpbl_arr = np.array(dpbl_vals)
                delta = dpbl_arr - hssl_arr

                # Paired t-test
                t_stat, p_value = stats.ttest_rel(dpbl_arr, hssl_arr)

                # Wilcoxon signed-rank (non-parametric alternative)
                try:
                    w_stat, w_pvalue = stats.wilcoxon(dpbl_arr, hssl_arr)
                except ValueError:
                    w_stat, w_pvalue = np.nan, np.nan

                # Cohen's d for paired samples
                d_mean = np.mean(delta)
                d_std = np.std(delta, ddof=1)
                cohens_d = d_mean / d_std if d_std > 0 else 0.0

                # Win/tie/loss count
                wins = int(np.sum(delta > 0.001))
                ties = int(np.sum(np.abs(delta) <= 0.001))
                losses = int(np.sum(delta < -0.001))

                agg[metric] = {
                    "hssl_mean": float(np.mean(hssl_arr)),
                    "hssl_std": float(np.std(hssl_arr, ddof=1)),
                    "hssl_dpbl_mean": float(np.mean(dpbl_arr)),
                    "hssl_dpbl_std": float(np.std(dpbl_arr, ddof=1)),
                    "delta_mean": float(d_mean),
                    "delta_std": float(d_std),
                    "t_statistic": float(t_stat),
                    "p_value_ttest": float(p_value),
                    "w_statistic": float(w_stat) if not np.isnan(w_stat) else None,
                    "p_value_wilcoxon": float(w_pvalue) if not np.isnan(w_pvalue) else None,
                    "cohens_d": float(cohens_d),
                    "significant_005": p_value < 0.05,
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "n_folds": len(hssl_vals)
                }
            else:
                agg[metric] = {"error": f"Not enough paired data (n={len(hssl_vals)})"}

        return agg

    def export_per_subject_csv(self, per_subject_rows):
        """Export per-subject comparison to CSV."""
        out_path = os.path.join(self.output_dir, "per_subject_comparison.csv")
        if not per_subject_rows:
            logger.warning("No per-subject data to export.")
            return None

        fieldnames = ["subject"]
        for metric in METRICS:
            for model in MODELS_TO_COMPARE:
                fieldnames.append(f"{model}_{metric}")
            fieldnames.append(f"delta_{metric}")

        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in per_subject_rows:
                formatted = {}
                for k, v in row.items():
                    if isinstance(v, float):
                        formatted[k] = f"{v:.4f}"
                    else:
                        formatted[k] = v
                writer.writerow(formatted)

        logger.info(f"Saved per-subject comparison -> {out_path}")
        return out_path

    def export_aggregate_json(self, agg_stats):
        """Export aggregate statistics to JSON."""
        out_path = os.path.join(self.output_dir, "aggregate_statistics.json")
        with open(out_path, 'w') as f:
            json.dump(agg_stats, f, indent=2, default=str)
        logger.info(f"Saved aggregate statistics -> {out_path}")
        return out_path

    def print_summary(self, agg_stats):
        """Print formatted summary to logger."""
        logger.info("=" * 80)
        logger.info("HSSL vs HSSL+DPBL COMPARISON SUMMARY")
        logger.info("=" * 80)
        logger.info(f"{'Metric':<20} {'HSSL':>14} {'HSSL+DPBL':>14} {'Delta':>12} {'p-value':>10} {'Sig?':>6} {'W/T/L':>10}")
        logger.info("-" * 80)

        for metric in METRICS:
            s = agg_stats.get(metric, {})
            if "error" in s:
                logger.info(f"{metric:<20} {s['error']}")
                continue

            hssl_str = f"{s['hssl_mean']:.4f}±{s['hssl_std']:.4f}"
            dpbl_str = f"{s['hssl_dpbl_mean']:.4f}±{s['hssl_dpbl_std']:.4f}"
            delta_str = f"{s['delta_mean']:+.4f}"
            p_str = f"{s['p_value_ttest']:.4f}"
            sig_str = "YES" if s['significant_005'] else "no"
            wtl_str = f"{s['wins']}/{s['ties']}/{s['losses']}"

            logger.info(f"{metric:<20} {hssl_str:>14} {dpbl_str:>14} {delta_str:>12} {p_str:>10} {sig_str:>6} {wtl_str:>10}")

        logger.info("=" * 80)
        
        # Effect size interpretation
        for metric in ["f1_score", "accuracy"]:
            s = agg_stats.get(metric, {})
            if "cohens_d" in s:
                d = abs(s["cohens_d"])
                if d < 0.2:
                    effect = "negligible"
                elif d < 0.5:
                    effect = "small"
                elif d < 0.8:
                    effect = "medium"
                else:
                    effect = "large"
                logger.info(f"Cohen's d for {metric}: {s['cohens_d']:.4f} ({effect} effect)")

    def run(self):
        """Full analysis pipeline."""
        logger.info("Starting HSSL vs HSSL+DPBL comparison analysis...")

        # 1. Load results
        results, subjects = self.load_fold_results()

        for model in MODELS_TO_COMPARE:
            n = len(results[model])
            logger.info(f"Loaded {n} fold results for {model}")
            if n == 0:
                logger.error(f"No results found for {model}. Run training first.")
                return

        # 2. Per-subject comparison
        per_subject_rows = self.per_subject_comparison(results, subjects)

        # 3. Aggregate statistics
        agg_stats = self.aggregate_statistics(per_subject_rows)

        # 4. Export
        self.export_per_subject_csv(per_subject_rows)
        self.export_aggregate_json(agg_stats)

        # 5. Print summary
        self.print_summary(agg_stats)

        logger.info("HSSL vs HSSL+DPBL comparison analysis complete.")
        return agg_stats


def main():
    analyzer = HSSLvsDPBLAnalyzer()
    analyzer.run()


if __name__ == "__main__":
    main()