"""
FASE 8 - Validasi Statistik
Membuktikan signifikansi hasil penelitian menggunakan uji statistik.
Includes: Normality, Friedman/ANOVA, Effect Size (Cohen's d, Hedges' g),
Post-hoc (Tukey/Dunn), Pairwise Contribution (t-test, Wilcoxon).

Tasks:
- Normality Test (Shapiro-Wilk)
- Multi-model Comparison (ANOVA / Friedman Test)
- Effect Size (Cohen's d, Hedges' g)
- Post-hoc Analysis (Tukey HSD / Dunn's Test)
- Contribution Analysis (Paired t-Test vs Wilcoxon)
- Export (statistical_results.csv)
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    import scikit_posthocs as sp
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

from src.logger import setup_logger

logger = setup_logger("StatisticalValidation")


def cohens_d(x, y):
    """Cohen's d effect size between two samples."""
    nx, ny = len(x), len(y)
    s = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / (nx+ny-2))
    if s == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / s


def hedges_g(x, y):
    """Hedges' g (bias-corrected Cohen's d)."""
    d = cohens_d(x, y)
    nx, ny = len(x), len(y)
    # Correction factor
    df = nx + ny - 2
    c = 1 - 3 / (4*df - 1)
    return d * c


def friedman_effect_size(chi2, k, n):
    """Kendall's W for Friedman test (effect size)."""
    return chi2 / (n * (k - 1))


class StatisticalValidator:
    def __init__(self, robustness_dir="results/robustness", reports_dir="reports"):
        self.robustness_dir = robustness_dir
        self.reports_dir = reports_dir
        self.output_dir = "results/statistical_analysis"
        os.makedirs(self.output_dir, exist_ok=True)
        self.iterations_data = self._load_iterations_data()

    def _load_iterations_data(self):
        data = []
        if not os.path.exists(self.robustness_dir):
            logger.warning(f"Robustness dir {self.robustness_dir} not found.")
            return pd.DataFrame()

        for root, dirs, files in os.walk(self.robustness_dir):
            for d in dirs:
                if d.startswith("iter_"):
                    iter_idx = d.split("_")[1]
                    summary_path = os.path.join(root, d, "model_summary.csv")
                    if os.path.exists(summary_path):
                        df = pd.read_csv(summary_path)
                        df['iteration'] = iter_idx
                        data.append(df)

        if not data:
            logger.warning("No iteration data. Run robustness_testing.py first.")
            return pd.DataFrame()

        combined_df = pd.concat(data, ignore_index=True)
        logger.info(f"Loaded data from {len(data)} iterations.")
        return combined_df

    def check_normality(self, df, metric="f1_score_mean"):
        """Shapiro-Wilk Test for normality per model."""
        normality_results = []
        models = df['model'].unique()

        for model in models:
            model_data = df[df['model'] == model][metric].dropna()
            if len(model_data) < 3:
                continue

            stat, p_value = stats.shapiro(model_data)
            is_normal = p_value > 0.05
            normality_results.append({
                "model": model,
                "shapiro_stat": stat,
                "p_value": p_value,
                "is_normal": is_normal
            })
            logger.info(f"[{model}] Shapiro-Wilk p={p_value:.4f} -> {'Normal' if is_normal else 'Not Normal'}")

        return pd.DataFrame(normality_results)

    def compare_models(self, df, metric="f1_score_mean", is_normal_all=True):
        """Multi-model comparison using ANOVA or Friedman + effect size."""
        models = df['model'].unique()
        data_groups = [df[df['model'] == m][metric].dropna().values for m in models]

        min_len = min([len(g) for g in data_groups])
        data_groups = [g[:min_len] for g in data_groups]

        results = {}
        if is_normal_all and len(models) > 2:
            stat, p_value = stats.f_oneway(*data_groups)
            results['test'] = 'ANOVA'
            results['statistic'] = stat
            results['p_value'] = p_value
            # Eta-squared effect size for ANOVA
            all_data = np.concatenate(data_groups)
            grand_mean = np.mean(all_data)
            ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in data_groups)
            ss_total = sum((v - grand_mean)**2 for v in all_data)
            results['effect_size'] = float(ss_between / ss_total) if ss_total > 0 else 0.0
            results['effect_type'] = 'eta_squared'
            logger.info(f"ANOVA p={p_value:.4f} eta²={results['effect_size']:.4f}")
        elif len(models) > 2:
            stat, p_value = stats.friedmanchisquare(*data_groups)
            results['test'] = 'Friedman'
            results['statistic'] = stat
            results['p_value'] = p_value
            # Kendall's W for Friedman
            results['effect_size'] = float(friedman_effect_size(stat, len(models), min_len))
            results['effect_type'] = 'kendall_w'
            logger.info(f"Friedman Test p={p_value:.4f} W={results['effect_size']:.4f}")
        else:
            results['test'] = 'N/A'
            results['p_value'] = 1.0
            results['effect_size'] = 0.0
            results['effect_type'] = 'none'

        return results

    def posthoc_analysis(self, df, metric="f1_score_mean", is_normal_all=True):
        """Tukey HSD or Dunn's Test."""
        if not HAS_STATSMODELS:
            logger.warning("statsmodels/scikit_posthocs not installed. Skipping post-hoc.")
            return None

        if is_normal_all:
            tukey = pairwise_tukeyhsd(endog=df[metric], groups=df['model'], alpha=0.05)
            res_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
            res_df['test'] = 'Tukey HSD'
            return res_df
        else:
            dunn = sp.posthoc_dunn(df, val_col=metric, group_col='model', p_adjust='bonferroni')
            dunn['test'] = 'Dunn'
            return dunn

    def compute_effect_sizes(self, df, metric="f1_score_mean"):
        """Pairwise Cohen's d and Hedges' g between all model pairs."""
        models = df['model'].unique()
        rows = []
        for i in range(len(models)):
            for j in range(i+1, len(models)):
                m1, m2 = models[i], models[j]
                d1 = df[df['model'] == m1][metric].dropna().values
                d2 = df[df['model'] == m2][metric].dropna().values
                min_len = min(len(d1), len(d2))
                d1, d2 = d1[:min_len], d2[:min_len]
                d = cohens_d(d1, d2)
                g = hedges_g(d1, d2)
                rows.append({
                    "model_a": m1, "model_b": m2,
                    "cohens_d": d, "hedges_g": g,
                    "mean_a": np.mean(d1), "mean_b": np.mean(d2),
                    "diff": np.mean(d1) - np.mean(d2)
                })
                logger.info(f"Effect {m1} vs {m2}: d={d:.4f} g={g:.4f}")
        return pd.DataFrame(rows)

    def pairwise_contribution(self, df, metric="f1_score_mean"):
        """Paired t-Test and Wilcoxon Signed-Rank between HSSL and HSSL+DPBL."""
        hssl = df[df['model'] == 'hssl'].sort_values('iteration')[metric].values
        hssl_dpbl = df[df['model'] == 'hssl+dpbl'].sort_values('iteration')[metric].values

        min_len = min(len(hssl), len(hssl_dpbl))
        if min_len < 3:
            return None

        hssl = hssl[:min_len]
        hssl_dpbl = hssl_dpbl[:min_len]

        t_stat, t_p = stats.ttest_rel(hssl, hssl_dpbl)
        w_stat, w_p = stats.wilcoxon(hssl, hssl_dpbl)
        d = cohens_d(hssl_dpbl, hssl)
        g = hedges_g(hssl_dpbl, hssl)

        return {
            "comparison": "HSSL vs HSSL+DPBL",
            "paired_ttest_stat": t_stat,
            "paired_ttest_p": t_p,
            "wilcoxon_stat": w_stat,
            "wilcoxon_p": w_p,
            "cohens_d": d,
            "hedges_g": g,
            "hssl_mean": float(np.mean(hssl)),
            "hssl_dpbl_mean": float(np.mean(hssl_dpbl)),
            "improvement": float(np.mean(hssl_dpbl) - np.mean(hssl))
        }

    def run(self):
        logger.info("=" * 60)
        logger.info("PHASE 8: STATISTICAL VALIDATION")
        logger.info("=" * 60)

        df = self.iterations_data
        if df.empty:
            return

        metric = "f1_score_mean"
        if metric not in df.columns:
            logger.error(f"Metric {metric} not found. Using accuracy if available.")
            metric = "accuracy_mean" if "accuracy_mean" in df.columns else df.columns[1]

        # 1. Normality
        logger.info("1. Normality Tests (Shapiro-Wilk)...")
        normality_df = self.check_normality(df, metric)
        if normality_df.empty:
            logger.error("Not enough data.")
            return

        is_normal_all = normality_df['is_normal'].all()
        normality_df.to_csv(os.path.join(self.output_dir, "normality_results.csv"), index=False)

        # 2. Multi-model comparison
        logger.info("2. Multi-model Comparison...")
        comparison_res = self.compare_models(df, metric, is_normal_all)
        with open(os.path.join(self.output_dir, "multimodel_comparison.json"), "w") as f:
            json.dump(comparison_res, f, indent=2)

        # 3. Post-hoc
        logger.info("3. Post-hoc Analysis...")
        posthoc_res = self.posthoc_analysis(df, metric, is_normal_all)
        if posthoc_res is not None:
            posthoc_res.to_csv(os.path.join(self.output_dir, "posthoc_results.csv"))

        # 4. Effect sizes (pairwise Cohen's d, Hedges' g)
        logger.info("4. Pairwise Effect Sizes (Cohen's d, Hedges' g)...")
        effect_df = self.compute_effect_sizes(df, metric)
        if not effect_df.empty:
            effect_df.to_csv(os.path.join(self.output_dir, "effect_sizes.csv"), index=False)

        # 5. Contribution analysis
        logger.info("5. Contribution Analysis (HSSL vs HSSL+DPBL)...")
        contrib_res = self.pairwise_contribution(df, metric)
        if contrib_res is not None:
            with open(os.path.join(self.output_dir, "contribution_analysis.json"), "w") as f:
                json.dump(contrib_res, f, indent=2)
            logger.info(f"HSSL vs HSSL+DPBL: improvement={contrib_res['improvement']:.4f}, "
                       f"Wilcoxon p={contrib_res['wilcoxon_p']:.4f}, "
                       f"Cohen's d={contrib_res['cohens_d']:.4f}")

        logger.info("Phase 8 done. Results in results/statistical_analysis/")


if __name__ == "__main__":
    validator = StatisticalValidator()
    validator.run()