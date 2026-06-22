"""
FASE 8 - Validasi Statistik
Membuktikan signifikansi hasil penelitian menggunakan uji statistik.

Tasks:
- Normality Test (Shapiro-Wilk)
- Multi-model Comparison (ANOVA / Friedman Test)
- Post-hoc Analysis (Tukey HSD / Dunn's Test)
- Contribution Analysis (Paired t-Test: HSSL vs HSSL+DPBL)
- Non-parametric Validation (Wilcoxon Signed-Rank: HSSL vs HSSL+DPBL)
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

class StatisticalValidator:
    def __init__(self, robustness_dir="results/robustness", reports_dir="reports"):
        self.robustness_dir = robustness_dir
        self.reports_dir = reports_dir
        self.output_dir = "results/statistical_analysis"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # We will use robustness results for statistical validation
        # Specifically, we need the raw iteration data to perform tests.
        # So we look for iter_*/model_summary.csv or iter_*/results
        self.iterations_data = self._load_iterations_data()
        
    def _load_iterations_data(self):
        data = []
        if not os.path.exists(self.robustness_dir):
            logger.warning(f"Robustness directory {self.robustness_dir} not found. Cannot perform statistical tests.")
            return pd.DataFrame()
            
        # Find all iter folders
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
            logger.warning("No iteration data found. Please run robustness_testing.py first.")
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
            logger.info(f"[{model}] Shapiro-Wilk p-value: {p_value:.4f} -> {'Normal' if is_normal else 'Not Normal'}")
            
        return pd.DataFrame(normality_results)

    def compare_models(self, df, metric="f1_score_mean", is_normal_all=True):
        """Multi-model comparison using ANOVA or Friedman."""
        models = df['model'].unique()
        data_groups = [df[df['model'] == m][metric].dropna().values for m in models]
        
        # Ensure all groups have the same length for Friedman or general comparison
        min_len = min([len(g) for g in data_groups])
        data_groups = [g[:min_len] for g in data_groups]
        
        results = {}
        if is_normal_all and len(models) > 2:
            # ANOVA
            stat, p_value = stats.f_oneway(*data_groups)
            results['test'] = 'ANOVA'
            results['statistic'] = stat
            results['p_value'] = p_value
            logger.info(f"ANOVA p-value: {p_value:.4f}")
        elif len(models) > 2:
            # Friedman Test
            stat, p_value = stats.friedmanchisquare(*data_groups)
            results['test'] = 'Friedman'
            results['statistic'] = stat
            results['p_value'] = p_value
            logger.info(f"Friedman Test p-value: {p_value:.4f}")
        else:
            results['test'] = 'N/A'
            results['p_value'] = 1.0
            
        return results

    def posthoc_analysis(self, df, metric="f1_score_mean", is_normal_all=True):
        """Tukey HSD or Dunn's Test."""
        if not HAS_STATSMODELS:
            logger.warning("statsmodels/scikit_posthocs not installed. Skipping post-hoc.")
            return None
            
        if is_normal_all:
            # Tukey HSD
            tukey = pairwise_tukeyhsd(endog=df[metric], groups=df['model'], alpha=0.05)
            # Create a simple representation
            res_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
            res_df['test'] = 'Tukey HSD'
            return res_df
        else:
            # Dunn's test using scikit-posthocs
            dunn = sp.posthoc_dunn(df, val_col=metric, group_col='model', p_adjust='bonferroni')
            dunn['test'] = 'Dunn'
            return dunn

    def pairwise_contribution(self, df, metric="f1_score_mean"):
        """Paired t-Test and Wilcoxon Signed-Rank between HSSL and HSSL+DPBL."""
        hssl = df[df['model'] == 'hssl'].sort_values('iteration')[metric].values
        hssl_dpbl = df[df['model'] == 'hssl+dpbl'].sort_values('iteration')[metric].values
        
        min_len = min(len(hssl), len(hssl_dpbl))
        if min_len < 3:
            return None
            
        hssl = hssl[:min_len]
        hssl_dpbl = hssl_dpbl[:min_len]
        
        # Paired t-test
        t_stat, t_p = stats.ttest_rel(hssl, hssl_dpbl)
        
        # Wilcoxon
        w_stat, w_p = stats.wilcoxon(hssl, hssl_dpbl)
        
        return {
            "comparison": "HSSL vs HSSL+DPBL",
            "paired_ttest_stat": t_stat,
            "paired_ttest_p": t_p,
            "wilcoxon_stat": w_stat,
            "wilcoxon_p": w_p,
            "hssl_mean": hssl.mean(),
            "hssl_dpbl_mean": hssl_dpbl.mean(),
            "improvement": hssl_dpbl.mean() - hssl.mean()
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
            logger.error(f"Metric {metric} not found in data. Using accuracy if available.")
            metric = "accuracy_mean" if "accuracy_mean" in df.columns else df.columns[1]
            
        # 1. Normality
        logger.info("1. Running Normality Tests (Shapiro-Wilk)...")
        normality_df = self.check_normality(df, metric)
        if normality_df.empty:
            logger.error("Not enough data for normality test.")
            return
            
        is_normal_all = normality_df['is_normal'].all()
        normality_df.to_csv(os.path.join(self.output_dir, "normality_results.csv"), index=False)
        
        # 2. Multi-model comparison
        logger.info("2. Running Multi-model Comparison...")
        comparison_res = self.compare_models(df, metric, is_normal_all)
        with open(os.path.join(self.output_dir, "multimodel_comparison.json"), "w") as f:
            json.dump(comparison_res, f, indent=2)
            
        # 3. Post-hoc
        logger.info("3. Running Post-hoc Analysis...")
        posthoc_res = self.posthoc_analysis(df, metric, is_normal_all)
        if posthoc_res is not None:
            posthoc_res.to_csv(os.path.join(self.output_dir, "posthoc_results.csv"))
            
        # 4. Contribution
        logger.info("4. Running Contribution Analysis (HSSL vs HSSL+DPBL)...")
        contrib_res = self.pairwise_contribution(df, metric)
        if contrib_res is not None:
            with open(os.path.join(self.output_dir, "contribution_analysis.json"), "w") as f:
                json.dump(contrib_res, f, indent=2)
                
            logger.info(f"HSSL vs HSSL+DPBL Improvement: {contrib_res['improvement']:.4f}")
            logger.info(f"Wilcoxon p-value: {contrib_res['wilcoxon_p']:.4f}")
            
        logger.info("Phase 8 completed. Results saved to results/statistical_analysis/")

if __name__ == "__main__":
    validator = StatisticalValidator()
    validator.run()