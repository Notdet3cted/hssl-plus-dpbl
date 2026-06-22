"""
FASE 9 - Reporting dan Visualisasi
Menghasilkan artefak akhir penelitian.

Tasks:
- Tables: Tabel performa seluruh model, robustness, hasil statistik
- Visualization: ROC Curve, PR Curve, Confusion Matrix, Boxplot perbandingan model
- Final Report: Generate laporan otomatis
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.logger import setup_logger

logger = setup_logger("ReportingVisualization")

class Reporter:
    def __init__(self, results_dir="results", stat_dir="results/statistical_analysis"):
        self.results_dir = results_dir
        self.stat_dir = stat_dir
        
        self.figures_dir = os.path.join(self.results_dir, "figures")
        self.tables_dir = os.path.join(self.results_dir, "final_tables")
        self.report_dir = os.path.join(self.results_dir, "final_report")
        
        for d in [self.figures_dir, self.tables_dir, self.report_dir]:
            os.makedirs(d, exist_ok=True)
            
    def generate_tables(self):
        logger.info("Generating Final Tables...")
        # 1. Performance Table
        perf_path = os.path.join(self.results_dir, "model_summary.csv")
        if os.path.exists(perf_path):
            df = pd.read_csv(perf_path)
            # Filter and reformat for final table
            cols = ['model'] + [c for c in df.columns if '_mean' in c or '_std' in c]
            final_perf = df[cols].copy()
            final_perf.to_csv(os.path.join(self.tables_dir, "final_performance.csv"), index=False)
            logger.info("Generated final_performance.csv")
            
        # 2. Robustness Table
        rob_path = os.path.join(self.results_dir, "robustness", "robustness_results.csv")
        if os.path.exists(rob_path):
            df = pd.read_csv(rob_path)
            final_rob = df.copy()
            final_rob.to_csv(os.path.join(self.tables_dir, "final_robustness.csv"), index=False)
            logger.info("Generated final_robustness.csv")
            
        # 3. Statistical Tables
        norm_path = os.path.join(self.stat_dir, "normality_results.csv")
        if os.path.exists(norm_path):
            df = pd.read_csv(norm_path)
            df.to_csv(os.path.join(self.tables_dir, "final_normality.csv"), index=False)
            
        contrib_path = os.path.join(self.stat_dir, "contribution_analysis.json")
        if os.path.exists(contrib_path):
            with open(contrib_path, "r") as f:
                contrib = json.load(f)
            # Convert single json to simple DF
            df = pd.DataFrame([contrib])
            df.to_csv(os.path.join(self.tables_dir, "final_contribution.csv"), index=False)
            
    def plot_boxplots(self):
        logger.info("Generating Boxplots...")
        robustness_dir = os.path.join(self.results_dir, "robustness")
        data = []
        if os.path.exists(robustness_dir):
            for root, dirs, files in os.walk(robustness_dir):
                for d in dirs:
                    if d.startswith("iter_"):
                        summary_path = os.path.join(root, d, "model_summary.csv")
                        if os.path.exists(summary_path):
                            df = pd.read_csv(summary_path)
                            data.append(df)
                            
        if data:
            combined_df = pd.concat(data, ignore_index=True)
            if 'f1_score_mean' in combined_df.columns:
                plt.figure(figsize=(10, 6))
                sns.boxplot(x='model', y='f1_score_mean', data=combined_df)
                plt.title("Model F1-Score Comparison across Robustness Iterations")
                plt.ylabel("F1-Score")
                plt.xlabel("Model")
                plt.savefig(os.path.join(self.figures_dir, "boxplot_f1_score.png"), dpi=300, bbox_inches="tight")
                plt.close()
                logger.info("Saved boxplot_f1_score.png")
            else:
                logger.warning("No f1_score_mean found for boxplot.")
        else:
            logger.warning("No iteration data found for boxplots.")
            
    def plot_confusion_matrices(self):
        logger.info("Generating Confusion Matrix Plots...")
        cm_dir = os.path.join(self.results_dir, "confusion_matrices")
        if not os.path.exists(cm_dir):
            return
            
        for f in os.listdir(cm_dir):
            if f.endswith(".json"):
                model = f.split("_")[0]
                with open(os.path.join(cm_dir, f), "r") as json_f:
                    cm = np.array(json.load(json_f))
                    
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f"Confusion Matrix: {model}")
                plt.ylabel("True Label")
                plt.xlabel("Predicted Label")
                plt.savefig(os.path.join(self.figures_dir, f"cm_{model}.png"), dpi=300, bbox_inches="tight")
                plt.close()
                logger.info(f"Saved cm_{model}.png")
                
    def generate_report(self):
        logger.info("Generating Final Report (Markdown)...")
        report_path = os.path.join(self.report_dir, "final_report.md")
        
        with open(report_path, "w") as f:
            f.write("# Personalized Stress Detection using HSSL + DPBL\n\n")
            f.write("## Final Research Report\n\n")
            
            f.write("### 1. Performance Summary\n")
            f.write("A summary of the model evaluation across all LOSO folds.\n\n")
            perf_path = os.path.join(self.tables_dir, "final_performance.csv")
            if os.path.exists(perf_path):
                df = pd.read_csv(perf_path)
                f.write(df.to_markdown(index=False) + "\n\n")
                
            f.write("### 2. Robustness Summary\n")
            f.write("Model stability over multiple experimental iterations.\n\n")
            rob_path = os.path.join(self.tables_dir, "final_robustness.csv")
            if os.path.exists(rob_path):
                df = pd.read_csv(rob_path)
                f.write(df.to_markdown(index=False) + "\n\n")
                
            f.write("### 3. Statistical Contribution Analysis\n")
            f.write("Evaluating the significance of DPBL personalization over base HSSL.\n\n")
            contrib_path = os.path.join(self.tables_dir, "final_contribution.csv")
            if os.path.exists(contrib_path):
                df = pd.read_csv(contrib_path)
                f.write(df.to_markdown(index=False) + "\n\n")
                
            f.write("### 4. Figures\n")
            f.write("Boxplots and Confusion Matrices are available in the `figures/` directory.\n")
            
        logger.info(f"Saved final report -> {report_path}")

    def run(self):
        logger.info("=" * 60)
        logger.info("PHASE 9: REPORTING AND VISUALIZATION")
        logger.info("=" * 60)
        
        self.generate_tables()
        self.plot_boxplots()
        self.plot_confusion_matrices()
        self.generate_report()
        
        logger.info("Phase 9 completed. All artifacts generated in results/.")

if __name__ == "__main__":
    reporter = Reporter()
    reporter.run()