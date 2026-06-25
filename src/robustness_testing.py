"""
FASE 7 - Robustness Testing

Jalankan eksperimen berulang dengan random seed berbeda untuk memastikan kestabilan model.
Di environment ini, pengguna meminta untuk menyiapkan kodenya dengan benar tetapi tidak perlu 
hardcode/paksa 30 iterasi eksperimen penuh saat dicoba (agar tidak terlalu berat). 
Namun, skrip siap digunakan untuk N iterasi.

Usage:
    python -m src.robustness_testing --n_iters 3 --epochs 10
"""

import os
import json
import csv
import argparse
import numpy as np
import traceback
from datetime import datetime

from src.logger import setup_logger
from src.train_classifiers import ClassifierTrainer
from src.evaluate_models import ModelEvaluator

logger = setup_logger("RobustnessTesting")

# Model yang akan diuji
MODELS = ["hssl+dpbl"] # Per requirements, only robust test HSSL+DPBL
METRICS = ["accuracy", "precision", "recall", "f1_score", "weighted_precision", "weighted_recall", "weighted_f1", "roc_auc", "pr_auc"]

class RobustnessTester:
    def __init__(self, base_seed=42, results_dir="results/robustness"):
        self.base_seed = base_seed
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Base trainer to get folds info
        self.trainer = ClassifierTrainer(seed=self.base_seed)
        self.subjects = list(self.trainer.folds.keys())

    def run_iteration(self, iter_idx, seed, epochs=30, skip_existing=True):
        """Menjalankan satu iterasi full LOSO training + evaluation dengan seed tertentu."""
        logger.info(f"\n{'='*50}\nSTARTING ITERATION {iter_idx} (Seed: {seed})\n{'='*50}")
        
        # Buat sub-folder khusus untuk iterasi ini agar hasil tidak tertimpa
        iter_results_dir = os.path.join(self.results_dir, f"iter_{iter_idx}_seed_{seed}")
        os.makedirs(iter_results_dir, exist_ok=True)
        
        iter_predictions_dir = os.path.join(iter_results_dir, "predictions")
        os.makedirs(iter_predictions_dir, exist_ok=True)
        
        iter_checkpoints_dir = os.path.join(iter_results_dir, "checkpoints")
        os.makedirs(iter_checkpoints_dir, exist_ok=True)
        
        # Inisialisasi trainer dengan custom directory untuk iterasi ini
        trainer = ClassifierTrainer(seed=seed)
        trainer.results_dir = iter_results_dir
        trainer.predictions_dir = iter_predictions_dir
        trainer.checkpoints_dir = iter_checkpoints_dir
        
        # Training loop untuk semua fold di iterasi ini
        for i, subj in enumerate(self.subjects):
            logger.info(f"[Iter {iter_idx} | Seed {seed}] Fold {i+1}/{len(self.subjects)}: Subject {subj}")
            
            if skip_existing:
                existing = [m for m in MODELS if os.path.exists(os.path.join(iter_results_dir, f"{m}_fold_{subj}.json"))]
                missing = [m for m in MODELS if m not in existing]
                if not missing:
                    logger.info(f"Skipping {subj} — all results exist for iter {iter_idx}.")
                    continue
            
            try:
                trainer.set_seed(seed)
                trainer.train_hssl_dpbl(test_subject=subj, epochs=epochs)
            except Exception as e:
                logger.error(f"[Iter {iter_idx}] Fold {subj} FAILED: {e}")
                traceback.print_exc()
                continue
                
        # Evaluasi untuk iterasi ini
        logger.info(f"Evaluating Iteration {iter_idx}...")
        evaluator = ModelEvaluator(results_dir=iter_results_dir)
        # Hack to only evaluate HSSL+DPBL since we patched Models globally in robustness_testing but ModelEvaluator has its own MODELS
        import src.evaluate_models as em
        original_models = em.MODELS
        em.MODELS = ["hssl+dpbl"]
        try:
            evaluator.run(from_predictions=False)
        finally:
            em.MODELS = original_models
        
        # Baca model_summary.csv dari iterasi ini untuk dikembalikan
        summary_path = os.path.join(iter_results_dir, "model_summary.csv")
        iter_summary = []
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Parse numerical values
                    parsed_row = {"model": row["model"]}
                    for k, v in row.items():
                        if k != "model" and v != "N/A" and v != "":
                            try:
                                parsed_row[k] = float(v)
                            except ValueError:
                                parsed_row[k] = None
                    iter_summary.append(parsed_row)
                    
        return iter_summary

    def compile_robustness_results(self, all_iters_results):
        """Mengkalkulasi mean, std, min, max dari seluruh iterasi per model."""
        compiled_results = []
        
        for model in MODELS:
            # Ambil hasil summary metrics untuk model ini dari semua iterasi
            model_metrics_across_iters = {m: [] for m in METRICS}
            
            for iter_result in all_iters_results:
                # Cari dict yang sesuai dengan model
                model_data = next((item for item in iter_result if item["model"] == model), None)
                if model_data:
                    for m in METRICS:
                        mean_key = f"{m}_mean"
                        if mean_key in model_data and model_data[mean_key] is not None:
                            model_metrics_across_iters[m].append(model_data[mean_key])
            
            # Hitung statistik robustness
            model_robustness = {"model": model}
            
            # Save raw iterations for CSV
            for i, it_res in enumerate(all_iters_results):
                model_data = next((item for item in it_res if item["model"] == model), None)
                if model_data:
                    for m in METRICS:
                        mean_key = f"{m}_mean"
                        if mean_key in model_data and model_data[mean_key] is not None:
                            model_robustness[f"{m}_iter_{i+1}"] = model_data[mean_key]
                        else:
                            model_robustness[f"{m}_iter_{i+1}"] = None

            for m in METRICS:
                vals = model_metrics_across_iters[m]
                if vals:
                    arr = np.array(vals)
                    mean_val = float(np.mean(arr))
                    std_val = float(np.std(arr, ddof=1)) if len(vals) > 1 else 0.0
                    n = len(vals)
                    
                    # CI 95% = mean +/- 1.96 * (std / sqrt(n))
                    ci_95 = 1.96 * (std_val / np.sqrt(n)) if n > 0 else 0.0

                    model_robustness[f"{m}_robust_mean"] = mean_val
                    model_robustness[f"{m}_robust_std"] = std_val
                    model_robustness[f"{m}_robust_min"] = float(np.min(arr))
                    model_robustness[f"{m}_robust_max"] = float(np.max(arr))
                    model_robustness[f"{m}_robust_ci95"] = float(ci_95)
                else:
                    for sfx in ["_robust_mean", "_robust_std", "_robust_min", "_robust_max", "_robust_ci95"]:
                        model_robustness[f"{m}{sfx}"] = None
                        
            compiled_results.append(model_robustness)
            
        return compiled_results

    def export_robustness_results(self, compiled_results, n_iters):
        """Menyimpan hasil agregasi robustness ke CSV."""
        out_path = os.path.join(self.results_dir, "robustness_results.csv")
        
        if not compiled_results:
            logger.warning("No robustness results to export.")
            return None
            
        # Build fieldnames
        fieldnames = ["model"]
        for m in METRICS:
            for i in range(1, n_iters + 1):
                fieldnames.append(f"{m}_iter_{i}")
            fieldnames.extend([f"{m}_robust_mean", f"{m}_robust_std", f"{m}_robust_min", f"{m}_robust_max", f"{m}_robust_ci95"])
            
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in compiled_results:
                # Format to 4 decimal places
                formatted_row = {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()}
                writer.writerow(formatted_row)
                
        logger.info(f"Saved robustness results -> {out_path}")
        
        # Print summary
        logger.info("=" * 80)
        logger.info("ROBUSTNESS TESTING SUMMARY (Mean ± Std)")
        logger.info("-" * 80)
        for r in compiled_results:
            f1_mean = r.get("f1_score_robust_mean")
            f1_std = r.get("f1_score_robust_std")
            
            if f1_mean is not None and f1_std is not None:
                logger.info(f"{r['model']:<12}: F1 = {f1_mean:.4f} ± {f1_std:.4f}")
            else:
                logger.info(f"{r['model']:<12}: F1 = N/A")
        logger.info("=" * 80)
        
        return out_path
        
    def run(self, n_iters=3, epochs=30, skip_existing=True):
        """Jalankan keseluruhan pipeline robustness testing."""
        logger.info(f"Starting Robustness Testing: {n_iters} iterations.")
        
        # Generate N random seeds (deterministic based on base_seed)
        rng = np.random.RandomState(self.base_seed)
        seeds = rng.randint(1000, 9999, size=n_iters).tolist()
        logger.info(f"Generated seeds for iterations: {seeds}")
        
        all_iters_results = []
        
        # 1. Jalankan eksperimen untuk tiap seed
        for idx, seed in enumerate(seeds):
            iter_idx = idx + 1
            iter_summary = self.run_iteration(
                iter_idx=iter_idx, 
                seed=seed, 
                epochs=epochs, 
                skip_existing=skip_existing
            )
            all_iters_results.append(iter_summary)
            
        # 2. Kalkulasi statistik antar iterasi
        logger.info("Compiling robustness statistics across all iterations...")
        compiled_results = self.compile_robustness_results(all_iters_results)
        
        # 3. Export hasil
        out_path = self.export_robustness_results(compiled_results, n_iters)
        
        # 4. Save metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "n_iters": n_iters,
            "base_seed": self.base_seed,
            "seeds_used": seeds,
            "models": MODELS,
            "epochs_per_iter": epochs
        }
        with open(os.path.join(self.results_dir, "robustness_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
            
        logger.info("Phase 7 Robustness Testing complete.")
        return out_path


def main():
    parser = argparse.ArgumentParser(description="Phase 7: Robustness Testing")
    parser.add_argument("--n_iters", type=int, default=30, help="Number of experimental iterations (default 30 for full robustness)")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per training")
    parser.add_argument("--base_seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--skip_existing", action="store_true", help="Skip if results already exist")
    parser.add_argument("--test_subject", type=str, default=None, help="Specific subject to test (e.g., S2) to save time")
    args = parser.parse_args()

    tester = RobustnessTester(base_seed=args.base_seed)
    if args.test_subject:
        tester.subjects = [args.test_subject]
        
    tester.run(n_iters=args.n_iters, epochs=args.epochs, skip_existing=args.skip_existing)

if __name__ == "__main__":
    main()