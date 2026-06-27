#!/usr/bin/env python3
"""
Colab-Optimized Pipeline Runner

Runs the WESAD HSSL+DPBL pipeline with settings optimized for Colab usage.
- Reduced epochs for faster iteration
- Checkpoint saving for resumability
- Automatic config adaptation for Drive paths
"""

import os
import argparse
import sys
from src.logger import setup_logger

logger = setup_logger("ColabPipelineRunner")

def run_command(command, step_name):
    logger.info(f"--- Starting {step_name} ---")
    logger.info(f"Command: {command}")
    ret = os.system(command)
    if ret != 0:
        logger.error(f"Error executing {step_name}. Exit code: {ret}")
        sys.exit(1)
    logger.info(f"--- Completed {step_name} ---\n")

def main():
    parser = argparse.ArgumentParser(description="Run WESAD HSSL+DPBL Pipeline on Colab")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "data_prep", "server_ssl", "server_a", "server_b", "server_c", "server_d", "eval"],
                        help="Execution mode")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs for model training (reduced for Colab)")
    parser.add_argument("--ssl_epochs", type=int, default=30, help="Epochs for SSL pre-training (reduced for Colab)")
    parser.add_argument("--robust_iter", type=int, default=5, help="Iterations for robustness testing (reduced for Colab)")
    parser.add_argument("--skip-setup", action="store_true", help="Skip kaggle_setup.py")
    args = parser.parse_args()
    
    # Run setup (mount Drive, clone repo) is handled in the notebook; optional additional setup here
    if not args.skip_setup:
        logger.info("Running optional setup...")
        ret = os.system("python kaggle_setup.py --skip-install")
        if ret != 0:
            logger.warning("Setup had issues, continuing anyway...")
    
    logger.info("=" * 60)
    logger.info(f"COLAB PIPELINE STARTING (Mode: {args.mode})")
    logger.info(f"Epochs: {args.epochs}, SSL Epochs: {args.ssl_epochs}, Robust Iter: {args.robust_iter}")
    logger.info("=" * 60)
    
    if args.mode in ["all", "data_prep"]:
        run_command("python -m src.data_acquisition", "Data Acquisition")
        run_command("python -m src.data_preprocessing", "Data Preprocessing")
        run_command("python -m src.loso_preparation", "LOSO Fold Preparation")
        run_command("python -m src.generate_windows", "Window Generation")
    
    if args.mode in ["all", "server_ssl"]:
        run_command(f"python -m src.train_ssl --epochs {args.ssl_epochs}", "SSL Pre-training (SimSiam)")
        run_command("python -m src.generate_ssl_embeddings", "Generate SSL Embeddings")
    
    if args.mode in ["all", "server_a"]:
        run_command(f"python -m src.run_all_folds --epochs {args.epochs} --models rf,cnn --skip_existing", "Train Classifiers (RF, CNN)")
        run_command(f"python -m src.run_all_folds --epochs {args.epochs} --models ssl --skip_existing", "Train Classifiers (SSL)")
    
    if args.mode in ["all", "server_b"]:
        run_command(f"python -m src.train_hssl --epochs {args.epochs}", "HSSL Pre-training")
        run_command("python -m src.generate_embeddings", "Generate HSSL Embeddings")
        run_command(f"python -m src.run_all_folds --epochs {args.epochs} --models hssl --skip_existing", "Train Classifiers (HSSL)")
    
    if args.mode in ["all", "server_c"]:
        run_command(f"python -m src.train_dpbl --epochs {args.epochs}", "DPBL Training")
        run_command("python -m src.generate_dpbl_embeddings", "Generate DPBL Embeddings")
        run_command(f"python -m src.run_all_folds --epochs {args.epochs} --models ssl+dpbl,hssl+dpbl --skip_existing", "Train Classifiers (SSL+DPBL, HSSL+DPBL)")
    
    if args.mode in ["all", "server_d"]:
        run_command(f"python -m src.robustness_testing --n_iters {args.robust_iter} --epochs {args.epochs}", "Robustness Testing")
    
    if args.mode in ["all", "eval"]:
        run_command("python -m src.evaluate_models", "Evaluate Models")
        run_command("python -m src.compare_hssl_dpbl", "Compare HSSL vs HSSL+DPBL")
        run_command("python -m src.statistical_validation", "Statistical Validation")
        run_command("python -m src.reporting_visualization", "Reporting & Visualization")
        run_command("python -m src.generate_dashboard", "Dashboard Generation")
    
    logger.info("=" * 60)
    logger.info(f"COLAB PIPELINE (Mode: {args.mode}) COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info("Results in /content/drive/MyDrive/hssl_dpbl_results/")
    logger.info("Dashboard in /content/drive/MyDrive/hssl_dpbl_results/interactive_dashboard.html")

if __name__ == "__main__":
    main()