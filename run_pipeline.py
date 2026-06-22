import os
import argparse
import sys

from src.logger import setup_logger

logger = setup_logger("PipelineRunner")

def run_command(command, step_name):
    logger.info(f"--- Starting {step_name} ---")
    ret = os.system(command)
    if ret != 0:
        logger.error(f"Error executing {step_name}. Exiting.")
        sys.exit(1)
    logger.info(f"--- Completed {step_name} ---\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full WESAD Stress Detection Pipeline")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs for training models")
    parser.add_argument("--robust_iter", type=int, default=30, help="Iterations for robustness testing")
    parser.add_argument("--mode", type=str, default="all", 
                        choices=["all", "data_prep", "server_a", "server_b", "server_c", "server_d", "eval"],
                        help="Execution mode for distributed training across multiple servers")
    args = parser.parse_args()

    if args.mode in ["all", "data_prep"]:
        # Step 1: Data Acquisition
        run_command("python -m src.data_acquisition", "Data Acquisition")
        # Step 2: Preprocessing
        run_command("python -m src.data_preprocessing", "Data Preprocessing")
        # Step 3: LOSO Preparation
        run_command("python -m src.loso_preparation", "LOSO Fold Preparation")
        # Step 4: Window Generation
        run_command("python -m src.generate_windows", "Window Generation")

    if args.mode in ["all", "server_a"]:
        # RF and CNN training
        run_command(f"python -m src.run_all_folds --epochs {args.epochs} --models rf,cnn --skip_existing", "Train Classifiers (RF, CNN)")

    if args.mode in ["all", "server_b"]:
        # HSSL Pre-training, Embeddings, and Classifier
        run_command(f"python -m src.train_hssl --epochs {args.epochs}", "HSSL Pre-training")
        run_command("python -m src.generate_embeddings", "Generate HSSL Embeddings")
        run_command(f"python -m src.run_all_folds --epochs {args.epochs} --models hssl --skip_existing", "Train Classifiers (HSSL)")

    if args.mode in ["all", "server_c"]:
        # DPBL Training, Embeddings, and Classifier
        # Note: server_c requires HSSL to be pre-trained. If running distributed, ensure 'checkpoints/' from server_b is copied here first.
        run_command(f"python -m src.train_dpbl --epochs {args.epochs}", "DPBL Training")
        run_command("python -m src.generate_dpbl_embeddings", "Generate DPBL Embeddings")
        run_command(f"python -m src.run_all_folds --epochs {args.epochs} --models hssl+dpbl --skip_existing", "Train Classifiers (HSSL+DPBL)")

    if args.mode in ["all", "server_d"]:
        # Robustness Testing
        # Note: server_d requires 'embeddings/' from server_c to be copied here first.
        run_command(f"python -m src.robustness_testing --n_iter {args.robust_iter} --epochs {args.epochs}", "Robustness Testing")

    if args.mode in ["all", "eval"]:
        # Final Evaluation and Reporting
        # Note: If distributed, ensure all 'results/' and 'checkpoints/' from servers A, B, C, D are gathered here first.
        run_command("python -m src.evaluate_models", "Evaluate Models")
        run_command("python -m src.statistical_validation", "Statistical Validation")
        run_command("python -m src.reporting_visualization", "Reporting & Visualization")
        run_command("python -m src.generate_dashboard", "Dashboard Generation")

    logger.info("==================================================")
    logger.info(f"PIPELINE (Mode: {args.mode}) COMPLETED SUCCESSFULLY!")
    if args.mode in ["all", "eval"]:
        logger.info("Please open results/interactive_dashboard.html to view the final results.")
    logger.info("==================================================")
