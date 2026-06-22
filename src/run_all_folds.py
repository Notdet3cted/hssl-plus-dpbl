"""
Run training & evaluation for ALL LOSO folds.
Usage:
    python src/run_all_folds.py --epochs 30 --seed 42
"""
import os
import json
import argparse
from src.logger import setup_logger
from src.train_classifiers import ClassifierTrainer

logger = setup_logger("RunAllFolds")

def main():
    parser = argparse.ArgumentParser(description="Run all LOSO folds")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip fold if result files already exist")
    parser.add_argument("--models", type=str, default="rf,cnn,hssl,hssl+dpbl", 
                        help="Comma-separated list of models to train")
    args = parser.parse_args()

    trainer = ClassifierTrainer(seed=args.seed)
    fold_subjects = list(trainer.folds.keys())
    logger.info(f"Total folds: {len(fold_subjects)} | Subjects: {fold_subjects}")
    logger.info(f"Epochs: {args.epochs} | Seed: {args.seed}")

    results_dir = trainer.results_dir
    models_to_run = [m.strip().lower() for m in args.models.split(',')]
    logger.info(f"Models to run: {models_to_run}")

    for i, subj in enumerate(fold_subjects):
        logger.info(f"===== Fold {i+1}/{len(fold_subjects)}: Test Subject {subj} =====")

        if args.skip_existing:
            existing = [m for m in models_to_run if os.path.exists(os.path.join(results_dir, f"{m}_fold_{subj}.json"))]
            missing = [m for m in models_to_run if m not in existing]
            if not missing:
                logger.info(f"Skipping {subj} — all results already exist.")
                continue
            else:
                logger.info(f"Fold {subj}: existing={existing}, missing={missing}")
                models_to_run_this_fold = missing
        else:
            models_to_run_this_fold = models_to_run

        try:
            trainer.set_seed(args.seed)
            trainer.run_all(test_subject=subj, epochs=args.epochs, models_to_run=models_to_run_this_fold)
            logger.info(f"Fold {subj} completed successfully.")
        except Exception as e:
            logger.error(f"Fold {subj} FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue

    logger.info("All folds training completed.")

if __name__ == "__main__":
    main()