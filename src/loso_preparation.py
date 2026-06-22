import os
import json
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.logger import setup_logger
from src.experiment_tracker import ExperimentTracker

class LOSOPreparation:
    def __init__(self):
        self.logger = setup_logger("LOSOPreparation")
        self.tracker = ExperimentTracker()
        self.processed_dir = self.tracker.config["paths"]["processed_data"]
        self.normalized_dir = self.tracker.config["paths"]["normalized_data"]
        self.reports_dir = self.tracker.config["paths"]["reports"]
        self.checkpoints_dir = self.tracker.config["paths"]["checkpoints"]
        self.subjects = [f"S{i}" for i in range(2, 18) if i != 12]
        
    def generate_folds(self):
        self.logger.info("Generating LOSO folds...")
        os.makedirs(self.reports_dir, exist_ok=True)
        folds_path = os.path.join(self.reports_dir, "loso_folds.json")
        
        valid_subjects = []
        for subject in self.subjects:
            proc_path = os.path.join(self.processed_dir, f"{subject}_processed.pkl")
            if os.path.exists(proc_path):
                # Validate that the file is not corrupted
                try:
                    with open(proc_path, 'rb') as f:
                        data = pickle.load(f)
                        if 'features' in data and 'labels' in data:
                            valid_subjects.append(subject)
                        else:
                            self.logger.error(f"Invalid processed file structure for {subject}")
                except Exception as e:
                    self.logger.error(f"Failed to read processed file for {subject}: {e}")
                
        if not valid_subjects:
            self.logger.error("No valid processed data found. Cannot generate LOSO folds.")
            return None
            
        folds = {}
        for test_subject in valid_subjects:
            train_subjects = [s for s in valid_subjects if s != test_subject]
            
            # Validation: Ensure test subject is not in train subjects (Leakage check)
            if test_subject in train_subjects:
                self.logger.error(f"Data leakage detected! {test_subject} is in both train and test.")
                return None
            
            # FIT SCALER ONLY ON TRAIN SUBJECTS TO PREVENT LEAKAGE
            self.logger.info(f"Fitting scaler for fold {test_subject} (Test Subject: {test_subject})")
            train_features = []
            for t_subj in train_subjects:
                proc_path = os.path.join(self.processed_dir, f"{t_subj}_processed.pkl")
                with open(proc_path, 'rb') as f:
                    data = pickle.load(f)
                    train_features.append(data["features"])
            
            scaler = StandardScaler()
            scaler.fit(np.vstack(train_features))
            
            # Save scaler for this specific fold
            os.makedirs(os.path.join(self.checkpoints_dir, "scalers"), exist_ok=True)
            scaler_path = os.path.join(self.checkpoints_dir, "scalers", f"scaler_fold_{test_subject}.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            # Normalize and save BOTH Train and Test data for this fold
            fold_norm_dir = os.path.join(self.normalized_dir, f"fold_{test_subject}")
            os.makedirs(fold_norm_dir, exist_ok=True)
            
            # Transform All Valid Subjects with the Train-Fitted Scaler
            for subj in valid_subjects:
                proc_path = os.path.join(self.processed_dir, f"{subj}_processed.pkl")
                norm_path = os.path.join(fold_norm_dir, f"{subj}_normalized.pkl")
                
                with open(proc_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Check for Infinite / NaN before transform
                if not np.isfinite(data["features"]).all():
                     self.logger.warning(f"NaN/Inf found in {subj} before scaling. Replacing with 0.")
                     data["features"] = np.nan_to_num(data["features"])
                     
                data["features"] = scaler.transform(data["features"])
                
                # Verify Normalized
                if not np.isfinite(data["features"]).all():
                     self.logger.error(f"NaN/Inf found in {subj} AFTER scaling. Leakage or invalid data.")
                
                with open(norm_path, 'wb') as f:
                    pickle.dump(data, f)
                
            folds[test_subject] = {
                "train": train_subjects,
                "test": [test_subject],
                "scaler_path": scaler_path,
                "normalized_data_dir": fold_norm_dir
            }
            
        with open(folds_path, 'w') as f:
            json.dump(folds, f, indent=4)
            
        self.logger.info(f"Generated {len(folds)} folds successfully.")
        self.logger.info(f"Folds saved to {folds_path}")
        
        return folds

if __name__ == "__main__":
    loso = LOSOPreparation()
    loso.generate_folds()