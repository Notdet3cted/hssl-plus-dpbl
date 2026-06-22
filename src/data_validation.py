import os
import pickle
import json
import numpy as np
from src.logger import setup_logger
from src.experiment_tracker import ExperimentTracker

class DataValidation:
    def __init__(self):
        self.logger = setup_logger("DataValidation")
        self.tracker = ExperimentTracker()
        self.raw_dir = self.tracker.config["paths"]["raw_data"]
        self.dataset_path = os.path.join(self.raw_dir, "WESAD")
        self.subjects = [f"S{i}" for i in range(2, 18) if i != 12]

    def validate_dataset(self):
        self.logger.info("Starting dataset validation...")
        validation_results = {}
        
        if not os.path.exists(self.dataset_path):
            self.logger.error("WESAD directory not found. Please ensure the dataset is extracted.")
            return False

        for subject in self.subjects:
            subj_path = os.path.join(self.dataset_path, subject, f"{subject}.pkl")
            if not os.path.exists(subj_path):
                self.logger.error(f"Missing file for {subject}: {subj_path}")
                validation_results[subject] = {"status": "missing"}
                continue

            try:
                with open(subj_path, 'rb') as f:
                    data = pickle.load(f, encoding='latin1')
                
                # Check required keys
                if 'signal' not in data or 'label' not in data:
                    self.logger.error(f"Missing 'signal' or 'label' in {subject}")
                    validation_results[subject] = {"status": "corrupt"}
                    continue
                
                label_dist = np.unique(data['label'], return_counts=True)
                dist_dict = {int(k): int(v) for k, v in zip(label_dist[0], label_dist[1])}
                
                validation_results[subject] = {
                    "status": "valid",
                    "label_distribution": dist_dict
                }
                self.logger.info(f"{subject} validated successfully.")
            except Exception as e:
                self.logger.error(f"Failed to read {subject}: {e}")
                validation_results[subject] = {"status": "error", "message": str(e)}

        report_path = os.path.join(self.tracker.config["paths"]["reports"], "dataset_validation.json")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(validation_results, f, indent=4)
        
        self.logger.info(f"Validation report saved to {report_path}")
        return validation_results

if __name__ == "__main__":
    validator = DataValidation()
    validator.validate_dataset()