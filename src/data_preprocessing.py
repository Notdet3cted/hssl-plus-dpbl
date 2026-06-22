import os
import pickle
import numpy as np
import pandas as pd
from src.logger import setup_logger
from src.experiment_tracker import ExperimentTracker

class DataPreprocessing:
    def __init__(self):
        self.logger = setup_logger("DataPreprocessing")
        self.tracker = ExperimentTracker()
        self.raw_dir = os.path.join(self.tracker.config["paths"]["raw_data"], "WESAD")
        self.processed_dir = self.tracker.config["paths"]["processed_data"]
        self.subjects = [f"S{i}" for i in range(2, 18) if i != 12]
        self.window_size = self.tracker.config.get("preprocessing", {}).get("window_size", 60)
        self.overlap = self.tracker.config.get("preprocessing", {}).get("overlap", 0.5)

    def process_all(self):
        self.logger.info("Starting data preprocessing...")
        for subject in self.subjects:
            self._process_subject(subject)
        self.logger.info("Preprocessing completed.")

    def _process_subject(self, subject):
        subj_path = os.path.join(self.raw_dir, subject, f"{subject}.pkl")
        out_path = os.path.join(self.processed_dir, f"{subject}_processed.pkl")

        if os.path.exists(out_path):
            self.logger.info(f"Processed data for {subject} already exists. Skipping.")
            return

        if not os.path.exists(subj_path):
            self.logger.error(f"Raw data for {subject} not found. Skipping.")
            return

        self.logger.info(f"Processing {subject}...")
        try:
            with open(subj_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')

            # Extracting real data. Using chest EDA as a basic feature for pipeline validation
            chest_data = data['signal']['chest']
            eda = chest_data['EDA']
            labels = data['label']
            
            # WESAD labels: 1=baseline, 2=stress, 3=amusement
            mask = np.isin(labels, [1, 2, 3])
            filtered_labels = labels[mask]
            binary_labels = np.where(filtered_labels == 2, 1, 0)
            
            features = eda[mask]
            
            processed_data = {
                "features": features,
                "labels": binary_labels
            }
            
            with open(out_path, 'wb') as f:
                pickle.dump(processed_data, f)
            
            self.logger.info(f"Saved processed data for {subject}.")
        except Exception as e:
            self.logger.error(f"Error processing {subject}: {e}")

if __name__ == "__main__":
    processor = DataPreprocessing()
    processor.process_all()