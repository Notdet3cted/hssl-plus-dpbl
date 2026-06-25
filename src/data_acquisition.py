import os
import subprocess
from src.logger import setup_logger
from src.experiment_tracker import ExperimentTracker

class DataAcquisition:
    def __init__(self):
        self.logger = setup_logger("DataAcquisition")
        self.tracker = ExperimentTracker()
        self.raw_dir = self.tracker.config["paths"]["raw_data"]
        self.extract_path = os.path.join(self.raw_dir, "WESAD")
        self.dataset_name = "orvile/wesad-wearable-stress-affect-detection-dataset"

        # Auto-detect Kaggle environment
        self.is_kaggle = os.path.exists('/kaggle/input/wesad-wearable-stress-affect-detection-dataset')
        if self.is_kaggle:
            self.logger.info("Kaggle environment detected — dataset already mounted as input.")
            self.raw_dir = "/kaggle/input/wesad-wearable-stress-affect-detection-dataset/"
            self.extract_path = self.raw_dir

    def download_dataset(self):
        # Skip jika di Kaggle (dataset sudah di-mount)
        if self.is_kaggle:
            self.logger.info("Skipping download — WESAD dataset is mounted as Kaggle input.")
            return

        # Skip jika sudah ada
        if os.path.exists(self.extract_path) and len(os.listdir(self.extract_path)) > 0:
            self.logger.info("Dataset already exists. Skipping download.")
            return

        self.logger.info(f"Downloading dataset {self.dataset_name} using Kaggle API...")
        try:
            subprocess.run([
                "kaggle", "datasets", "download", "-d", self.dataset_name,
                "-p", self.raw_dir, "--unzip"
            ], check=True)
            self.logger.info("Download and extraction completed.")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to download dataset via Kaggle API: {e}")
            self.logger.error("Please ensure you have Kaggle API credentials configured (kaggle.json).")

    def validate_structure(self):
        subjects = [f"S{i}" for i in range(2, 18) if i != 12]  # S12 doesn't exist in WESAD
        valid = True
        for subject in subjects:
            subj_path = os.path.join(self.extract_path, subject)
            if not os.path.exists(subj_path):
                self.logger.error(f"Missing subject folder: {subject}")
                valid = False

        if valid:
            self.logger.info(f"Structure validated. Found {len(subjects)} subjects.")
        return valid

if __name__ == "__main__":
    acq = DataAcquisition()
    acq.download_dataset()
    acq.validate_structure()