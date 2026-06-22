import json
import os
import yaml
from datetime import datetime
from src.logger import setup_logger

class ExperimentTracker:
    def __init__(self, config_path="config/config.yaml"):
        self.logger = setup_logger("ExperimentTracker")
        self.config_path = config_path
        self.config = self._load_config()
        self.experiment_dir = self._setup_experiment_dir()
        self._save_metadata()

    def _load_config(self):
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return {}

    def _setup_experiment_dir(self):
        base_dir = self.config.get("paths", {}).get("experiments", "experiments/")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = os.path.join(base_dir, f"exp_{timestamp}")
        os.makedirs(exp_dir, exist_ok=True)
        self.logger.info(f"Experiment directory created at {exp_dir}")
        return exp_dir

    def _save_metadata(self):
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "version": "1.0.0" # Hardcoded for now
        }
        meta_path = os.path.join(self.experiment_dir, "metadata.json")
        try:
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=4)
            self.logger.info("Metadata saved.")
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")

if __name__ == "__main__":
    tracker = ExperimentTracker()