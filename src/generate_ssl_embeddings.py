import os
import json
import pickle
import torch
from torch.utils.data import DataLoader
from src.logger import setup_logger
from src.experiment_tracker import ExperimentTracker
from src.models.ssl.simsiam import SSLEncoder
from src.hssl_dataset import WindowLabelDataset

class SSLEmbeddingGenerator:
    def __init__(self):
        self.logger = setup_logger("SSLEmbeddingGenerator")
        self.tracker = ExperimentTracker()
        self.checkpoints_dir = self.tracker.config["paths"].get("checkpoints", "checkpoints")
        self.reports_dir = self.tracker.config["paths"].get("reports", "reports")
        self.embeddings_dir = self.tracker.config["paths"].get("embeddings_ssl", "embeddings/ssl/")
        os.makedirs(self.embeddings_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate_fold(self, test_subject, window_size=None):
        if window_size is None:
            window_size = self.tracker.config.get("preprocessing", {}).get("window_size", 700)
        self.logger.info(f"Generating SSL embeddings for Fold: {test_subject}")
        ssl_ckpt_dir = os.path.join(self.checkpoints_dir, f"ssl_fold_{test_subject}")
        best_path = os.path.join(ssl_ckpt_dir, "best.pt")
        
        folds_path = os.path.join(self.reports_dir, "loso_folds.json")
        with open(folds_path, 'r') as f:
            folds = json.load(f)
            
        if test_subject not in folds:
            raise ValueError(f"Fold for {test_subject} not found.")
            
        norm_dir = folds[test_subject]["normalized_data_dir"]
        subjects = folds[test_subject]["train"] + folds[test_subject]["test"]
        
        # Skip-check: if ALL subjects already have embeddings, skip
        fold_emb_dir = os.path.join(self.embeddings_dir, f"fold_{test_subject}")
        all_exist = all(os.path.exists(os.path.join(fold_emb_dir, f"{s}_embeddings.pkl")) for s in subjects)
        if all_exist:
            self.logger.info(f"All SSL embeddings for fold {test_subject} already exist. Skipping.")
            return

        sample_path = os.path.join(norm_dir, f"{subjects[0]}_normalized.pkl")
        with open(sample_path, 'rb') as f:
            sample_data = pickle.load(f)
        feat = sample_data["features"]
        channels = feat.shape[1] if feat.ndim > 1 else 1
        
        self.model = SSLEncoder(input_channels=channels).to(self.device)
        if os.path.exists(best_path):
            self.model.load_state_dict(torch.load(best_path, map_location=self.device))
            self.model.eval()
            self.logger.info(f"Loaded best SSL checkpoint from {best_path}.")
        else:
            self.logger.error(f"No checkpoint at {best_path}. Run SSL pre-training first.")
            return
            
        os.makedirs(fold_emb_dir, exist_ok=True)
        
        for subj in subjects:
            out_path = os.path.join(fold_emb_dir, f"{subj}_embeddings.pkl")
            if os.path.exists(out_path):
                self.logger.info(f"{subj} embeddings already exist. Skipping.")
                continue

            path = os.path.join(norm_dir, f"{subj}_normalized.pkl")
            with open(path, 'rb') as f:
                data = pickle.load(f)
                
            dataset = WindowLabelDataset(data["features"], data.get("labels"), window_size=window_size)
            if len(dataset) == 0:
                self.logger.warning(f"No windows for {subj}. Skipping.")
                continue
                
            dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
            
            all_macro = []
            all_labels = []
            
            with torch.no_grad():
                for x, l in dataloader:
                    if x.ndim == 2:
                        x = x.unsqueeze(1)
                    elif x.ndim == 3 and x.shape[2] < x.shape[1]:
                        x = x.transpose(1, 2)
                    x = x.to(self.device)
                    h, _, _ = self.model(x)
                    all_macro.append(h.cpu().numpy())
                    all_labels.append(l.cpu().numpy())
                    
            import numpy as np
            res = {
                "macro_ssl": np.concatenate(all_macro, axis=0),
                "labels": np.concatenate(all_labels, axis=0)
            }
            
            with open(out_path, 'wb') as f:
                pickle.dump(res, f)
                
            self.logger.info(f"Saved SSL embeddings for {subj} -> {out_path} (SSL shape: {res['macro_ssl'].shape})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_subject", type=str, default="S2")
    args = parser.parse_args()
    
    generator = SSLEmbeddingGenerator()
    generator.generate_fold(test_subject=args.test_subject)