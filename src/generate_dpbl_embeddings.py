import os
import json
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.logger import setup_logger
from src.experiment_tracker import ExperimentTracker
from src.models.dpbl import DPBL, BaselineTracker
from src.train_dpbl import DpblDataset

class DpblEmbeddingGenerator:
    def __init__(self):
        self.logger = setup_logger("DpblEmbeddingGenerator")
        self.tracker = ExperimentTracker()
        self.checkpoints_dir = self.tracker.config["paths"].get("checkpoints", "checkpoints")
        self.reports_dir = self.tracker.config["paths"].get("reports", "reports")
        self.hssl_embeddings_dir = self.tracker.config["paths"].get("embeddings_hssl", "embeddings/hssl/")
        self.dpbl_embeddings_dir = self.tracker.config["paths"].get("embeddings_hssl_dpbl", "embeddings/hssl_dpbl/")
        os.makedirs(self.dpbl_embeddings_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tracker_bsl = BaselineTracker()
        
    def compute_baseline_no_label(self, emb_data, n_calibration=30):
        """Compute baseline from first n_calibration windows (no labels)."""
        macro = emb_data["macro"]
        n = min(n_calibration, len(macro))
        return np.mean(macro[:n], axis=0)

    def get_window_labels(self, subject, norm_dir):
        """Extract window labels from normalized data."""
        norm_path = os.path.join(norm_dir, f"{subject}_normalized.pkl")
        with open(norm_path, 'rb') as f:
            data = pickle.load(f)
        labels = data["labels"]
        window_size = self.tracker.config.get("preprocessing", {}).get("window_size", 700)
        overlap = 0.5
        step = max(1, int(window_size * (1 - overlap)))
        window_labels = []
        n_samples = len(labels)
        for i in range(0, n_samples - window_size + 1, step):
            w_lbls = labels[i:i+window_size]
            val, counts = np.unique(w_lbls, return_counts=True)
            window_labels.append(val[np.argmax(counts)])
        return np.array(window_labels)

    def generate_fold(self, test_subject):
        self.logger.info(f"Generating DPBL embeddings for Fold: {test_subject}")
        fold_ckpt_dir = os.path.join(self.checkpoints_dir, f"dpbl_fold_{test_subject}")
        best_path = os.path.join(fold_ckpt_dir, "best.pt")
        
        folds_path = os.path.join(self.reports_dir, "loso_folds.json")
        with open(folds_path, 'r') as f:
            folds = json.load(f)
            
        if test_subject not in folds:
            raise ValueError(f"Fold for {test_subject} not found.")
            
        norm_dir = folds[test_subject]["normalized_data_dir"]
        subjects = folds[test_subject]["train"] + folds[test_subject]["test"]
        
        fold_emb_dir_in = os.path.join(self.hssl_embeddings_dir, f"fold_{test_subject}")
        fold_emb_dir_out = os.path.join(self.dpbl_embeddings_dir, f"fold_{test_subject}")
        os.makedirs(fold_emb_dir_out, exist_ok=True)
        
        # Determine embedding dimension
        sample_path = os.path.join(fold_emb_dir_in, f"{subjects[0]}_embeddings.pkl")
        with open(sample_path, 'rb') as f:
            sample_data = pickle.load(f)
        embedding_dim = sample_data["macro"].shape[1]
        
        self.model = DPBL(embedding_dim=embedding_dim).to(self.device)
        
        tracker_path = os.path.join(fold_ckpt_dir, "baseline_tracker.pkl")
        
        if os.path.exists(best_path):
            self.model.load_state_dict(torch.load(best_path, map_location=self.device))
            self.model.eval()
            self.logger.info(f"Loaded best DPBL checkpoint from {best_path}.")
        else:
            self.logger.error(f"No checkpoint found at {best_path}. Run training first.")
            return
            
        if os.path.exists(tracker_path):
            self.tracker_bsl.load(tracker_path)
            self.logger.info(f"Loaded baseline tracker from {tracker_path}.")

        for subj in subjects:
            emb_path_in = os.path.join(fold_emb_dir_in, f"{subj}_embeddings.pkl")
            if not os.path.exists(emb_path_in):
                continue
                
            with open(emb_path_in, 'rb') as f:
                emb_data = pickle.load(f)
                
            labels = self.get_window_labels(subj, norm_dir)
            
            # Compute baseline from first N windows (no labels) for all subjects
            # For test subjects that weren't in training, this is the only option
            baseline = self.compute_baseline_no_label(emb_data, n_calibration=30)
                
            dataset = DpblDataset(emb_data, labels, baseline)
            dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
            
            all_pers = []
            
            with torch.no_grad():
                for emb, lbl, bsl in dataloader:
                    emb, bsl = emb.to(self.device), bsl.to(self.device)
                    pers_emb = self.model(emb, bsl)
                    all_pers.append(pers_emb.cpu().numpy())
                    
            res_macro = np.concatenate(all_pers, axis=0)
            
            # Repackage with original micro if needed, and new personalized macro
            res = {
                "micro": emb_data.get("micro", None),
                "macro_hssl": emb_data["macro"][:len(res_macro)],
                "macro_dpbl": res_macro,
                "labels": labels[:len(res_macro)]
            }
            
            out_path = os.path.join(fold_emb_dir_out, f"{subj}_embeddings.pkl")
            with open(out_path, 'wb') as f:
                pickle.dump(res, f)
                
            self.logger.info(f"Saved DPBL embeddings for {subj} -> {out_path} (Shape: {res_macro.shape})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_subject", type=str, default="S2")
    args = parser.parse_args()
    
    generator = DpblEmbeddingGenerator()
    generator.generate_fold(test_subject=args.test_subject)