import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from src.logger import setup_logger
from src.experiment_tracker import ExperimentTracker
from src.models.dpbl import DPBL, BaselineTracker

class DpblDataset(Dataset):
    def __init__(self, embeddings_dict, labels, baseline):
        """
        embeddings_dict: Dict containing "macro" embeddings of shape (N, dim)
        labels: Array of labels corresponding to windows
        baseline: The subject's baseline array of shape (dim,)
        """
        self.embeddings = embeddings_dict["macro"]
        
        # Determine actual number of valid samples
        n_samples = min(len(self.embeddings), len(labels))
        
        self.embeddings = self.embeddings[:n_samples]
        self.labels = labels[:n_samples]
        self.baseline = baseline
        
    def __len__(self):
        return len(self.embeddings)
        
    def __getitem__(self, idx):
        emb = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        lbl = torch.tensor(self.labels[idx], dtype=torch.long)
        bsl = torch.tensor(self.baseline, dtype=torch.float32)
        return emb, lbl, bsl

class DPBLTrainer:
    def __init__(self):
        self.logger = setup_logger("DPBLTrainer")
        self.tracker = ExperimentTracker()
        self.checkpoints_dir = self.tracker.config["paths"].get("checkpoints", "checkpoints")
        self.reports_dir = self.tracker.config["paths"].get("reports", "reports")
        self.hssl_embeddings_dir = os.path.join("embeddings", "hssl")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tracker_bsl = BaselineTracker()
        self.logger.info(f"Initialized DPBL Trainer on {self.device}")

    def get_labels_and_baseline(self, subject, norm_dir):
        """
        Extracts labels and calculates baseline from the initial 'baseline' task of WESAD.
        For simulation, we approximate baseline class as label 1 (baseline condition in WESAD).
        """
        norm_path = os.path.join(norm_dir, f"{subject}_normalized.pkl")
        with open(norm_path, 'rb') as f:
            data = pickle.load(f)
            
        labels = data["labels"]
        # In WESAD, Label 1 is baseline.
        # Find indices where label == 1. Wait, WESAD labels are per-sample. 
        # But HSSL windows were created with step overlap.
        # Approximation for mockup: We'll assume the first 100 windows are baseline 
        # or we calculate exact window labels using majority vote.
        window_size = 700
        overlap = 0.5
        step = max(1, int(window_size * (1 - overlap)))
        
        window_labels = []
        n_samples = len(labels)
        for i in range(0, n_samples - window_size + 1, step):
            w_lbls = labels[i:i+window_size]
            # Majority vote
            val, counts = np.unique(w_lbls, return_counts=True)
            window_labels.append(val[np.argmax(counts)])
            
        return np.array(window_labels)

    def train_fold(self, test_subject, epochs=10, batch_size=128):
        self.logger.info(f"Starting DPBL Training for Fold: {test_subject}")
        fold_ckpt_dir = os.path.join(self.checkpoints_dir, f"dpbl_fold_{test_subject}")
        os.makedirs(fold_ckpt_dir, exist_ok=True)
        
        folds_path = os.path.join(self.reports_dir, "loso_folds.json")
        with open(folds_path, 'r') as f:
            folds = json.load(f)
            
        train_subjects = folds[test_subject]["train"]
        norm_dir = folds[test_subject]["normalized_data_dir"]
        fold_emb_dir = os.path.join(self.hssl_embeddings_dir, f"fold_{test_subject}")
        
        all_datasets = []
        embedding_dim = None
        
        for subj in train_subjects:
            emb_path = os.path.join(fold_emb_dir, f"{subj}_embeddings.pkl")
            if not os.path.exists(emb_path):
                self.logger.warning(f"Embeddings for {subj} not found. Skipping.")
                continue
                
            with open(emb_path, 'rb') as f:
                emb_data = pickle.load(f)
                
            if embedding_dim is None:
                embedding_dim = emb_data["macro"].shape[1]
                
            labels = self.get_labels_and_baseline(subj, norm_dir)
            
            # Baseline is mean of embeddings where label == 1 (Baseline class in WESAD)
            # If no label 1, fallback to first 50 embeddings
            base_idx = np.where(labels[:len(emb_data["macro"])] == 1)[0]
            if len(base_idx) > 0:
                baseline = self.tracker_bsl.update_baseline(subj, emb_data["macro"][base_idx])
            else:
                baseline = self.tracker_bsl.update_baseline(subj, emb_data["macro"][:50])
                
            dataset = DpblDataset(emb_data, labels, baseline)
            all_datasets.append(dataset)
            
        if not all_datasets:
            self.logger.error("No valid datasets loaded for DPBL.")
            return
            
        full_dataset = torch.utils.data.ConcatDataset(all_datasets)
        dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
        self.logger.info(f"Total DPBL training windows: {len(full_dataset)}")
        
        self.model = DPBL(embedding_dim=embedding_dim).to(self.device)
        
        # Split into train/val
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.logger.info(f"Total DPBL windows -> Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
        self.model = DPBL(embedding_dim=embedding_dim).to(self.device)
        
        # Mapping WESAD labels: 1=Baseline, 2=Stress, 3=Amusement, 4=Meditation.
        # Max label is 4, so we need 5 classes (0-4) or remap 1-4 to 0-3.
        # Let's assume labels are 0-4
        classifier = nn.Linear(embedding_dim, 5).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(list(self.model.parameters()) + list(classifier.parameters()), lr=0.001)
        
        # Resume Checkpoint
        start_epoch = 1
        best_loss = float('inf')
        latest_path = os.path.join(fold_ckpt_dir, "latest.pt")
        best_path = os.path.join(fold_ckpt_dir, "best.pt")
        
        if os.path.exists(latest_path):
            self.logger.info(f"Resuming DPBL from {latest_path}")
            ckpt = torch.load(latest_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            classifier.load_state_dict(ckpt['classifier_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_loss = ckpt.get('best_loss', best_loss)
            
        self.model.train()
        classifier.train()
        
        from sklearn.metrics import f1_score
        
        for epoch in range(start_epoch, epochs + 1):
            self.model.train()
            classifier.train()
            epoch_loss = 0.0
            
            for emb, lbl, bsl in train_loader:
                emb, lbl, bsl = emb.to(self.device), lbl.to(self.device), bsl.to(self.device)
                
                optimizer.zero_grad()
                
                pers_emb = self.model(emb, bsl)
                preds = classifier(pers_emb)
                
                loss = criterion(preds, lbl)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(train_loader)
            
            # Validation
            self.model.eval()
            classifier.eval()
            val_loss = 0.0
            all_preds = []
            all_lbls = []
            
            with torch.no_grad():
                for emb, lbl, bsl in val_loader:
                    emb, lbl, bsl = emb.to(self.device), lbl.to(self.device), bsl.to(self.device)
                    pers_emb = self.model(emb, bsl)
                    preds = classifier(pers_emb)
                    loss = criterion(preds, lbl)
                    val_loss += loss.item()
                    
                    all_preds.extend(torch.argmax(preds, dim=1).cpu().numpy())
                    all_lbls.extend(lbl.cpu().numpy())
                    
            avg_val_loss = val_loss / len(val_loader)
            val_f1 = f1_score(all_lbls, all_preds, average='macro', zero_division=0)
            
            self.logger.info(f"DPBL Epoch {epoch}/{epochs} - Train Loss: {avg_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val F1: {val_f1:.4f}")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'val_loss': avg_val_loss,
                'val_f1': val_f1,
                'best_loss': min(best_loss, avg_val_loss)
            }, latest_path)
            
            # Save baseline tracker
            self.tracker_bsl.save(os.path.join(fold_ckpt_dir, "baseline_tracker.pkl"))
            
            # Using val_loss for best checkpoint (could also use val_f1)
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(self.model.state_dict(), best_path)
                torch.save(classifier.state_dict(), os.path.join(fold_ckpt_dir, "classifier_best.pt"))
                self.logger.info(f"[*] New best DPBL checkpoint saved! Val Loss: {best_loss:.4f} | Val F1: {val_f1:.4f}")

        self.logger.info(f"DPBL Training completed for fold {test_subject}.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_subject", type=str, default="S2")
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    
    trainer = DPBLTrainer()
    trainer.train_fold(test_subject=args.test_subject, epochs=args.epochs)