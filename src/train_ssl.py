import os
import json
import pickle
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from src.logger import setup_logger
from src.experiment_tracker import ExperimentTracker
from src.models.ssl.simsiam import SSLEncoder, SimSiamLoss

class SSLDataset(Dataset):
    """Dataset for SSL pre-training with SimSiam augmentations."""
    def __init__(self, data_list, window_size=700):
        self.windows = []
        self.subject_ids = []
        for data in data_list:
            features = data["features"]
            n = features.shape[0]
            for i in range(0, n - window_size + 1, int(window_size * 0.5)):
                w = features[i:i+window_size]
                if w.shape[0] == window_size:
                    self.windows.append(w)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        x = self.windows[idx]
        # Two random crops/augmentations for SimSiam
        x_tensor = torch.tensor(x, dtype=torch.float32)
        x1 = x_tensor + torch.randn_like(x_tensor) * 0.02
        x2 = x_tensor + torch.randn_like(x_tensor) * 0.02
        return x_tensor, x1, x2


class SSLTrainer:
    def __init__(self):
        self.logger = setup_logger("SSLTrainer")
        self.tracker = ExperimentTracker()
        self.checkpoints_dir = self.tracker.config["paths"].get("checkpoints", "checkpoints")
        self.reports_dir = self.tracker.config["paths"].get("reports", "reports")
        self.processed_dir = self.tracker.config["paths"].get("processed_data", "data/processed/")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"SSL Pre-trainer on {self.device}")

    def train_ssl(self, epochs=50, batch_size=128):
        self.logger.info("=" * 60)
        self.logger.info("SSL PRE-TRAINING (SimSiam baseline)")
        self.logger.info("=" * 60)
        
        folds_path = os.path.join(self.reports_dir, "loso_folds.json")
        with open(folds_path, 'r') as f:
            folds = json.load(f)
        
        # Use all folds' train subjects for SSL pre-training
        all_subjects = set()
        for fold, v in folds.items():
            for s in v["train"]:
                all_subjects.add(s)
                
        self.logger.info(f"SSL pre-training on {len(all_subjects)} subjects: {sorted(all_subjects)}")
        
        all_data = []
        norm_dirs_seen = set()
        for fold, v in folds.items():
            nd = v["normalized_data_dir"]
            if nd not in norm_dirs_seen:
                norm_dirs_seen.add(nd)
                for subj in sorted(all_subjects):
                    path = os.path.join(nd, f"{subj}_normalized.pkl")
                    if os.path.exists(path):
                        with open(path, 'rb') as f:
                            all_data.append(pickle.load(f))
                        
        if not all_data:
            self.logger.error("No data found for SSL pre-training")
            return
            
        window_size = self.tracker.config.get("preprocessing", {}).get("window_size", 700)
        dataset = SSLDataset(all_data, window_size=window_size)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.logger.info(f"SSL windows: {len(dataset)}")
        
        # Determine input channels from first data
        sample = all_data[0]["features"]
        channels = sample.shape[1] if sample.ndim > 1 else 1
        
        model = SSLEncoder(input_channels=channels).to(self.device)
        criterion = SimSiamLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        ssl_ckpt_dir = os.path.join(self.checkpoints_dir, "ssl_pretrain")
        os.makedirs(ssl_ckpt_dir, exist_ok=True)
        
        best_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            for _, x1, x2 in dataloader:
                if x1.ndim == 2:
                    x1 = x1.unsqueeze(1)
                    x2 = x2.unsqueeze(1)
                elif x1.ndim == 3 and x1.shape[2] < x1.shape[1]:
                    x1 = x1.transpose(1, 2)
                    x2 = x2.transpose(1, 2)
                    
                x1, x2 = x1.to(self.device), x2.to(self.device)
                _, z1, p1 = model(x1)
                _, z2, p2 = model(x2)
                loss = criterion(p1, z2, p2, z1)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / max(len(dataloader), 1)
            scheduler.step()
            
            self.logger.info(f"[SSL] Epoch {epoch}/{epochs} Loss: {avg_loss:.4f} LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Save best
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), os.path.join(ssl_ckpt_dir, "best.pt"))
                self.logger.info(f"[*] New best SSL checkpoint! Loss: {best_loss:.4f}")
        
        torch.save(model.state_dict(), os.path.join(ssl_ckpt_dir, "final.pt"))
        self.logger.info(f"SSL pre-training complete. Best loss: {best_loss:.4f}")
        
    def run(self, epochs=50, batch_size=128):
        self.train_ssl(epochs=epochs, batch_size=batch_size)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    
    trainer = SSLTrainer()
    trainer.run(epochs=args.epochs, batch_size=args.batch_size)