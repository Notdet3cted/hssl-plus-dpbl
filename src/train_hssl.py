import os
import json
import pickle
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from src.logger import setup_logger
from src.experiment_tracker import ExperimentTracker
from src.models.hssl import HSSLEncoder
from src.models.ssl_loss import NTXentLoss
from src.augmentations import SignalAugmentations

class WESADDataset(Dataset):
    def __init__(self, data_list, window_size=700, overlap=0.5):
        """
        data_list: list of loaded processed/normalized pickle dicts
        window_size: 700 (1 second of EDA at 700Hz as default example)
        """
        self.windows = []
        step = max(1, int(window_size * (1 - overlap)))
        
        self.labels = []
        for d in data_list:
            feat = d["features"]
            lbl = d.get("labels", None)
            
            # Ensure shape is (Time, Channels)
            if feat.ndim == 1:
                feat = feat.reshape(-1, 1)
                
            n_samples = feat.shape[0]
            for i in range(0, n_samples - window_size + 1, step):
                w = feat[i:i+window_size]
                self.windows.append(w)
                
                if lbl is not None:
                    w_lbls = lbl[i:i+window_size]
                    val, counts = np.unique(w_lbls, return_counts=True)
                    self.labels.append(val[np.argmax(counts)])
                else:
                    self.labels.append(-1)
                
    def __len__(self):
        return len(self.windows)
        
    def __getitem__(self, idx):
        # Convert to (Channels, Sequence_Length)
        w = self.windows[idx]
        l = self.labels[idx]
        return torch.tensor(w, dtype=torch.float32).transpose(0, 1), torch.tensor(l, dtype=torch.long)

class HSSLTrainer:
    def __init__(self):
        self.logger = setup_logger("HSSLTrainer")
        self.tracker = ExperimentTracker()
        self.checkpoints_dir = self.tracker.config["paths"].get("checkpoints", "checkpoints")
        self.reports_dir = self.tracker.config["paths"].get("reports", "reports")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Initialized HSSL Trainer on {self.device}")
        
    def load_fold_data(self, test_subject):
        folds_path = os.path.join(self.reports_dir, "loso_folds.json")
        with open(folds_path, 'r') as f:
            folds = json.load(f)
            
        if test_subject not in folds:
            raise ValueError(f"Fold for test subject {test_subject} not found.")
            
        train_subjects = folds[test_subject]["train"]
        norm_dir = folds[test_subject]["normalized_data_dir"]
        
        train_data = []
        for subj in train_subjects:
            path = os.path.join(norm_dir, f"{subj}_normalized.pkl")
            with open(path, 'rb') as f:
                train_data.append(pickle.load(f))
                
        return train_data

    def train_fold(self, test_subject, epochs=50, batch_size=128, window_size=700):
        self.logger.info(f"Starting HSSL Training for Fold: {test_subject}")
        fold_ckpt_dir = os.path.join(self.checkpoints_dir, f"hssl_fold_{test_subject}")
        os.makedirs(fold_ckpt_dir, exist_ok=True)
        
        # Load and prepare data
        self.logger.info("Loading normalized fold data...")
        train_data = self.load_fold_data(test_subject)
        
        dataset = WESADDataset(train_data, window_size=window_size)
        if len(dataset) == 0:
             self.logger.error("Dataset is empty after windowing. Check data preprocessing.")
             return
             
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.logger.info(f"Total training windows: {len(dataset)} ({len(dataloader)} batches)")
        
        # Determine channels dynamically based on dataset
        sample_batch, _ = next(iter(dataloader))
        channels = sample_batch.shape[1]
        self.logger.info(f"Detected channels: {channels}, Sequence length: {sample_batch.shape[2]}")
        
        self.model = HSSLEncoder(input_channels=channels).to(self.device)
        self.criterion = NTXentLoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Log Shapes (Debugging)
        self.model.eval()
        with torch.no_grad():
            x_test = sample_batch[:2].to(self.device)
            micro_h, macro_h, z = self.model(x_test)
            self.logger.info("--- Shape Logging ---")
            self.logger.info(f"Input Shape: {x_test.shape}")
            self.logger.info(f"Micro Representation Shape: {micro_h.shape}")
            self.logger.info(f"Macro Representation Shape: {macro_h.shape}")
            self.logger.info(f"Projected Embedding Shape: {z.shape}")
            self.logger.info("---------------------")
        self.model.train()
        
        # Resume Checkpoint
        start_epoch = 1
        best_loss = float('inf')
        latest_path = os.path.join(fold_ckpt_dir, "latest.pt")
        best_path = os.path.join(fold_ckpt_dir, "best.pt")
        
        if os.path.exists(latest_path):
            self.logger.info(f"Found checkpoint at {latest_path}. Resuming...")
            checkpoint = torch.load(latest_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('best_loss', best_loss)
            self.logger.info(f"Resumed from epoch {start_epoch - 1} with best loss {best_loss:.4f}")
        
        # Training Loop
        for epoch in range(start_epoch, epochs + 1):
            epoch_loss = 0.0
            
            for batch_idx, (x, _) in enumerate(dataloader):
                x = x.to(self.device)
                x_i, x_j = SignalAugmentations.get_views(x)
                
                self.optimizer.zero_grad()
                
                _, _, z_i = self.model(x_i)
                _, _, z_j = self.model(x_j)
                
                loss = self.criterion(z_i, z_j)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(dataloader)
            self.logger.info(f"Epoch {epoch}/{epochs} - Avg Contrastive Loss: {avg_loss:.4f}")
            
            # Save Latest Checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': avg_loss,
                'best_loss': min(best_loss, avg_loss)
            }, latest_path)
            
            # Save Best Checkpoint
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), best_path)
                self.logger.info(f"[*] New best checkpoint saved! Loss: {best_loss:.4f}")
                
        self.logger.info(f"Training completed for fold {test_subject}. Best loss: {best_loss:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_subject", type=str, default="S2", help="Subject to use as test fold")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to train (default small for test)")
    args = parser.parse_args()
    
    trainer = HSSLTrainer()
    trainer.train_fold(test_subject=args.test_subject, epochs=args.epochs)