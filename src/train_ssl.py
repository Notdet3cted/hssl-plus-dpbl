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
    """Memory-efficient SSL dataset. Stores metadata only, samples windows on-the-fly."""
    def __init__(self, data_list, window_size=700, overlap=0.5):
        self.data_refs = []
        self.window_counts = []
        self.window_size = window_size
        self.step = max(1, int(window_size * (1 - overlap)))

        for data in data_list:
            features = data["features"]
            if features.ndim == 1:
                features = features.reshape(-1, 1)
            n = features.shape[0]
            if n >= window_size:
                n_windows = (n - window_size) // self.step + 1
                self.data_refs.append(features)
                self.window_counts.append(n_windows)

        if not self.data_refs:
            self.cum_counts = np.array([0])
            self.total = 0
        else:
            self.cum_counts = np.cumsum([0] + self.window_counts)
            self.total = self.cum_counts[-1]

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        data_idx = np.searchsorted(self.cum_counts, idx, side='right') - 1
        local_idx = idx - self.cum_counts[data_idx]
        start = local_idx * self.step
        features = self.data_refs[data_idx]
        w = features[start:start + self.window_size]

        x_tensor = torch.tensor(w, dtype=torch.float32)
        x1 = x_tensor + torch.randn_like(x_tensor) * 0.02
        x2 = x_tensor + torch.randn_like(x_tensor) * 0.02
        return x1, x2


class SSLTrainer:
    def __init__(self):
        self.logger = setup_logger("SSLTrainer")
        self.tracker = ExperimentTracker()
        self.checkpoints_dir = self.tracker.config["paths"].get("checkpoints", "checkpoints")
        self.reports_dir = self.tracker.config["paths"].get("reports", "reports")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"SSL Pre-trainer on {self.device}")

    def load_fold_data(self, test_subject):
        folds_path = os.path.join(self.reports_dir, "loso_folds.json")
        with open(folds_path, 'r') as f:
            folds = json.load(f)

        if test_subject not in folds:
            raise ValueError(f"Fold for test subject {test_subject} not found.")

        train_subjects = folds[test_subject]["train"]
        norm_dir = folds[test_subject]["normalized_data_dir"]

        self.logger.info(f"SSL pre-training on {len(train_subjects)} subjects for fold {test_subject}: {train_subjects}")

        train_data = []
        for subj in train_subjects:
            path = os.path.join(norm_dir, f"{subj}_normalized.pkl")
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    train_data.append(pickle.load(f))

        return train_data

    def train_fold(self, test_subject, epochs=50, batch_size=128):
        self.logger.info(f"Starting SSL Pre-training for Fold: {test_subject}")
        fold_ckpt_dir = os.path.join(self.checkpoints_dir, f"ssl_fold_{test_subject}")
        os.makedirs(fold_ckpt_dir, exist_ok=True)

        # Skip if best checkpoint already exists
        best_path = os.path.join(fold_ckpt_dir, "best.pt")
        if os.path.exists(best_path):
            self.logger.info(f"Fold {test_subject} already completed (best.pt exists). Skipping.")
            return

        # Load data
        train_data = self.load_fold_data(test_subject)
        if not train_data:
            self.logger.error("No training data found.")
            return

        window_size = self.tracker.config.get("preprocessing", {}).get("window_size", 700)
        dataset = SSLDataset(train_data, window_size=window_size)
        if len(dataset) == 0:
            self.logger.error("Dataset is empty. Check data.")
            return

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
        self.logger.info(f"Total training windows: {len(dataset)} ({len(dataloader)} batches)")

        # Determine input channels
        sample = train_data[0]["features"]
        channels = sample.shape[1] if sample.ndim > 1 else 1

        model = SSLEncoder(input_channels=channels).to(self.device)
        criterion = SimSiamLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Resume from checkpoint if exists
        latest_path = os.path.join(fold_ckpt_dir, "latest.pt")
        start_epoch = 1
        best_loss = float('inf')

        if os.path.exists(latest_path):
            self.logger.info(f"Found checkpoint at {latest_path}. Resuming...")
            checkpoint = torch.load(latest_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('best_loss', best_loss)
            self.logger.info(f"Resumed from epoch {start_epoch - 1} with best loss {best_loss:.4f}")

        for epoch in range(start_epoch, epochs + 1):
            model.train()
            epoch_loss = 0.0
            for x1, x2 in dataloader:
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

            self.logger.info(f"[SSL] Fold {test_subject} Epoch {epoch}/{epochs} Loss: {avg_loss:.4f} LR: {scheduler.get_last_lr()[0]:.6f}")

            # Save latest checkpoint (with optimizer state for resume)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'best_loss': min(best_loss, avg_loss)
            }, latest_path)

            # Save best
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), best_path)
                self.logger.info(f"[*] New best SSL checkpoint! Loss: {best_loss:.4f}")

        # Save final
        torch.save(model.state_dict(), os.path.join(fold_ckpt_dir, "final.pt"))
        self.logger.info(f"SSL pre-training for fold {test_subject} complete. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_subject", type=str, default=None, help="Fold subject for LOSO training (None = legacy global training)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    trainer = SSLTrainer()
    if args.test_subject:
        trainer.train_fold(test_subject=args.test_subject, epochs=args.epochs, batch_size=args.batch_size)
    else:
        trainer.logger.error("--test_subject is required for LOSO SSL training.")
        exit(1)
