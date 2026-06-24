import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from src.logger import setup_logger
from src.experiment_tracker import ExperimentTracker
from src.models.cnn import Simple1DCNN

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class EarlyStopping:
    def __init__(self, patience=5, mode='max'):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif (self.mode == 'max' and score < self.best_score) or \
             (self.mode == 'min' and score > self.best_score):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class ClassifierTrainer:
    def __init__(self, seed=42):
        self.logger = setup_logger("ClassifierTrainer")
        self.tracker = ExperimentTracker()
        self.seed = seed
        self.set_seed(seed)
        self.reports_dir = self.tracker.config["paths"].get("reports", "reports")
        self.checkpoints_dir = self.tracker.config["paths"].get("checkpoints", "checkpoints")
        self.windowed_dir = os.path.join("data", "windowed")
        self.hssl_embeddings_dir = os.path.join("embeddings", "hssl")
        self.dpbl_embeddings_dir = os.path.join("embeddings", "hssl_dpbl")
        self.ssl_embeddings_dir = os.path.join("embeddings", "ssl")
        self.ssl_dpbl_embeddings_dir = os.path.join("embeddings", "ssl_dpbl")
        self.results_dir = os.path.join("results")
        self.predictions_dir = os.path.join("results", "predictions")
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.predictions_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        folds_path = os.path.join(self.reports_dir, "loso_folds.json")
        with open(folds_path, 'r') as f:
            self.folds = json.load(f)

    def set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def load_windowed_data(self, test_subject):
        train_subjs = self.folds[test_subject]["train"]
        test_subjs = self.folds[test_subject]["test"]
        def _load_group(subjs):
            X_all, y_all = [], []
            for subj in subjs:
                path = os.path.join(self.windowed_dir, f"{subj}_windows.pkl")
                if not os.path.exists(path):
                    continue
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                X_all.append(data["features"])
                y_all.append(data["labels"])
            return np.concatenate(X_all), np.concatenate(y_all)
        X_train, y_train = _load_group(train_subjs)
        X_test, y_test = _load_group(test_subjs)
        return X_train, y_train, X_test, y_test

    def load_embeddings(self, test_subject, emb_dir, use_dpbl=False):
        train_subjs = self.folds[test_subject]["train"]
        test_subjs = self.folds[test_subject]["test"]
        fold_dir = os.path.join(emb_dir, f"fold_{test_subject}")
        def _load_group(subjs):
            X_all, y_all = [], []
            for subj in subjs:
                path = os.path.join(fold_dir, f"{subj}_embeddings.pkl")
                if not os.path.exists(path):
                    continue
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                # DPBL dir: use macro_dpbl. SSL dir: use macro_ssl. HSSL dir: use macro.
                if use_dpbl and "macro_dpbl" in data:
                    key = "macro_dpbl"
                elif "macro_ssl" in data:
                    key = "macro_ssl"
                else:
                    key = "macro"
                X_all.append(data[key])
                y_all.append(data["labels"][:len(data[key])])
            return np.concatenate(X_all), np.concatenate(y_all)
        X_train, y_train = _load_group(train_subjs)
        X_test, y_test = _load_group(test_subjs)
        return X_train, y_train, X_test, y_test

    def split_val(self, X, y, val_ratio=0.15):
        n = len(y)
        idx = np.random.permutation(n)
        val_size = int(n * val_ratio)
        val_idx, train_idx = idx[:val_size], idx[val_size:]
        return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

    def evaluate(self, y_true, y_prob, num_classes=2):
        y_pred = np.argmax(y_prob, axis=1)
        res = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
            "recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average='macro', zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }
        try:
            if num_classes > 2:
                y_true_bin = label_binarize(y_true, classes=range(num_classes))
                res["roc_auc"] = roc_auc_score(y_true_bin, y_prob, multi_class='ovr', average='macro')
                res["pr_auc"] = average_precision_score(y_true_bin, y_prob, average='macro')
            else:
                res["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])
                res["pr_auc"] = average_precision_score(y_true, y_prob[:, 1])
        except Exception:
            res["roc_auc"] = 0.0
            res["pr_auc"] = 0.0
        return res

    def save_predictions(self, model_name, test_subject, y_true, y_prob):
        y_pred = np.argmax(y_prob, axis=1)
        out = {
            "subject": test_subject,
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
            "y_prob": y_prob.tolist()
        }
        path = os.path.join(self.predictions_dir, f"{model_name}_fold_{test_subject}.json")
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)

    def train_rf(self, test_subject):
        self.logger.info(f"[RF] Fold {test_subject}")
        X_train, y_train, X_test, y_test = self.load_windowed_data(test_subject)
        X_tr, y_tr, X_val, y_val = self.split_val(X_train, y_train)
        X_tr_f = X_tr.reshape(len(X_tr), -1)
        X_val_f = X_val.reshape(len(X_val), -1)
        X_te_f = X_test.reshape(len(X_test), -1)
        rf = RandomForestClassifier(n_estimators=200, random_state=self.seed, n_jobs=-1, class_weight='balanced')
        rf.fit(X_tr_f, y_tr)
        val_prob = rf.predict_proba(X_val_f)
        val_f1 = f1_score(y_val, np.argmax(val_prob, axis=1), average='macro', zero_division=0)
        self.logger.info(f"[RF] Val F1: {val_f1:.4f}")
        rf.fit(np.concatenate([X_tr_f, X_val_f]), np.concatenate([y_tr, y_val]))
        test_prob = rf.predict_proba(X_te_f)
        num_classes = test_prob.shape[1]
        metrics = self.evaluate(y_test, test_prob, num_classes)
        self.logger.info(f"[RF] Test F1: {metrics['f1_score']:.4f}")
        with open(os.path.join(self.results_dir, f"rf_fold_{test_subject}.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
        with open(os.path.join(self.checkpoints_dir, f"rf_fold_{test_subject}.pkl"), 'wb') as f:
            pickle.dump(rf, f)
        self.save_predictions("rf", test_subject, y_test, test_prob)

    def train_cnn(self, test_subject, epochs=30):
        self.logger.info(f"[CNN] Fold {test_subject}")
        X_train, y_train, X_test, y_test = self.load_windowed_data(test_subject)
        if X_train.ndim == 2:
            X_train = np.expand_dims(X_train, 1)
            X_test = np.expand_dims(X_test, 1)
        elif X_train.ndim == 3 and X_train.shape[2] < X_train.shape[1]:
            X_train = np.transpose(X_train, (0, 2, 1))
            X_test = np.transpose(X_test, (0, 2, 1))
        X_tr, y_tr, X_val, y_val = self.split_val(X_train, y_train)
        num_classes = max(np.max(y_train), np.max(y_test)) + 1
        model = Simple1DCNN(input_channels=X_tr.shape[1], seq_len=X_tr.shape[2], num_classes=num_classes).to(self.device)
        classes = np.unique(y_tr)
        weights = compute_class_weight('balanced', classes=classes, y=y_tr)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(self.device))
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        es = EarlyStopping(patience=7, mode='max')
        best_state = None
        train_loader = DataLoader(SimpleDataset(X_tr, y_tr), batch_size=128, shuffle=True)
        val_loader = DataLoader(SimpleDataset(X_val, y_val), batch_size=128)
        
        train_losses = []
        val_f1s = []
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            train_losses.append(epoch_loss / len(train_loader))
            
            model.eval()
            v_probs = []
            with torch.no_grad():
                for x, _ in val_loader:
                    v_probs.extend(torch.softmax(model(x.to(self.device)), dim=1).cpu().numpy())
            v_f1 = float(f1_score(y_val, np.argmax(v_probs, axis=1), average='macro', zero_division=0))
            val_f1s.append(v_f1)
            scheduler.step(v_f1)
            es(v_f1)
            if v_f1 >= es.best_score:
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.logger.info(f"[CNN] Epoch {epoch+1}/{epochs} Val F1: {v_f1:.4f}")
            if es.early_stop:
                self.logger.info("[CNN] Early stopping")
                break
                
        os.makedirs(os.path.join(self.results_dir, "learning_curves"), exist_ok=True)
        curve_data = {"train_loss": train_losses, "val_f1": val_f1s}
        with open(os.path.join(self.results_dir, "learning_curves", f"cnn_fold_{test_subject}.json"), 'w') as f:
            json.dump(curve_data, f, indent=2)

        model.load_state_dict(best_state)
        model.eval()
        test_loader = DataLoader(SimpleDataset(X_test, y_test), batch_size=128)
        t_probs = []
        with torch.no_grad():
            for x, _ in test_loader:
                t_probs.extend(torch.softmax(model(x.to(self.device)), dim=1).cpu().numpy())
        t_probs = np.array(t_probs)
        metrics = self.evaluate(y_test, t_probs, num_classes)
        self.logger.info(f"[CNN] Test F1: {metrics['f1_score']:.4f}")
        with open(os.path.join(self.results_dir, f"cnn_fold_{test_subject}.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
        torch.save(best_state, os.path.join(self.checkpoints_dir, f"cnn_fold_{test_subject}.pt"))
        self.save_predictions("cnn", test_subject, y_test, t_probs)

    def train_embedding_classifier(self, test_subject, emb_dir, model_name, epochs=30):
        self.logger.info(f"[{model_name}] Fold {test_subject}")
        X_train, y_train, X_test, y_test = self.load_embeddings(test_subject, emb_dir)
        if len(X_train) == 0:
            self.logger.error(f"No embeddings for {model_name}")
            return
        X_tr, y_tr, X_val, y_val = self.split_val(X_train, y_train)
        num_classes = max(np.max(y_train), np.max(y_test)) + 1
        classifier = nn.Sequential(
            nn.Linear(X_train.shape[1], 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        ).to(self.device)
        classes = np.unique(y_tr)
        weights = compute_class_weight('balanced', classes=classes, y=y_tr)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(self.device))
        optimizer = optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        es = EarlyStopping(patience=7, mode='max')
        best_state = None
        train_loader = DataLoader(SimpleDataset(X_tr, y_tr), batch_size=128, shuffle=True)
        val_loader = DataLoader(SimpleDataset(X_val, y_val), batch_size=128)
        
        train_losses = []
        val_f1s = []
        
        for epoch in range(epochs):
            classifier.train()
            epoch_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(classifier(x), y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            train_losses.append(epoch_loss / len(train_loader))
            
            classifier.eval()
            v_probs = []
            with torch.no_grad():
                for x, _ in val_loader:
                    v_probs.extend(torch.softmax(classifier(x.to(self.device)), dim=1).cpu().numpy())
            v_f1 = float(f1_score(y_val, np.argmax(v_probs, axis=1), average='macro', zero_division=0))
            val_f1s.append(v_f1)
            scheduler.step(v_f1)
            es(v_f1)
            if v_f1 >= es.best_score:
                best_state = {k: v.cpu().clone() for k, v in classifier.state_dict().items()}
            self.logger.info(f"[{model_name}] Epoch {epoch+1}/{epochs} Val F1: {v_f1:.4f}")
            if es.early_stop:
                self.logger.info(f"[{model_name}] Early stopping")
                break
                
        os.makedirs(os.path.join(self.results_dir, "learning_curves"), exist_ok=True)
        curve_data = {"train_loss": train_losses, "val_f1": val_f1s}
        with open(os.path.join(self.results_dir, "learning_curves", f"{model_name.lower()}_fold_{test_subject}.json"), 'w') as f:
            json.dump(curve_data, f, indent=2)

        classifier.load_state_dict(best_state)
        classifier.eval()
        test_loader = DataLoader(SimpleDataset(X_test, y_test), batch_size=128)
        t_probs = []
        with torch.no_grad():
            for x, _ in test_loader:
                t_probs.extend(torch.softmax(classifier(x.to(self.device)), dim=1).cpu().numpy())
        t_probs = np.array(t_probs)
        metrics = self.evaluate(y_test, t_probs, num_classes)
        self.logger.info(f"[{model_name}] Test F1: {metrics['f1_score']:.4f}")
        with open(os.path.join(self.results_dir, f"{model_name.lower()}_fold_{test_subject}.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
        torch.save(best_state, os.path.join(self.checkpoints_dir, f"{model_name.lower()}_clf_fold_{test_subject}.pt"))
        self.save_predictions(model_name.lower(), test_subject, y_test, t_probs)

    def run_all(self, test_subject, epochs=30, models_to_run=None):
        if models_to_run is None:
            models_to_run = ["rf", "cnn", "ssl", "hssl", "ssl+dpbl", "hssl+dpbl"]
            
        if "rf" in models_to_run:
            self.train_rf(test_subject)
        if "cnn" in models_to_run:
            self.train_cnn(test_subject, epochs=epochs)
        if "ssl" in models_to_run:
            self.train_embedding_classifier(test_subject, self.ssl_embeddings_dir, "SSL", epochs=epochs)
        if "hssl" in models_to_run:
            self.train_embedding_classifier(test_subject, self.hssl_embeddings_dir, "HSSL", epochs=epochs)
        if "ssl+dpbl" in models_to_run:
            self.train_embedding_classifier(test_subject, self.ssl_dpbl_embeddings_dir, "SSL+DPBL", epochs=epochs)
        if "hssl+dpbl" in models_to_run:
            self.train_embedding_classifier(test_subject, self.dpbl_embeddings_dir, "HSSL+DPBL", epochs=epochs)

    def train_hssl_dpbl(self, test_subject, epochs=30):
        self.train_embedding_classifier(test_subject, self.dpbl_embeddings_dir, "HSSL+DPBL", epochs=epochs)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_subject", type=str, default="S2")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", type=str, default="rf,cnn,hssl,hssl+dpbl", help="Comma-separated list of models to train")
    args = parser.parse_args()
    
    models_list = [m.strip().lower() for m in args.models.split(',')]
    trainer = ClassifierTrainer(seed=args.seed)
    trainer.run_all(test_subject=args.test_subject, epochs=args.epochs, models_to_run=models_list)
