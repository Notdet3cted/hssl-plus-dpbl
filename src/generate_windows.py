import os
import json
import pickle
import yaml
import numpy as np
from src.logger import setup_logger

logger = setup_logger("GenerateWindows")

def _load_window_size():
    try:
        with open("config/config.yaml", 'r') as f:
            cfg = yaml.safe_load(f)
        return cfg.get("preprocessing", {}).get("window_size", 700)
    except Exception:
        return 700

def extract_windows(features, labels, window_size=700, overlap=0.5):
    step = max(1, int(window_size * (1 - overlap)))
    n_samples = len(labels)
    X_win, y_win = [], []
    for i in range(0, n_samples - window_size + 1, step):
        w_feat = features[i:i+window_size]
        w_lbls = labels[i:i+window_size]
        val, counts = np.unique(w_lbls, return_counts=True)
        y_win.append(val[np.argmax(counts)])
        X_win.append(w_feat)
    return np.array(X_win), np.array(y_win)

def generate_all():
    folds_path = "reports/loso_folds.json"
    with open(folds_path, 'r') as f:
        folds = json.load(f)
    
    subjs = folds["S2"]["train"] + folds["S2"]["test"]
    norm_dir = folds["S2"]["normalized_data_dir"]
    
    out_dir = os.path.join("data", "windowed")
    os.makedirs(out_dir, exist_ok=True)
    
    for subj in subjs:
        path = os.path.join(norm_dir, f"{subj}_normalized.pkl")
        out_path = os.path.join(out_dir, f"{subj}_windows.pkl")
        if os.path.exists(out_path):
            continue
            
        with open(path, 'rb') as f:
            data = pickle.load(f)
            
        ws = _load_window_size()
        X, y = extract_windows(data["features"], data["labels"], window_size=ws)
        with open(out_path, 'wb') as f:
            pickle.dump({"features": X, "labels": y}, f)
        logger.info(f"Saved windows for {subj}: X {X.shape}, y {y.shape}")

if __name__ == "__main__":
    generate_all()