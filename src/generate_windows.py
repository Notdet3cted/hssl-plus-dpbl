import os
import json
import pickle
import numpy as np
from src.logger import setup_logger
from src.experiment_tracker import ExperimentTracker

logger = setup_logger("GenerateWindows")

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
    tracker = ExperimentTracker()
    reports_dir = tracker.config["paths"]["reports"]
    windowed_dir = tracker.config["paths"]["windowed"]
    config = tracker.config

    folds_path = os.path.join(reports_dir, "loso_folds.json")
    with open(folds_path, 'r') as f:
        folds = json.load(f)
    
        # Use the first fold key dynamically (any subject works)
        first_fold_key = next(iter(folds))
        subjs = folds[first_fold_key]["train"] + folds[first_fold_key]["test"]
        # Use the same normalized data directory as the first fold key
        norm_dir = folds[first_fold_key]["normalized_data_dir"]
    
    os.makedirs(windowed_dir, exist_ok=True)
    
    ws = config.get("preprocessing", {}).get("window_size", 700)
    
    for subj in subjs:
        path = os.path.join(norm_dir, f"{subj}_normalized.pkl")
        out_path = os.path.join(windowed_dir, f"{subj}_windows.pkl")
        if os.path.exists(out_path):
            continue
            
        with open(path, 'rb') as f:
            data = pickle.load(f)
            
        X, y = extract_windows(data["features"], data["labels"], window_size=ws)
        with open(out_path, 'wb') as f:
            pickle.dump({"features": X, "labels": y}, f)
        logger.info(f"Saved windows for {subj}: X {X.shape}, y {y.shape}")

if __name__ == "__main__":
    generate_all()
