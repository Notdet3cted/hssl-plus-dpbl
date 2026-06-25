#!/usr/bin/env python3
"""
Kaggle Setup Script for WESAD HSSL+DPBL Pipeline

This script adapts the project for Kaggle environment:
- Updates config paths to use /kaggle/working/ for outputs
- Creates symlinks if needed
- Sets up environment variables
- Installs dependencies if needed

Usage:
    python kaggle_setup.py [--config-path config/config.yaml]
"""

import os
import sys
import yaml
import shutil
import argparse
from pathlib import Path


def setup_kaggle_paths(config_path="config/config.yaml"):
    """Update config.yaml paths for Kaggle environment."""
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Kaggle-specific path adjustments
    kaggle_base = "/kaggle/working"
    
    # Update paths to use Kaggle working directory for outputs
    path_updates = {
        "checkpoints": os.path.join(kaggle_base, "checkpoints"),
        "embeddings": os.path.join(kaggle_base, "embeddings"),
        "embeddings_hssl": os.path.join(kaggle_base, "embeddings", "hssl"),
        "embeddings_ssl": os.path.join(kaggle_base, "embeddings", "ssl"),
        "embeddings_hssl_dpbl": os.path.join(kaggle_base, "embeddings", "hssl_dpbl"),
        "embeddings_ssl_dpbl": os.path.join(kaggle_base, "embeddings", "ssl_dpbl"),
        "logs": os.path.join(kaggle_base, "logs"),
        "reports": os.path.join(kaggle_base, "reports"),
        "results": os.path.join(kaggle_base, "results"),
        "results_predictions": os.path.join(kaggle_base, "results", "predictions"),
        "experiments": os.path.join(kaggle_base, "experiments"),
        "dashboard": os.path.join(kaggle_base, "dashboard"),
    }
    
    # Keep input data paths as-is (Kaggle input datasets)
    if "paths" not in config:
        config["paths"] = {}
    
    config["paths"].update(path_updates)
    
    # Add Kaggle-specific config
    config["kaggle"] = {
        "input_data_path": "/kaggle/input/wesad-processed",  # Adjust as needed
        "use_gpu": True,
        "max_runtime_hours": 12,
    }
    
    # Write updated config
    backup_path = config_path + ".backup"
    shutil.copy2(config_path, backup_path)
    print(f"Backup saved to {backup_path}")
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Updated config saved to {config_path}")
    print("Kaggle paths configured:")
    for key, val in path_updates.items():
        print(f"  {key}: {val}")
    
    return config


def create_directories(config):
    """Create all necessary directories in Kaggle environment."""
    for key, path in config["paths"].items():
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")


def install_dependencies():
    """Install required packages for Kaggle."""
    import subprocess
    
    # Check if we're in Kaggle
    if not os.path.exists("/kaggle"):
        print("Not in Kaggle environment, skipping dependency install")
        return
    
    # Core dependencies (most should be pre-installed in Kaggle)
    packages = [
        "torch",
        "torchvision",
        "torchaudio",
        "scikit-learn",
        "scipy",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "plotly",
        "pyyaml",
        "tqdm",
    ]
    
    for pkg in packages:
        try:
            __import__(pkg.replace("-", "_"))
            print(f"{pkg} already installed")
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
    
    print("Dependencies ready")


def setup_symlinks():
    """Set up symlinks for data if needed."""
    # If input data is in /kaggle/input, create symlink to data/raw
    input_path = "/kaggle/input/wesad-processed"
    raw_path = "data/raw"
    
    if os.path.exists(input_path) and not os.path.exists(raw_path):
        os.makedirs("data", exist_ok=True)
        os.symlink(input_path, raw_path)
        print(f"Created symlink: {raw_path} -> {input_path}")
    elif os.path.exists(input_path):
        print(f"Data directory already exists at {raw_path}")


def main():
    parser = argparse.ArgumentParser(description="Setup WESAD HSSL+DPBL for Kaggle")
    parser.add_argument("--config-path", default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--skip-install", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-dirs", action="store_true", help="Skip directory creation")
    args = parser.parse_args()
    
    print("=" * 60)
    print("KAGGLE SETUP FOR WESAD HSSL+DPBL")
    print("=" * 60)
    
    # Update config
    config = setup_kaggle_paths(args.config_path)
    
    # Create directories
    if not args.skip_dirs:
        create_directories(config)
    
    # Setup symlinks
    setup_symlinks()
    
    # Install dependencies
    if not args.skip_install:
        install_dependencies()
    
    print("\n" + "=" * 60)
    print("KAGGLE SETUP COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run data preparation: python -m src.data_preprocessing")
    print("  2. Or run full pipeline: python run_pipeline.py --mode all")
    print("\nNote: Adjust --epochs and --ssl_epochs for Kaggle time limits")


if __name__ == "__main__":
    main()