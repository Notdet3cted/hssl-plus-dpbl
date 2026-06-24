import torch
import torch.nn as nn
from src.experiment_tracker import ExperimentTracker

class Simple1DCNN(nn.Module):
    """
    Flexible 1D CNN for time-series classification.
    Channel sizes and kernel sizes are config-driven.
    Default: [64, 128, 256] channels, [7, 5, 3] kernels.
    """
    def __init__(self, input_channels, seq_len, num_classes=2,
                 channels=None, kernels=None, dropout=0.4):
        super(Simple1DCNN, self).__init__()
        
        # Load from config if not specified
        if channels is None or kernels is None:
            try:
                tracker = ExperimentTracker()
                cfg = tracker.config.get("models", {}).get("cnn", {})
                if channels is None:
                    channels = cfg.get("channels", [64, 128, 256])
                if kernels is None:
                    kernels = cfg.get("kernels", [7, 5, 3])
                if dropout is None:
                    dropout = cfg.get("dropout", 0.4)
            except Exception:
                channels = [64, 128, 256]
                kernels = [7, 5, 3]
                dropout = 0.4
        
        self.channels = channels
        self.kernels = kernels
        self.dropout = dropout
        
        # Build conv blocks dynamically
        blocks = []
        in_ch = input_channels
        for i, (ch, ks) in enumerate(zip(channels, kernels)):
            padding = ks // 2
            stride = 2 if i == 0 else 1
            blocks.extend([
                nn.Conv1d(in_ch, ch, kernel_size=ks, stride=stride, padding=padding),
                nn.BatchNorm1d(ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ])
            in_ch = ch
        
        # Adaptive pooling collapses time → 1
        blocks.append(nn.AdaptiveAvgPool1d(1))
        self.conv_block = nn.Sequential(*blocks)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        x: (Batch, Channels, Time)
        """
        features = self.conv_block(x)
        features = features.squeeze(-1)
        logits = self.classifier(features)
        return logits

    def forward_features(self, x):
        """Extract features before classifier (for embeddings)."""
        features = self.conv_block(x)
        features = features.squeeze(-1)
        return features