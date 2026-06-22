import torch
import torch.nn as nn

class Simple1DCNN(nn.Module):
    """
    1D CNN with 3 conv blocks for time-series classification.
    """
    def __init__(self, input_channels, seq_len, num_classes=2):
        super(Simple1DCNN, self).__init__()
        self.conv_block = nn.Sequential(
            # Block 1
            nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Block 2
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Block 3
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
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