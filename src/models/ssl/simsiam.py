import torch
import torch.nn as nn

class SSLEncoder(nn.Module):
    """
    SSL baseline encoder using SimSiam-style self-supervised learning.
    No hierarchical structure — simple CNN encoder + projection.
    Used as the 'no HSSL' baseline for ablation study.
    """
    def __init__(self, input_channels=3, hidden_dim=64, out_dim=128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv1d(input_channels, hidden_dim, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Block 2
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Block 3
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, out_dim)
        )
        
        self.prediction = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.BatchNorm1d(out_dim // 2),
            nn.ReLU(),
            nn.Linear(out_dim // 2, out_dim)
        )
    
    def forward(self, x):
        h = self.encoder(x)
        h = h.squeeze(-1)
        z = self.projection(h)
        p = self.prediction(z)
        return h, z, p


class SimSiamLoss(nn.Module):
    """Negative cosine similarity (SimSiam loss)."""
    def forward(self, p1, z2, p2, z1):
        def neg_cos_sim(p, z):
            z = z.detach()  # stop gradient
            p = nn.functional.normalize(p, dim=1)
            z = nn.functional.normalize(z, dim=1)
            return -(p * z).sum(dim=1).mean()
        return (neg_cos_sim(p1, z2) + neg_cos_sim(p2, z1)) / 2