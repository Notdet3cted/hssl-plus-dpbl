import torch
import torch.nn as nn

class HSSLEncoder(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=64, out_dim=128):
        """
        input_channels: 3 untuk WESAD (misal: EDA, BVP, TEMP atau channel gabungan yang akan digunakan)
        """
        super(HSSLEncoder, self).__init__()
        
        # 1. Micro-Representation Level (Local features / Short temporal patterns)
        self.micro_encoder = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU()
        )
        
        # 2. Macro-Representation Level (Global features / Long temporal context)
        # Using dilated convolutions to capture broader context without losing resolution,
        # then aggregating globally.
        self.macro_encoder = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # Aggregating temporal dimension
        )
        
        # Projection Head for Contrastive Learning
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, out_dim)
        )

    def forward(self, x):
        # x shape: (Batch, Channels, Sequence_Length)
        
        # Extract Micro-representations
        micro_h = self.micro_encoder(x)
        
        # Extract Macro-representations
        macro_h = self.macro_encoder(micro_h)
        
        # Flatten for projection
        h = macro_h.squeeze(-1) # shape: (Batch, hidden_dim * 4)
        
        # Non-linear projection
        z = self.projection_head(h)
        
        # Returning BOTH micro and macro representations, along with projection
        return micro_h, h, z
