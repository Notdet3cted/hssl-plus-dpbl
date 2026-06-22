import torch
import torch.nn as nn
import numpy as np

class DPBL(nn.Module):
    """
    Domain-Projected Behavior Learning (DPBL) Module.
    Takes generic embeddings (e.g., from HSSL) and adapts them based on personalized baseline behavior.
    """
    def __init__(self, embedding_dim=256, hidden_dim=128):
        super(DPBL, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Projection layer to compute deviation representations
        self.deviation_projector = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim), # Takes [generic_emb, generic_emb - baseline]
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Gating mechanism to merge generic and deviation representations
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid()
        )

    def forward(self, embeddings, baseline):
        """
        embeddings: Tensor of shape (Batch, embedding_dim)
        baseline: Tensor of shape (1, embedding_dim) or (Batch, embedding_dim)
        """
        # 1. Compute physical deviation
        deviation = embeddings - baseline
        
        # 2. Project deviation
        combined_input = torch.cat([embeddings, deviation], dim=1)
        projected_deviation = self.deviation_projector(combined_input)
        
        # 3. Gating mechanism
        gate_input = torch.cat([embeddings, projected_deviation], dim=1)
        g = self.gate(gate_input)
        
        # 4. Personalized Representation
        personalized_emb = g * embeddings + (1 - g) * projected_deviation
        
        return personalized_emb

import pickle

class BaselineTracker:
    """
    Tracks and computes the personalized baseline for subjects.
    For stress detection, baseline is typically computed from the first few minutes (or a baseline phase)
    of the subject's data.
    """
    def __init__(self):
        self.baselines = {}
        
    def update_baseline(self, subject_id, embeddings):
        """
        embeddings: numpy array of shape (N, embedding_dim)
        Updates or sets the baseline using the mean of provided embeddings.
        """
        current_mean = np.mean(embeddings, axis=0)
        
        if subject_id in self.baselines:
            # Moving average or simple overwrite depending on use-case.
            # Here we just overwrite with the pure baseline session representations.
            self.baselines[subject_id] = current_mean
        else:
            self.baselines[subject_id] = current_mean
            
        return self.baselines[subject_id]
        
    def get_baseline(self, subject_id):
        if subject_id not in self.baselines:
            raise ValueError(f"Baseline for {subject_id} not initialized.")
        return self.baselines[subject_id]

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.baselines, f)
            
    def load(self, path):
        with open(path, 'rb') as f:
            self.baselines = pickle.load(f)
