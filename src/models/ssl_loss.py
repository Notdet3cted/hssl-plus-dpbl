import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = torch.matmul(representations, representations.T)
        
        logits = similarity_matrix / self.temperature
        logits.fill_diagonal_(-float('inf'))
        
        targets = torch.empty(2 * batch_size, dtype=torch.long, device=z_i.device)
        targets[:batch_size] = torch.arange(batch_size, 2 * batch_size, device=z_i.device)
        targets[batch_size:] = torch.arange(batch_size, device=z_i.device)
        
        loss = self.criterion(logits, targets)
        return loss