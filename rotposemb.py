import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len, n_rotations=16):
        super().__init__()
        self.n_rotations = n_rotations
        self.max_len = max_len
        self.d_model = d_model
        
        self.basis = nn.Parameter(torch.randn(self.n_rotations, self.d_model))
        self.coeffs = nn.Parameter(torch.randn(self.n_rotations))
        
    def forward(self, x):
        # x = [sent len, batch size]
        position = torch.arange(0, x.size(0), dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * -(np.log(self.max_len) / self.d_model))
        pos = position.unsqueeze(-1) * div_term.unsqueeze(0)
        
        rotations = torch.einsum('ij,j->ij', self.basis, self.coeffs)
        rotations = rotations.sum(dim=0)
        
        rotary_pos = torch.einsum('ij,j->ij', pos, rotations[:x.size(0)])
        
        return rotary_pos
