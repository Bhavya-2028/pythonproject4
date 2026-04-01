import torch
import torch.nn as nn

class ECGGenerator(nn.Module):
    def __init__(self, latent_dim=100, seq_len=1250):
        super().__init__()
        self.seq_len = seq_len

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, seq_len),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)