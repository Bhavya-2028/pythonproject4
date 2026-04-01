import torch
import torch.nn as nn

class ECGCritic(nn.Module):
    def __init__(self, seq_len=1250):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(seq_len, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.model(x)