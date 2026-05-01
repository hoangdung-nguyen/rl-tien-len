import torch
import torch.nn as nn


class TienLenNet(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        # State (159) + Action (52) = 211
        input_dim = state_shape[0] + action_shape[0]

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Outputs a single scalar score
        )

    def forward(self, state, action):
        # state: [batch, 159], action: [batch, 52]
        x = torch.cat([state, action], dim=-1)
        return self.mlp(x)
