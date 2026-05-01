import torch
import torch.nn as nn

class TienLenNet(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        # State (159) + Action (67) = 226
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
        x = torch.cat([state, action], dim=-1)
        return self.mlp(x)

class TienLenLSTMNet(nn.Module):
    def __init__(self, state_dim=159, action_dim=60):
        super().__init__()
        self.lstm = nn.LSTM(input_size=action_dim, hidden_size=128, batch_first=True)

        # Input = LSTM output (128) + Snapshot Obs (159) + Current Action (60)
        input_dim = 128 + state_dim + action_dim
        self.dense1 = nn.Linear(input_dim, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)  # Outputs the Q-value (Expected Payoff)

    def forward(self, z, x_state, x_action):
        """
        z: History sequence [batch, 15, 60]
        x_state: Current snapshot [batch, 159]
        x_action: Current action features [batch, 60]
        """
        # lstm_out: [batch, 15, 128], (h_n, c_n): [1, batch, 128]
        _, (h_n, _) = self.lstm(z)
        lstm_out = h_n[-1]

        x = torch.cat([lstm_out, x_state, x_action], dim=-1)
        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        x = torch.relu(self.dense3(x))
        x = torch.relu(self.dense4(x))
        x = torch.relu(self.dense5(x))
        return self.dense6(x)
