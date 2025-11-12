import torch
import torch.nn as nn
import torch.optim as optim


class SimpleSignalNet(nn.Module):
    def __init__(self, signal_length=1024, hidden_size=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(signal_length, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Output layer
        self.pos_head = nn.Linear(hidden_size // 2, signal_length)
        self.height_head = nn.Linear(hidden_size // 2, 3)
        self.width_head = nn.Linear(hidden_size // 2, 3)

    def forward(self, x):
        x = self.net(x)
        pos = torch.sigmoid(self.pos_head(x))
        height = self.height_head(x)
        width = self.width_head(x)
        return pos, height, width
    