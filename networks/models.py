import torch
import torch.nn as nn
import torch.optim as optim


class SimpleSignalNet(nn.Module):
    """A minimal feed-forward network for signal regression/classification, 
    basically a simple multi layer perception, just to test."""
    def __init__(self, input_size=1024, hidden_size=128, output_size=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        return self.model(x)
    


class SignalNet(nn.Module):
    def __init__(self, input_size=1024, hidden_size=128, output_size=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        return self.model(x)
