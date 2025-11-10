import torch
import torch.nn as nn
import torch.optim as optim


class PeakDetectionModel(nn.Module):
    def __init__(self, signal_length=1024, max_peaks=3):
        super(PeakDetectionModel, self).__init__()
        self.signal_length = signal_length
        self.max_peaks = max_peaks

        # Feature extractor
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=13),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=13),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, ceil_mode=True),
        )

        # Compute the flattened size after conv+pool
        # Assuming input length = signal_length
        dummy = torch.zeros(1, 1, signal_length)
        with torch.no_grad():
            out = self.net(dummy)
            flattened_size = out.shape[1] * out.shape[2]

        # Fully connected intermediate layer
        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 2048),
            nn.ReLU()
        )

        # Output heads
        self.fc_positions = nn.Linear(2048, signal_length)  # BCELoss target: [batch, signal_length]
        self.fc_heights = nn.Linear(2048, max_peaks)        # L1Loss target: [batch, max_peaks]
        self.fc_width = nn.Linear(2048, max_peaks)          # L1Loss target: [batch, max_peaks]
        self.sigmoid = nn.Sigmoid()                         # Only for positions

    def forward(self, x):
        features = self.net(x)
        features = features.view(features.shape[0], -1)  # flatten
        features = self.fc(features)

        positions = self.sigmoid(self.fc_positions(features))  # [batch, signal_length]
        heights = torch.relu(self.fc_heights(features))        # [batch, max_peaks]
        width = torch.relu(self.fc_width(features))            # [batch, max_peaks]

        return positions, heights, width
