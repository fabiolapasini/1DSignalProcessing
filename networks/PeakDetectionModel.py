import torch
import torch.nn as nn
import torch.optim as optim


class PeakDetectionModel(nn.Module):
    def __init__(self, signal_length=1024, max_peaks=3, dropout=0.3):
        super(PeakDetectionModel, self).__init__()
        self.signal_length = signal_length
        self.max_peaks = max_peaks

        # Feature extractor
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15),
            nn.ReLU(),
            # nn.Dropout(dropout / 2),  
            nn.Conv1d(32, 64, kernel_size=15),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, ceil_mode=True),
            # nn.Dropout(dropout / 2)
        )

        '''
        (Lout = Lin - kernel_size + 1)
        after conv1: 1024 - 15 + 1 = 1010 
        after conv2: 1012 - 15 + 1 = 998
        after pool:  998 / 2 = 498

        with torch.no_grad():
            dummy = torch.zeros(1, 1, signal_length)
            out = self.net(dummy)
            flatten_dim = out.view(1, -1).shape[1]
        '''

        # Fully connected intermediate layer
        self.fc = nn.Sequential(
            nn.Linear((64 * 498), 2048),
            nn.ReLU()
        )

        # Output heads
        self.fc_positions = nn.Linear(2048, signal_length)  # BCELoss target: [batch, signal_length]
        self.fc_heights = nn.Linear(2048, max_peaks)        # L1Loss target: [batch, max_peaks]
        self.fc_width = nn.Linear(2048, max_peaks)          # L1Loss target: [batch, max_peaks]
        self.sigmoid = nn.Sigmoid()                         # Only for positions

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.shape[0], -1)  # flatten
        x = self.fc(x)

        positions = self.sigmoid(self.fc_positions(x))  # [batch, signal_length]
        heights = torch.relu(self.fc_heights(x))        # [batch, max_peaks]
        width = torch.relu(self.fc_width(x))            # [batch, max_peaks]

        return positions, heights, width
