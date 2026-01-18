import torch
import torch.nn as nn

class PeakDetectionModel(nn.Module):
    """
    1D CNN for peak detection in signals.
    Predicts peak positions (binary per sample), heights, and widths for up to max_peaks.
    """
    def __init__(self, signal_length=1024, max_peaks=3, dropout=0.3):
        super(PeakDetectionModel, self).__init__()
        self.signal_length = signal_length
        self.max_peaks = max_peaks
        
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=13),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=13),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, ceil_mode=True),
        )
        
        self.encoder[0].name = "entry_layer"
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, signal_length)
            out = self.encoder(dummy)
            flatten_dim = out.view(1, -1).shape[1]
        
        self.fc = nn.Sequential(
            nn.Linear(flatten_dim, 2048),
            nn.ReLU()
        )
        
        self.fc_positions = nn.Linear(2048, signal_length)
        self.fc_heights = nn.Linear(2048, max_peaks)
        self.fc_widths = nn.Linear(2048, max_peaks)
        
        self.fc_positions.name = "final_layer_positions"
        self.fc_heights.name = "final_layer_heights"
        self.fc_widths.name = "final_layer_widths"
    
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        
        positions = torch.sigmoid(self.fc_positions(x))
        heights = torch.relu(self.fc_heights(x))
        widths = torch.relu(self.fc_widths(x))
        
        return positions, heights, widths