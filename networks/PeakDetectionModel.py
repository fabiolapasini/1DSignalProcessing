import torch
import torch.nn as nn

class PeakDetectionModel(nn.Module):
    def __init__(self, max_signal_length=1024, n_anchors=1):
        super(PeakDetectionModel, self).__init__()
        self.max_signal_length = max_signal_length
        self.n_anchors = n_anchors
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        
        self.output_positions = nn.Conv1d(in_channels=128, out_channels=n_anchors, kernel_size=3, padding=1)
        
        self.output_heights = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=n_anchors, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.output_widths = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=n_anchors, kernel_size=3, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        
        positions = self.output_positions(x)
        heights = self.output_heights(x)
        widths = self.output_widths(x)
        
        return positions, heights, widths