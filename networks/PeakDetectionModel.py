import torch
import torch.nn as nn

class PeakDetectionModel(nn.Module):
    def __init__(self, max_signal_length=1024, n_anchors=1):
        super(PeakDetectionModel, self).__init__()
        self.max_signal_length = max_signal_length
        self.n_anchors = n_anchors
        
        # Encoder condiviso
        self.encoder = nn.Sequential(
            # Input shape: (b, 1, 1024)
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            # Output shape: (b, 32, 512)
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            # Output shape: (b, 64, 256)
            
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            # Output shape: (b, 64, 128)
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            # Output shape: (b, 128, 64)
            
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            # Output shape: (b, 128, 32)
        )
        
        # Output heads separati
        self.output_positions = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=n_anchors, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        self.output_heights = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=n_anchors, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        self.output_widths = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=n_anchors, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Input: (batch, 1, 1024)
        x = self.encoder(x)
        # x shape: (batch, 128, 32)
        
        positions = self.output_positions(x) 
        heights = self.output_heights(x)      
        widths = self.output_widths(x) 
        
        return positions, heights, widths
