import torch
import torch.nn as nn
import torch.optim as optim



class PeakDetectionModel(nn.Module):
    def __init__(self, signal_length=1024, max_peaks=3, dropout=0.3):
        super(PeakDetectionModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=13)   # kernel size 13 as max width of peak is 10
        self.conv2 = nn.Conv1d(32, 64, kernel_size=13)

        self.pool = nn.MaxPool1d(kernel_size=2, ceil_mode=True)
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear((64 * 500), 2048)
        # self.fc_num_peaks = nn.Linear(64, 1)
        self.fc_positions = nn.Linear(2048, signal_length)
        self.fc_heights = nn.Linear(2048, max_peaks)
        self.fc_width = nn.Linear(2048, max_peaks)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = torch.relu(self.fc1(x))
        # num_peaks = self.fc_num_peaks(x)
        positions = self.sigmoid(self.fc_positions(x))
        heights = torch.relu(self.fc_heights(x))
        width = torch.relu(self.fc_width(x))
        return positions, heights, width



'''class PeakDetectionModel(nn.Module):
    def __init__(self, signal_length=1024, max_peaks=3, dropout=0.3):
        super(PeakDetectionModel, self).__init__()
        self.signal_length = signal_length
        self.max_peaks = max_peaks
        
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        
        # Automatically get dimensin after encoder step
        with torch.no_grad():
            dummy = torch.zeros(1, 1, signal_length)
            out = self.encoder(dummy)
            flatten_dim = out.view(1, -1).shape[1]
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(flatten_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Separate output heads
        self.fc_positions = nn.Linear(1024, signal_length)
        self.fc_heights = nn.Linear(1024, max_peaks)
        self.fc_width = nn.Linear(1024, max_peaks)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        
        # Flatten
        x = x.view(x.shape[0], -1)
        
        # FC layer
        x = self.fc(x)
        
        # Output heads
        positions = self.sigmoid(self.fc_positions(x))
        heights = torch.tanh(self.fc_heights(x)) * 1.5
        widths = torch.relu(self.fc_width(x))
        
        return positions, heights, widths
'''



'''class PeakDetectionModel(nn.Module):
    def __init__(self, signal_length=1024, max_peaks=3, dropout=0.3):
        super(PeakDetectionModel, self).__init__()
        self.signal_length = signal_length
        self.max_peaks = max_peaks
        
        # Feature extractor
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=15),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, ceil_mode=True),
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, signal_length)
            out = self.net(dummy)
            flatten_dim = out.view(1, -1).shape[1]
        
        self.fc = nn.Sequential(
            nn.Linear(flatten_dim, 2048),
            nn.ReLU(),
            nn.Dropout(dropout)  
        )
        
        self.fc_positions = nn.Linear(2048, signal_length)
        self.fc_heights = nn.Linear(2048, max_peaks)
        self.fc_width = nn.Linear(2048, max_peaks)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.net(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        
        positions = self.sigmoid(self.fc_positions(x))
        heights = torch.tanh(self.fc_heights(x)) * 1.5
        width = torch.relu(self.fc_width(x))
        
        return positions, heights, width'''


