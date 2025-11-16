import numpy as np
import torch
from torch.utils.data import Dataset

class SignalDataset(Dataset):
    def __init__(self, signal_path: str, info_path: str,
                 chunks: int = 1024, max_peaks: int = 3,
                 normalize: bool = False):
        self.signal_path = signal_path
        self.info_path = info_path
        self.chunks = chunks
        self.max_peaks = max_peaks
        self.normalize = normalize
        
        self.signal_data = self._load_signals()
        self.info_data = self._load_info()
        
        assert len(self.signal_data) == len(self.info_data), \
            f"Mismatch: {len(self.signal_data)} signals vs {len(self.info_data)} infos"
        
        if self.normalize:
            self.signal_mean = self.signal_data.mean()
            self.signal_std = self.signal_data.std()
        else:
            self.signal_mean = 0.0
            self.signal_std = 1.0
    
    def _load_signals(self) -> np.ndarray:
        with open(self.signal_path, "rb") as f:
            signal_data = np.fromfile(f, dtype=np.uint8)
            signal_data = signal_data.reshape(-1, self.chunks).astype(np.float32)
        return signal_data
    
    def _load_info(self) -> np.ndarray:
        with open(self.info_path, "rb") as f:
            info_data = np.fromfile(f, dtype=np.float32)
            info_data = info_data.reshape(-1, 11).astype(np.float32)
        return info_data
    
    def __len__(self):
        return len(self.signal_data)
    
    def __getitem__(self, idx):
        signal = self.signal_data[idx].copy()
        info = self.info_data[idx].copy()
        
        if self.normalize:
            signal = (signal - self.signal_mean) / (self.signal_std + 1e-8)
        
        num_peaks = min(int(info[1]), self.max_peaks) 
        peak_pos = torch.zeros(self.chunks, dtype=torch.float32)
        peak_height = torch.zeros(self.max_peaks, dtype=torch.float32)
        peak_width = torch.zeros(self.max_peaks, dtype=torch.float32)
        
        for i in range(num_peaks):
            pos = int(round(info[2 + i * 3])) 
            if 0 <= pos < self.chunks:
                peak_pos[pos] = 1.0
            
            peak_height[i] = float(info[3 + i * 3])
            peak_width[i] = float(info[4 + i * 3])
        
        return (
            torch.from_numpy(signal).float(),
            peak_pos,
            peak_height,
            peak_width
        )