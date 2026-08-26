import numpy as np
import torch
from torch.utils.data import Dataset

class SignalDataset(Dataset):
    def __init__(self, signal_path: str, info_path: str,
                 chunks: int = 1024, max_peaks: int = 3):
        self.signal_path = signal_path
        self.info_path = info_path
        self.chunks = chunks
        self.max_peaks = max_peaks
        
        self.signal_data = self._load_signals()
        self.info_data = self._load_info()
        
        self.signal_mean = 0.0
        self.signal_std = 1.0
    
    def _load_signals(self) -> np.ndarray:
        with open(self.signal_path, "rb") as f:
            signal_data = np.fromfile(f, dtype=np.uint8) # float32
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

        '''
        "signal_id": signal_id,
        "length": length,
        "n_peaks": n_peaks,
        "pos_1": None, "height_1": None, "width_1": None,
        "pos_2": None, "height_2": None, "width_2": None,
        "pos_3": None, "height_3": None, "width_3": None
        '''
        
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