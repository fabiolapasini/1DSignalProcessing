import numpy as np
import torch
from torch.utils.data import Dataset


class SignalDataset(Dataset):
    def __init__(
        self,
        signal_path: str,
        info_path: str,
        max_signal_length: int = 1024,
        max_peaks: int = 3
    ):
        self.signal_path = signal_path
        self.info_path = info_path
        self.max_signal_length = max_signal_length
        self.max_peaks = max_peaks

        self.signal_data = self._load_signals()
        self.info_data = self._load_info()

    def _load_signals(self) -> np.ndarray:
        with open(self.signal_path, "rb") as f:
            data = np.fromfile(f, dtype=np.uint8)
            data = data.reshape(-1, self.max_signal_length)
        return data

    def _load_info(self) -> np.ndarray:
        with open(self.info_path, "rb") as f:
            data = np.fromfile(f, dtype=np.float32)
            data = data.reshape(-1, 2 + self.max_peaks * 3)
        return data

    def __len__(self):
        return len(self.signal_data)

    def __getitem__(self, idx):
        info = self.info_data[idx]
        raw_signal = self.signal_data[idx]

        signal_length = int(info[0])
        n_peaks = int(info[1])

        signal = raw_signal[:signal_length].astype(np.float32)
        peaks = info[2:].reshape(self.max_peaks, 3)[:n_peaks]

        return {
            "signal": torch.from_numpy(signal),
            "peaks": torch.from_numpy(peaks)
        }
    


def collate_fn(batch):
    """Custom collate function to handle variable-length signals and peaks."""
    signals = [item["signal"] for item in batch]
    peaks_list = [item["peaks"] for item in batch]
    
    # Pad signals to max length in batch
    max_signal_len = max(s.size(0) for s in signals)
    padded_signals = torch.zeros(len(signals), max_signal_len)
    signal_lengths = torch.zeros(len(signals), dtype=torch.long)
    
    for i, s in enumerate(signals):
        length = s.size(0)
        padded_signals[i, :length] = s
        signal_lengths[i] = length
    
    # Pad peaks to max number in batch
    max_peaks = max(p.size(0) for p in peaks_list)
    if max_peaks == 0:
        max_peaks = 1  # Evita errori con batch senza picchi
    
    padded_peaks = torch.zeros(len(peaks_list), max_peaks, 3)
    n_peaks = torch.zeros(len(peaks_list), dtype=torch.long)
    
    for i, p in enumerate(peaks_list):
        if p.size(0) > 0:
            padded_peaks[i, :p.size(0), :] = p
            n_peaks[i] = p.size(0)
    
    return {
        "signal": padded_signals,
        "peaks": padded_peaks,
        "signal_lengths": signal_lengths,
        "n_peaks": n_peaks
    }

