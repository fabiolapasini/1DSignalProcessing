
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


class SignalDataset(Dataset):
    """
    PyTorch Dataset per il caricamento e l'augmentazione dei segnali
    da file binari (.raw) e delle corrispondenti info.
    """
    def __init__(self, signal_path: str, info_path: str,
                chunks: int = 1024, max_peaks: int = 3,
                augment: bool = False):
        """
        Parameters:
            signal_path (str): Path to the signal binary file.
            info_path (str): Path to the info binary file.
            chunks (int): Signal length per sample.
            max_peaks (int): Maximum number of peaks parsed from info.
            augment (bool): Whether to include flipped data augmentation.
        """
        self.signal_path = signal_path
        self.info_path = info_path
        self.chunks = chunks
        self.max_peaks = max_peaks
        self.augment = augment

        # Load data
        self.signal_data = self._load_signals()
        self.info_data = self._load_info()

    # ------------------------- DATA LOADING -------------------------

    def _load_signals(self) -> np.ndarray:
        """Load raw signal data and reshape into chunks."""
        with open(self.signal_path, "rb") as f:
            signal_data = np.fromfile(f, dtype=np.uint8)
            # List comprehension
            # signal_data = [signal_data[i*self.chunks: (i+1)*self.chunks] for i in range(len(signal_data)//self.chunks)]
            signal_data = signal_data.reshape(-1, self.chunks).astype(np.float32)
        return signal_data

    def _load_info(self) -> np.ndarray:
        """Load raw info data."""
        with open(self.info_path, "rb") as f:
            info_data = np.fromfile(f, dtype=np.float32)
            # 11 elem for each signal
            info_data = info_data.reshape(-1, 11).astype(np.float32)
        return info_data

    # ------------------------- PYTORCH API -------------------------

    def __len__(self):
        return len(self.signal_data)

    '''
    __getitem__() is a special method in Python that allows us to access an element 
    from an object using square brackets, similar to how we access items 
    in a list, tuple, or dictionary. It is commonly used to retrieve items 
    from containers or objects that support indexing or key-based access
    '''
    def __getitem__(self, idx):
        signal = self.signal_data[idx]
        info = self.info_data[idx]

        # --- Random vertical flip ---
        if self.augment and np.random.rand() < 0.5:
            signal = -signal
            info = info.copy()
            num_peaks = int(info[1])
            for i in range(num_peaks):
                info[3 + i * 3] *= -1  # invert height only

        # --- Normalize ---
        # signal = (signal - signal.mean()) / (signal.std() + 1e-8)

        # --- Parse targets ---
        num_peaks = int(info[1])
        peak_pos = torch.zeros(self.chunks, dtype=torch.float32)
        peak_height = torch.zeros(self.max_peaks, dtype=torch.float32)
        peak_width = torch.zeros(self.max_peaks, dtype=torch.float32)
        for i in range(num_peaks):
            pos = round(info[2 + i * 3])
            if pos < self.chunks:
                peak_pos[pos] = 1
            peak_height[i] = float(info[3 + i * 3])
            peak_width[i] = float(info[4 + i * 3])

        return (
            torch.from_numpy(signal),
            peak_pos,
            peak_height,
            peak_width
        )


