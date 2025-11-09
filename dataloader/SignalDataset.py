
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

        # Perform augmentation if requested
        if self.augment:
            self.signal_data, self.info_data = self._augment_with_flipped_data()

    # ------------------------- DATA LOADING -------------------------

    def _load_signals(self) -> np.ndarray:
        """Load raw signal data and reshape into chunks."""
        with open(self.signal_path, "rb") as f:
            signal_data = np.fromfile(f, dtype=np.uint8)
        signal_data = signal_data.reshape(-1, self.chunks).astype(np.float32)
        return signal_data

    def _load_info(self) -> np.ndarray:
        """Load raw info data."""
        with open(self.info_path, "rb") as f:
            info_data = np.fromfile(f, dtype=np.float32)
        return info_data

    # ------------------------- INFO PARSING -------------------------

    def _parse_info_array(self, info_array: np.ndarray) -> pl.DataFrame:
        """Parse info array into structured Polars DataFrame."""
        parsed = []
        i = 0
        signal_id = 0
        total_len = len(info_array)

        while i < total_len:
            if i + 1 >= total_len:
                break
            length = int(info_array[i])
            n_peaks = int(info_array[i + 1])
            i += 2

            entry = {
                "signal_id": signal_id,
                "length": length,
                "n_peaks": n_peaks,
                "pos_1": None, "height_1": None, "width_1": None,
                "pos_2": None, "height_2": None, "width_2": None,
                "pos_3": None, "height_3": None, "width_3": None,
            }

            for p in range(min(n_peaks, self.max_peaks)):
                if i + 2 >= total_len:
                    break
                entry[f"pos_{p+1}"] = float(info_array[i])
                entry[f"height_{p+1}"] = float(info_array[i + 1])
                entry[f"width_{p+1}"] = float(info_array[i + 2])
                i += 3

            parsed.append(entry)
            signal_id += 1

        return pl.DataFrame(parsed)

    # ------------------------- AUGMENTATION -------------------------

    def _augment_with_flipped_data(self):
        """Create augmented version (original + flipped)."""
        flipped_signals = -self.signal_data
        df_info = self._parse_info_array(self.info_data)

        flipped_info = df_info.with_columns(
            (pl.col("signal_id") + len(df_info)).alias("signal_id")
        )

        # Flip all height columns
        for col in flipped_info.columns:
            if "height" in col:
                flipped_info = flipped_info.with_columns(pl.col(col) * -1)

        augmented_signals = np.concatenate(
            [self.signal_data, flipped_signals], axis=0
        )
        augmented_info = pl.concat([df_info, flipped_info], how="vertical")

        print(f"[INFO] Augmentation complete → signals: {augmented_signals.shape}, info: {augmented_info.shape}")
        return augmented_signals, augmented_info

    # ------------------------- PYTORCH API -------------------------

    def __len__(self):
        return len(self.signal_data)

    def __getitem__(self, idx):
        signal = torch.tensor(self.signal_data[idx], dtype=torch.float32)
        # Optionally, convert relevant info to tensor
        return signal
