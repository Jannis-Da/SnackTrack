from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np

class SnackTrackDataset(Dataset):
    """Snack Track dataset."""

    def __init__(self, metadata):
        """
        Args:
            metadata: (np.array) Metadata of the dataset.
        """
        self.metadata = metadata

    def __len__(self):
        """
        Returns length of the dataset.
        """
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Returns a sample from the dataset.
        Args:
            idx: (int) Index of the sample.
        """
        row = self.metadata.iloc[idx]
        spectrogram = np.load(row["spectrogram_path"])  # Load spectrogramd
        label = row["label"]

        # Map label to numeric
        label_map = {"chewing": 0, "swallowing": 1, "others": 2, "resting": 3}
        label = label_map[label]

        spectrogram = np.expand_dims(spectrogram, axis=0)  # Add channel dimension

        return (torch.tensor(spectrogram, dtype=torch.float32), label)
