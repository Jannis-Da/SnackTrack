from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np

class SnackTrackDataset(Dataset):
    """Snack Track dataset."""

    def __init__(self, metadata, transform=None):
        """
        Arguments:
            root_dir (string): Directory with all spectograms.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.metadata = metadata
        self.transform = transform 

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        spectrogram = np.load(row["spectrogram_path"])  # Load spectrogramd
        label = row["label"]

        # Map label to numeric
        label_map = {"chewing": 0, "swallowing": 1, "others": 2, "resting": 3}
        label = label_map[label]

        spectrogram = np.expand_dims(spectrogram, axis=0)  # Add channel dimension

        return (torch.tensor(spectrogram, dtype=torch.float32), label)
