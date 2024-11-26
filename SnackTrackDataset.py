from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np

class SnackTrackDataset(Dataset):
    """Snack Track dataset."""

    def __init__(self, metadata_dir, transform=None):
        """
        Arguments:
            root_dir (string): Directory with all spectograms.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.metadata = pd.read_csv(metadata_dir)
        self.transform = transform 

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        spectrogram = np.load(row["spectrogram_path"])  # Load spectrogram
        label = row["label"]

        # Map label to numeric
        label_map = {"chewing": 1, "swallowing": 2, "others": 3, "resting": 4}
        label = label_map[label]

        return (torch.tensor(spectrogram, dtype=torch.float32), label)
