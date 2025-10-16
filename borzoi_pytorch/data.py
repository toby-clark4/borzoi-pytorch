import os
import numpy as np
import pandas as pd
import torch
import h5py
import baskerville_dna as dna
from torch.utils.data import Dataset


def extract_window(chrom: str, pos: int, window_size: int = 524288):
    half_window = (window_size - 1) // 2
    start = max(0, pos - half_window)
    end = pos + half_window
    return chrom, start, end


def make_onehot(genome, chrom: str, start: int, end: int, seq_len: int = 524288):
    if start < 0:
        seq_dna = "N" * (-start) + genome.fetch(chrom, 0, end)
    else:
        seq_dna = genome.fetch(chrom, start, end)

    if len(seq_dna) < seq_len:
        seq_dna += "N" * (seq_len - len(seq_dna))

    seq_1hot = dna.dna_1hot(seq_dna)
    return seq_1hot


def process_sequence(genome, chrom: str, start: int, end: int, seq_len: int = 524288):
    input_seq_len = end - start
    start -= (seq_len - input_seq_len) // 2
    end += (seq_len - input_seq_len) // 2

    onehot_sequence = make_onehot(genome, chrom, start, end, seq_len)

    onehot_sequence = np.transpose(onehot_sequence)  # Gives shape (4, seq_len)

    return onehot_sequence


class BorzoiDataCollator:
    """
    Hugging Face–compatible data collator for Borzoi fine-tuning.
    """

    def __init__(self, genome, seq_len=524288, device=None):
        self.genome = genome
        self.seq_len = seq_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, features):
        """
        Args:
            features (list[dict]): Each example should have keys:
                - 'chrom'
                - 'pos'
                - 'label' (optional)
        Returns:
            dict[str, torch.Tensor]
        """
        batch_inputs = []
        batch_labels = []

        for f in features:
            chrom, pos = f["chrom"], f["pos"]
            label = f.get("label", None)

            # Compute start/end window
            chrom, start, end = extract_window(chrom, pos, self.seq_len)

            # Get one-hot encoded DNA sequence
            seq_1hot = process_sequence(self.genome, chrom, start, end, self.seq_len)

            # Convert to tensor and add batch dimension
            x = torch.tensor(seq_1hot, dtype=torch.float32).unsqueeze(0)  # (1, 4, L)
            batch_inputs.append(x)

            if label is not None:
                batch_labels.append(torch.tensor(label, dtype=torch.float32))

        # Stack into batch tensors
        x_batch = torch.cat(batch_inputs, dim=0)  # .to(self.device)  # (B, 4, L)
        batch = {"x": x_batch}

        if batch_labels:
            batch["labels"] = torch.stack(batch_labels)  # .to(self.device)

        return batch


class BorzoiRegressionDataset(Dataset):
    """
    PyTorch Dataset to load chromosome and position data for single-value-prediction
    """

    def __init__(
        self,
        csv_path: str,
        pos_col: str = "MAPINFO",
        target_col: str = "mean_beta",
        subset_seqs: int = 0,
    ):
        self.data = pd.read_csv(csv_path)
        if subset_seqs > 0:
            self.data = self.data.sample(n=subset_seqs, random_state=42)

        self.chroms = self.data["REFSEQ_chr"].tolist()
        self.positions = self.data[pos_col].tolist()
        self.targets = self.data[target_col].tolist()

    def __len__(self):
        return len(self.chroms)

    def __getitem__(self, idx):
        return {
            "chrom": self.chroms[idx],
            "pos": self.positions[idx],
            "label": self.targets[idx],
        }


class BorzoiH5Dataset(Dataset):
    """
    Pytorch Dataset to load sequences and methylation values from an H5 file
    """

    def __init__(self, h5_path: str):
        assert os.path.exists(h5_path), f"H5 file {h5_path} does not exist"
        self.h5_path = h5_path
        self._file = None  # Allows only opening once per worker

    def _get_file(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __len__(self):
        f = self._get_file()
        n = len(f["sequence"])
        f.close()
        return n

    def __getitem__(self, idx):
        f = self._get_file()
        x = f["sequence"][idx]
        y = f["me_track"][idx]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()
