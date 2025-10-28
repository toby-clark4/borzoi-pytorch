import os
import random
import numpy as np
import pandas as pd
import torch
import h5py
from . import baskerville_dna as dna
from torch.utils.data import Dataset


def extract_window(chrom: str, pos: int, window_size: int = 524288):
    half_window = (window_size) // 2
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


def process_sequence(genome, chrom: str, start: int, end: int, seq_len: int = 524288, augment = False):
    input_seq_len = end - start
    start -= (seq_len - input_seq_len) // 2
    end += (seq_len - input_seq_len) // 2

    onehot_sequence = make_onehot(genome, chrom, start, end, seq_len)

    if augment:
        onehot_sequence = get_shift_and_augment(onehot_sequence).copy()

    onehot_sequence = np.transpose(onehot_sequence)  # Gives shape (4, seq_len)

    return onehot_sequence

def get_shift_and_augment(onehot_sequence):
    if random.random() < 0.5:
        fwdrc=True # Use forward
    else:
        fwdrc=False # Use reverse complement
    shift = random.randint(-3, 3)

    onehot_sequence = dna.hot1_augment(onehot_sequence, fwdrc=fwdrc, shift=shift)

    return onehot_sequence


def make_variant_onehot(genome, chrom: str, start: int, end: int, allele1: str, allele2: str, snp_pos: int, seq_len: int = 524288):
    """
    Takes information about a SNP variant and returns one-hot encoded sequences of each allele. 
    """
    if start < 0:
        seq_dna = "N" * (-start) + genome.fetch(chrom, 0, end)
    else:
        seq_dna = genome.fetch(chrom, start, end)
    
    if len(seq_dna) < seq_len:
        seq_dna += "N" * (seq_len - len(seq_dna))
    
    snp_idx = snp_pos - start

    allele1_seq = seq_dna[:snp_idx] + allele1 + seq_dna[snp_idx+1:]
    allele2_seq = seq_dna[:snp_idx] + allele2 + seq_dna[snp_idx+1:]

    allele1_1hot = dna.dna_1hot(allele1_seq)
    allele2_1hot = dna.dna_1hot(allele2_seq)

    return allele1_1hot, allele2_1hot


def process_variant(genome, chrom: str, start: int, end: int, snp_pos: int, allele1: str, allele2: str, seq_len: int = 524288):
    """
    Formats sequences to the right length, processes and transposes to the right shape.
    """
    input_seq_len = end - start
    start -= (seq_len - input_seq_len) // 2
    end += (seq_len - input_seq_len) // 2

    allele1_1hot, allele2_1hot = make_variant_onehot(genome, chrom, start, end, allele1, allele2, snp_pos, seq_len)

    allele1_1hot = np.transpose(allele1_1hot)
    allele2_1hot = np.transpose(allele2_1hot)

    return allele1_1hot, allele2_1hot


def bin_methylation(window_sites: pd.DataFrame, start_coordinate: int = 0, seq_len: int = 524_288, window_to_predict: int = 196_608, resolution: int=32):
    """
    Takes a dataframe of methylation sites in a window and maps to a numpy array
    with methylation values binned at the specified resolution (in bp)
    """
    target_length = seq_len // resolution
    out_channel = np.zeros(target_length)
    window_sites['bin_loc'] = (window_sites.index - start_coordinate) // 32
    for bin_loc, group in window_sites.groupby('bin_loc'):
        out_channel[bin_loc] = group['mean_beta'].mean() # type: ignore

    # Extract the centre window output by the model
    _, start, end = extract_window('N', pos = target_length // 2, window_size = window_to_predict // resolution)
    out_channel = out_channel[start:end]

    return out_channel
  

class BorzoiDataCollator:
    """
    Hugging Face–compatible data collator for Borzoi fine-tuning.
    """

    def __init__(self, genome, model, seq_len=524288, device=None):
        self.genome = genome
        self.seq_len = seq_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model # Used to toggle data augmentation

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
            # Apply augmentation only during training
            seq_1hot = process_sequence(self.genome, chrom, start, end, self.seq_len, augment=self.model.training)

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


class BorzoiVariantDataCollator(BorzoiDataCollator):
    """
    Hugging Face–compatible data collator for Borzoi fine-tuning.
    """

    def __init__(self, genome, siamese = False, seq_len=524288, device=None):
        super().__init__(genome, seq_len, device)
        self.siamese = siamese

    def __call__(self, features):
        """
        Args:
            features (list[dict]): Each example should have keys:
                - 'chrom'
                - 'snp_pos'
                - 'cpg_pos'
                - 'allele1'
                - 'allele2'
                - 'label' (optional)
        Returns:
            dict[str, torch.Tensor]
        """
        batch_inputs = []
        batch_labels = []

        for f in features:
            chrom, snp_pos, cpg_pos, allele1, allele2 = f["chrom"], f["snp_pos"], f["cpg_pos"], f["allele1"], f["allele2"]
            label = f.get("label", None)

            # Compute start/end window
            chrom, start, end = extract_window(chrom, cpg_pos, self.seq_len)

            # Get one-hot encoded DNA sequence
            allele1_1hot, allele2_1hot = process_variant(self.genome, chrom, start, end, snp_pos, allele1, allele2, self.seq_len)

            # Convert to tensor and add batch dimension
            x1 = torch.tensor(allele1_1hot, dtype=torch.float32).unsqueeze(0)  # (1, 4, L)
            x2 = torch.tensor(allele2_1hot, dtype=torch.float32).unsqueeze(0)
            
            if self.siamese: # concatenate inputs to be processed together.
                x = torch.cat((x1, x2), dim=1)
                batch_inputs.append(x)
            else:
                batch_inputs.append(x1)
                batch_inputs.append(x2)

            if label is not None:
                # Twice to match dim
                batch_labels.append(torch.tensor(label, dtype=torch.float32))
                if not self.siamese:
                    batch_labels.append(torch.tensor(label, dtype=torch.float32))

        # Stack into batch tensors
        x_batch = torch.cat(batch_inputs, dim=0) # .to(self.device)  # (B, 4, L)
        batch = {"x": x_batch}

        if batch_labels:
            batch["labels"] = torch.stack(batch_labels) # .to(self.device)

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
        random_state: int = 42,
    ):
        self.data = pd.read_csv(csv_path)
        if subset_seqs > 0:
            self.data = self.data.sample(n=subset_seqs, random_state=random_state)

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


class BorzoiVariantDataset(Dataset):
    def __init__(
            self,
            csv_path: str,
            snp_pos_col: str = "pos",
            cpg_pos_col: str = "MAPINFO",
            target_col: str = "beta_a1",
            subset_seqs: int = 0,
            random_state: int = 42,
    ):
        self.data = pd.read_csv(csv_path)
        if subset_seqs > 0:
            self.data = self.data.sample(n=subset_seqs, random_state=random_state).reset_index(drop=True)
        
        self.chroms = self.data["REFSEQ_chr"]
        self.snp_pos = self.data[snp_pos_col]
        self.cpg_pos = self.data[cpg_pos_col]
        self.targets = self.data[target_col]
        self.a1s = self.data["allele1"]
        self.a2s = self.data["allele2"]
    
    def __len__(self):
        return len(self.chroms)
    
    def __getitem__(self, idx):
        return {
            "chrom": self.chroms[idx],
            "snp_pos": self.snp_pos[idx],
            "cpg_pos": self.cpg_pos[idx],
            "allele1": self.a1s[idx],
            "allele2": self.a2s[idx],
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
        with h5py.File(self.h5_path, 'r') as f:
            n = len(f["sequence"])
        return n

    def __getitem__(self, idx):
        f = self._get_file()
        x = f["sequence"][idx]
        y = f["me_track"][idx]
        return {
            "x": torch.from_numpy(x).float(),
            "labels": torch.from_numpy(y).float()
        }
