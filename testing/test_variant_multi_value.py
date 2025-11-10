import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from borzoi_pytorch import Borzoi
from borzoi_pytorch.data import BorzoiH5Dataset

device = torch.device('cuda')
model = Borzoi.from_pretrained('/home/tobyc/data/borzoi-pytorch/assets/multi-meBorzoi_ernest_mse')
model.to(device)
model.eval()

variant_ds = BorzoiH5Dataset("../data/me_chip/snps.h5")

batch_size=4
variant_loader = DataLoader(
    variant_ds,
    batch_size=batch_size,
    shuffle=False,
)

pair_mean_diffs = []

with torch.no_grad():
    for batch in tqdm(variant_loader):
        # Move batch tensors to device
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        logits = outputs.logits  # shape (batch_flat, ...)

        # Ensure even batch size
        Bflat = logits.shape[0]
        assert Bflat % 2 == 0, "Batch length must be even (pairs)."

        # reshape into (Bpairs, 2, ...)
        Bpairs = Bflat // 2
        logits_pairs = logits.reshape(Bpairs, 2, *logits.shape[1:])

        # allele0 - allele1 for each pair
        diffs = logits_pairs[:, 0] - logits_pairs[:, 1]  # shape (Bpairs, ...)

        # collapse remaining dims and compute mean per pair
        diffs_flat = diffs.view(diffs.shape[0], -1)         # (Bpairs, N)
        mean_per_pair = diffs_flat.mean(dim=1)              # (Bpairs,)

        pair_mean_diffs.extend(mean_per_pair.cpu().tolist())
        break

# Save results back to your dataframe
print(pair_mean_diffs)