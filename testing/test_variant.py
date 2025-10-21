import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import pysam

from borzoi_pytorch import Borzoi
from borzoi_pytorch.data import BorzoiVariantDataCollator, BorzoiVariantDataset

genome_path = '/home/tobyc/data/borzoi-pytorch/data/ref_genomes/GRCh37/GCF_000001405.13/GCF_000001405.13_GRCh37_genomic.fna'
device = torch.device('cuda')
model = Borzoi.from_pretrained('/home/tobyc/data/borzoi-pytorch/assets/meBorzoi_ernest_10k')
model.to(device)
model.eval()

variant_ds = BorzoiVariantDataset('../data/godmc/processed_example.csv')
variant_data = pd.read_csv('../data/godmc/processed_example.csv')

batch_size=1
collate_fn = BorzoiVariantDataCollator(pysam.FastaFile(genome_path))
variant_loader = DataLoader(
    variant_ds,
    batch_size=batch_size,
    shuffle=False,
    collate_fn = collate_fn,
)

all_diffs = []

with torch.no_grad():
    for batch in tqdm(variant_loader):
        outputs = model(**batch)
        logits = outputs.logits
        diff = logits[0] - logits[1]
        all_diffs.append(diff.cpu().item())

variant_data['delta_pred'] = all_diffs

variant_data.to_csv('../results/me_chip/meBorzoi_ernest_10k_variants.csv')        
