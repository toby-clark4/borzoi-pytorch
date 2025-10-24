import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import pysam

from borzoi_pytorch import Borzoi
from borzoi_pytorch.data import BorzoiVariantDataCollator, BorzoiVariantDataset

genome_path = '/home/tobyc/data/borzoi-pytorch/data/ref_genomes/GRCh37/GCF_000001405.13/GCF_000001405.13_GRCh37_genomic.fna'
device = torch.device('cuda')
model = Borzoi.from_pretrained('/home/tobyc/data/borzoi-pytorch/assets/meBorzoi_ernest_10k_sigmoid')
model.to(device)
model.eval()

variant_ds = BorzoiVariantDataset('/home/tobyc/data/borzoi-pytorch/data/godmc/assoc_metal_all_under_1k.csv', subset_seqs=10000)
variant_data = pd.read_csv('/home/tobyc/data/borzoi-pytorch/data/godmc/assoc_metal_all_under_1k.csv', nrows=10000)

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

variant_data.to_csv('../results/me_chip/meBorzoi_ernest_10k_close_variants.csv')        
