import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import pysam

from borzoi_pytorch import Borzoi
from borzoi_pytorch.data import BorzoiVariantDataset, BorzoiVariantMultiDataCollator

device = torch.device('cuda')
model = Borzoi.from_pretrained('/home/tobyc/data/borzoi-pytorch/assets/multi-meBorzoi_ernest_mse')
model.to(device)
model.eval()

genome = pysam.FastaFile('/home/tobyc/data/borzoi-pytorch/data/ref_genomes/GRCh37/GCF_000001405.13/GCF_000001405.13_GRCh37_genomic.fna')

variant_ds = BorzoiVariantDataset("../data/godmc/snps_for_vep_multi.csv.gz", cpg_pos_col='cpg_locations', target_col='mean_betas')
data_collator = BorzoiVariantMultiDataCollator(genome=genome, model=model)

batch_size=4
variant_loader = DataLoader(
    variant_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    collate_fn=data_collator,
)

pair_mean_masked_diffs = []
pair_mean_diffs = []

with torch.no_grad():
    for batch in tqdm(variant_loader):
        # Move batch tensors to device
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        logits = outputs.logits
        labels = batch['labels'] 
        
        mask = labels != 0

        curr_batch = logits.shape[0] // 2
        # Reshape to (batch_size, 2, seq_len)
        logits_pairs = logits.reshape(curr_batch, 2, *logits.shape[1:])
        mask_pairs = mask.reshape(curr_batch, 2, *logits.shape[1:])
        
        # Compute diffs across sequence
        diffs = logits_pairs[:, 0, :] - logits_pairs[:, 1, :]

        # Mask out unmeasured positions and compute diffs
        masked_logit_pairs = logits_pairs * mask_pairs
        masked_diffs = masked_logit_pairs[:, 0, :] - masked_logit_pairs[:, 1, :]
        
        # sum the differences
        sum_per_pair = diffs.sum(dim=-1)
        sum_per_pair_masked = masked_diffs.sum(dim=-1)

        # Extend lists
        pair_mean_diffs.extend(sum_per_pair.cpu().tolist())
        pair_mean_masked_diffs.extend(sum_per_pair_masked.cpu().tolist())


ds_csv = variant_ds.data
ds_csv['predicted_mean_diff'] = pair_mean_diffs
ds_csv['predicted_mean_masked_diff'] = pair_mean_masked_diffs
ds_csv.to_csv('../results/godmc/vep_multi_mechip_predictions.csv.gz', compression='gzip')