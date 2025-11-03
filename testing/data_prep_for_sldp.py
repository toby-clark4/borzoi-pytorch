# Formats a csv of variant results to give necessary output files for SLDP.

import pandas as pd
import os

data_path = "../results/me_chip/meBorzoi_ernest_10k_close_variants.csv"
out_dir = "../results/vep/10k_augmented/"
name = "100k_close_variants"
os.makedirs(out_dir, exist_ok=True)
os.makedirs(f"{out_dir}/annots/{name}", exist_ok=True)

sumstats_map = {'rsid': 'SNP', 'chr': 'CHR',
                'MAPINFO': 'BP', 'allele1': 'A1',
                'allele2': 'A2', 'beta_a1': 'BETA',
                'pval': 'P', 'samplesize': 'N',
                'se': 'SE', 'freq_a1': 'FRQ'}

data = pd.read_csv(data_path, index_col=0)

# Give columns the correct names
data = data.rename(columns = sumstats_map)

data = data[~data['SNP'].duplicated()]  # Remove duplicated SNPs

# Save chromosome data to individual TSV files for SLDP
for i in range(1, 23):
    chr_data = data[data['CHR'] == i].reset_index(drop=True).copy()
    chr_data = chr_data[['SNP', 'A1', 'A2', 'delta_pred']]
    chr_data.to_csv(f"{out_dir}/annots/{name}/{i}.sannot.gz", sep = '\t', index=False, compression='gzip')

# Calculate Z scores and save summary stats
data['Z'] = data['BETA'] / data['SE']
data_formatted = data[['SNP', 'A1', 'A2', 'Z', 'N']]
data_formatted.to_csv(f"{out_dir}/{name}.sumstats.gz", sep = '\t', index=False, compression='gzip')
