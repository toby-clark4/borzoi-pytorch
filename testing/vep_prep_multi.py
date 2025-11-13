import pandas as pd
# import numpy as np
# import h5py
# import pysam
# import gc
# from tqdm import tqdm
# from borzoi_pytorch.data import extract_window, process_variant, bin_methylation, refseq_map


# target_len = 6144 # to match Borzoi output resolution.
# seq_len = 524_288

mean_betas = pd.read_csv('../data/me_chip/annotated_beta_values.csv', index_col=0)
mean_betas = mean_betas[['REFSEQ_chr', 'mean_beta', 'MAPINFO']]

data = pd.read_csv('../data/godmc/assoc_meta_all_filtered.csv', index_col=0)

data = data[['REFSEQ_chr', 'MAPINFO', 'snp', 'pos', 'allele1', 'allele2']]

merged = data.merge(
    mean_betas,
    on=['REFSEQ_chr', 'MAPINFO'],
    how='inner',     
)

per_snp = (
    merged
    .groupby(['snp', 'REFSEQ_chr'], as_index=False)
    .agg({
        'MAPINFO': list,
        'mean_beta': list,
        'pos': 'first',
        'allele1': 'first',
        'allele2': 'first'
    })
)

per_snp = per_snp.rename(columns={
    'MAPINFO': 'cpg_locations',
    'mean_beta': 'mean_betas'
})

print(len(per_snp))

per_snp.to_csv('../data/godmc/snps_for_vep_multi.csv.gz', compression='gzip')



'''
with h5py.File(f"../data/godmc/snps.h5", "w") as f:
    f.create_dataset(
        "sequence",
        shape=(0, 4, seq_len),
        maxshape=(None, 4, seq_len),
        dtype=np.uint8,
        compression='lzf',
        chunks=(1, 4, seq_len),            
    )
    f.create_dataset(
        "me_track",
        shape=(0, target_len),
        maxshape=(None, target_len),
        dtype=np.float32,
        compression="lzf",
        chunks=(1, target_len)
    )
    f.create_dataset("chromosome", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype(encoding='utf-8'))
    f.create_dataset("snp", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype(encoding='utf-8'))

BATCH_SIZE = 1500  # Write every 1000 SNPs (adjust based on available RAM)

def _write_batch(f, seqs, targets, chroms, snps):
    """Helper function to write a batch to HDF5"""
    if not seqs:
        return
        
    seqs_arr = np.stack(seqs)
    targets_arr = np.stack(targets)
    chroms_arr = np.array(chroms, dtype=h5py.string_dtype(encoding="utf-8"))
    snps_arr = np.array(snps, dtype=h5py.string_dtype(encoding="utf-8"))

    n_new = seqs_arr.shape[0]
    
    start_idx = f["sequence"].shape[0]


    for name, data_arr in zip(['sequence', 'me_track', 'chromosome', 'snp'], 
                               [seqs_arr, targets_arr, chroms_arr, snps_arr]):
        f[name].resize(start_idx + n_new, axis=0)
        f[name][-n_new:] = data_arr


with h5py.File(f"../data/godmc/snps.h5", "a") as f:
    for chrom in refseq_map.values():
        print(f"Processing chromosome {chrom}")

        # Load only this chromosome's data
        data = pd.read_csv('../data/godmc/assoc_meta_all_filtered.csv', index_col=0)
        chr_data = data[data['REFSEQ_chr'] == chrom].set_index('MAPINFO')
        del data  # Free memory immediately

        chr_me_mask = mean_betas['REFSEQ_chr'] == chrom
        chr_me = mean_betas[chr_me_mask]

        chr_sequence = genome.fetch(chrom).upper()

        seqs, targets, chroms, snps = [], [], [], []
        
        for snp, snp_data in tqdm(chr_data.groupby('snp')):
            row = snp_data.iloc[0]
            snp_pos = row['pos']
            
            me_vals = chr_me.loc[snp_data.index]
            chrom, start, end = extract_window(chrom, snp_pos, seq_len)
            allele1_seq, allele2_seq = process_variant(
                genome, chrom, start, end, snp_pos, 
                row['allele1'], row['allele2']
            )
            out_channel = bin_methylation(me_vals, start_coordinate=start, seq_len=seq_len, resolution=32)
            
            seqs.append(allele1_seq)
            seqs.append(allele2_seq)
            targets.append(out_channel)
            targets.append(out_channel)
            chroms.extend([chrom, chrom])
            snps.extend([snp, snp])
            
            # Write batch when given number of sequences have been processed
            if len(seqs) >= BATCH_SIZE * 2:
                _write_batch(f, seqs, targets, chroms, snps)
                seqs, targets, chroms, snps = [], [], [], []  # Clear memory
        
        # Write remaining data for this chromosome
        _write_batch(f, seqs, targets, chroms, snps)
        
        # Explicitly free memory
        del chr_sequence, chr_data, chr_me
        gc.collect()

'''
"""
with h5py.File(f"../data/me_chip/snps.h5", "a") as f:
    for chrom in refseq_map.values():
        print(f"Processing chromosome {chrom}")

        data = pd.read_csv('../data/godmc/assoc_meta_all_filtered.csv', index_col=0)

        chr_data = data[data['REFSEQ_chr'] == chrom].set_index('MAPINFO')

        del data

        chr_me_mask = mean_betas['REFSEQ_chr'] == chrom
        chr_me = mean_betas[chr_me_mask]

        chr_sequence = genome.fetch(chrom).upper()

        seqs, targets, chroms, snps = [], [], [], []

        # Group by SNP once to avoid repeated filtering
        
        for snp, snp_data in tqdm(chr_data.groupby('snp')):
            row = snp_data.iloc[0]
            snp_pos = row['pos']
            
            me_vals = chr_me.loc[snp_data.index]
            # Extract a window around the SNP to centre the SNP in the sequence
            chrom, start, end = extract_window(chrom, snp_pos, seq_len)
            # One-hot encode the two sequences
            allele1_seq, allele2_seq = process_variant(genome, chrom, start, end, snp_pos, row['allele1'], row['allele2'], chrom_seq=chr_sequence)
            out_channel = bin_methylation(me_vals, start_coordinate=start, seq_len=seq_len, resolution=32)
            seqs.append(allele1_seq)
            seqs.append(allele2_seq)
            targets.append(out_channel)
            targets.append(out_channel)
            chroms.extend([chrom, chrom])
            snps.extend([snp, snp])
        


        if not seqs:
            continue

        seqs = np.stack(seqs)
        targets = np.stack(targets)
        chroms = np.array(chroms, dtype=h5py.string_dtype(encoding="utf-8"))
        snps = np.array(snps, dtype=h5py.string_dtype(encoding="utf-8"))

        n_new = seqs.shape[0]
        start_idx = f["sequence"].shape[0]

        for name, data_arr in zip(['sequence', 'me_track', 'chromosome', 'snp'], [seqs, targets, chroms, snps]):
            f[name].resize(start_idx + n_new, axis=0)
            f[name][-n_new:] = data_arr
"""
