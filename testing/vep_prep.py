import pandas as pd

full_data = pd.read_csv('/home/tobyc/data/borzoi-pytorch/data/godmc/assoc_meta_all.csv.gz', compression='gzip')
snps = pd.read_csv('/home/tobyc/data/borzoi-pytorch/data/godmc/snps.csv.gz')
manifest = pd.read_csv('../data/me_chip/manifest/GPL13534_HumanMethylation450_15017482_v.1.1.csv', skiprows=7)
manifest.set_index('IlmnID', inplace=True)

snps = snps[snps['assoc_class'] == 'cis_only']
snps = snps[snps['snp_qc'] == 'PASS']
snps = snps[snps['type'] == 'SNP']

df_merged = full_data.merge(snps, how='inner', left_on='snp', right_on='name', suffixes=('', '_dup')).reset_index(drop=True)
df_merged = df_merged.loc[:, ~df_merged.columns.str.endswith('_dup')]

df_merged['MAPINFO'] = df_merged['cpg'].map(manifest['MAPINFO']).astype(int)
df_merged['cpg_chr'] = [manifest.loc[cpg]['CHR'] for cpg in df_merged['cpg']]

df_merged = df_merged[df_merged['cpg_chr'] == df_merged['chr']]

df_merged['snp_to_cpg'] = (df_merged['MAPINFO'] - df_merged['pos']).abs()
df_merged = df_merged[df_merged['snp_to_cpg'] <= 50_000]

refseq_map = {
    "1": "NC_000001.10",
    "2": "NC_000002.11",
    "3": "NC_000003.11",
    "4": "NC_000004.11",
    "5": "NC_000005.9",
    "6": "NC_000006.11",
    "7": "NC_000007.13",
    "8": "NC_000008.10",
    "9": "NC_000009.11",
    "10": "NC_000010.10",
    "11": "NC_000011.9",
    "12": "NC_000012.11",
    "13": "NC_000013.10",
    "14": "NC_000014.8",
    "15": "NC_000015.9",
    "16": "NC_000016.9",
    "17": "NC_000017.10",
    "18": "NC_000018.9",
    "19": "NC_000019.9",
    "20": "NC_000020.10",
    "21": "NC_000021.8",
    "22": "NC_000022.10",
    "X": "NC_000023.10",
    "Y": "NC_000024.9"
}

df_merged['REFSEQ_chr'] = df_merged['chr'].astype(str).map(refseq_map)
df_merged = df_merged.dropna(subset=['REFSEQ_chr'])
df_merged.reset_index(drop=True).to_csv('../data/godmc/processed_example.csv', index=False)

