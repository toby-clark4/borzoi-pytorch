import pandas as pd

data = pd.read_csv('../data/godmc/assoc_meta_all_under_1k.csv', index_col=0)

ds_train = data[~data['chr'].isin([3, 12, 5, 10])]
ds_val = data[data['chr'].isin([3, 12])]
ds_test = data[data['chr'].isin([5, 10])]
ds_train.to_csv('../data/siamese/1k_train.csv', index=False)
ds_val.to_csv('../data/siamese/1k_val.csv', index=False)
ds_test.to_csv('../data/siamese/1k_test.csv', index=False)