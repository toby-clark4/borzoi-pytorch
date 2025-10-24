import os

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

import pysam

from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

from borzoi_pytorch import Borzoi
from borzoi_pytorch.config_borzoi import BorzoiConfig
from borzoi_pytorch.data import BorzoiDataCollator, BorzoiRegressionDataset

os.environ['WANDB_PROJECT'] = 'meBorzoi'
    
class BorzoiTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        labels = inputs.pop("labels")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            
            # Compute loss
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.squeeze(), labels.squeeze())
        
        return (loss, logits, labels)

data_dir = '../data/me_chip'
res_dir = '../results/me_chip'
model_base_path = '../assets'
genome_path = '/home/tobyc/data/borzoi-pytorch/data/ref_genomes/GRCh37/GCF_000001405.13/GCF_000001405.13_GRCh37_genomic.fna'
name = 'meBorzoi_ernest_10k_2window'
model_dir = f'{model_base_path}/{name}'
checkpoint_dir = f'{model_base_path}/checkpoints/{name}'
os.makedirs(model_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(res_dir, exist_ok=True)

device = torch.device('cuda')

config = BorzoiConfig.from_pretrained('johahi/borzoi-replicate-0')
config.enable_human_head = False
config.enable_methylation_head = True
model = Borzoi.from_pretrained('johahi/borzoi-replicate-0', config=config)
model.to(device) # type: ignore

ds_train = BorzoiRegressionDataset(f'{data_dir}/ernest_train.csv', subset_seqs=10_000)
ds_val = BorzoiRegressionDataset(f'{data_dir}/ernest_val.csv', subset_seqs=1_000)
ds_test = BorzoiRegressionDataset(f'{data_dir}/ernest_test.csv', subset_seqs=1_000)

data_collator = BorzoiDataCollator(pysam.FastaFile(genome_path), seq_len=524288, device=device)

lora_config = LoraConfig(
    task_type="SEQ_CLS",
    r=16,
    lora_alpha=16,
    target_modules = ['to_q', 'to_k', 'to_v'],
    modules_to_save = ['methylation_head'],
    lora_dropout=0.1,
)

model = get_peft_model(model, lora_config)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    logits = logits.flatten()
    labels = labels.flatten()

    try:
        pearson_corr = pearsonr(logits, labels)[0].item()
        spearman_corr = spearmanr(logits, labels)[0].item()

        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
        }
    except:
        return {"pearson":0.0, "spearmanr":0.0}
   

training_args = TrainingArguments(
    output_dir = checkpoint_dir,
    eval_strategy='epoch',
    save_strategy='epoch',
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    weight_decay = 1e-3,
    warmup_ratio=0.05,
    logging_steps=25,
    max_grad_norm=1.0,
    bf16=False,
    seed=42,
    load_best_model_at_end=True,
    report_to=["wandb"],
    label_names=["chrom", "pos", "label"],
    dataloader_num_workers=4,
    run_name=name,
)


trainer = BorzoiTrainer(
    model=model,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

pred, true, metrics = trainer.predict(ds_test)

model.cpu()
merged_model = model.merge_and_unload()
merged_model.save_pretrained(model_dir)

pd.DataFrame({'pred': np.squeeze(pred), 'true': true}).to_csv(f'{res_dir}/{name}.csv')

print(f'Test pearson r: {metrics['test_pearson']}')
print(f'Test spearman r: {metrics['test_spearmanr']}')



