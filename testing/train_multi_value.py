import os

import torch
import torch.nn as nn
from torchmetrics.functional import pearson_corrcoef
import numpy as np

from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

from borzoi_pytorch import Borzoi
from borzoi_pytorch.config_borzoi import BorzoiConfig
from borzoi_pytorch.data import BorzoiDataCollator, BorzoiH5Dataset
from borzoi_pytorch.pytorch_borzoi_utils import SparseMSELoss

os.environ['WANDB_PROJECT'] = 'meBorzoi'
    
class BorzoiTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        labels = inputs.pop("labels")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            
            # Compute loss
            loss_fct = SparseMSELoss()
            loss = loss_fct(logits, labels)
        
        return (loss, logits, labels)

data_dir = '../data/me_chip'
res_dir = '../results/me_chip'
model_base_path = '../assets'
name = 'multi-meBorzoi_ernest_10k'
model_dir = f'{model_base_path}/{name}'
checkpoint_dir = f'{model_base_path}/checkpoints/{name}'
os.makedirs(model_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(res_dir, exist_ok=True)

device = torch.device('cuda')

config = BorzoiConfig.from_pretrained('johahi/borzoi-replicate-0')
config.enable_human_head = False
config.enable_methylation_head = True
config.single_target = False
model = Borzoi.from_pretrained('johahi/borzoi-replicate-0', config=config)
model.to(device) # type: ignore

ds_train = BorzoiH5Dataset(f'{data_dir}/ernest_train.h5')
ds_val = BorzoiH5Dataset(f'{data_dir}/ernest_val.h5')
ds_test = BorzoiH5Dataset(f'{data_dir}/ernest_test.h5')

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
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)

    # Apply sigmoid if model outputs logits
    logits = torch.sigmoid(logits)

    # Compute per-sequence Pearson r
    batch_size = logits.shape[0]
    r_values = []
    for i in range(batch_size):
        r = pearson_corrcoef(logits[i], labels[i])
        r_values.append(r)

    r_mean = torch.stack(r_values).mean()

    return {"mean_pearson": r_mean.item()}
   

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
    label_names=["me_track"],
    dataloader_num_workers=4,
)



trainer = BorzoiTrainer(
    model=model,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    compute_metrics=compute_metrics,
)

trainer.train()

pred, true, metrics = trainer.predict(ds_test)

model.cpu()
merged_model = model.merge_and_unload()
merged_model.save_pretrained(model_dir)

pred = pred.detach().cpu().numpy()
true = true.detach().cpu().numpy()
np.savez(f"{res_dir}/{name}.npz", pred=pred, true=true)

print(f'Test pearson r: {metrics['test_mean_pearson']}')



