import warnings
warnings.simplefilter(action='ignore')
import scanpy as sc
import torch
import scarches as sca
import pandas as pd
import numpy as np
import gdown
from sys import argv
import os

adata_input, training_dir, condition_key = argv[1:]
# adata_input, training_dir, condition_key = ['adata_train_gmt.h5ad', 'model_train', 'Tissue_Source']

adata = sc.read(adata_input)
adata.obs['batch'] = adata.obs[condition_key]

# model train
early_stopping_kwargs = {
    "early_stopping_metric": "val_unweighted_loss",
    "threshold": 0,
    "patience": 50,
    "reduce_lr": True,
    "lr_patience": 13,
    "lr_factor": 0.1,
}

intr_cvae = sca.models.EXPIMAP(
    adata=adata,
    condition_key=condition_key,
    hidden_layer_sizes=[512, 512, 512, 512, 512],
    recon_loss='nb',
    mask=adata.varm['I'].T,
    soft_mask=False,
    use_hsic=False,
    hsic_one_vs_all = False
)

intr_cvae.train(
    n_epochs=500,
    alpha_epoch_anneal=130,
    alpha=0.95, ## 
    alpha_kl=0.1, ## 
    weight_decay=0.,  ##
    early_stopping_kwargs=early_stopping_kwargs,
    use_early_stopping=True,
    monitor_only_val=False,
    print_stats = True,
    seed=2020
)

intr_cvae.save(training_dir)
