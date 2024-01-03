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

adata_input, training_dir, adata_predict, adata_predict_output, predict_dir = argv[1:]
# adata_input, training_dir, adata_predict, adata_predict_output, predict_dir = ['adata_train_gmt.h5ad', 'model_train', 'adata_predict.h5ad', 'adata_predict_gmt.h5ad', 'model_predict']

adata = sc.read(adata_input)
intr_cvae = sca.models.EXPIMAP.load(training_dir, adata)

predict = sc.read(adata_predict)
predict = predict[:, predict.var_names & adata.var_names].copy()
df_predict = pd.DataFrame.sparse.from_spmatrix(predict.X, columns = predict.var_names, index = predict.obs_names)
df_predictadd = pd.DataFrame(index = predict.obs.index, columns = adata.var_names.difference(predict.var_names))
df_predict = df_predict.merge(df_predictadd, left_index = True, right_index = True, how = 'left')
df_predict = df_predict.fillna(0)
df_predict = df_predict[adata.var_names]
predict = sc.AnnData(df_predict, obs = predict.obs)
predict.X = predict.X.astype('int32')
predict.layers["counts"] = predict.X.copy()
predict.obs['batch'] = 'Query'
predict.uns['terms'] = adata.uns['terms']
predict.write(adata_predict_output, compression = True)

# model predict
p_intr_cvae = sca.models.EXPIMAP.load_query_data(predict, intr_cvae)

p_intr_cvae.train(n_epochs=500, 
                  alpha_epoch_anneal=130, 
                  weight_decay=0., 
                  alpha_kl=0.01,
                  seed=2020, 
                  use_early_stopping=True,
                  print_n_deactive=True,
                  print_stats=True )

p_intr_cvae.save(predict_dir)
