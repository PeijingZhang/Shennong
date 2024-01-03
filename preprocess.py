import warnings
warnings.simplefilter(action='ignore')
import scanpy as sc
import pandas as pd
import numpy as np
from sys import argv

adata_input, cmap_file, cmap_file_output, adata_output = argv[1:]
# adata_input, cmap_file, cmap_file_output, adata_output = ['adata_train.h5ad', 'perturbation.gmt', 'perturbation.gmt.h5ad', 'adata_train_gmt.h5ad']


#########
files=cmap_file
min_genes=1
max_genes=None
varm_key='I'
uns_key='terms'
clean=False
genes_use_upper=False

#########
files = [files] if isinstance(files, str) else files
annot = []

for file in files:
     with open(file) as f:
          p_f = [l.upper() for l in f] if genes_use_upper else f
          terms = [l.strip('\n').split() for l in p_f]
     if clean:
          terms = [[term[0].split('_', 1)[-1][:30]]+term[1:] for term in terms if term]
     annot+=terms


######### load data
adata = sc.read(adata_input)
I = sc.read(cmap_file_output)

geneuse = adata.var_names.intersection(I.var_names)
geneadd = adata.var_names.difference(I.var_names)
Iadd = pd.DataFrame(0, index = I.obs_names, columns = geneadd)
Iadd = Iadd.astype('int32')
I = pd.DataFrame(I.X, index=I.obs_names, columns=I.var_names)
I = I[geneuse]
I= I.astype('int32')
I = I.merge(Iadd, left_index = True, right_index = True, how = 'left')
I = I.fillna(0)
I = I[adata.var_names]
I=I.T
# print(I.shape)

######### select terms
I = np.asarray(I, dtype='int32')
mask = I.sum(0) > min_genes
if max_genes is not None:
     mask &= I.sum(0) < max_genes

I = I[:, mask]
# print(I.shape)

######### merge with adata
adata.varm[varm_key] = I
adata.uns[uns_key] = [term[0] for i, term in enumerate(annot) if i not in np.where(~mask)[0]]

adata._inplace_subset_var(adata.varm['I'].sum(1)>0)
# print(adata)

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=8500, flavor="seurat_v3", subset=True) ## hvg 

select_terms = adata.varm['I'].sum(0)>12
adata.uns['terms'] = np.array(adata.uns['terms'])[select_terms].tolist()
adata.varm['I'] = adata.varm['I'][:, select_terms]

adata._inplace_subset_var(adata.varm['I'].sum(1)>0)
adata.X = adata.layers["counts"].copy()

adata.write(adata_output, compression = True)
