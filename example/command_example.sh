#### Load train data and do preprocessing
python ../script/preprocess.py adata_train.h5ad ../data/perturbation_test.gmt ../data/perturbation_test.gmt.h5ad adata_train_gmt.h5ad

#### Create model and train it
python ../script/train.py adata_train_gmt.h5ad model_train Tissue_Source

#### Load query data for predict mapping & Initlizling the model for predict training
python ../script/predict.py adata_train_gmt.h5ad model_train adata_predict.h5ad adata_predict_gmt.h5ad model_predict
