#### Load train data and do preprocessing
python ./script/preprocess.py ./example/adata_train.h5ad ./data/perturbation_test.gmt ./data/perturbation_test.gmt.h5ad ./example/adata_train_gmt.h5ad

#### Create model and train it
python ./script/train.py ./example/adata_train_gmt.h5ad ./example/model_train Tissue_Source

#### Load query data for predict mapping & Initlizling the model for predict training
python ./script/predict.py ./example/adata_train_gmt.h5ad ./example/model_train ./example/adata_predict.h5ad ./example/adata_predict_gmt.h5ad ./example/model_predict

