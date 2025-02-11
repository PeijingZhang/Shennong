# Shennong
A deep learning framework for <i>in silico</i> screening of anticancer drugs at the single-cell level.

<a><img src="https://bis.zju.edu.cn/shennong/assets/img/pipeline.png"></a>

<p>We introduce a deep learning framework named <b>Shennong</b> for <i>in silico</i> screening of anticancer drugs for targeting each of the landscape cell clusters. Utilizing Shennong, we could predict individual cell responses to pharmacologic compounds, evaluate drug candidates’ tissue damaging effects, and investigate their corresponding action mechanisms. Prioritized compounds in Shennong’s prediction results include FDA-approved drugs currently undergoing clinical trials for new indications, as well as drug candidates reporting anti-tumor activity. Furthermore, the tissue damaging effect prediction aligns with documented injuries and terminated discovery events. This robust and explainable framework has the potential to accelerate the drug discovery process and enhance the accuracy and efficiency of drug screening.</p>

<p>The training and prediction results could be obtained and queried on our website (<a href="http://bis.zju.edu.cn/shennong/index.html" target="_blank">http://bis.zju.edu.cn/shennong/index.html</a>).</p>

<p>Citation: Peijing Zhang†, Xueyi Wang†, Xufeng Cen†, Qi Zhang†, Yuting Fu, Yuqing Mei, Xinru Wang, Renying Wang, Jingjing Wang, Hongwei Ouyang, Tingbo Liang*, Hongguang Xia*, Xiaoping Han*, and Guoji Guo*. <b>A deep learning framework for in silico screening of anticancer drugs at the single-cell level</b>. <b><i>National Science Review</i></b>, 2024, 12(2):nwae451. DOI: <a href="https://doi.org/10.1093/nsr/nwae451" target="_blank">https://doi.org/10.1093/nsr/nwae451</a>.</p>

## Requirements
Python packages  
```
scanpy >= 1.9.2
scarches >= 0.5.7
torch >= 1.13.1
pandas >= 1.5.3
numpy >= 1.23.5
gdown >= 4.6.3
```
## Tutorial
The scripts `command_example.sh` or `example.ipynb` shows how to predict individual cell responses to pharmacologic compounds with **Shennong** framework. The visualization of training and prediction results can be found in `example.ipynb`.
#### Data required
- <b>scRNA data</b>
- <b>Perturbation data</b>  
<a href="http://bis.zju.edu.cn/shennong/data/perturbation.gmt" target="_blank">perturbation.gmt</a> <a href="http://bis.zju.edu.cn/shennong/data/perturbation.gmt.h5ad" target="_blank">perturbation.gmt.h5ad</a> (High-confidence signatures of CMap with already preproceessed)
#### Pipeline
##### 1 Load train data and do preprocessing
```
python ../script/preprocess.py adata_train.h5ad ../data/perturbation_test.gmt ../data/perturbation_test.gmt.h5ad adata_train_gmt.h5ad
```  
The processed data would be saved in `adata_train_gmt.h5ad`.
##### 2 Create model and training
```
python ../script/train.py adata_train_gmt.h5ad model_train Tissue_Source
```  
The trained model would be saved in the `model_train/` directory. Tissue_Source is the value of <b>batch_key</b>.
##### 3 Load query data for prediction mapping & Initlizling the model for prediction training
```
python ../script/predict.py adata_train_gmt.h5ad model_train adata_predict.h5ad adata_predict_gmt.h5ad model_predict
```  
The predicted model would be saved in the `model_predict/` directory.
#### Visualization
The relevant code could be found in `example.ipynb` and the `code_scRNA/Application of framework/` directory of <a href="https://figshare.com/s/ac34f719115943d1d46c" target="_blank">figshare</a>, including plotting the latent space of the dataset and the influence term score for each cell.
