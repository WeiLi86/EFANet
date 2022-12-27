# FRANet model for CTR prediction
This is a repository designed for CTR prediction methods, it includes our CTR prediction methods:
  - **EFANet**: Enhanced-Field-Aware interaction With Filed-Attention For CTR Prediction [paper](https://arxiv.org/abs/1910.05552)

and some other representative baselines:
  - **AdnFM**: AdnFM: An Attentive DenseNet based Factorization Machine for Click-Through-Rate Prediction [paper](https://dl.acm.org/doi/pdf/10.1145/3512850.3512852)
  - **FRNet**: Enhancing CTR Prediction with Context-Aware Feature Representation Learning [paper](https://arxiv.org/pdf/2204.08758)
  - **XCrossNet**: XCrossNet: Feature Structure-Oriented Learning for Click-Through Rate Prediction [paper](https://arxiv.org/pdf/2104.10907)
  - **FINT**: FINT: FIELD-AWARE INTERACTION NEURAL NETWORK FOR CTR PREDICTION [paper](https://arxiv.org/pdf/2107.01999)
  - **Fi-gnn**: Modeling Feature Interactions via Graph Neural Networks for CTR Prediction [paper](https://arxiv.org/abs/1910.05552)
  - **AutoInt**: AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks [paper](https://arxiv.org/abs/1810.11921)
  - **InterHAt**: Interpretable Click-Through Rate Prediction through Hierarchical Attention [paper](https://dl.acm.org/doi/abs/10.1145/3336191.3371785)

You can find the model files of the above methods in [code](code/) for Criteo and Avazu datasets and [movielens/code](movielens/movie_code/) for MovieLens-1M dataset:
```bash
├── code                   
│   ├── fignn.py
│   ├── efanet.py                
│   ├── fint.py
│   ├── autoint.py                   
│   ├── interHAt.py            
│   ├── xcrossnet.py            
│   ├── frnet.py            
```


## Requirements: 
* **Tensorflow 1.5.0**
* Python 3.6
* CUDA 9.0+ (For GPU)

## Usage
Our code is based on [AutoInt](https://github.com/DeepGraphLearning/RecommenderSystems/tree/master/featureRec).
### Input Format
The required input data is in the following format:
* train_x: matrix with shape *(num_sample, num_field)*. train_x[s][t] is the feature value of feature field t of sample s in the dataset. The default value for categorical feature is 1.
* train_i: matrix with shape *(num_sample, num_field)*. train_i[s][t] is the feature index of feature field t of sample s in the dataset. The maximal value of train_i is the feature size.
* train_y: label of each sample in the dataset.

If you want to know how to preprocess the data, please refer to `data/Dataprocess/Criteo/preprocess.py`

### Example
There are three public real-world datasets(Avazu, Criteo, MovieLens-1M) that you can use. You can run the code on MovieLens-1M dataset directly in `/movielens`. The other three datasets are super huge, and they can not be fit into the memory as a whole. Therefore, we split the whole dataset into 10 parts and we use the first file as test set and the second file as valid set. We provide the codes for preprocessing these three datasets in `data/Dataprocess`. If you want to reuse these codes, you should first run `preprocess.py` to generate `train_x.txt, train_i.txt, train_y.txt` as described in `Input Format`. Then you should run `data/Dataprocesss/Kfold_split/StratifiedKfold.py` to split the whole dataset into ten folds. Finally you can run `scale.py` to scale the numerical value(optional).

To help test the correctness of the code and familarize yourself with the code, we upload the first `10000` samples of `Criteo` dataset in `train_examples.txt`. And we provide the scripts for preprocessing and training.(Please refer to `	data/sample_preprocess.sh` and `run_criteo.sh`, you may need to modify the path in `config.py` and `run_criteo.sh`). 

After you run the `data/sample_preprocess.sh`, you should get a folder named `Criteo` which contains `part*, feature_size.npy, fold_index.npy, train_*.txt`. `feature_size.npy` contains the number of total features which will be used to initialize the model. `train_*.txt` is the whole dataset.

Here's how to run the preprocessing.

```
cd data
mkdir Criteo
python ./Dataprocess/Criteo/preprocess.py
python ./Dataprocess/Kfold_split/stratifiedKfold.py
python ./Dataprocess/Criteo/scale.py
```

Here's how to train GraphFM on Criteo dataset.
```
CUDA_VISIBLE_DEVICES=$GPU python -m code.train \
--model_type EFANet \
                        --data_path $YOUR_DATA_PATH --data Criteo \
                        --blocks 3 --heads 2 --block_shape "[64, 64, 64]" \
                        --is_save --has_residual \
                        --save_path ./models/EFANet/Criteo/b3h2_64x64x64/ \
                        --field_size 39  --run_times 1 \
                        --epoch 2 --batch_size 1024 \
```

Here's how to train GraphFM on Avazu dataset.
```
CUDA_VISIBLE_DEVICES=$GPU python -m code.train \
--model_type EFANet \
                        --data_path $YOUR_DATA_PATH --data Avazu \
                        --blocks 3 --heads 2 --block_shape "[64, 64, 64]" \
                        --is_save --has_residual \
                        --save_path ./models/EFANet/Avazu/b3h2_64x64x64/ \
                        --field_size 23  --run_times 1 \
                        --epoch 2 --batch_size 1024 \
```

You can run the training on the relatively small MovieLens dataset in `/movielens`.


You should see the output like this:

```
...
train logs
...
start testing!...
restored from ./models/Criteo/b3h2_64x64x64/1/
test-result = 0.8088, test-logloss = 0.4430
test_auc [0.8088305055534442]
test_log_loss [0.44297631300399626]
avg_auc 0.8088305055534442
avg_log_loss 0.44297631300399626
```

## Acknowledgement
This implementation is based on Weiping Song and Chence Shi's [AutoInt](https://github.com/DeepGraphLearning/RecommenderSystems/tree/master/featureRec). Thanks for their sharing and contribution.
