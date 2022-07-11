# Signal Classification

🔸 This repository shows the pipeline developed for the signal classification of a pressure signal with 8 channels.

🔸 The datasets are private and not available for download.

🔸 The target classification is binary and renamed for privacy as 'class 0' and 'class 1', although it is easily adaptable for multiclass classification

<img src="https://user-images.githubusercontent.com/95075305178309732-0c254f86-df91-417d-81e8-1b8fff13cf67.gif" width="600">

### Prepare data and resample

###### This script joins the signal data (inside folder data/raw/dat_files) with its general features (inside folder data/raw), cleans the dataset, renames the target features and resample the signals by either interpolating or removing extra points (with care to not remove significant points). It saves the cleaned dataset in the folder data/interm

- target: path and filename of general input data
- features: path to filename of input .csv files with signal data
- output: path and filename of pickle data
- final_points: number of points in each signal after resampling (default=average)

```python run_generate_dataframe.py --target data/raw/data.csv --features data/raw/dat_files/csv_files --output data/interm/resampled.pkl -n final_points```


### Extract features

###### This script reads the cleaned dataset from folder data/interm. It extracts features (mean, std, energy, entropy, zero cross count, median, max, min, skewness, kurtosis) for each nsegments of the signal. We can choose to retrieve the data from the raw or the resampled signal. We can also choose to remove outliers using the LocalOutlierFactor(). The dataset is saved in a dictionary with other relevant keys, and this output dictionary is saved as a pickle in the folder data/processed.

- target: path and filename of input data
- nsegments: number of segments to split signal data (default=10)
- signal: 'raw' / 'resample' (default=raw)
- remove: choose to remove outliers 'yes' / 'no' (default=no)
- output: path only

Note: nohup runs python in background

```nohup python run_feature_extraction.py --target data/interm/resampled.pkl  --nsegments 10 --signal raw --remove no --output data/processed  > data/processed/run_feature_extraction.log &```

### Select features (not required)

###### This script does a feature selection according to the correlation. It is nor required, but tends to improve the results. It removes highly correlated features and features with very little correlation with target. The filtered dataset is saved in an output dictionary as a pickle in the folder data/processed.

- target: path and filename of input data
- uppercorrelation: threshold to remove highly correlated features (default=0.85)
- lowercorrelation: threshold to remove features only slightly correlated with target (default=0.05)
- output: path only

```nohup python run_feature_selection.py --target data/processed/data_extracted_features.pkl  --uppercorrelation 0.85 --lowercorrelation 0.05 --output data/processed  > run_feature_selection.log &```

### Model selection (not required)

###### This script evaluates several models (see buil_models.py) using the RepeatedStratifiedKFold and default parameters. It gives a sense of which models might perform best and should be evaluated further. The results are saved in the folder models/select

- target: path and filename of input data
- scale: normalize data ('yes'/'no' , default='yes')
- nsplits: RepeatedStratifiedKFold number of splits (default=5)
- nrepeats: RepeatedStratifiedKFold number of repeats (default=10)
- cpus: number of cpus to use in the models and in the RepeatedStratifiedKFold (default=4)
- output: path only

```python run_model_selection.py --target data/processed/data_extracted_selected_features.pkl  --scale yes --nsplits 5 --nrepeats 10 --cpus 4 --output models/select```

### Model tuning

###### This script tunes the required model using RandomizedSearchCV. The param_grid can be changed in src/models/paramgrid_config.py. The results are evaluated against a test set (10%) and saved in the folder models/tune. The best estimator and its parameters are saved in an output dictionary, saved as a pickle.

- target: path and filename of input data
- scale: normalize data ('yes'/'no' , default='yes')
- model: model to tune (knn,svm,rf,gb,xgb  ; if necessary change param_grid in src/models/paramgrid_config.py)
- nsplits: StratifiedKFold number of splits (default=5)
- niters: number of iterations of RandomizedSearchCV (default=1000)
- cpus: number of cpus to use in the RandomizedSearchCV (default=10)
- output: path only

```python run_model_tuning.py --target data/processed/data_extracted_selected_features.pkl  --scale yes --model rf --nsplits 5  --niters 3000 --cpus 10 --output models/tune```

### Model train

###### This script should be used when the model and its parameters are chosen. For now, since we expect more data to arrive, it gets the best set of parameters from the folder models/tune/...and retrains the model. The test set accuracy and confusion matrix are obtained. The results are saved into the folder models/train

- target: path and filename of input data
- best model: path to best_model.pkl (typically inside folder models/tune)
- scale: normalize data ('yes'/'no' , default='yes')
- testset: Test set size (default=0.1)
- random: random split of data into training/test set (default=no)
- cpus: number of cpus (default=4)
- output: path only

```python run_train_model.py --target data/processed/data_extracted_selected_features.pkl --bestmodel models/tune/RandomForest/2022-07-08_16-52-08/best_rf.pkl --scale yes --testset 0.1 --random no --cpus 5 --output models/train```