import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import logging
import os

from src.models import build_models
from src.models import paramgrid_config
from datetime import datetime

def ModelTuning(target=None,scale=None,model=None,nsplits=None,niters=None,cpus=None,output=None):

    CT = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    if model=='knn':
        m=build_models.knn(n_jobs=4)
        param_grid=paramgrid_config.grid_knn()
        directory='KNearestNeighbors'

    elif model=='svm':
        m=build_models.svm()
        param_grid=paramgrid_config.grid_svm()
        directory='SVM'

    elif model=='rf':
        m=build_models.rf(n_jobs=4)
        param_grid=paramgrid_config.grid_random_forest()
        directory='RandomForest'

    elif model=='gb':
        m=build_models.gbc()
        param_grid=paramgrid_config.grid_gbc()
        directory='GradientBoost'

    elif model=="xgb":
        m=build_models.xgboost(n_jobs=4)
        param_grid=paramgrid_config.grid_xgboost()
        directory='XGBoost'

    opath = os.path.join(output, directory)

     # Create output directory if does not exist
    if not os.path.exists(opath):
        os.makedirs(opath)

     #create directory for each run
    os.makedirs(os.path.join(opath,CT))

    #save log file into the run folder
    logging.basicConfig(level=logging.INFO, filename=os.path.join(opath,CT,'model_tunning.log'), filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")

    # Load data (deserialize)
    with open(target, 'rb') as handle:
        dict_data = pickle.load(handle)

    data=dict_data['data'][0]

    logging.info(f"data filename: {target}\n")
    logging.info(f"Normalized data: {scale}\n")
    logging.info(f"Config: {dict_data['config'][0]}\n")

    #Get dataset
    y=np.array(data['target'])
    X=np.array(data.drop('target', axis=1))

    #get train and test data
    sss = StratifiedShuffleSplit(n_splits=2, test_size=0.1, random_state=0)

    for train_index, test_index in sss.split(X, y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    #check shapes
    logging.info('\nX_train: '+str(X_train.shape)+'\n'+
                'y_train: '+str(y_train.shape))

    logging.info('\nX_test: '+str(X_test.shape)+'\n'+
                'y_test: '+str(y_test.shape))

    logging.info(f"Model: {directory}\n")

    import warnings
    warnings.filterwarnings("ignore")

    # define the pipeline
    steps = list()
    if scale=='yes':
        steps.append(('scaler', MinMaxScaler()))
    steps.append(('model', m))
    pipeline = Pipeline(steps=steps)

    # define the evaluation procedure
    cv = StratifiedKFold(n_splits=nsplits)

    logging.info("Start randomized search...")

    # tune model
    rs=RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter = niters,scoring = 'accuracy',cv=cv, n_jobs=cpus)

    rs.fit(X_train,y_train)

    logging.info("...finish\n")

    logging.info('Best parameters:\n'+str(rs.best_params_))
    logging.info('\nBest score:\n'+str(rs.best_score_)+'\n\n')

    #create an output dictionary
    output_dict = {"CT":[],"best_model":[],"parameters":[],"config":[]}

    config=dict_data['config'][0]
    config.append(directory)

    output_dict["CT"].append(CT)
    output_dict["config"].append(config)
    output_dict["best_model"].append(rs.best_estimator_)
    output_dict["parameters"].append(rs.best_params_)

    #save best model to pickle
    with open(os.path.join(opath,CT,'best_'+model+'.pkl'), 'wb') as handle:
        pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #pickle.dump(rs.best_estimator_, open(os.path.join(opath,CT,'best_'+model+'.sav'), 'wb'))

    #create gitignore to avoid pushing model files
    lines = ['*.pkl', '!.gitignore']
    with open(os.path.join(opath,CT,'.gitignore'), 'w') as f:
        f.write('\n'.join(lines))

    #Check accuracy of test set
    if scale=='yes':
        scaler=MinMaxScaler()
        X_train_scaled=scaler.fit_transform(X_train)
        X_test_scaled=scaler.transform(X_test)
    else:
        X_train_scaled=X_train
        X_test_scaled=X_test

    m.fit(X_train_scaled,y_train)
    y_pred=m.predict(X_test_scaled)

    acc=accuracy_score(y_test, y_pred)
    cm=confusion_matrix(y_test, y_pred, labels=rs.classes_)

    logging.info('\nTest set accuracies BEST ESTIMATOR\n'+
    directory+': '+str(acc)+'\n')

    mapping=dict_data["label_dict"][0][0]
    categories=np.array(pd.DataFrame(rs.classes_)[0].map(mapping))

    #confusion matrix

    fig, ax = plot_confusion_matrix(conf_mat=cm,
                                        figsize=(6,6),
                                        class_names=categories,
                                        show_absolute=True,
                                        show_normed=True,
                                        colorbar = True)

    plt.tight_layout()
    fig.savefig(os.path.join(opath,CT,'confusion_matrix.pdf'))


