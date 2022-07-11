import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from mlxtend.plotting import plot_confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import regex as re
from datetime import datetime
import logging
import os
from src.models import build_models

def ModelTrain(input=None,bestmodel=None,scale=None,testset=None,random=None,cpus=None,output=None):
    CT = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    model=re.findall(r'_(\w+(?=.pkl))',bestmodel)[0]


    if model=='knn':
        m=build_models.knn(n_jobs=4)
        directory='KNearestNeighbors'

    elif model=='svm':
        m=build_models.svm()
        directory='SVM'

    elif model=='rf':
        m=build_models.rf(n_jobs=4)
        directory='RandomForest'

    elif model=='gb':
        m=build_models.gbc()
        directory='GradientBoost'

    elif model=="xgb":
        m=build_models.xgboost(n_jobs=4)
        directory='XGBoost'

    opath = os.path.join(output, directory)

     # Create output directory if does not exist
    if not os.path.exists(opath):
        os.makedirs(opath)

     #create directory for each run
    os.makedirs(os.path.join(opath,CT))

    #save log file into the run folder
    logging.basicConfig(level=logging.INFO, filename=os.path.join(opath,CT,'model_train.log'), filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")

    # Load data (deserialize)
    with open(input, 'rb') as handle:
        dict_data = pickle.load(handle)

    data=dict_data['data'][0]

    logging.info(f"data filename: {input}\n")

    #Get dataset
    y=np.array(data['target'])
    X=np.array(data.drop('target', axis=1))

    #get train and test data
    if random=='yes':
        sss = StratifiedShuffleSplit(n_splits=2, test_size=testset)
    else:
        sss = StratifiedShuffleSplit(n_splits=2, test_size=testset, random_state=0)


    for train_index, test_index in sss.split(X, y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    #check shapes
    logging.info('\nX_train: '+str(X_train.shape)+'\n'+
                'y_train: '+str(y_train.shape))

    logging.info('\nX_test: '+str(X_test.shape)+'\n'+
                'y_test: '+str(y_test.shape))


    #get model dictionary
    with open(bestmodel, 'rb') as handle:
        dict_model = pickle.load(handle)

    logging.info(f"Config: {dict_model['config'][0]}\n")
    logging.info(f"Parameters: {dict_model['parameters'][0]}\n")

    #rename keys to trim the model__ that needed to be added because of the RepeatedKFold pipeline
    params=dict_model['parameters'][0]
    newkeys=[re.sub(r'model__', '', file) for file in list(params.keys())]
    for i,key in enumerate(list(params.keys())):
        params[newkeys[i]] = params.pop(key)

    #set best set of parameters
    m.set_params(**params)

    #Train model for new dataset
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
    cm=confusion_matrix(y_test, y_pred, labels=m.classes_)

    logging.info('\nTest set accuracies BEST ESTIMATOR\n'+
    directory+': '+str(acc)+'\n')

    mapping=dict_data["label_dict"][0][0]
    categories=np.array(pd.DataFrame(m.classes_)[0].map(mapping))

    #confusion matrix

    fig, ax = plot_confusion_matrix(conf_mat=cm,
                                        figsize=(6,6),
                                        class_names=categories,
                                        show_absolute=True,
                                        show_normed=True,
                                        colorbar = True)

    plt.tight_layout()
    fig.savefig(os.path.join(opath,CT,'confusion_matrix.pdf'))



