import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import logging
import os

from src.models import build_models
from datetime import datetime

def ModelSelection(target=None, scale=None, nsplits=None, nrepeats=None,cpus=None,output=None):

    CT = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    directory='ModelSelection_'+CT

     # Create output directory
    opath = os.path.join(output, directory)
    os.mkdir(opath)

    logging.basicConfig(level=logging.DEBUG, filename=os.path.join(opath, 'model_selection.log'), filemode="a+",
                            format="%(asctime)-15s %(levelname)-8s %(message)s")


    logging.info("\n_______________Start Model Selection__________________\n")



     # Load data (deserialize)
    with open(target, 'rb') as handle:
        dict_data = pickle.load(handle)

    data=dict_data['data'][0]

    #normalized data
    norm='yes'
    if scale:
        norm=scale

    logging.info(f"data filename: {target}\n")

    logging.info(f"Normalized data: {norm}\n")

    logging.info(f"Config: {dict_data['config'][0]}\n")

    #create models with default parameters
    #Note: the models were defined so that we can test every parameter available with scikit learn. Non defined parameters are set to scikit learn default
    ncpus=4
    if cpus:
        ncpus=cpus

    knn=build_models.knn(n_jobs=ncpus)
    svm=build_models.svm()
    rf=build_models.rf(n_jobs=ncpus)
    gb=build_models.gbc()
    dummy=build_models.dummy()
    xgb=build_models.xgboost(n_jobs=ncpus)

    models=[knn,svm,rf,gb,dummy,xgb]

    #create dictionary with models:
    labels=['knn','svm','rf','gb','dummy','xgb']
    model_dict=dict(zip(labels, models))

    #Get dataset
    y=np.array(data['target'])
    X=np.array(data.drop('target', axis=1))

    #check shapes
    logging.info('\nX: '+str(X.shape)+'\n'+
            'y: '+str(y.shape))

    #nsplits
    n_splits=5
    if nsplits:
        n_splits=nsplits

    n_repeats=10
    if nrepeats:
        n_repeats=nrepeats

    lscore=[]
    avg_score=[]

    for m,model in enumerate(models):
        # define the pipeline
        steps = list()
        if norm=='yes':
            steps.append(('scaler', MinMaxScaler()))
        steps.append(('model', model))
        pipeline = Pipeline(steps=steps)

        # define the evaluation procedure
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)

        # evaluate the model using cross-validation
        scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=ncpus)


        avg=sum(scores)/len(scores)
        lscore.append(scores)
        avg_score.append(avg)

        # Print the output.
        logging.info('\n--------'+list(model_dict.keys())[m]+'---------\n'+
        'Maximum Accuracy: '+
            str(pd.DataFrame(scores).max()[0])+
        '\nMinimum Accuracy: '+
            str(pd.DataFrame(scores).min()[0])+
        '\nStandard Deviation: '+str(pd.DataFrame(scores).std()[0])+
        '\n\nOverall Accuracy: '+
            str(pd.DataFrame(scores).mean()[0])+'\n\n')

    avg_score=pd.DataFrame(avg_score, index=model_dict.keys(),columns=['avg_scores'])
    avg_score.sort_values(by=['avg_scores'], ascending=False,inplace=True)

    avg_score.to_csv(os.path.join(opath, 'accuracies.csv'))

    #create gitignore to avoid csv
    lines = ['*.csv', '!.gitignore']
    with open(os.path.join(opath, '.gitignore'), 'w') as f:
        f.write('\n'.join(lines))

