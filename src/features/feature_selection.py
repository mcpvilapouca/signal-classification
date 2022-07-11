import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import os
import logging
logging.getLogger().setLevel(logging.INFO)

def FeatureSelection(target=None, uppercorrelation=None,lowercorrelation=None,output=None):

    logging.info('Start feature selection\n')

    # Load data (deserialize)
    with open(target, 'rb') as handle:
        dict_data = pickle.load(handle)

    data=dict_data['data'][0]

    X=data.drop('target', axis=1)
    y=data.target

    #get correlations of each features in dataset
    corrmat = data.corr()
    top_corr_features = corrmat.index

    #define features liist
    features=list(X.columns)
    logging.info(f'Initial number of features: {len(features)}\n')

    #Remove features that are highly correlated with each other and that are not correlated with the target
    #exclude highly correlated features
    a=pd.DataFrame()
    for col in corrmat.columns:
        a=pd.concat([a,pd.DataFrame(list(corrmat[col][corrmat[col].gt(uppercorrelation) | corrmat[col].lt(-uppercorrelation)].index))], ignore_index=True, axis=1)
        if col=='target':
            #remote features not correlated with target
            b=pd.DataFrame(list(corrmat[col][corrmat[col].gt(-lowercorrelation) & corrmat[col].lt(lowercorrelation)].index))

    a.rename(columns=dict(zip(list(a.columns), list(corrmat.columns))), inplace=True)

    features_to_remove=[]
    for col in a.columns:
        if a[col].dropna().shape[0]>1:
            features_to_remove.append(list(a[col].dropna().iloc[1:]))

    features_to_remove.append(list(b[0]))

    #flatten list
    features_to_remove = [x for xs in features_to_remove for x in xs]


    #remove duplicates and target
    features_to_remove=list(set(features_to_remove))

    #remove target if necessary
    if ('target' in features_to_remove):
        features_to_remove.remove('target')

    #get the selected features
    features=[x for x in features if x not in features_to_remove]

    logging.info(f'Final number of features: {len(features)}\n')

    #filter data for the new features
    X=X[features]

    data=pd.concat([X,y], axis=1)

    #create an output dictionary
    output_dict = {"CT":[],"data":[],"features":[],"config":[],"label_dict":[]}

    config=dict_data['config'][0]
    config.append('Feature Selection: yes')
    config.append('Upper correlation: '+str(uppercorrelation))
    config.append('Lower correlation: '+str(lowercorrelation))

    CT = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    output_dict["CT"].append(CT)
    output_dict["data"].append(data)
    output_dict["features"].append(features)
    output_dict["label_dict"].append(dict_data['label_dict'])
    output_dict["config"].append(config)

    #save output dictionary to pickle
    filename='data_extracted_selected_features_'+str(CT)+'.pkl'
    with open(os.path.join(output, filename), 'wb') as handle:
        pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info('Saved to pickle\n')