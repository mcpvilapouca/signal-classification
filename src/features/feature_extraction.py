import logging
logging.getLogger().setLevel(logging.INFO)
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import LocalOutlierFactor
from src.features.operations import decompact_df,ENE,ENT,ZCC,MEAN,STD,MEDIAN,MAXVAL,MINVAL,SKEW,KURTOSIS,get_segments
import pickle
import pandas as pd
from datetime import datetime
import warnings
import os

def FeatureExtraction(target=None,nsegments=None,signal=None,remove=None,output=None):

    logging.info('Start feature extraction\n')

    #read intermediate data
    df=pd.read_pickle(target)

    #choose channels depending on raw or resampled data (default='raw')

    if signal=='processed':
        channels=['p1','p2','p3','p4','p5','p6','p7','p8']
    elif signal=='raw':
        channels=['op1','op2','op3','op4','op5','op6','op7','op8']
        df=df.apply(decompact_df, column_name='signal_data', channels=channels, axis=1)

    #create segments within raw data (default=10)
    logging.info(f'number of segments: {nsegments}\n')
#
    df=df.apply(get_segments,channels=channels,nseg=nsegments,axis=1)
#
    #Extract features and insert them in dataframe data
    data=pd.DataFrame()

    data['target']=df['target']

    seg_channels=[]
    for channel in channels:
        for i in range(nsegments):
            seg_channels.append(channel+'_'+str(i))

    logging.info('extracting features...')

    #ignore Performance warnings
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    #ENE : sum of the squared values of each segment of the signal
    for channel in seg_channels:
        data['ENE_'+channel]=df.apply(ENE, col=channel,axis=1)
        data['ENT_'+channel]=df.apply(ENT, col=channel,axis=1)
        data['ZCC_'+channel]=df.apply(ZCC, col=channel,axis=1)
        data['mean_'+channel]=df.apply(MEAN, col=channel, axis=1)
        data['std_'+channel]=df.apply(MEAN, col=channel, axis=1)
        data['median_'+channel]=df.apply(MEDIAN, col=channel, axis=1)
        data['max_'+channel]=df.apply(MAXVAL, col=channel, axis=1)
        data['skewness_'+channel]=df.apply(SKEW, col=channel, axis=1)
        data['kurtosis_'+channel]=df.apply(KURTOSIS, col=channel, axis=1)

    logging.info('...finish extracting features\n')

    #use label encoder to encode target column
    le= LabelEncoder()
    data['target']=le.fit_transform(df['target'])

    #create dictionary with the target mapping
    mapping = dict(zip(le.transform(le.classes_),le.classes_))

    #Remove Outliers (default='no')
    if remove=='yes':
        X=data.drop('target',axis=1)
        y=data.target
        # identify outliers in the training dataset
        lof = LocalOutlierFactor()
        yhat = lof.fit_predict(X)
        # select all rows that are not outliers
        mask = yhat != -1
        X, y = X[mask], y[mask]

        data=pd.concat([X,y], axis=1)


    #create an output dictionary
    output_dict = {"CT":[],"data":[],"features":[],"config":[],"label_dict":[]}

    config=['Remove outliers: '+remove]
    config.append('Segment number: '+str(nsegments))
    config.append('Signal type: '+signal)

    CT = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    features=list(data.drop('target', axis=1).columns)

    output_dict["CT"].append(CT)
    output_dict["data"].append(data)
    output_dict["features"].append(features)
    output_dict["label_dict"].append(mapping)
    output_dict["config"].append(config)

    #save output dictionary to pickle
    filename='data_extracted_features_'+str(CT)+'.pkl'
    with open(os.path.join(output, filename), 'wb') as handle:
        pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info('Saved to pickle\n')
