#import ray
#ray.init(num_cpus=4)
#import modin.pandas as pd
import pandas as pd
import numpy as np
import random
from os import path
import logging
logging.getLogger().setLevel(logging.INFO)

def filter_entries(df,d):
    #remove nans
    df = df.dropna(subset=d.keys())
    df=df.reset_index(drop=True)

    #rename column of .dat and target file names
    df.rename(columns = d, inplace = True)
    return df

def add_path2file(df,path_to_file):

    df['file']=df.dat.apply(lambda x: [path_to_file+x[i] for i in range(len(x))] )

    #muda a extensão
    df.file=df.file.apply(lambda x: [x[i][:-4]+'.csv' for i in range(len(x))])

    return df

def target_clean_filter(df,ltargets):
    #lower case of 'target' text and remove unwanted spaces
    df['target']=df['target'].str.lower()
    df['target']=df['target'].str.strip()

    #replace accents and ?
    df['target']=df['target'].replace(['á','à','ã','â','ó','õ','ò','ô','é','ê','è','í','ì','ç','\?'],
                                     ['a','a','a','a','o','o','o','o','e','e','e','i','i','c',''], regex=True)
    #homogeneize
    df['target']=df['target'].replace(['.*cl.*ss.*0','.*cl.*ss.*1'],
                                         ['class 0','class 1'], regex=True)
    #rename equivalent confitions
    df['target']=df['target'].replace(['.*cl.*ss.*zero','.*cl.*ss.*one'],
                                         ['class 0','class 1'], regex=True)

    string=r'(?:'+ltargets[0]
    for i in ltargets[1:]:
        string=string+'|'+i
    string=string+')'

    df['target']=df['target'].str.findall(string)
    df['target'] = [''.join(map(str, l)) for l in df['target']]

    df['target']=df['target'].replace([''],[np.NaN])

    df=df[df['target'].notna()]

    df=df.reset_index(drop=True)
    return df

def split_dats(row,col):
    row[col]=[row[col].strip() for row[col] in row[col].split(';')]
    row[col]=[i for i in row[col] if i]
    return row[col]

def get_signals(row):
    colnames=['p1', 'p2', 'p3','p4','p5','p6','p7','p8']
    #read from csv file
    data=pd.read_csv(row['file'][0],index_col=False,skiprows=[0],usecols = [j for j in range(2,10)],names=colnames)
    if len(row['file'])>1:
        for i in range(len(row['file'])-1):
            data=pd.concat([data,pd.read_csv(row['file'][i+1],index_col=False,skiprows=[0],usecols = [j for j in range(2,10)],names=colnames)])


    #add number of points to dataframe
    row['n_points']=data.shape[0]

    #add signals do dataframe as lists of lists
    row['signal_data']=data.transpose().values.tolist()

    #calculate time from 4Hz accquisition
    freq=4
    dt=1/freq
    T=dt*data.shape[0]

    #add time as a list
    row['time']=list(np.arange(0, T, dt))

    return row

def apply_interpolation(row,final_points):
    logging.info(f'patient_id: {row.name}')
    a=pd.DataFrame(row['signal_data']).T
    c=pd.DataFrame(row['time'])

    b,d=interpolate_signal(final_points,a,c)

    row['int_signal_data']=b.transpose().values.tolist()
    row['int_time']=d.transpose().values.tolist()[0]
    return row


def interpolate_signal(final_points,a,c):

    n=a.shape[0]

    if n<final_points: # no_points smaller than final points -> create points by interpolation

        p2f=final_points-n

        while p2f>0:
            if p2f>n-1:
                fp=n-1
            else:
                fp=p2f

            points=np.arange(0,n+fp,1)
            b=pd.DataFrame(index=range(0,n+fp),columns=[0,1,2,3,4,5,6,7])
            d=pd.DataFrame(index=range(0,n+fp),columns=[0])

            if fp==1:
                delta=round((n+fp)/(fp+1))
                idelta=[delta]
            else:
                delta=round((n+fp)/fp)
                #index to fill
                idelta=[]
                count=-1
                i=-1
                while len(idelta)<fp:
                    i=i+1
                    count=count+1
                    if count==delta-1:
                        idelta.append(i)
                        count=0

            #index to insert original data
            ii=[]
            for i in list(points):
                if i not in idelta:
                    ii.append(i)

            #insert data into empty dataframe
            j=-1
            for i in ii:
                j=j+1
                b.iloc[i]=a.iloc[j]
                d.iloc[i]=c.iloc[j]

            #interpolate
            for col in b.columns:
                b[col]= pd.to_numeric(b[col], downcast='float')
                b[col]=b[col].interpolate(method='linear')


            d[0]= pd.to_numeric(d[0], downcast='float')

            d[0]=d[0].interpolate(method='linear')

            #update variables for next cycle
            n=len(list(points))
            p2f=final_points-n
            a=b
            c=d

    elif n==final_points:
        b=a
        d=c

        #convert to floats
        for col in b.columns:
            b[col]= pd.to_numeric(b[col], downcast='float')

        d[0]= pd.to_numeric(d[0], downcast='float')

    else: # no_points larger than final points -> eliminate points
        b=a
        d=c

        p2e=n-final_points

        warning_count=-1
        idelta=[]

        tol=check_tol(a)

        while len(idelta)<p2e:

            val=random.randint(0, n-1)

            #check if variation is to steep to eliminate
            if val==n-1:
                x=pd.DataFrame(a.iloc[val-1,:]).T
                x=pd.concat([x,pd.DataFrame(a.iloc[val,:]).T])
            elif val==0:
                x=pd.concat([x,pd.DataFrame(a.iloc[val,:]).T])
                x=pd.concat([x,pd.DataFrame(a.iloc[val+1,:]).T])
            else:
                x=pd.DataFrame(a.iloc[val-1,:]).T
                x=pd.concat([x,pd.DataFrame(a.iloc[val,:]).T])
                x=pd.concat([x,pd.DataFrame(a.iloc[val+1,:]).T])

            if x.diff().max().max()<tol:
                idelta.append(val)
                #remove duplicates
                idelta=list(set(idelta))
            else:
                warning_count=warning_count+1

            if warning_count>5*p2e:
                raise Exception('Error: Point elimination caused too many iterations. \n Repeat resampling increasing the tolerance or reducing the number of points to eliminate \n')

            elif warning_count>3*p2e:
                raise Exception('Warning: Point elimination is causing too many iterations. \n Check tolerance or reduce the number of points to eliminate \n')


        #create dataframe and drop rows from idelta
        for k in idelta:
            b=b.drop(k)
            d=d.drop(k)
        b=b.reset_index(drop=True)
        d=d.reset_index(drop=True)

        #convert to floats
        for col in b.columns:
            b[col]= pd.to_numeric(b[col], downcast='float')

        d[0]= pd.to_numeric(d[0], downcast='float')

    return b,d

def check_tol(a):
    l=[]
    nchannels=8
    for i in range(nchannels):
        ii=a.diff().nlargest(1,[i])[i].index[0]
        l.append(a.diff().nlargest(1,[i]).iloc[:,i][ii])
    l=np.array(l)
    stdev=np.std(l)
    avg=np.mean(l)
    tol=avg-stdev

    return tol

def decompact_df(row,column_name):
    row['p1']=row[column_name][0]
    row['p2']=row[column_name][1]
    row['p3']=row[column_name][2]
    row['p4']=row[column_name][3]
    row['p5']=row[column_name][4]
    row['p6']=row[column_name][5]
    row['p7']=row[column_name][6]
    row['p8']=row[column_name][7]
    return row