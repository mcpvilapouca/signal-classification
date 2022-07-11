import logging
from src.data.read_file import read_raw_data
from src.data.operations import filter_entries,add_path2file,target_clean_filter,apply_interpolation,decompact_df,get_signals,split_dats
logging.getLogger().setLevel(logging.INFO)


LTARGETS=['class 0','class 1']

def GenerateDataFrame(target=None,ltargets=None,features=None,output=None,nodes=None):
    #read data table of targets
    df=read_raw_data(target)
    #logging.info(f"{df}")

    #get only linked patients and rename target columns from 'Hipótese' to 'target'
    d_map = {'.dat': 'dat', 'Hipótese':'target'}
    df=filter_entries(df,d_map)

    #split dat files into a list of dats
    df['dat']=df.apply(split_dats, col='dat' ,axis=1)

    #add column of .csv file locations. column "file"
    df=add_path2file(df,features)
    logging.info(df.file)

    #check unique groups of classification
    df['target'].nunique()
    logging.info(f"{df['target'].nunique()}\n")
    logging.info(f"Before Cleaning: {df.groupby('target').size()}\n")

    ltargets=['class 0','class 1']
    if not ltargets:
        ltargets=LTARGETS

    #clean and filter target columns
    df=target_clean_filter(df,ltargets)

    #check unique groups of classification after cleaning
    df['target'].nunique()
    logging.info(f"{df['target'].nunique()}\n")
    logging.info(f"After Cleaning: {df.groupby('target').size()}\n")

    #get signals
    df=df.apply(get_signals,axis=1)

    #Print Max and min number of points and extremes to define final_points
    min_npoints=int(df.n_points.min())
    max_npoints=int(df.n_points.max())

    logging.info(f'min_npoints: {min_npoints}')
    logging.info(f'max_npoints: {max_npoints}\n')


    # TO DEFINE: final_points --> number of points after resampling

    #final points, at most (2*min_points)
    #final points, at least (1/2)*(max_points)

    final_points=int(0.5*min_npoints+0.5*max_npoints)
    if nodes:
        final_points = nodes


    logging.info(f'final_points: {final_points}\n')

    #apply interpolation a,c,tol
    logging.info(f'Start resampling ...\n')
    df=df.apply(apply_interpolation,final_points=final_points,axis=1)
    logging.info(f'... Finish resampling')
    #check histogram of number of points
    #hist_npoints(df)

    #plot data from patient=patient_id: plot_data(patient_id,df)
    #plot_data(0,df)

    #check_interpolation_wplot(patient_id,signal_id,df)
    #check_interpolation_wplot(688,df)

    #decompact interpolated signal data into columns
    df=df.apply(decompact_df,column_name='int_signal_data',axis=1)

    #save to pickle
    cols_of_interest=['target','p1','p2','p3','p4','p5','p6','p7','p8','int_time','signal_data','time']
    df[cols_of_interest].to_pickle(output)
    logging.info('\n')
    logging.info(f'Finish: data saved to pickle')