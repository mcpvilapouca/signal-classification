import pandas as pd

def read_raw_data(file):
    if file.endswith('.csv'):
        df=pd.read_csv(file)
    elif file.endswith('.xlsx'):
        df=pd.read_excel(file)
    else:
        raise Exception(f'It seems that the targets file {file} does not have the right format')
    return df