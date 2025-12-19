'''
script for loading raw data from csv files
'''
import pandas as pd
import os

def load_raw_data(file_path):
    '''
    check if file exists
    '''
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"data file not found at: {file_path}")
    
    # load csv using iso dates
    df = pd.read_csv(file_path)
    
    # convert time to datetime and set as index
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    
    # ensure columns are lowercase for consistency
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    return df

if __name__ == "__main__":
    # simple test for the loader
    test_path = "data/raw/BITSTAMP_BTCUSD, 1D - ISOtime.csv"
    data = load_raw_data(test_path)
    print(data.head())