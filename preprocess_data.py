# Pre-process the data
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def get_preprocessed_data():
    df = pd.read_csv('data/unemployment_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])  
    df.set_index('Date', inplace=True)
    df = df[['Unemployment']]
    df.dropna(how='any', inplace=True)

    # Preprocessing
    scaler = MinMaxScaler()
    data_cleaned = scaler.fit_transform(df)
    return(data_cleaned, df, scaler) # df returns unscaled data