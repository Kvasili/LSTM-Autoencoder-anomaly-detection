'''
    @Author: XXXXX 

    USAGE
    python reactor_anomaly_detection_evaluate_SCRAM.py 


'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Input, Dropout, RepeatVector, TimeDistributed, Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
import datetime
from tensorflow.keras.models import load_model
import joblib

start = datetime.datetime.now()

config = {

    "model_name": "lstm_autoencoder_SCRAM_v2.model",
    "model_normalization": "standard_normalization_SCRAM_v2.pkl",
    "paht_to_data": "./data/scram_fake_data_small.csv",
    "critical_signal": 0

}


def find_Null_values(df):
    '''
        This function finds Null values in the given dataset
        across the rows and columns and returns a new dataframe
        by eliminating the whole row
        without Null values

    '''
    df = pd.DataFrame(df)
    df.dropna(inplace=True)

    return df


def load_data(filename, cols_to_be_read, percentage):
    '''
        This function loads the data from a .csv file to a pandas dataframe. 
        Percentage parameter defines the number of rows to be loaded
    '''

    df = pd.read_csv(filename)
    df.dropna(inplace=True)
    df = df.loc[:, cols_to_be_read]  # 'index',

    length = len(df)

    try:
        if 0 < percentage <= 1.0:

            number_of_rows = int(percentage*length)
            df = df[:number_of_rows]

            return df
    except:
        raise ValueError("values in percentage should be in the range (0, 1]")


def to_sequences(x, seq_size):
    x_values = []

    for i in range(len(x)-seq_size):

        x_values.append(x.iloc[i:(i+seq_size)].values)

    return np.array(x_values)


def main():
    column_names = ["nfd-1-cps", "nfd-1-cr", "rr-active-state", "rr-position",
                    "ss1-active-state", "ss1-position", "ss2-active-state", "ss2-position", "manual-scram"]

    print("...START TRAINING.....")

    filename_abnormal = config["paht_to_data"]
    df_abnormal = load_data(
        filename_abnormal, column_names, 0.01)
    print(df_abnormal.head())

    # normalize the data
    normalization_model = joblib.load(
        './models/'+config['model_normalization'])
    test_scaled = normalization_model.transform(df_abnormal)
    test_scaled = pd.DataFrame(test_scaled)

    testX = to_sequences(test_scaled, seq_size=10)
    print('Test X: ')
    print(testX.shape)

    print("****************************************")
    # load model
    model = load_model('./models/'+config['model_name'])
    critical_feature_index = config["critical_signal"]

    testPredict = model.predict(testX)
    print('Predictions: ')
    print(testPredict.shape)

    test_reconstruction_errors = np.abs(
        testPredict[:, :, critical_feature_index] - testX[:, :, critical_feature_index])

    # This will produce a reconstruction error for each row of the initial dataset
    print('Test reconstruntion errors:')
    print(test_reconstruction_errors.shape)

    max_trainMAE = 1.9
    anomaly_df = test_scaled[10:].copy()
    anomaly_df.columns = column_names

    # the anomaly in every time sequence is defined as the average reconstruction error at each 10s sequence
    anomaly_df['reconstruction_error'] = np.mean(
        test_reconstruction_errors, axis=1)

    anomaly_df['max_trainMAE'] = max_trainMAE

    # Replace the maximum value in column 'A' with 1
    anomaly_df["manual-scram"].replace(
        anomaly_df["manual-scram"].max(), 1, inplace=True)

    anomaly_df["manual-scram"].replace(
        anomaly_df["manual-scram"].min(), 0, inplace=True)

    # anomaly is defined as comparison with the reconstructed datapoint
    anomaly_df['anomaly'] = anomaly_df['reconstruction_error'] > anomaly_df['max_trainMAE']
    anomaly_df['anomaly'] = anomaly_df['anomaly'].astype(int)

    # # Define the columns you want to plot
    columns_to_plot = ["nfd-1-cps", "rr-position", "ss2-position",
                       "manual-scram", 'anomaly']

    colors = {
        "nfd-1-cps": "blue",
        "rr-position": "green",
        "ss2-position": "purple",
        "manual-scram": "orange",
        "anomaly": "red"
    }

    # Plot each column on the same graph
    for col in columns_to_plot:
        plt.plot(anomaly_df.index,
                 anomaly_df[col], label=col, color=colors[col])

    # Add labels and title
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized sensor values')
    plt.title('Selected Columns and Anomalies Over Time')

    # Add a legend to distinguish between the columns
    plt.legend()
    plt.show()

    # end of training time
    end = datetime.datetime.now()
    print("[INFO] training completed after: {}".format(end-start))


if __name__ == "__main__":
    main()

'''

    USAGE
    python reactor_anomaly_detection_evaluate_SCRAM.py

'''
