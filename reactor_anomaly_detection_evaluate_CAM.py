'''
    @Author:



    USAGE
    python reactor_anomaly_detection_evaluate.py 

'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Input, Dropout, RepeatVector, TimeDistributed, Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix,  classification_report
import joblib
import datetime
from tensorflow.keras.models import load_model


start = datetime.datetime.now()

config = {

    "model_name": "lstm_autoencoder_CAM.model",
    "model_normalization": "MinMax_normalization_CAM.pkl"

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


def train_test_split(df, percentage):
    # Function that splits the larger normal dataset into training, testing, and validation
    split_index = int(len(df) * percentage)

    return df[:split_index], df[split_index:]


def normalize_data(train_df, val_df, Monel_name):
    # Normalize data
    scaler = MinMaxScaler()
    # scaler = StandardScaler()

    train_scaled = scaler.fit_transform(train_df)
    val_scaled = scaler.transform(val_df)
    # test_df_scaled = scaler.transform(test_df)
    # Export the Normalized model to be used later
    joblib.dump(scaler, Monel_name)

    return pd.DataFrame(train_scaled), pd.DataFrame(val_scaled)


def plot_data(x, y):
    plt.plot(x, y)
    plt.legend()
    plt.show()


def to_sequences(x, seq_size):
    x_values = []

    for i in range(len(x)-seq_size):
        # print(i)
        x_values.append(x.iloc[i:(i+seq_size)].values)

    return np.array(x_values)

# def Lstm_autoenoder(trainX, trainY, testX):


def main():

    column_names = ["CAM-CNT", "NFD-1-CPS", "cam-cnt_abnormal"]  #

    print("...START TRAINING.....")

    # filename_abnormal = 'C:\\Users\\kvasi\\OneDrive - purdue.edu\\projects\\Autonomous Control System\\data\\cam_mixed.csv'
    filename_abnormal = './data/CAM_sensor_mixed_data.csv'

    df_abnormal = load_data(
        filename_abnormal, column_names, 0.5)

    print(df_abnormal.head())
    print('Mix data shape: ')
    print(df_abnormal.shape)

    # # normalize the data
    normalization_model = joblib.load(
        './models/'+config['model_normalization'])
    # exclude the prediction for the normalization
    test_scaled = normalization_model.transform(df_abnormal.iloc[:, :-1])
    test_scaled = pd.DataFrame(test_scaled)
    print(test_scaled.head())

    testX = to_sequences(test_scaled, seq_size=10)
    print('Test X: ')
    print(testX.shape)

    # print("****************************************")
    # # load model
    model = load_model('./models/'+config['model_name'])
    critical_feature_index = 0

    testPredict = model.predict(testX)
    print('Predictions: ')
    print(testPredict.shape)

    test_reconstruction_errors = np.abs(
        testPredict[:, :, critical_feature_index] - testX[:, :, critical_feature_index])

    # # This will produce a reconstruction error for each row of the initial dataset
    print('Test reconstruntion errors:')
    print(test_reconstruction_errors.shape)

    # this threshold is based on the distribution error of the Normal dataset
    max_trainMAE = 1.0
    anomaly_df = test_scaled[10:].copy()
    anomaly_df.columns = column_names[:-1]
    anomaly_df['cam-cnt_abnormal'] = df_abnormal.iloc[10:, -1]

    # # print(anomaly_df.head())

    # # the anomaly in every time sequence is defined as the average reconstruction error at each 10s sequence
    anomaly_df['reconstruction_error'] = np.mean(
        test_reconstruction_errors, axis=1)

    anomaly_df['max_trainMAE'] = max_trainMAE

    anomaly_df['anomaly'] = anomaly_df['reconstruction_error'] > anomaly_df['max_trainMAE']
    anomaly_df['anomaly'] = anomaly_df['anomaly'].astype(int)

    anomaly_df.dropna(inplace=True)
    print('anomaly df after dropping NA: ')
    print(anomaly_df.head())

    predicted_values = anomaly_df['anomaly']
    actual_values = anomaly_df['cam-cnt_abnormal']

    print('Classification results: ')
    accuracy = accuracy_score(actual_values, predicted_values)
    print(f'accuracy: {accuracy}')
    conf_matrix = confusion_matrix(actual_values, predicted_values)
    print(conf_matrix)

    # Generate classification report
    report = classification_report(
        actual_values, predicted_values, target_names=['Normal', 'Anomaly'])

    print(report)

    print(anomaly_df['anomaly'])
    num_anomalies = (anomaly_df['anomaly'] == 1).sum()
    print('**********************************************')
    print("Number of anomalies:", num_anomalies)
    # Replace 1s with 1000 in 'anomaly_numeric'
    anomaly_df['anomaly'] = anomaly_df['anomaly'].replace(
        1, 10000)

    # print(anomaly_df.head())
    # print(anomaly_df.shape)

    # # Define the columns you want to plot
    columns_to_plot = ["anomaly", "CAM-CNT", "NFD-1-CPS"]

    # # Rename 'anomaly_numeric' to 'anomaly' in the dataframe for consistency
    # anomaly_df = anomaly_df.rename(columns={'anomaly_numeric': 'anomaly'})

    colors = {
        "CAM-CNT": "blue",
        "NFD-1-CPS": "green",
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
    @Author:
    Konstantinos Vasili


    USAGE
    python reactor_anomaly_detection_evaluate_CAM.py

'''
