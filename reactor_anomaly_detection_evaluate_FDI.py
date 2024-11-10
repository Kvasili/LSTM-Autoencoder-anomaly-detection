'''
    @Author:


    USAGE
    python reactor_anomaly_detection_evaluate_FDI.py 

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

    "model_name": "lstm_autoencoder_FDI.model",
    "model_normalization": "standard_normalization_FDI.pkl"

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


def load_data_min(filename, cols_to_be_read, percentage):
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
    index_values = x["index"]

    # print(index_values)

    for i in range(len(x)-seq_size):
        # print(i)
        seq = x.iloc[i:(i + seq_size)]
        seq_index = list(index_values[i:(i + seq_size)])
        # print(seq_index)

        # Check if all indices in seq_index are successive
        is_continuous = True

        for j in range(1, len(seq_index)):
            if seq_index[j] - seq_index[j - 1] != 1:
                is_continuous = False
                break

        if is_continuous:
            # only append into list if all time values are continuous
            # Append only the data, excluding the index
            x_values.append(seq.drop(columns=["index"]).values)
            # print(seq)

    return np.array(x_values)


def array_to_dataframe(array, original_columns):
    # Create an empty list to hold the reconstructed rows
    reconstructed_rows = []

    # Iterate through the sequences and append only the unique parts
    for i in range(len(array)):
        if i == 0:
            # Add the entire first sequence
            reconstructed_rows.extend(array[i])
        else:
            # Add only the last row of each subsequent sequence to avoid duplication
            reconstructed_rows.append(array[i][-1])

    # Convert the reconstructed rows into a DataFrame
    df_reconstructed = pd.DataFrame(
        reconstructed_rows, columns=original_columns)

    return df_reconstructed


def main():
    column_names = ["index", "nfd-1-cps", "nfd-1-cr", "rr-active-state", "rr-position",
                    "ss1-active-state", "ss1-position", "ss2-active-state", "ss2-position", "manual-scram"]  #

    print("...START TRAINING.....")

    filename_abnormal = './data/FDI_fake_scrams_ch1_ch1cr.csv'
    df_abnormal_original = load_data_min(
        filename_abnormal, column_names, 1)
    # exclude the last value with manual SCRAM
    df_abnormal = df_abnormal_original.iloc[:, 1:-1]

    # normalize the data
    normalization_model = joblib.load(
        './models/'+config['model_normalization'])
    test_scaled = normalization_model.transform(df_abnormal)
    test_scaled = pd.DataFrame(test_scaled)
    # print(test_scaled.head())

    # print(df_abnormal_original['index'])

    test_scaled = pd.concat(
        [df_abnormal_original['index'], test_scaled, df_abnormal_original['manual-scram']], axis=1)
    print(test_scaled.head())

    # print("scaled")
    # print(test_scaled)
    # print(test_scaled.shape)

    # this creates the dataset in appropriate format but also removes the index column
    testX = to_sequences(test_scaled, seq_size=10)
    print('Test X: ')
    print(testX.shape)

#     print("****************************************")
#     # load model
    model = load_model('./models/'+config['model_name'])
    critical_feature_index = 0

    # # This does not have the manual scram
    testPredict = model.predict(testX[:, :, :-1])
    print('Predictions: ')
    print(testPredict.shape)

    test_reconstruction_errors = np.abs(
        testPredict[:, :, critical_feature_index] - testX[:, :, critical_feature_index])

    print(test_reconstruction_errors)

    df_reconstructed = array_to_dataframe(testX, column_names[1:])

    print(df_reconstructed.shape)

    # This will produce a reconstruction error for each row of the initial dataset
    print('Test reconstruntion errors:')
    print(test_reconstruction_errors.shape)

    max_trainMAE = 0.2
    anomaly_df = df_reconstructed[10:].copy()
    print(anomaly_df.shape)

    # anomaly_df.columns = column_names[:-1]
    # anomaly_df['manual-scram'] = df_abnormal_original.iloc[10:, -1]

    # print(anomaly_df.head())

    # the anomaly in every time sequence is defined as the average reconstruction error at each 10s sequence
    anomaly_df['reconstruction_error'] = np.mean(
        test_reconstruction_errors, axis=1)[:-1]

    anomaly_df['max_trainMAE'] = max_trainMAE

    anomaly_df['anomaly'] = anomaly_df['reconstruction_error'] > anomaly_df['max_trainMAE']
    anomaly_df['anomaly'] = anomaly_df['anomaly'].astype(int)
    # print(anomaly_df['anomaly_numeric'])
    num_anomalies = (anomaly_df['anomaly'] == 1).sum()
    print('**********************************************')
    print("Number of anomalies:", num_anomalies)
    # Replace 1s with 1000 in 'anomaly_numeric'
    # anomaly_df['anomaly'] = anomaly_df['anomaly'].replace(
    #     1, 5)
    anomaly_df.loc[:, 'anomaly'] = anomaly_df['anomaly'].replace(1, 5)

    # print(anomaly_df.head())
    # print(anomaly_df.shape)

    # # Define the columns you want to plot
    columns_to_plot = ["nfd-1-cps", "rr-position", "ss2-position",
                       "manual-scram", 'anomaly']

    # # Rename 'anomaly_numeric' to 'anomaly' in the dataframe for consistency
    # anomaly_df = anomaly_df.rename(columns={'anomaly_numeric': 'anomaly'})

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

# # '''
# #     @Author:
# #     Konstantinos Vasili


# #     USAGE
# #     python reactor_anomaly_detection_evaluate_FDI.py

# # '''
