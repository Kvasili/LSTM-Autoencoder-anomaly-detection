'''
    @Author: XXXXXXXXX
    @Date: 

    @Description: This code for training an LSTM Autoencoder for 
    detecting abnormalities in multivariate nuclear timeseries data



    USAGE
    Copy and paste the following command in a cmd environment
    Make sure you have installed all the appropriate libaries
    I ran the code in a conda environment where I have installed tensorflow with GPU

    python reactor_anomaly_detection_v2.py 

'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, RepeatVector, TimeDistributed, Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
import datetime

# Configuration dictionary
config = {
    "number_of_epochs": 10,
    # Index of the critical feature (neutron counts or CAM counts)
    "critical_feature": 0,
    # Name for trained LSTM model
    # "lstm_autoencoder_CAM_test.model",
    "model_name": "lstm_autoencoder_FDI.model ",
    # Name for the normalization model
    "model_normalization": "standard_normalization_FDI.pkl",
    "save_models": False,
    "save_norm_models": False
}


def find_null_values(df):
    """
    Removes rows with null values from the DataFrame.
    """
    df = pd.DataFrame(df)
    df.dropna(inplace=True)

    return df


def load_data(filename, cols_to_be_read, percentage=1.0):
    '''
        This function loads the data from a .csv file to a pandas dataframe. 
        Percentage parameter defines the number of rows to be loaded
    '''

    df = pd.read_csv(filename)
    df.dropna(inplace=True)
    df = df.loc[:, cols_to_be_read]

    length = len(df)

    try:
        if 0 < percentage <= 1.0:

            number_of_rows = int(percentage*length)
            df = df[:number_of_rows]

            return df
    except:
        raise ValueError("values in percentage should be in the range (0, 1]")


def split_data(df, train_size=0.8):
    """
    Splits the DataFrame into training and validation sets.
    """
    split_index = int(len(df) * train_size)
    return df[:split_index], df[split_index:]


def normalize_data(train_df, val_df, normalization_mode, normalization_model_name):
    """
    Normalizes the training and validation data using MinMax or Standard scaling.
    """
    # Normalize data
    if normalization_mode == 'MinMax':
        scaler = MinMaxScaler()
    elif normalization_mode == 'Standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Provide MinMax or Standard normalization methods.")

    train_scaled = scaler.fit_transform(train_df)
    val_scaled = scaler.transform(val_df)
    # test_df_scaled = scaler.transform(test_df)
    # Export the Normalized model to be used later
    if config['save_norm_models'] == True:
        try:
            joblib.dump(scaler, './models/' + normalization_model_name)
            print("[INFO] Normalization model saved.")
        except:
            print('[Error] Normalization model could not be saved.')

    return pd.DataFrame(train_scaled), pd.DataFrame(val_scaled)


def to_sequences(data, seq_size=10):
    """
    Converts a DataFrame into sequences for LSTM input.
    """
    x_values = []

    for i in range(len(data)-seq_size):
        # print(i)
        x_values.append(data.iloc[i:(i+seq_size)].values)

    return np.array(x_values)


def plot_metric(history, metric='loss'):
    """
    Plots training and validation metrics.
    """
    plt.plot(history.history[metric], label='Training')
    plt.plot(history.history[f'val_{metric}'], label='Validation')
    plt.title(f'{metric.capitalize()} over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.show()


def main():
    print("...START TRAINING.....")
    start_time = datetime.datetime.now()

    # # Load datasets
    # # Uncomment each dataset and run
    # Dataset 1: SCRAM dataset
    # filename_normal = './data/scram_real_data_small.csv'
    # filename_abnormal = './data/scram_fake_data_small.csv'
    # columns = ["nfd-1-cps", "nfd-1-cr", "rr-active-state", "rr-position",
    #            "ss1-active-state", "ss1-position", "ss2-active-state", "ss2-position", "manual-scram"]

    # # Dataset 2: FDI dataset
    filename_normal = './data/FDI_surrogate_signals_as.csv'
    filename_abnormal = './data/FDI_fake_scrams_ch1_ch1cr.csv'
    columns = ["nfd-1-cps", "nfd-1-cr", "rr-active-state", "rr-position",
               "ss1-active-state", "ss1-position", "ss2-active-state", "ss2-position"]  #

    # Dataset 3: CAM dataset
    # CAM dataset
    # filename_normal = './data/CAM_sensor_normal.csv'
    # filename_abnormal = './data/CAM_sensor_mixed_data.csv'
    # columns = ["CAM-CNT", "NFD-1-CPS"]

    df_normal = load_data(filename_normal, columns, 1)
    df_abnormal = load_data(filename_abnormal, columns, 1)

    print("Normal data shape:", df_normal.shape)
    print("Abnormal data shape:", df_abnormal.shape)

    # normalize the data
    train_scaled, test_scaled = normalize_data(
        df_normal, df_abnormal, 'MinMax', config['model_normalization'])

    # Prepare sequences for LSTM input
    trainX = to_sequences(train_scaled, seq_size=10)
    testX = to_sequences(test_scaled, seq_size=10)

    print('Train X shape:', trainX.shape)
    print('Test X shape:', testX.shape)

    # Define LSTM Autoencoder model
    model = Sequential([
        LSTM(128, activation='sigmoid', return_sequences=True,
             input_shape=(trainX.shape[1], trainX.shape[2])),
        Dropout(0.2),
        LSTM(64, activation='sigmoid', return_sequences=True),
        LSTM(64, activation='sigmoid', return_sequences=False),  # Bottleneck layer
        RepeatVector(trainX.shape[1]),
        LSTM(64, activation='sigmoid', return_sequences=True),
        LSTM(128, activation='sigmoid', return_sequences=True),
        Dropout(0.2),
        TimeDistributed(Dense(trainX.shape[2]))
    ])

    model.compile(optimizer='adam', loss='mse')
    model.summary()

    # Train the model
    history = model.fit(trainX, trainX, epochs=config['number_of_epochs'],
                        batch_size=32, validation_split=0.1, verbose=1)

    # Save the trained model
    if config['save_models'] == True:
        try:
            model.save('./models/' + config['model_name'])
            print("[INFO] Model saved.")
        except:
            print('[Error] Model could not be saved.')

    # Plot training history
    plot_metric(history, 'loss')

    # Calculate reconstruction errors and plot histograms
    for dataset, label, color in [(trainX, 'Normal', 'blue'), (testX, 'Abnormal', 'green')]:
        predictions = model.predict(dataset)
        reconstruction_errors = np.abs(
            predictions[:, :, config['critical_feature']] - dataset[:, :, config['critical_feature']])
        # for each row a mean reconstruction error of the 10 previous rows will be considered
        # other options are to choose the maximum error or to vote
        mae = np.mean(reconstruction_errors, axis=1)
        plt.hist(mae, bins=30, color=color, alpha=0.7)
        plt.title(f'Reconstruction Error {label} Dataset')
        plt.xlabel('Mean Absolute Error')
        plt.ylabel('Frequency')
        plt.show()

    end_time = datetime.datetime.now()
    print(f"[INFO] Training completed in: {end_time - start_time}")


if __name__ == "__main__":
    main()


'''
    @Author: XXXXXXXXX
    @Date: 

    @Description: This code for training an LSTM Autoencoder for 
    detecting abnormalities in multivariate nuclear timeseries data



    USAGE
    Copy and paste the following command in a cmd environment
    Make sure you have installed all the appropriate libaries
    I ran the code in a conda environment where I have installed tensorflow with GPU

    python reactor_anomaly_detection_v2.py 

'''
