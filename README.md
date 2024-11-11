# LSTM-Autoencoder-anomaly-detection
2 codes were implemented - one for the training - one for evaluating the abnormal events
The code for the evaluation is repeated 3 times one for each dataset, but with the dataset details
Simply copy and paste the follwoing in a cmd and the results will pop up. 
Every code has already the predefined paths for the datasets and the reading of the models

#1.   reactor_anomaly_detection_training.py 

This code to train the network 
uncomment the different datasets
each dataset will export a model for the training, a model for the normalization of the data, and a reconstruction error distribution


#2. reactor_anomaly_detection_evaluate_SCRAM.py

This code reads the model exported during training for the SCRAM dataset
Reads the mix SCRAM dataset with the appropriate features and exports a graph with the results


#3. reactor_anomaly_detection_evaluate_FDI.py

This code for the FDI dataset

#4. reactor_anomaly_detection_evaluate_CAM.py
This code for the CAM dataset



