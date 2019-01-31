import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

'''
A program that creates scaled testing data out of a given csv file
'''

# Load training data
training_data_df = pd.read_csv("prices.csv")
print(training_data_df)
training_data_trimmed_df = training_data_df.drop('symbol', axis=1)
print(training_data_trimmed_df)

# Create scaler for data
scaler = MinMaxScaler(feature_range=(0, 1))

scaled_training = scaler.fit_transform(training_data_trimmed_df)

# TODO find test data, determine which data set to use for training

# create dataframe out of scaled data so we can turn it into a csv
scaled_training_df = pd.DataFrame(scaled_training, columns=training_data_df.columns.values)

scaled_training_df.to_csv('scaled_training_data.csv', index=False)
