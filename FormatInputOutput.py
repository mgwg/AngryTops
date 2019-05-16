# Use this script to read in the panda array, seperate input and output, and
# seperate test / training data from each other
import pandas as pd
import numpy as np
from features import *
from sklearn.utils import shuffle

def get_input_output(training_split=0.5, shuffle=True):
    """
    Return the training and testing data
    Training Data: Array of 36 elements. I am debating reshaping to matrix of (6 x 6)
    Testing Data: Matrix of Shape (5 x 6)
    """
    df = pd.read_csv("csv/topreco.csv", names=column_names)
    # Drop Unwanted Features
    df.drop('jets_n')
    df.drop('bjets_n')
    # Shuffle the DataSet
    if shuffle:
        df = shuffle(df)
    # Seperate the input and output columns
    input = df[input_columns].values
    output = df[output_columns].values
    output = output.reshape(output.shape[0], 5, 6)
    # Seperate training and testing data
    assert 0 < training_split < 1, "Invalid training_split given"
    cut = df.shape[0] * training_split
    training_input = input[:cut]
    training_output = output[:cut]
    testing_input = input[cut:]
    testing_output = output[cut:]
    return (training_input, training_output), (testing_input, testing_output)
