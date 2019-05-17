# Use this script to read in the panda array, seperate input and output, and
# seperate test / training data from each other
import pandas as pd
import numpy as np
from features import *
from sklearn.utils import shuffle

def get_input_output(training_split=0.75, shuff=True):
    """
    Return the training and testing data
    Training Data: Array of 36 elements. I am debating reshaping to matrix of (6 x 6)
    Testing Data: Matrix of Shape (4 x 6)
    """
    df = pd.read_csv("csv/topreco.csv", names=column_names)
    # Drop Unwanted Features
    df.drop('jets_n', 1)
    df.drop('bjets_n', 1)
    # Shuffle the DataSet
    if shuff:
        df = shuffle(df)
    # Seperate the input and output columns
    input = df[input_columns].values
    output = df[output_columns].values
    output = output.reshape(output.shape[0], 6, 4)
    # Seperate training and testing data
    assert 0 < training_split < 1, "Invalid training_split given"
    cut = np.int(np.round(df.shape[0] * training_split))
    training_input = input[:cut]
    training_output = output[:cut]
    testing_input = input[cut:]
    testing_output = output[cut:]
    return (training_input, training_output), (testing_input, testing_output)

# Testing to see if this works
if __name__=='__main__':
    (training_input, training_output), (testing_input, testing_output) = get_input_output()
    print(training_input.shape)
    print(training_output.shape)
    print(testing_input.shape)
    print(testing_output.shape)
