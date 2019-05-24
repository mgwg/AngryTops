# Use this script to read in the panda array, seperate input and output, and
# seperate test / training data from each other
import pandas as pd
import numpy as np
from features import *
from sklearn.utils import shuffle
import sklearn.preprocessing

input_filename = "csv/topreco.csv"
#input_filename = "csv/topreco_augmented1.csv"

def get_input_output(training_split=0.9, shuff=False):
    """
    Return the training and testing data
    Training Data: Array of 36 elements. I am debating reshaping to matrix of (6 x 6)
    Testing Data: Matrix of Shape (4 x 6)
    """
    df = pd.read_csv(input_filename, names=column_names)
    # Drop Unwanted Features
    df.drop('jets_n', 1)
    df.drop('bjets_n', 1)
    # Shuffle the DataSet
    if shuff:
        df = shuffle(df)
    # Load jets, leptons and output columns
    jets = df[input_features_jets].values
    lep = df[input_features_lep].values
    truth = df[output_columns].values
    btag = jets[:,-1].reshape(jets.shape[0], 1)

    # Normalize and retrieve the standard scalar
    # Note: We do not want to normalize the BTag column (the last one) in jets
    jets_momentum, jets_scalar = normalize(jets[:,:-1])
    lep_norm, lep_scalar = normalize(lep)
    jets_norm = np.concatenate((jets_momentum, btag), axis=1)

    # Combine into input and output arrays of correct shape
    input = np.concatenate((lep_norm, jets_norm), axis=1)
    output, output_scalar = normalize(truth)
    output = output.reshape(output.shape[0], 6, 4)

    # Seperate training and testing data
    assert 0 < training_split < 1, "Invalid training_split given"
    cut = np.int(np.round(df.shape[0] * training_split))
    training_input = input[:cut]
    training_output = output[:cut]
    testing_input = input[cut:]
    testing_output = output[cut:]
    return (training_input, training_output), (testing_input, testing_output), \
           (jets_scalar, lep_scalar, output_scalar)

def normalize(arr):
    """Normalize the arr with StandardScalar and return the normalized array
    and the scalar"""
    scalar = sklearn.preprocessing.StandardScaler()
    new_arr = scalar.fit_transform(arr)
    return new_arr, scalar

# Testing to see if this works
if __name__=='__main__':
    (training_input, training_output), (testing_input, testing_output), \
           (jets_scalar, lep_scalar, output_scalar) = get_input_output()
    print(np.any(np.isnan(training_input)))
    print(np.any(np.isnan(training_output)))
    print(training_output[3])
