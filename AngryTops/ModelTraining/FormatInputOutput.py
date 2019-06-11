"""Use this script to read in the panda array, seperate input and output, and
seperate test / training data from each other
Meant to be run from the parent directory
"""
import pandas as pd
import numpy as np
import sys
from features import *
from sklearn.utils import shuffle
import sklearn.preprocessing

#input_filename = "csv/topreco.csv"
#input_filename = "csv/topreco_augmented1.csv"

def get_input_output(input_filename='topreco_augmented1_5dec.csv', \
                     training_split=0.9, shuff=False, type="minmax", rep="cart"):
    """
    Return the training and testing data
    Training Data: Array of 36 elements. I am debating reshaping to matrix of (6 x 6)
    Testing Data: Matrix of Shape (4 x 6)
    """
    input_filename = "../csv/{}".format(input_filename)
    df = pd.read_csv(input_filename, names=column_names)
    # Shuffle the DataSet
    if shuff:
        df = shuffle(df)
    # Load jets, leptons and output columns
    event_info = df[features_event_info].values
    if rep == "ptetaphiE":
        jets = df[input_features_jets_ptetaphi].values
        lep = df[input_features_lep_ptetaphi].values
        truth = df[output_columns_ptetaphiE].values
    elif rep == 'ptetaphiM':
        jets = df[input_features_jets_ptetaphi].values
        lep = df[input_features_lep_ptetaphi].values
        truth = df[output_columns_ptetaphiM].values
    else:
        jets = df[input_features_jets].values
        lep = df[input_features_lep].values
        truth = df[output_columns].values
    btag = df[btags].values
    btag = btag.reshape(btag.shape[0], btag.shape[1], 1)

    # Normalize and retrieve the standard scalar
    # Note: We do not want to normalize the BTag column (the last one) in jets
    jets_momentum, jets_scalar = normalize(jets, type)
    jets_momentum = jets_momentum.reshape(jets.shape[0], 5, 5)
    jets_norm = np.concatenate((jets_momentum, btag), axis=2)

    # Lepton
    lep_norm, lep_scalar = normalize(lep, type)
    lep_norm = lep_norm.reshape(lep.shape[0], 1, lep.shape[1])

    # Combine into input and output arrays of correct shape
    # For new, we flatten the input. Can change in the future
    input = np.concatenate((lep_norm, jets_norm), axis=1)
    input = input.reshape(input.shape[0], 36)
    output, output_scalar = normalize(truth, type)
    output = output.reshape(output.shape[0], 6, 4)

    # Seperate training and testing data
    assert 0 < training_split < 1, "Invalid training_split given"
    cut = np.int(np.round(df.shape[0] * training_split))
    training_input = input[:cut]
    training_output = output[:cut]
    testing_input = input[cut:]
    testing_output = output[cut:]
    event_training = event_info[cut:]
    event_testing = event_info[:cut]
    return (training_input, training_output), (testing_input, testing_output), \
           (jets_scalar, lep_scalar, output_scalar), (event_training, event_testing)


def normalize(arr, type="minmax"):
    """Normalize the arr with StandardScalar and return the normalized array
    and the scalar"""
    if type == "standard":
        scalar = sklearn.preprocessing.StandardScaler()
    else:
        scalar = sklearn.preprocessing.MinMaxScaler()
    new_arr = scalar.fit_transform(arr)
    return new_arr, scalar

# Testing to see if this works
if __name__=='__main__':
    (training_input, training_output), (testing_input, testing_output), \
           (jets_scalar, lep_scalar, output_scalar), (event_training, event_testing)= get_input_output(rep="ptetaphiM")
    print(np.any(np.isnan(training_input)))
    print(np.any(np.isnan(training_output)))
    print(training_output[3])
