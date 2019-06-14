"""Use this script to read in the panda array, seperate input and output, and
seperate test / training data from each other
Meant to be run from the parent directory
"""
import pandas as pd
import numpy as np
import sys
from AngryTops.features import *
from sklearn.utils import shuffle
import sklearn.preprocessing

def get_input_output(input_filename, training_split=0.9, **kwargs):
    """
    Return the training and testing data
    Training Data: Array of 36 elements. I am debating reshaping to matrix of (6 x 6)
    Testing Data: Matrix of Shape (6 x 4)
    """
    # Inputs
    scaling = kwargs['scaling']
    rep = kwargs['rep']
    multi_input = kwargs['multi_input']

    # Load jets, leptons and output columns of the correct representation
    input_filename = "../csv/{}".format(input_filename)
    df = pd.read_csv(input_filename, names=column_names)
    event_info = df[features_event_info].values
    lep = df[representations[rep][0]].values
    jets = df[representations[rep][1]].values
    truth = df[representations[rep][2]].values
    btag = df[btags].values
    btag = btag.reshape(btag.shape[0], btag.shape[1], 1)

    # Normalize and retrieve the standard scalar
    # Note: We do not want to normalize the BTag column (the last one) in jets
    jets_momentum, jets_scalar = normalize(jets, scaling)
    jets_momentum = jets_momentum.reshape(jets.shape[0], 5, -1)
    jets_norm = np.concatenate((jets_momentum, btag), axis=2)
    jets_norm = jets_norm.reshape(jets_norm.shape[0], jets_norm.shape[1] * jets_norm.shape[2])
    lep_norm, lep_scalar = normalize(lep, scaling)

    # INPUT
    assert 0 < training_split < 1, "Invalid training_split given"
    cut = np.int(np.round(df.shape[0] * training_split))
    if multi_input:
        input = [lep_norm, jets_norm]
        training_input = [lep_norm[:cut], jets_norm[:cut]]
        testing_input = [lep_norm[cut:], jets_norm[cut:]]
    else:
        input = np.concatenate((lep_norm, jets_norm), axis=1)
        training_input = input[:cut]
        testing_input = input[cut:]

    # OUTPUT
    output, output_scalar = normalize(truth, scaling)
    output = output.reshape(output.shape[0], 6, -1)
    training_output = output[:cut]
    testing_output = output[cut:]

    event_training = event_info[cut:]
    event_testing = event_info[:cut]
    return (training_input, training_output), (testing_input, testing_output), \
           (jets_scalar, lep_scalar, output_scalar), (event_training, event_testing)


def normalize(arr, scaling):
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
    (jets_scalar, lep_scalar, output_scalar), \
    (event_training, event_testing)= get_input_output(input_filename="topreco_5dec.csv", scaling='standard', rep="pxpypzE")
    print(np.any(np.isnan(training_input)))
    print(np.any(np.isnan(training_output)))
