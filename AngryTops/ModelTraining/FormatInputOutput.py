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

def get_input_output(input_filename, training_split=0.9, single_output=None, **kwargs):
    """
    Return the training and testing data
    Training Data: Array of 36 elements. I am debating reshaping to matrix of (6 x 6)
    Testing Data: Matrix of Shape (6 x 4)
    """
    # Inputs
    scaling = kwargs['scaling']
    rep = kwargs['rep']
    multi_input = kwargs['multi_input']
    sort_jets = kwargs['sort_jets']
    if 'single_output' in kwargs.keys(): single_output = kwargs['single_output']

    # Load jets, leptons and output columns of the correct representation
    input_filename = "/home/fsyed/AngryTops/csv/{}".format(input_filename)
    df = pd.read_csv(input_filename, names=column_names)
    if 'shuffle' in kwargs.keys():
        print("Shuffling training/testing data")
        df = shuffle(df)
    lep = df[representations[rep][0]].values
    jets = df[representations[rep][1]].values
    if single_output is None:
        truth = df[representations[rep][2]].values
    else:
        truth = df[single_output].values
        truth = truth.reshape(truth.shape[0], -1)
    btag = df[btags].values
    btag = btag.reshape(btag.shape[0], btag.shape[1], 1)

    # Normalize and retrieve the standard scalar
    # Note: We do not want to normalize the BTag column (the last one) in jets
    jets_momentum, jets_scalar = normalize(jets, scaling)
    jets_momentum = jets_momentum.reshape(jets.shape[0], 5, -1)
    jets_norm = np.concatenate((jets_momentum, btag), axis=2)
    if sort_jets:
        for i in range(jets_norm.shape[0]):
            jets_norm[i] = jets_norm[i][jets_norm[i][:,-2].argsort(kind='mergesort')]
            jets_norm[i] = jets_norm[i][jets_norm[i][:,-1].argsort(kind='mergesort')]
    jets_norm = jets_norm.reshape(jets_norm.shape[0], jets_norm.shape[1] * jets_norm.shape[2])
    lep_norm, lep_scalar = normalize(lep, scaling)

    # CUT
    assert 0 < training_split < 1, "Invalid training_split given"
    print("Training_split: ", training_split)
    cut = np.int(np.round(df.shape[0] * training_split))

    # MET Info
    met_info = df[input_event_info]
    training_event_info = met_info[:cut]
    testing_event_info = met_info[cut:]
    if multi_input:
        training_input = [training_event_info, jets_norm[:cut]]
        testing_input = [testing_event_info, jets_norm[cut:]]
    else:
        input = np.concatenate((lep_norm, jets_norm), axis=1)
        #input = input[input[:,-2].argsort(kind='mergesort')]
        #input = input[input[:,-1].argsort(kind='mergesort')]
        training_input = input[:cut]
        testing_input = input[cut:]

    # EVENT INFO
    event_info = df[features_event_info].values
    event_training = event_info[:cut]
    event_testing = event_info[cut:]

    # OUTPUT
    output, output_scalar = normalize(truth, scaling)
    if not single_output:
        output = output.reshape(output.shape[0], 6, -1)
    training_output = output[:cut]
    testing_output = output[cut:]

    return (training_input, training_output), (testing_input, testing_output), \
           (jets_scalar, lep_scalar, output_scalar), (event_training, event_testing)


def normalize(arr, scaling):
    """Normalize the arr with StandardScalar and return the normalized array
    and the scalar"""
    if scaling == "standard":
        scalar = sklearn.preprocessing.StandardScaler()
    elif scaling == 'minmax':
        scalar = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
    else:
        return arr.copy(), None
    new_arr = scalar.fit_transform(arr)
    return new_arr, scalar

# Testing to see if this works
if __name__=='__main__':
    (training_input, training_output), (testing_input, testing_output), \
           (jets_scalar, lep_scalar, output_scalar), (event_training, event_testing) = \
    get_input_output(input_filename="topreco_5dec2.csv", scaling='minmax',
    rep="pxpypzEM", multi_input=True, sort_jets=False)
    print(event_training.shape)
    print(event_testing.shape)
