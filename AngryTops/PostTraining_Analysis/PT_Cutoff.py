"""This script finds the data points predicted by the model with pT values below
the pT cutoff"""
import numpy as np
import pandas as pd
import pickle
from AngryTops import features

def select_pT_events():
    """Create a smaller csv file where events are only those which the model
    failed to learn the pT cutoffs"""
    # Load csv file
    csv_filename = "../csv/topreco_5dec2.csv"
    data = pd.read_csv(csv_filename, names=features.column_names)
    data = data.set_index(["runNumber", "eventNumber", "target_W_had_Px"])

    # File inputs
    model_dir = 'PostTraining_Analysis/models/Retrained_LSTM'
    rep = 'pxpypzEM'
    scaling = 'minmax'
    pT_cutoff = 20

    # Load Predictions
    predictions = np.load('{}/predictions.npz'.format(model_dir))
    true = predictions['true']
    y_fitted = predictions['pred']
    event_info = predictions['events']
    old_shape = (true.shape[1], true.shape[2])

    # Scalars
    scaler_filename = "{}/scalers.pkl".format(model_dir)
    with open( scaler_filename, "rb" ) as file_scaler:
        jets_scalar = pickle.load(file_scaler)
        lep_scalar = pickle.load(file_scaler)
        output_scalar = pickle.load(file_scaler)

    # Rescale the truth array
    true = true.reshape(true.shape[0], true.shape[1]*true.shape[2])
    true = output_scalar.inverse_transform(true)
    true = true.reshape(true.shape[0], old_shape[0] * old_shape[1])

    # Rescale the fitted array
    y_fitted = y_fitted.reshape(y_fitted.shape[0], y_fitted.shape[1]*y_fitted.shape[2])
    y_fitted = output_scalar.inverse_transform(y_fitted)
    y_fitted = y_fitted.reshape(y_fitted.shape[0], old_shape[0] * old_shape[1])

    # Construct Panda DataFrame
    column_names = np.array(features.output_columns_pxpypz).flatten()
    df = pd.DataFrame(data=y_fitted, columns=column_names)

    # Add a pT column
    b_had_Pt = np.sqrt(df['target_b_had_Px']**2 + df['target_b_had_Py']**2)
    df['target_b_had_Pt'] = b_had_Pt
    target_W_had_Px = true[:,0]
    df['target_W_had_Px'] = target_W_had_Px

    # Add event info columns to dataframe
    df.insert(0, "runNumber", event_info[:,0], True)
    df.insert(1, "eventNumber", event_info[:,1], True)
    df.insert(2, "weight", event_info[:,2], True)
    df = df[df['target_b_had_Pt'] < 20]
    runNumber = np.array(df["runNumber"], dtype='int64').flatten()
    eventNumber = np.array(df["eventNumber"], dtype='int64').flatten()
    W_had_Px = np.array(df["target_W_had_Px"], dtype='float64').flatten()
    print(df.head())

    # Isolate dataframe to only those with predicted pT less than 20
    indices = [(runNumber[i], eventNumber[i], W_had_Px[i]) for i in range(len(runNumber))]
    data = data.loc[indices]
    data = data.reset_index()
    data = data.dropna(axis=0)
    print("Shape of dataframe: ", data.shape)

    # Save the smaller csv training file
    data.to_csv("../csv/b_had_pT.csv", header=False, index=False)

if __name__ == "__main__":
    select_pT_events()
