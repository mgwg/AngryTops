"""Go through all of the previous EPOCHES and output a curve of the Chi2 Values
for each histogram"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from glob import glob
import os
from ROOT import *
from AngryTops.ModelTraining.models import models

plt.rc('legend',fontsize=22)
plt.rcParams.update({'font.size': 22})

m_t = 172.5
m_W = 80.4
m_b = 4.95

attributes = [
'W_had_px', 'W_had_py', 'W_had_pz', 'W_had_pt', 'W_had_y', 'W_had_y', 'W_had_phi', 'W_had_E',
'W_lep_px', 'W_lep_py', 'W_lep_pz', 'W_lep_pt', 'W_lep_y', 'W_lep_y', 'W_lep_phi', 'W_lep_E',
'b_had_px', 'b_had_py', 'b_had_pz', 'b_had_pt', 'b_had_y', 'b_had_y', 'b_had_phi', 'b_had_E',
'b_lep_px', 'b_lep_py', 'b_lep_pz', 'b_lep_pt', 'b_lep_y', 'b_lep_y', 'b_lep_phi', 'b_lep_E',
't_had_px', 't_had_py', 't_had_pz', 't_had_pt', 't_had_y', 't_had_y', 't_had_phi', 't_had_E',
't_lep_px', 't_lep_py', 't_lep_pz', 't_lep_pt', 't_lep_y', 't_lep_y', 't_lep_phi', 't_lep_E']

def IterateEpoches(train_dir, representation, model_name, **kwargs):
    # Dictionary of
    chi2tests = {}
    for att in attributes:
        chi2tests[att] = []
    # Load Scalars + Predictions
    predictions = np.load('{}/predictions.npz'.format(train_dir))
    scaler_filename = "{}/scalers.pkl".format(train_dir)
    with open( scaler_filename, "rb" ) as file_scaler:
      jets_scalar = pickle.load(file_scaler)
      lep_scalar = pickle.load(file_scaler)
      output_scalar = pickle.load(file_scaler)

    # Load Truth Array
    true = predictions['true']
    old_shape = (true.shape[1], true.shape[2])
    true = true.reshape(true.shape[0], true.shape[1]*true.shape[2])
    true = output_scalar.inverse_transform(true)
    true = true.reshape(true.shape[0], old_shape[0], old_shape[1])

    # Load Input
    lep = predictions['lep']
    jet = predictions['jets']
    lep = lep_scalar.inverse_transform(lep)
    jet = jets_scalar.inverse_transform(jet)
    input = [lep, jet]

    # Make histogram of truth values
    truth_histograms = construct_histogram_dict(true, label='true')

    # Iterate over checkpoints
    checkpoints = np.sort(glob(train_dir + "/weights-improvement-*"))
    model = models[model_name](**kwargs)
    model = model.load_weights(train_dir + '/model_weights.h5')
    for checkpoint in checkpoints:
        print("Current CheckPoint: ", checkpoint)
        model.load_weights(checkpoint)
        y_fitted = model.predict(input)
        fitted_histograms = construct_histogram_dict(y_fitted, label='fitted')
        for att in attributes:
            X2 = truth_histograms[att].Chi2Test(fitted_histograms[att], "UU NORM CHI2/NDF")
            chi2tests[att].append(X2)

    x2_pickle = "{}/x2_epoches.pkl".format(train_dir)
    with open( x2_pickle, "wb" ) as file_scaler:
        pickle.dump(chi2tests, file_scaler, protocol=2)

    make_plots(chi2tests)




################################################################################
def make_plots(chi2tests):
    os.mkdir(train_dir + "/x2_tests")
    os.chdir(train_dor + "/x2_tests")
    for key in chi2tests.keys():
        arr = chi2tests[key]
        xaxis = np.arange(1, arr.size + 1, 1)
        plt.plot(xaxis, arr, label=key)
        plt.xlabel("EPOCH NUMBER")
        plt.ylabel("X2 Value")
        plt.savefig(key)




def construct_histogram_dict(true, label):
    # Divide out truth arrays
    y_W_had = true[:,0,:]
    y_W_lep = true[:,1,:]
    y_b_had = true[:,2,:]
    y_b_lep = true[:,3,:]
    y_t_had = true[:,4,:]
    y_t_lep = true[:,5,:]

    # Create empty histograms
    truth_histograms = {}
    for att in attributes:
        if att[-2:] == 'px' or att[-2:] == 'py' or att[-2:] == 'pz':
            truth_histograms[att] = TH1F(att + "_" + label,  ";" + att + " [GeV]", 50, -1000., 1000.)
        elif att[-2:] == 'pt':
            truth_histograms[att] = TH1F(att + "_" + label,  ";" + att + " [GeV]", 50, 0., 500.)
        elif att[-2:] == 'y':
            truth_histograms[att] = TH1F(att + "_" + label,   ";" + att + " #eta", 25, -5., 5.)
        elif att[-3:] == 'phi':
            truth_histograms[att] = TH1F(att + "_" + label, ";" + att + " #phi", 16, -3.2, 3.2)
        else:
            truth_histograms[att] = TH1F(att + "_" + label,   ";" + att + " [GeV]", 50, 0., 500.)

    # Iterate through events
    for i in range(true.shape[0]):
        W_had   = MakeP4( y_W_had[i], m_W, representation)
        W_lep   = MakeP4( y_W_lep[i], m_W, representation)
        b_had   = MakeP4( y_b_had[i], m_b, representation)
        b_lep   = MakeP4( y_b_lep[i], m_b, representation)
        t_had   = MakeP4( y_t_had[i], m_t, representation)
        t_lep   = MakeP4( y_t_lep[i], m_t, representation)

        truth_histograms['W_had_px'].Fill(  W_had.Px(),  1 )
        truth_histograms['W_had_py'].Fill(  W_had.Py(),  1 )
        truth_histograms['W_had_pz'].Fill(  W_had.Pz(),  1 )
        truth_histograms['W_had_pt'].Fill(  W_had.Pt(),  1 )
        truth_histograms['W_had_y'].Fill(   W_had.Rapidity(), 1 )
        truth_histograms['W_had_phi'].Fill( W_had.Phi(), 1 )
        truth_histograms['W_had_E'].Fill(   W_had.E(),   1 )
        truth_histograms['W_had_m'].Fill(   W_had.M(),   1 )

        truth_histograms['b_had_px'].Fill(  b_had.Px(),  1 )
        truth_histograms['b_had_py'].Fill(  b_had.Py(),  1 )
        truth_histograms['b_had_pz'].Fill(  b_had.Pz(),  1 )
        truth_histograms['b_had_pt'].Fill(  b_had.Pt(),  1 )
        truth_histograms['b_had_y_'].Fill(   b_had.Rapidity(), 1 )
        truth_histograms['b_had_phi'].Fill( b_had.Phi(), 1 )
        truth_histograms['b_had_E'].Fill(   b_had.E(),   1 )
        truth_histograms['b_had_m'].Fill(   b_had.M(),   1 )

        truth_histograms['t_had_px'].Fill(  t_had.Px(),  1 )
        truth_histograms['t_had_py'].Fill(  t_had.Py(),  1 )
        truth_histograms['t_had_pz'].Fill(  t_had.Pz(),  1 )
        truth_histograms['t_had_pt'].Fill(  t_had.Pt(),  1 )
        truth_histograms['t_had_y'].Fill(   t_had.Rapidity(), 1 )
        truth_histograms['t_had_phi'].Fill( t_had.Phi(), 1 )
        truth_histograms['t_had_E'].Fill(   t_had.E(),   1 )
        truth_histograms['t_had_m'].Fill(   t_had.M(),   1 )

        truth_histograms['W_lep_px'].Fill(  W_lep.Px(),  1 )
        truth_histograms['W_lep_py'].Fill(  W_lep.Py(),  1 )
        truth_histograms['W_lep_pz'].Fill(  W_lep.Pz(),  1 )
        truth_histograms['W_lep_pt'].Fill(  W_lep.Pt(),  1 )
        truth_histograms['W_lep_y'].Fill(   W_lep.Rapidity(), 1 )
        truth_histograms['W_lep_phi'].Fill( W_lep.Phi(), 1 )
        truth_histograms['W_lep_E'].Fill(   W_lep.E(),   1 )
        truth_histograms['W_lep_m'].Fill(   W_lep.M(),   1 )

        truth_histograms['b_lep_px'].Fill(  b_lep.Px(),  1 )
        truth_histograms['b_lep_py'].Fill(  b_lep.Py(),  1 )
        truth_histograms['b_lep_pz'].Fill(  b_lep.Pz(),  1 )
        truth_histograms['b_lep_pt'].Fill(  b_lep.Pt(),  1 )
        truth_histograms['b_lep_y'].Fill(   b_lep.Rapidity(), 1 )
        truth_histograms['b_lep_phi'].Fill( b_lep.Phi(), 1 )
        truth_histograms['b_lep_E'].Fill(   b_lep.E(),   1 )
        truth_histograms['b_lep_m'].Fill(   b_lep.M(),   1 )

        truth_histograms['t_lep_px'].Fill(  t_lep.Px(),  1 )
        truth_histograms['t_lep_py'].Fill(  t_lep.Py(),  1 )
        truth_histograms['t_lep_pz'].Fill(  t_lep.Pz(),  1 )
        truth_histograms['t_lep_pt'].Fill(  t_lep.Pt(),  1 )
        truth_histograms['t_lep_y'].Fill(   t_lep.Rapidity(), 1 )
        truth_histograms['t_lep_phi'].Fill( t_lep.Phi(), 1 )
        truth_histograms['t_lep_E'].Fill(   t_lep.E(),   1 )
        truth_histograms['t_lep_m'].Fill(   t_lep.M(),   1 )
    return truth_histograms

def Normalize( h, sf=1.0 ):
  if h == None: return
  A = h.Integral()
  if A == 0.: return
  h.Scale( sf / A )

def MakeP4(y, m, representation):
    p4 = TLorentzVector()
    p0 = y[0]
    p1 = y[1]
    p2 = y[2]
    if representation == "pxpypzE":
        E  = y[3]
        p4.SetPxPyPzE(p0, p1, p2, E)
    elif representation == "ptetaphiE":
        E  = y[3]
        p4.SetPtEtaPhiE(p0, p1, p2, E)
    elif representation == "ptetaphiM":
        M  = y[3]
        p4.SetPtEtaPhiM(p0, p1, p2, M)
    elif representation == "pxpypz":
        E = np.sqrt(p0**2 + p1**2 + p2**2 + m**2)
        p4.SetPxPyPzE(p0, p1, p2, E)
    elif representation == "ptetaphi":
        p4.SetPtEtaPhiM(p0, p1, p2, m)
    else:
        raise Exception("Invalid Representation Given: {}".format(representation))
    return p4

if __name__ == "__main__":
    IterateEpoches(train_dir, representation, model_name)
