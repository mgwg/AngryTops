"""Go through all of the previous EPOCHES and output a curve of the Chi2 Values
for each histogram"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from glob import glob
import os
import pickle
from ROOT import *
from AngryTops.ModelTraining.models import models

plt.rc('legend',fontsize=22)
plt.rcParams.update({'font.size': 22})

m_t = 172.5
m_W = 80.4
m_b = 4.95

attributes = [
'W_had_px', 'W_had_py', 'W_had_pz', 'W_had_pt', 'W_had_y', 'W_had_phi', 'W_had_E',
'W_lep_px', 'W_lep_py', 'W_lep_pz', 'W_lep_pt', 'W_lep_y', 'W_lep_phi', 'W_lep_E',
'b_had_px', 'b_had_py', 'b_had_pz', 'b_had_pt', 'b_had_y', 'b_had_phi', 'b_had_E',
'b_lep_px', 'b_lep_py', 'b_lep_pz', 'b_lep_pt', 'b_lep_y', 'b_lep_phi', 'b_lep_E',
't_had_px', 't_had_py', 't_had_pz', 't_had_pt', 't_had_y','t_had_phi', 't_had_E',
't_lep_px', 't_lep_py', 't_lep_pz', 't_lep_pt', 't_lep_y', 't_lep_phi', 't_lep_E']

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
    jet = predictions['jet']
    lep = lep_scalar.inverse_transform(lep)
    jet = jet.reshape(jet.shape[0], 5, -1)
    btag = jet[:,:, -1]
    btag = btag.reshape(btag.shape[0], btag.shape[1], 1)
    jet = jet[:,:,:-1]
    jet = jet.reshape(jet.shape[0], -1)
    jet = jets_scalar.inverse_transform(jet)
    jet = jet.reshape(jet.shape[0], 5, -1)
    jet = np.concatenate((jet, btag), axis=2)
    jet = jet.reshape(jet.shape[0], -1)
    input = [lep, jet]

    print("Successfully Loaded input and output")
    # Make histogram of truth values
    truth_histograms = construct_histogram_dict(true, label='true', representation=representation)
    print("Made truth histogram")

    # Iterate over checkpoints
    xaxis = []
    checkpoints = np.sort(glob(train_dir + "/weights-improvement-*"))
    model = models[model_name](**kwargs)
    model.load_weights(train_dir + '/model_weights.h5')
    max_evals = len(checkpoints)
    if 'max_evals' in kwargs.keys(): max_evals = kwargs['max_evals']
    for k in range(max_evals):
        checkpoint_name = '.'.join(checkpoints[k].split(".")[:-1])
        print("Current CheckPoint: ", checkpoint_name)
        try:
            model.load_weights(checkpoint_name)
            y_fitted = model.predict(input)
            xaxis.append(k)
            fitted_histograms = construct_histogram_dict(y_fitted, label='fitted', representation=representation)
            for att in attributes:
                X2 = truth_histograms[att].Chi2Test(fitted_histograms[att], "UU NORM CHI2/NDF")
                chi2tests[att].append(X2)
                print("EPOCHE #: {}    Attribute: {}     X2: {:.2f}".format(k, att, X2))
            if k < 10:
              PrintOut(MakeP4(true[k,4,:], m_t, representation), MakeP4(y_fitted[k,4,:], m_t, representation))
              h_true = truth_histograms['W_had_y']
              h_fitted = fitted_histograms['W_had_y']
              SetTH1FStyle( h_true,  color=kGray+2, fillstyle=1001, fillcolor=kGray, linewidth=3)
              SetTH1FStyle( h_fitted, color=kBlack, markersize=0, markerstyle=20, linewidth=3 )

              # DRAW HISTOGRAMS
              c, pad0, pad1 = MakeCanvas()
              pad0.cd()
              gStyle.SetOptTitle(0)
              h_true.Draw("h")
              h_fitted.Draw("h same")
              hmax = 1.5 * max([h_true.GetMaximum(), h_fitted.GetMaximum()])
              h_fitted.SetMaximum(hmax)
              h_true.SetMaximum(hmax)
              h_fitted.SetMinimum(0.)
              h_true.SetMinimum(0.)

              # Legend
              leg = TLegend( 0.20, 0.80, 0.50, 0.90 )
              leg.SetFillColor(0)
              leg.SetFillStyle(0)
              leg.SetBorderSize(0)
              leg.SetTextFont(42)
              leg.SetTextSize(0.05)
              leg.AddEntry( h_true, "MG5+Py8", "f" )
              leg.AddEntry( h_fitted, "fitted", "f" )
              leg.SetY1( leg.GetY1() - 0.05 * leg.GetNRows() )
              leg.Draw()

              # Statistical tests
              KS = h_true.KolmogorovTest( h_fitted )
              X2 = h_true.Chi2Test( h_fitted, "UU NORM CHI2/NDF" ) # UU NORM
              l = TLatex()
              l.SetNDC()
              l.SetTextFont(42)
              l.SetTextColor(kBlack)
              l.DrawLatex( 0.7, 0.80, "KS test: %.2f" % KS )
              l.DrawLatex( 0.7, 0.75, "#chi^{2}/NDF = %.2f" % X2 )

              # TITLE HISTOGRAM W/ CAPTION
              gPad.RedrawAxis()
              newpad = TPad("newpad","a caption",0.1,0,1,1)
              newpad.SetFillStyle(4000)
              newpad.Draw()
              newpad.cd()
              title = TPaveLabel(0.1,0.94,0.9,0.99,caption)
              title.SetFillColor(16)
              title.SetTextFont(52)
              title.Draw()

              # SAVE AND CLOSE HISTOGRAM
              gPad.RedrawAxis()
              pad1.cd()
              yrange = [0.4, 1.6]
              frame, tot_unc, ratio = DrawRatio(h_fitted, h_true, xtitle, yrange)
              gPad.RedrawAxis()
              c.cd()
              c.SaveAs("histogram.png")
              pad0.Close()
              pad1.Close()
              c.Close()


        except Exception as e:
            print(e)
            print("Invalid checkpoint encountered. Skipping checkpoint %i" % k)

    x2_pickle = "{}/x2_epoches.pkl".format(train_dir)
    with open( x2_pickle, "wb" ) as file_scaler:
        pickle.dump(chi2tests, file_scaler, protocol=2)

    make_plots(chi2tests, xaxis, train_dir)




################################################################################
def make_plots(chi2tests, xaxis, train_dir):
    strFile = train_dir + "/x2_tests"
    if os.path.isfile(strFile):
        os.mkdir(train_dir + "/x2_tests")
    os.chdir(strFile)
    for key in chi2tests.keys():
        arr = chi2tests[key]
        plt.plot(xaxis, arr, label=key)
        plt.xlabel("EPOCH NUMBER")
        plt.ylabel("X2 Value")
        if os.path.isfile(key):
            os.remove(key)
        plt.savefig(key)




def construct_histogram_dict(arr, label, representation):
    # Divide out truth arrays
    y_W_had = arr[:,0,:]
    y_W_lep = arr[:,1,:]
    y_b_had = arr[:,2,:]
    y_b_lep = arr[:,3,:]
    y_t_had = arr[:,4,:]
    y_t_lep = arr[:,5,:]

    # Create empty histograms
    histograms = {}
    for att in attributes:
        if att[-2:] == 'px' or att[-2:] == 'py' or att[-2:] == 'pz':
            histograms[att] = TH1F(att + "_" + label,  ";" + att + " [GeV]", 50, -1000., 1000.)
        elif att[-2:] == 'pt':
            histograms[att] = TH1F(att + "_" + label,  ";" + att + " [GeV]", 50, 0., 500.)
        elif att[-1:] == 'y':
            histograms[att] = TH1F(att + "_" + label,   ";" + att + " #eta", 25, -5., 5.)
        elif att[-3:] == 'phi':
            histograms[att] = TH1F(att + "_" + label, ";" + att + " #phi", 16, -3.2, 3.2)
        else:
            histograms[att] = TH1F(att + "_" + label,   ";" + att + " [GeV]", 50, 0., 500.)

    # Iterate through events
    for i in range(arr.shape[0]):
        W_had   = MakeP4( y_W_had[i], m_W, representation)
        W_lep   = MakeP4( y_W_lep[i], m_W, representation)
        b_had   = MakeP4( y_b_had[i], m_b, representation)
        b_lep   = MakeP4( y_b_lep[i], m_b, representation)
        t_had   = MakeP4( y_t_had[i], m_t, representation)
        t_lep   = MakeP4( y_t_lep[i], m_t, representation)

        histograms['W_had_px'].Fill(  W_had.Px(),  1 )
        histograms['W_had_py'].Fill(  W_had.Py(),  1 )
        histograms['W_had_pz'].Fill(  W_had.Pz(),  1 )
        histograms['W_had_pt'].Fill(  W_had.Pt(),  1 )
        histograms['W_had_y'].Fill(   W_had.Rapidity(), 1 )
        histograms['W_had_phi'].Fill( W_had.Phi(), 1 )
        histograms['W_had_E'].Fill(   W_had.E(),   1 )

        histograms['b_had_px'].Fill(  b_had.Px(),  1 )
        histograms['b_had_py'].Fill(  b_had.Py(),  1 )
        histograms['b_had_pz'].Fill(  b_had.Pz(),  1 )
        histograms['b_had_pt'].Fill(  b_had.Pt(),  1 )
        histograms['b_had_y'].Fill(   b_had.Rapidity(), 1 )
        histograms['b_had_phi'].Fill( b_had.Phi(), 1 )
        histograms['b_had_E'].Fill(   b_had.E(),   1 )

        histograms['t_had_px'].Fill(  t_had.Px(),  1 )
        histograms['t_had_py'].Fill(  t_had.Py(),  1 )
        histograms['t_had_pz'].Fill(  t_had.Pz(),  1 )
        histograms['t_had_pt'].Fill(  t_had.Pt(),  1 )
        histograms['t_had_y'].Fill(   t_had.Rapidity(), 1 )
        histograms['t_had_phi'].Fill( t_had.Phi(), 1 )
        histograms['t_had_E'].Fill(   t_had.E(),   1 )

        histograms['W_lep_px'].Fill(  W_lep.Px(),  1 )
        histograms['W_lep_py'].Fill(  W_lep.Py(),  1 )
        histograms['W_lep_pz'].Fill(  W_lep.Pz(),  1 )
        histograms['W_lep_pt'].Fill(  W_lep.Pt(),  1 )
        histograms['W_lep_y'].Fill(   W_lep.Rapidity(), 1 )
        histograms['W_lep_phi'].Fill( W_lep.Phi(), 1 )
        histograms['W_lep_E'].Fill(   W_lep.E(),   1 )

        histograms['b_lep_px'].Fill(  b_lep.Px(),  1 )
        histograms['b_lep_py'].Fill(  b_lep.Py(),  1 )
        histograms['b_lep_pz'].Fill(  b_lep.Pz(),  1 )
        histograms['b_lep_pt'].Fill(  b_lep.Pt(),  1 )
        histograms['b_lep_y'].Fill(   b_lep.Rapidity(), 1 )
        histograms['b_lep_phi'].Fill( b_lep.Phi(), 1 )
        histograms['b_lep_E'].Fill(   b_lep.E(),   1 )

        histograms['t_lep_px'].Fill(  t_lep.Px(),  1 )
        histograms['t_lep_py'].Fill(  t_lep.Py(),  1 )
        histograms['t_lep_pz'].Fill(  t_lep.Pz(),  1 )
        histograms['t_lep_pt'].Fill(  t_lep.Pt(),  1 )
        histograms['t_lep_y'].Fill(   t_lep.Rapidity(), 1 )
        histograms['t_lep_phi'].Fill( t_lep.Phi(), 1 )
        histograms['t_lep_E'].Fill(   t_lep.E(),   1 )

    for key in histograms.keys():
        Normalize(histograms[key])
        histograms[key].Sumw2()
        histograms[key].SetMarkerColor(kRed)
        histograms[key].SetLineColor(kRed)
        histograms[key].SetMarkerStyle(24)

    return histograms

def Normalize( h, sf=1.0 ):
    if h == None: return
    A = h.Integral()
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

def PrintOut( p4_true, p4_fitted):
  print("true=( %4.1f, %3.2f, %3.2f, %4.1f ; %3.1f ) :: fitted=( %4.1f, %3.2f, %3.2f, %4.1f ; %3.1f )" % \
               (p4_true.Px(),   p4_true.Py(),   p4_true.Pz(),   p4_true.Pt(),   p4_true.Rapidity(), \
                p4_fitted.Px(), p4_fitted.Py(), p4_fitted.Pz(), p4_fitted.Pt(), p4_fitted.Rapidity() ))

if __name__ == "__main__":
    IterateEpoches('../../CheckPoints/dense_multi1.1000Epoches', 'pxpypz', 'dense_multi1', learn_rate=10e-5, max_evals=1)
