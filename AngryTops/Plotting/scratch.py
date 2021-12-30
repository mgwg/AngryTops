import os, sys
import numpy as np
from ROOT import *
import pickle
from AngryTops.Plotting.identification_helper import MakeP4
from AngryTops.features import *
from array import array

################################################################################
# CONSTANTS
infilename = "../CheckPoints/Nov06/predictions_May21.root"
representation = "pxpypzEM"
scaling = True              # whether the dataset has been passed through a scaling function or not

m_t = 172.5
m_W = 80.4
m_b = 4.95

# t = TFile.Open(infilename)
infile = TFile(infilename, "READ")
t = infile.Get("nominal")

################################################################################
# GET VALUES FROM TREE
jet1px = t.AsMatrix(["jet1_px_obs"]).flatten()
jet2px = t.AsMatrix(["jet2_px_obs"]).flatten()
jet3px = t.AsMatrix(["jet3_px_obs"]).flatten()
jet4px = t.AsMatrix(["jet4_px_obs"]).flatten()
jet5px = t.AsMatrix(["jet5_px_obs"]).flatten()

jet1py = t.AsMatrix(["jet1_py_obs"]).flatten()
jet2py = t.AsMatrix(["jet2_py_obs"]).flatten()
jet3py = t.AsMatrix(["jet3_py_obs"]).flatten()
jet4py = t.AsMatrix(["jet4_py_obs"]).flatten()
jet5py = t.AsMatrix(["jet5_py_obs"]).flatten()

jet1pz = t.AsMatrix(["jet1_pz_obs"]).flatten()
jet2pz = t.AsMatrix(["jet2_pz_obs"]).flatten()
jet3pz = t.AsMatrix(["jet3_pz_obs"]).flatten()
jet4pz = t.AsMatrix(["jet4_pz_obs"]).flatten()
jet5pz = t.AsMatrix(["jet5_pz_obs"]).flatten()

jet1pt = t.AsMatrix(["jet1_pt_obs"]).flatten()
jet2pt = t.AsMatrix(["jet2_pt_obs"]).flatten()
jet3pt = t.AsMatrix(["jet3_pt_obs"]).flatten()
jet4pt = t.AsMatrix(["jet4_pt_obs"]).flatten()
jet5pt = t.AsMatrix(["jet5_pt_obs"]).flatten()

jet1p = np.stack([jet1px, jet1py, jet1pz], axis=1)
jet2p = np.stack([jet2px, jet2py, jet2pz], axis=1)
jet3p = np.stack([jet3px, jet3py, jet3pz], axis=1)
jet4p = np.stack([jet4px, jet4py, jet4pz], axis=1)
jet5p = np.stack([jet5px, jet5py, jet5pz], axis=1)

jet1m = t.AsMatrix(["jet1_m_obs"]).flatten()
jet2m = t.AsMatrix(["jet2_m_obs"]).flatten()
jet3m = t.AsMatrix(["jet3_m_obs"]).flatten()
jet4m = t.AsMatrix(["jet4_m_obs"]).flatten()
jet5m = t.AsMatrix(["jet5_m_obs"]).flatten()

jet1btag = t.AsMatrix(["jet1_btag_obs"]).flatten()
jet2btag = t.AsMatrix(["jet2_btag_obs"]).flatten()
jet3btag = t.AsMatrix(["jet3_btag_obs"]).flatten()
jet4btag = t.AsMatrix(["jet4_btag_obs"]).flatten()
jet5btag = t.AsMatrix(["jet5_btag_obs"]).flatten()

hadWpt = t.AsMatrix(["W_had_pt_true"]).flatten()
hadbpt = t.AsMatrix(["b_had_pt_true"]).flatten()
lepbpt = t.AsMatrix(["b_lep_pt_true"]).flatten()

n_events = t.GetEntries()

jets = []
for i in range(n_events):
    jet1 = MakeP4(jet1p[i], jet1m[i], representation)
    jet2 = MakeP4(jet2p[i], jet2m[i], representation)
    jet3 = MakeP4(jet3p[i], jet3m[i], representation)
    jet4 = MakeP4(jet4p[i], jet4m[i], representation)
    jet5 = MakeP4(jet5p[i], jet5m[i], representation)
    jets.append([jet1, jet2, jet3, jet4, jet5])

jets_btag = np.stack([jet1btag, jet2btag, jet3btag, jet4btag, jet5btag], axis = 1)
# find index of btagged jets
btag = np.where(jets_btag) # returns array of event number and jet_number giving the index of b-tagged jets
# find index of non-btagged jets
non_btag = np.where(jets_btag==0) 