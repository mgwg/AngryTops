#!/usr/bin/env python
import ROOT
from ROOT import TLorentzVector
from collections import Counter
from array import array
from math import pow, sqrt
import numpy as np

GeV = 1e3
TeV = 1e6
n_jets_per_event = 5
n_features_per_jet = 9

#####################

def Normalize(h, sf=1.0, opt="width"):
    area = h.Integral(opt)
    h.Scale(sf / area)

#~~~~~~~~~~~~~~~~~~~~~~


def FlipEta(ljets=[]):
    for lj in ljets:
        lj.SetPtEtaPhiE(lj.Pt(), -lj.Eta(), lj.Phi(), lj.E())

###############################
# ADDED BY FARDIN

def MakeInput( jets, W_had, b_had, t_had, W_lep, b_lep, t_lep ):
    """Format output, returning a jet matrix (detector level), W, b quark and
    lepton arrays (particle leve) and t-quark momenta (parton level)
    """
    # INPUT DATA FOR RNN
    # Populate 5 x 6 matrix of jet foration
    # Is there possibly a bug here? What format is Px? Is sjets[0][0] = sjets[1][0] = sjets[2][0] = ...
    sjets = np.zeros( [ n_jets_per_event, n_features_per_jet ] )
    # At first, everything was divided by GeV. This shouldn't be the case, as
    # the Delphes Tree already outputs everything in GeV according to the
    # documentation
    for i in range(len(jets)):
        jet = jets[i]
        sjets[i][0] = jet.Px()
        sjets[i][1] = jet.Py()
        sjets[i][2] = jet.Pz()
        sjets[i][3] = jet.E()
        sjets[i][4] = jet.M()
        sjets[i][5] = jet.btag
        sjets[i][6] = jet.Pt()
        sjets[i][7] = jet.Eta()
        sjets[i][8] = jet.Phi()
        sjets[i][9] = jet.

    # OUTPUT DATA FOR RNN
    # Arrays containing W, b quarks and lepton information
    target_W_had = np.zeros([8])
    target_b_had = np.zeros([8])
    target_t_had = np.zeros([8])
    target_W_lep = np.zeros([8])
    target_b_lep = np.zeros([8])
    target_t_lep = np.zeros([8])

    # T Hadron
    target_t_had[0] = t_had.Px()
    target_t_had[1] = t_had.Py()
    target_t_had[2] = t_had.Pz()
    target_t_had[3] = t_had.E()
    target_t_had[4] = t_had.M()
    target_t_had[5] = t_had.Pt()
    target_t_had[6] = t_had.Eta()
    target_t_had[7] = t_had.Phi()

    # Hadronically Decay W Boson
    target_W_had[0] = W_had.Px()
    target_W_had[1] = W_had.Py()
    target_W_had[2] = W_had.Pz()
    target_W_had[3] = W_had.E()
    target_W_had[4] = W_had.M()
    target_W_had[5] = W_had.Pt()
    target_W_had[6] = W_had.Eta()
    target_W_had[7] = W_had.Phi()

    # Hadronic b quark
    target_b_had[0] = b_had.Px()
    target_b_had[1] = b_had.Py()
    target_b_had[2] = b_had.Pz()
    target_b_had[3] = b_had.E()
    target_b_had[4] = b_had.M()
    target_b_had[5] = b_had.Pt()
    target_b_had[6] = b_had.Eta()
    target_b_had[7] = b_had.Phi()

    # Leptonic t quark
    target_t_lep[0] = t_lep.Px()
    target_t_lep[1] = t_lep.Py()
    target_t_lep[2] = t_lep.Pz()
    target_t_lep[3] = t_lep.E()
    target_t_lep[4] = t_lep.M()
    target_t_lep[5] = t_lep.Pt()
    target_t_lep[6] = t_lep.Eta()
    target_t_lep[7] = t_lep.Phi()

    # Leptonically decaying W quark
    target_W_lep[0] = W_lep.Px()
    target_W_lep[1] = W_lep.Py()
    target_W_lep[2] = W_lep.Pz()
    target_W_lep[3] = W_lep.E()
    target_W_lep[4] = W_lep.M()
    target_W_lep[5] = W_lep.Pt()
    target_W_lep[6] = W_lep.Eta()
    target_W_lep[7] = W_lep.Phi()

    # Leptonic b quark
    target_b_lep[0] = b_lep.Px()
    target_b_lep[1] = b_lep.Py()
    target_b_lep[2] = b_lep.Pz()
    target_b_lep[3] = b_lep.E()
    target_b_lep[4] = b_lep.M()
    target_b_lep[5] = b_lep.Pt()
    target_b_lep[6] = b_lep.Eta()
    target_b_lep[7] = b_lep.Phi()

    return sjets, target_W_had, target_b_had, target_t_had, target_W_lep, target_b_lep, target_t_lep
