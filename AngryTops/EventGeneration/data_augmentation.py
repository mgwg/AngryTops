"""Data Augmentation Functions"""
import ROOT
from ROOT import TLorentzVector, TVector2
from collections import Counter
from array import array
from math import pow, sqrt
import numpy as np
from sklearn import preprocessing


def RotateEvent(lep, jets, met_phi, phi):
    """Takes in LorentzVector lep, jets and rotates each along the Z axis by
    an angle phi
    @==========================================================
    @ Parameters
    lep: LorentzVector containing lepton information
    jets: Array of LorentzVectors containing jet information
    phi: Angle between 0 and 2 pi
    @==========================================================
    @ Return
    A rotated LorentzVector
    """
    # Missing Azimuthal Energy
    met_phi = TVector2.Phi_mpi_pi(met_phi + phi)

    # Lepton
    lep_new = TLorentzVector(lep)
    lep_new.RotateZ(phi)

    # Jets
    jets_new = []
    for j in jets:
        jets_new += [ TLorentzVector(j) ]
        j_new = jets_new[-1]
        j_new.btag = j.btag
        j_new.RotateZ(phi)

    return lep_new, jets_new, met_phi

def FlipEta(lep, jets):
    lep_new = TLorentzVector()
    lep_new.SetPtEtaPhiE(lep.Pt(), -lep.Eta(), lep.Phi(), lep.E())
    new_jets = []
    for lj in jets:
        new_jets += [TLorentzVector(lj)]
        new_jet = new_jets[-1]
        new_jet.SetPtEtaPhiE(lj.Pt(), -lj.Eta(), lj.Phi(), lj.E())
        new_jet.btag = lj.btag

    return lep_new, new_jets
