"""Data Augmentation Functions"""
import ROOT
from ROOT import TLorentzVector
from collections import Counter
from array import array
from math import pow, sqrt
import numpy as np
from sklearn import preprocessing


def RotateEvent(lep, jets, met, phi):
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
    met_phi = TVector2.Phi_mpi_pi(met_phi_original + phi)

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

    return lep_new, jets_new

def FlipEta(lep, jets):
    lep = let.SetPtEtaPhiE(lep.Pt(), -lep.Eta(), lep.Phi(), lep.E())
    for lj in jets:
        lj.SetPtEtaPhiE(lj.Pt(), -lj.Eta(), lj.Phi(), lj.E())
