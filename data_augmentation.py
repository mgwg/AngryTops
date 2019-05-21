"""Data Augmentation Functions"""
import ROOT
from ROOT import TLorentzVector
from collections import Counter
from array import array
from math import pow, sqrt
import numpy as np


def RotateEvent( lep, jets, phi ):
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

    lep_new = TLorentzVector(lep)
    lep_new.RotateZ(phi)

    jets_new = []
    for j in jets:
        jets_new += [ TLorentzVector(j) ]
        j_new = jets_new[-1]
        j_new.btag = j.btag
        j_new.RotateZ(phi)

    return lep_new, jets_new
