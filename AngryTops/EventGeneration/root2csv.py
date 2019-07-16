#!/usr/bin/env python
import os, sys
import csv

import ROOT
from ROOT import TLorentzVector, gROOT, TChain, TVector2
from tree_traversal import GetIndices
from helper_functions import *
import numpy as np
import traceback
from data_augmentation import *

gROOT.SetBatch(True)

from AngryTops.features import *

###############################
# CONSTANTS
GeV = 1e3
TeV = 1e6

# Artificially increase training data size by 5 by rotating events differently 5 different ways
n_data_aug = 1
if len(sys.argv) > 3:
    n_data_aug = int(sys.argv[3])

# Maximum number of entries
n_evt_max = -1
#if len(sys.argv) > 2: n_evt_max = int( sys.argv[2] )

###############################
# BUILDING OUTPUTFILE

# List of filenames
filelistname = sys.argv[1]

# Output filename
outfilename = sys.argv[2]
#outfilename = "csv/topreco_augmented1.csv"
#outfilename = "csv/topreco.csv"
outfile = open( outfilename, "wt" )
csvwriter = csv.writer( outfile )
print ("INFO: output file:", outfilename)

###############################
# BUILDING OUTPUTFILE

# Not entirely sure how TCHAIN works, but I am guessing the truth/nominal input determines which file it loads in?
tree = TChain("Delphes", "Delphes")
f = open( filelistname, 'r' )
for fname in f.readlines():
   fname = fname.strip()
   tree.AddFile( fname )

n_entries = tree.GetEntries()
print("INFO: entries found:", n_entries)

###############################
# LOOPING THROUGH EVENTS

# Cap on number of reconstructed events.
if n_evt_max > 0: n_entries = min( [ n_evt_max, n_entries ] )
print("INFO: looping over %i reco-level events" % n_entries)
print("INFO: using data augmentation: rotateZ %ix" % n_data_aug)

# Number of events which are actually copied over
n_good = 0

# Looping through the reconstructed entries
for ientry in range(n_entries):
    ##############################################################
    # Withdraw next event
    tree.GetEntry(ientry)
    runNumber = tree.GetTreeNumber()
    eventNumber = tree.GetLeaf("Event.Number").GetValue()
    weight = tree.GetLeaf("Event.Weight").GetValue()

    # Printing how far along in the loop we are
    if (n_entries < 10) or ((ientry+1) % int(float(n_entries)/10.) == 0):
        perc = 100. * ientry / float(n_entries)
        print("INFO: Event %-9i  (%3.0f %%)" % (ientry, perc))

    # Number of muons, leptons, jets and bjets (bjet_n set later)
    # For now, I am cutting out reactions with electrons, or more than two
    mu_n = tree.GetLeaf("Muon.PT").GetLen()
    jets_n  = tree.GetLeaf("Jet.PT").GetLen()
    bjets_n = 0

    # If more than one lepton of less than 4 jets, cut
    if mu_n != 1:
        print("Incorrect number of muons: {}. Applying cuts".format(mu_n))
        continue
    if jets_n < 4:
        print("Missing jets: {}. Applying cuts".format(jets_n))
        continue

    ##############################################################
    # Muon vector. Replaced E w/ T
    lep = TLorentzVector()
    lep.SetPtEtaPhiE( tree.GetLeaf("Muon.PT").GetValue(0),
                      tree.GetLeaf("Muon.Eta").GetValue(0),
                      tree.GetLeaf("Muon.Phi").GetValue(0),
                      0
                      )
    lep.sumPT = tree.GetLeaf("Muon.SumPt").GetValue(0)
    if lep.Pt() < 20:
        print("Lepton PT below threshold: {}. Applying cuts".format(lep.Pt()))
        continue # Fail to get a muon passing the threshold
    if np.abs(lep.Eta()) > 2.5:
        print("Lepton Eta value above threshold: {}. Apply cuts".format(lep.Eta()))
        continue

    # Missing Energy values
    met_met = tree.GetLeaf("MissingET.MET").GetValue(0)
    met_phi = tree.GetLeaf("MissingET.Phi").GetValue(0)
    met_eta = tree.GetLeaf("MissingET.Eta").GetValue(0)

    # Append jets, check prob of being a bjet, and update bjet number
    # This is what will be fed into the RNN
    # Replaced the mv2c10 value with the bjet tag value, as that is what is
    # recoreded by Delphes
    jets = []
    for i in range(jets_n):
        if i >= n_jets_per_event: break
        if tree.GetLeaf("Jet.PT").GetValue(i) > 20 or np.abs(tree.GetLeaf("Jet.Eta").GetValue(i)) < 2.5 :
            jets += [ TLorentzVector() ]
            j = jets[-1]
            j.index = i
            j.SetPtEtaPhiM(
            tree.GetLeaf("Jet.PT").GetValue(i),
            tree.GetLeaf("Jet.Eta").GetValue(i),
            tree.GetLeaf("Jet.Phi").GetValue(i),
            tree.GetLeaf("Jet.Mass").GetValue(i))
            j.btag = tree.GetLeaf("Jet.BTag").GetValue(i)
            j.charge = tree.GetLeaf("Jet.Charge").GetValue(i)
            j.nneutrals = tree.GetLeaf("Jet.NNeutrals").GetValue(i)
            j.deltaeta = tree.GetLeaf("Jet.DeltaEta").GetValue(i)
            j.deltaphi = tree.GetLeaf("Jet.DeltaPhi").GetValue(i)
            if j.btag > 0.0: bjets_n += 1
    # Cut based on number of passed jets
    jets_n = len(jets)
    if jets_n < 4:
        print("Missing jets: {}. Applying cuts".format(jets_n))
        continue

    ##############################################################
    # Build output data we are trying to predict with RNN
    try:
        indices = GetIndices(tree, ientry)
    except Exception as e:
        print("Exception thrown when retrieving indices")
        print(e)
        print(traceback.format_exc())
        continue

    t_had = TLorentzVector()
    t_lep = TLorentzVector()
    W_had = TLorentzVector()
    W_lep = TLorentzVector()
    b_had = TLorentzVector()
    b_lep = TLorentzVector()

    t_had.SetPtEtaPhiM( tree.GetLeaf("Particle.PT").GetValue(indices['t_had']),
                        tree.GetLeaf("Particle.Eta").GetValue(indices['t_had']),
                        tree.GetLeaf("Particle.Phi").GetValue(indices['t_had']),
                        tree.GetLeaf("Particle.Mass").GetValue(indices['t_had'])
                        )

    W_had.SetPtEtaPhiM( tree.GetLeaf("Particle.PT").GetValue(indices['W_had']),
                        tree.GetLeaf("Particle.Eta").GetValue(indices['W_had']),
                        tree.GetLeaf("Particle.Phi").GetValue(indices['W_had']),
                        tree.GetLeaf("Particle.Mass").GetValue(indices['W_had'])
                        )

    t_lep.SetPtEtaPhiM( tree.GetLeaf("Particle.PT").GetValue(indices['t_lep']),
                        tree.GetLeaf("Particle.Eta").GetValue(indices['t_lep']),
                        tree.GetLeaf("Particle.Phi").GetValue(indices['t_lep']),
                        tree.GetLeaf("Particle.Mass").GetValue(indices['t_lep']))

    W_lep.SetPtEtaPhiM( tree.GetLeaf("Particle.PT").GetValue(indices['W_lep']),
                        tree.GetLeaf("Particle.Eta").GetValue(indices['W_lep']),
                        tree.GetLeaf("Particle.Phi").GetValue(indices['W_lep']),
                        tree.GetLeaf("Particle.Mass").GetValue(indices['W_lep']))

    b_had.SetPtEtaPhiM( tree.GetLeaf("Particle.PT").GetValue(indices['b_had']),
                        tree.GetLeaf("Particle.Eta").GetValue(indices['b_had']),
                        tree.GetLeaf("Particle.Phi").GetValue(indices['b_had']),
                        tree.GetLeaf("Particle.Mass").GetValue(indices['b_had']))

    b_lep.SetPtEtaPhiM( tree.GetLeaf("Particle.PT").GetValue(indices['b_lep']),
                        tree.GetLeaf("Particle.Eta").GetValue(indices['b_lep']),
                        tree.GetLeaf("Particle.Phi").GetValue(indices['b_lep']),
                        tree.GetLeaf("Particle.Mass").GetValue(indices['b_lep']))

    ##############################################################
    # CUTS USING PARTICLE LEVEL OBJECTS
    if (t_had.Pz() == 0.) or (t_had.M() != t_had.M()):
        print("Invalid t_had values, P_z = {0}, M = {1}".format(t_had.Pz(), t_had.M()))
        continue
    if (t_lep.Pz() == 0.) or (t_lep.M() != t_lep.M()):
        print("Invalid t_lep values, P_z = {0}, M = {1}".format(t_lep.Pz(), t_lep.M()))
        continue
    if W_had.Pt() < 20:
        print("Invalid W_had.pt: {}".format(W_had.Pt()))
        continue
    if W_lep.Pt() < 20:
        print("Invalid W_lep.pt: {}".format(W_lep.Pt()))
        continue
    if b_had.Pt() < 20:
        print("Invalid b_had.pt: {}".format(b_had.Pt()))
        continue
    if b_lep.Pt() < 20:
        print("Invalid b_lep.pt: {}".format(b_lep.Pt()))
        continue
    if np.abs(W_had.Eta()) > 2.5:
        print("Invalid W_had.eta: {}".format(W_had.Eta()))
        continue
    if np.abs(W_lep.Eta()) > 2.5:
        print("Invalid W_lep.eta: {}".format(W_lep.Eta()))
        continue
    if np.abs(b_had.Eta()) > 2.5:
        print("Invalid b_had.eta: {}".format(b_had.Eta()))
        continue
    if np.abs(b_lep.Eta()) > 2.5:
        print("Invalid b_lep.eta: {}".format(b_lep.Eta()))
        continue

    ##############################################################
    # Augment Data By Rotating 5 Different Ways
    n_good += 1
    phi = 0
    flip_eta = False
    # Set the phi angle of the lepton to zero
    if len(sys.argv) > 4 and sys.argv[4] == 'aa': phi = -1 * lep.Phi()
    if len(sys.argv) > 5 and sys.argv[5] == 'bb': flip_eta = True
    print("Writing new row to csv file")
    for i in range(n_data_aug):
    # make event wrapper
        lep_aug, jets_aug, met_phi_aug = RotateEvent(lep, jets, met_phi, phi)
        lep_flipped, jets_flipped = FlipEta(lep_aug, jets_aug)
        sjets0, target_W_had, target_b_had, target_t_had, target_W_lep, target_b_lep, target_t_lep = MakeInput(jets_aug, W_had, b_had, t_had, W_lep, b_lep, t_lep )
        sjets1, _, _, _, _, _, _ = MakeInput(jets_flipped, W_had, b_had, t_had, W_lep, b_lep, t_lep )

    # write out
        csvwriter.writerow( (
        "%i" % runNumber, "%i" % eventNumber, "%.5f" % weight, "%i" % jets_n, "%i" % bjets_n,
        "%.5f" % lep_aug.Px(),     "%.5f" % lep_aug.Py(),     "%.5f" % lep_aug.Pz(),     "%.5f" % lep_aug.E(),      "%.5f" % met_met,      "%.5f" % met_phi_aug,
        "%.5f" % sjets0[0][0],  "%.5f" % sjets0[0][1],  "%.5f" % sjets0[0][2],  "%.5f" % sjets0[0][3],  "%.5f" % sjets0[0][4],  "%.5f" % sjets0[0][5],
        "%.5f" % sjets0[1][0],  "%.5f" % sjets0[1][1],  "%.5f" % sjets0[1][2],  "%.5f" % sjets0[1][3],  "%.5f" % sjets0[1][4],  "%.5f" % sjets0[1][5],
        "%.5f" % sjets0[2][0],  "%.5f" % sjets0[2][1],  "%.5f" % sjets0[2][2],  "%.5f" % sjets0[2][3],  "%.5f" % sjets0[2][4],  "%.5f" % sjets0[2][5],
        "%.5f" % sjets0[3][0],  "%.5f" % sjets0[3][1],  "%.5f" % sjets0[3][2],  "%.5f" % sjets0[3][3],  "%.5f" % sjets0[3][4],  "%.5f" % sjets0[3][5],
        "%.5f" % sjets0[4][0],  "%.5f" % sjets0[4][1],  "%.5f" % sjets0[4][2],  "%.5f" % sjets0[4][3],  "%.5f" % sjets0[4][4],  "%.5f" % sjets0[4][5],
        "%.5f" % target_W_had[0], "%.5f" % target_W_had[1], "%.5f" % target_W_had[2], "%.5f" % target_W_had[3], "%.5f" % target_W_had[4],
        "%.5f" % target_W_lep[0], "%.5f" % target_W_lep[1], "%.5f" % target_W_lep[2], "%.5f" % target_W_lep[3], "%.5f" % target_W_lep[4],
        "%.5f" % target_b_had[0], "%.5f" % target_b_had[1], "%.5f" % target_b_had[2], "%.5f" % target_b_had[3], "%.5f" % target_b_had[4],
        "%.5f" % target_b_lep[0], "%.5f" % target_b_lep[1], "%.5f" % target_b_lep[2], "%.5f" % target_b_lep[3], "%.5f" % target_b_lep[4],
        "%.5f" % target_t_had[0], "%.5f" % target_t_had[1], "%.5f" % target_t_had[2], "%.5f" % target_t_had[3], "%.5f" % target_t_had[4],
        "%.5f" % target_t_lep[0], "%.5f" % target_t_lep[1], "%.5f" % target_t_lep[2], "%.5f" % target_t_lep[3], "%.5f" % target_t_lep[4],
        "%.5f" % lep_aug.Pt(),     "%.5f" % lep_aug.Eta(),     "%.5f" % lep_aug.Phi(),
        "%.5f" % sjets0[0][6],  "%.5f" % sjets0[0][7],  "%.5f" % sjets0[0][8],
        "%.5f" % sjets0[1][6],  "%.5f" % sjets0[1][7],  "%.5f" % sjets0[1][8],
        "%.5f" % sjets0[2][6],  "%.5f" % sjets0[2][7],  "%.5f" % sjets0[2][8],
        "%.5f" % sjets0[3][6],  "%.5f" % sjets0[3][7],  "%.5f" % sjets0[3][8],
        "%.5f" % sjets0[4][6],  "%.5f" % sjets0[4][7],  "%.5f" % sjets0[4][8],
        "%.5f" % target_W_had[5], "%.5f" % target_W_had[6], "%.5f" % target_W_had[7],
        "%.5f" % target_W_lep[5], "%.5f" % target_W_lep[6], "%.5f" % target_W_lep[7],
        "%.5f" % target_b_had[5], "%.5f" % target_b_had[6], "%.5f" % target_b_had[7],
        "%.5f" % target_b_lep[5], "%.5f" % target_b_lep[6], "%.5f" % target_b_lep[7],
        "%.5f" % target_t_had[5], "%.5f" % target_t_had[6], "%.5f" % target_t_had[7],
        "%.5f" % target_t_lep[5], "%.5f" % target_t_lep[6], "%.5f" % target_t_lep[7],
        "%.5f" % lep.sumPT, "%.5f" % met_eta,
        "%.5f" % sjets0[0][9],  "%.5f" % sjets0[0][10],  "%.5f" % sjets0[0][11], "%.5f" % sjets0[0][12],
        "%.5f" % sjets0[1][9],  "%.5f" % sjets0[1][10],  "%.5f" % sjets0[1][11], "%.5f" % sjets0[1][12],
        "%.5f" % sjets0[2][9],  "%.5f" % sjets0[2][10],  "%.5f" % sjets0[2][11], "%.5f" % sjets0[2][12],
        "%.5f" % sjets0[3][9],  "%.5f" % sjets0[3][10],  "%.5f" % sjets0[3][11], "%.5f" % sjets0[3][12],
        "%.5f" % sjets0[4][9],  "%.5f" % sjets0[4][10],  "%.5f" % sjets0[4][11], "%.5f" % sjets0[4][12]
            ) )

        if flip_eta:
            csvwriter.writerow( (
            "%i" % runNumber, "%i" % eventNumber, "%.5f" % weight, "%i" % jets_n, "%i" % bjets_n,
            "%.5f" % lep_flipped.Px(),     "%.5f" % lep_flipped.Py(),     "%.5f" % lep_flipped.Pz(),     "%.5f" % lep_flipped.E(),      "%.5f" % met_met,      "%.5f" % met_phi_aug,
            "%.5f" % sjets1[0][0],  "%.5f" % sjets1[0][1],  "%.5f" % sjets1[0][2],  "%.5f" % sjets1[0][3],  "%.5f" % sjets1[0][4],  "%.5f" % sjets1[0][5],
            "%.5f" % sjets1[1][0],  "%.5f" % sjets1[1][1],  "%.5f" % sjets1[1][2],  "%.5f" % sjets1[1][3],  "%.5f" % sjets1[1][4],  "%.5f" % sjets1[1][5],
            "%.5f" % sjets1[2][0],  "%.5f" % sjets1[2][1],  "%.5f" % sjets1[2][2],  "%.5f" % sjets1[2][3],  "%.5f" % sjets1[2][4],  "%.5f" % sjets1[2][5],
            "%.5f" % sjets1[3][0],  "%.5f" % sjets1[3][1],  "%.5f" % sjets1[3][2],  "%.5f" % sjets1[3][3],  "%.5f" % sjets1[3][4],  "%.5f" % sjets1[3][5],
            "%.5f" % sjets1[4][0],  "%.5f" % sjets1[4][1],  "%.5f" % sjets1[4][2],  "%.5f" % sjets1[4][3],  "%.5f" % sjets1[4][4],  "%.5f" % sjets1[4][5],
            "%.5f" % target_W_had[0], "%.5f" % target_W_had[1], "%.5f" % target_W_had[2], "%.5f" % target_W_had[3], "%.5f" % target_W_had[4],
            "%.5f" % target_W_lep[0], "%.5f" % target_W_lep[1], "%.5f" % target_W_lep[2], "%.5f" % target_W_lep[3], "%.5f" % target_W_lep[4],
            "%.5f" % target_b_had[0], "%.5f" % target_b_had[1], "%.5f" % target_b_had[2], "%.5f" % target_b_had[3], "%.5f" % target_b_had[4],
            "%.5f" % target_b_lep[0], "%.5f" % target_b_lep[1], "%.5f" % target_b_lep[2], "%.5f" % target_b_lep[3], "%.5f" % target_b_lep[4],
            "%.5f" % target_t_had[0], "%.5f" % target_t_had[1], "%.5f" % target_t_had[2], "%.5f" % target_t_had[3], "%.5f" % target_t_had[4],
            "%.5f" % target_t_lep[0], "%.5f" % target_t_lep[1], "%.5f" % target_t_lep[2], "%.5f" % target_t_lep[3], "%.5f" % target_t_lep[4],
            "%.5f" % lep_flipped.Pt(),     "%.5f" % lep_flipped.Eta(),     "%.5f" % lep_flipped.Phi(),
            "%.5f" % sjets1[0][6],  "%.5f" % sjets1[0][7],  "%.5f" % sjets1[0][8],
            "%.5f" % sjets1[1][6],  "%.5f" % sjets1[1][7],  "%.5f" % sjets1[1][8],
            "%.5f" % sjets1[2][6],  "%.5f" % sjets1[2][7],  "%.5f" % sjets1[2][8],
            "%.5f" % sjets1[3][6],  "%.5f" % sjets1[3][7],  "%.5f" % sjets1[3][8],
            "%.5f" % sjets1[4][6],  "%.5f" % sjets1[4][7],  "%.5f" % sjets1[4][8],
            "%.5f" % target_W_had[5], "%.5f" % target_W_had[6], "%.5f" % target_W_had[7],
            "%.5f" % target_W_lep[5], "%.5f" % target_W_lep[6], "%.5f" % target_W_lep[7],
            "%.5f" % target_b_had[5], "%.5f" % target_b_had[6], "%.5f" % target_b_had[7],
            "%.5f" % target_b_lep[5], "%.5f" % target_b_lep[6], "%.5f" % target_b_lep[7],
            "%.5f" % target_t_had[5], "%.5f" % target_t_had[6], "%.5f" % target_t_had[7],
            "%.5f" % target_t_lep[5], "%.5f" % target_t_lep[6], "%.5f" % target_t_lep[7],
            "%.5f" % lep.sumPT, "%.5f" % met_eta,
            "%.5f" % sjets1[0][9],  "%.5f" % sjets1[0][10],  "%.5f" % sjets1[0][11], "%.5f" % sjets1[0][12],
            "%.5f" % sjets1[1][9],  "%.5f" % sjets1[1][10],  "%.5f" % sjets1[1][11], "%.5f" % sjets1[1][12],
            "%.5f" % sjets1[2][9],  "%.5f" % sjets1[2][10],  "%.5f" % sjets1[2][11], "%.5f" % sjets1[2][12],
            "%.5f" % sjets1[3][9],  "%.5f" % sjets1[3][10],  "%.5f" % sjets1[3][11], "%.5f" % sjets1[3][12],
            "%.5f" % sjets1[4][9],  "%.5f" % sjets1[4][10],  "%.5f" % sjets1[4][11], "%.5f" % sjets1[4][12]
                ) )
        # Change the angle to rotate by
        phi = np.random.uniform(- np.pi, np.pi)


##############################################################
# Close Program
outfile.close()

f_good = 100. * n_good / n_entries
print("INFO: output file:", outfilename)
print("INFO: %i entries written (%.2f %%)" % ( n_good, f_good))
