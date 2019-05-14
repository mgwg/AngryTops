
import ROOT
from ROOT import TLorentzVector
from collections import Counter
from array import array
from math import pow, sqrt
import numpy as np

GeV = 1e3
TeV = 1e6

#####################

def Normalize(h, sf=1.0, opt="width"):
    area = h.Integral(opt)
    h.Scale(sf / area)

#####################


def GetEventWeight(tree, syst="nominal"):
    w = 1.

    w_pileup = tree.weight_pileup
    w_jvt = tree.weight_jvt
    w_btag = tree.weight_bTagSF_MV2c10_70
    w_others = 1.
    isOtherSyst = True

    if syst == "nominal":
        pass

    elif syst in ["pileup_UP", "pileup_DOWN"]:
        w_pileup = tree.weight_pileup_UP if syst == "pileup_UP" else tree.weight_pileup_DOWN

    elif syst in ["jvt_UP", "jvt_DOWN"]:
        w_jvt = tree.weight_jvt_UP if syst == "jvt_UP" else tree.weight_jvt_DOWN

    elif syst.startswith("bTagSF"):
        if "eigenvars" in syst:
            syst_branch = "weight_%s" % syst
            exec("w_btag = tree.%s" % syst_branch)
        else:
            k = int(syst.split('_')[-1])
            syst_btag = syst.replace("_up_%i" % k, "_up").replace(
                "_down_%i" % k, "_down")
            syst_branch = "weight_%s" % syst_btag
            exec("w_btag = tree.%s[%i]" % (syst_branch, k))
    else:
        if syst in systematics_weight:
            syst_branch = "weight_%s" % syst
            exec("w_others = tree.%s" % syst_branch)

    w *= w_pileup * w_jvt * w_btag * w_others

    return w

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def MakeEventJets(tree, b_tag_cut=0.83):
    ljets = []

    ljets_n = len(tree.ljet_pt)
    for i in range(ljets_n):
        ljets += [ntzVector()]
        lj = ljets[-1]
        lj.SetPtEtaPhiM(
            tree.ljet_pt[i]/GeV, tree.ljet_eta[i], tree.ljet_phi[i], tree.ljet_m[i]/GeV)
        lj.index = i
        lj.tau2 = tree.ljet_tau2[i]
        lj.tau3 = tree.ljet_tau3[i]
        lj.tau32 = tree.ljet_tau32[i]
        #lj.tau21 = tree.ljet_tau21[i]
        #lj.sd12  = tree.ljet_sd12[i]
        #lj.sd23  = tree.ljet_sd23[i]
        #lj.Qw    = tree.ljet_Qw[i]
        #lj.ntrk  = tree.ljet_QGTaggerNTrack[i]

    return ljets


###############################


def RotateJets(ljets=[], phi=None):
    if phi == None:
        phi = -ljets[0].Phi()

    dPhi = ljets[0].DeltaPhi(ljets[1])
    ljets[0].SetPhi(0)
    ljets[1].SetPhi( abs(dPhi ) )

#    for lj in ljets:
#        lj.RotateZ(phi)

    return phi

#~~~~~~~~~~~~~~~~~~~~~~


def FlipEta(ljets=[]):
    for lj in ljets:
        lj.SetPtEtaPhiE(lj.Pt(), -lj.Eta(), lj.Phi(), lj.E())

####################

def save_training_history( history, filename = "GAN/training_history.root", verbose=True ):
   training_root = TFile.Open( filename, "RECREATE" )

   if verbose == True:
      print "INFO: saving training history..."

   h_d_lr = TGraphErrors()
   h_g_lr = TGraphErrors()
   h_d_loss = TGraphErrors()
   h_d_loss_r = TGraphErrors()
   h_d_loss_f = TGraphErrors()
   h_d_acc = TGraphErrors()
   h_g_loss = TGraphErrors()
   h_d_acc = TGraphErrors()
   h_d_acc_f = TGraphErrors()
   h_d_acc_r = TGraphErrors()

   n_epochs = len(history['d_loss'])
   for i in range(n_epochs):
       d_lr = history['d_lr'][i]
       g_lr = history['g_lr'][i]
       d_loss = history['d_loss'][i]
       d_loss_r = history['d_loss_r'][i]
       d_loss_f = history['d_loss_f'][i]
       d_acc = history['d_acc'][i]
       d_acc_f = history['d_acc_f'][i]
       d_acc_r = history['d_acc_r'][i]
       g_loss = history['g_loss'][i]

       h_d_lr.SetPoint(i, i, d_lr)
       h_g_lr.SetPoint(i, i, g_lr)
       h_d_loss.SetPoint(i, i, d_loss)
       h_d_loss_r.SetPoint(i, i, d_loss_r)
       h_d_loss_f.SetPoint(i, i, d_loss_f)
       h_d_acc.SetPoint(i, i, d_acc)
       h_d_acc_f.SetPoint(i, i, d_acc_f)
       h_d_acc_r.SetPoint(i, i, d_acc_r)
       h_g_loss.SetPoint(i, i, g_loss)

   h_d_lr.Write("d_lr")
   h_g_lr.Write("g_lr")
   h_d_loss.Write("d_loss")
   h_d_loss_r.Write("d_loss_r")
   h_d_loss_f.Write("d_loss_f")
   h_g_loss.Write("g_loss")
   h_d_acc.Write("d_acc")
   h_d_acc_f.Write("d_acc_f")
   h_d_acc_r.Write("d_acc_r")

   training_root.Write()
   training_root.Close()

   if verbose == True:
      print "INFO: training history saved to file:", training_root.GetName()

###############################
# ADDED BY FARDIN

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

    lep_new            = TLorentzVector( lep )
    lep_new.q          = lep.q
    lep_new.flav       = lep.flav
    lep_new.topoetcone = lep.topoetcone
    lep_new.ptvarcone  = lep.ptvarcone
    lep_new.d0sig      = lep.d0sig

    lep_new.RotateZ( phi )

    jets_new = []
    for j in jets:
        jets_new += [ TLorentzVector(j) ]
        j_new = jets_new[-1]

        j_new.mv2c10 = j.mv2c10
        j_new.index  = j.index

        j_new.RotateZ( phi )

    return lep_new, jets_new

###############################

def MakeInput( jets, W_had, b_had, t_had, W_lep, b_lep, t_lep ):
    """Format output, returning a jet matrix (detector level), W, b quark and lepton arrays (particle leve) and t-quark momenta (parton level)
    """
    # INPUT DATA FOR RNN
    # Populate 5 x 6 matrix of jet foration
    # Is there possibly a bug here? What format is Px? Is sjets[0][0] = sjets[1][0] = sjets[2][0] = ...
    sjets = np.zeros( [ n_jets_per_event, n_features_per_jet ] )
    for i in range(len(jets)):
        jet = jets[i]
        sjets[i][0] = jet.Px()/GeV
        sjets[i][1] = jet.Py()/GeV
        sjets[i][2] = jet.Pz()/GeV
        sjets[i][3] = jet.E()/GeV
        sjets[i][4] = jet.M()/GeV
        sjets[i][5] = jet.mv2c10

    # OUTPUT DATA FOR RNN
    # Arrays containing W, b quarks and lepton information
    target_W_had = np.zeros( [5] )
    target_b_had = np.zeros( [5] )
    target_t_had = np.zeros( [5] )
    target_W_lep = np.zeros( [5] )
    target_b_lep = np.zeros( [5] )
    target_t_lep = np.zeros( [5] )

    # T Hadron
    target_t_had[0] = t_had.Px()/GeV
    target_t_had[1] = t_had.Py()/GeV
    target_t_had[2] = t_had.Pz()/GeV
    target_t_had[3] = t_had.E()/GeV
    target_t_had[4] = t_had.M()/GeV

    # Hadronically Decay W Boson
    target_W_had[0] = W_had.Px()/GeV
    target_W_had[1] = W_had.Py()/GeV
    target_W_had[2] = W_had.Pz()/GeV
    target_W_had[3] = W_had.E()/GeV
    target_W_had[4] = W_had.M()/GeV

    # Hadronic b quark
    target_b_had[0] = b_had.Px()/GeV
    target_b_had[1] = b_had.Py()/GeV
    target_b_had[2] = b_had.Pz()/GeV
    target_b_had[3] = b_had.E()/GeV
    target_b_had[4] = b_had.M()/GeV

    # Leptonic t quark
    target_t_lep[0] = t_lep.Px()/GeV
    target_t_lep[1] = t_lep.Py()/GeV
    target_t_lep[2] = t_lep.Pz()/GeV
    target_t_lep[3] = t_lep.E()/GeV
    target_t_lep[4] = t_lep.M()/GeV

    # Leptonically decaying W quark
    target_W_lep[0] = W_lep.Px()/GeV
    target_W_lep[1] = W_lep.Py()/GeV
    target_W_lep[2] = W_lep.Pz()/GeV
    target_W_lep[3] = W_lep.E()/GeV
    target_W_lep[4] = W_lep.M()/GeV

    # Leptonic b quark
    target_b_lep[0] = b_lep.Px()/GeV
    target_b_lep[1] = b_lep.Py()/GeV
    target_b_lep[2] = b_lep.Pz()/GeV
    target_b_lep[3] = b_lep.E()/GeV
    target_b_lep[4] = b_lep.M()/GeV

    return sjets, target_W_had, target_b_had, target_t_had, target_W_lep, target_b_lep, target_t_lep
