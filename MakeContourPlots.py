from ROOT import TCanvas, TFile, TProfile, TNtuple, TH1F, TH2F, TChain
from ROOT import gROOT, gBenchmark, gRandom, gSystem, Double
from features import *

def draw_contour(attribute_name, treename='fitted.root'):
    # Create a new canvas, and customize it.
    c1 = TCanvas( 'c1', 'Dynamic Filling Example', 200, 10, 700, 500 )
    c1.SetFillColor( 42 )
    c1.GetFrame().SetFillColor( 21 )
    c1.GetFrame().SetBorderSize( 6 )
    c1.GetFrame().SetBorderMode( -1 )

    # Open tree
    ttree = TChain('nominal', 'nominal')
    ttree.AddFile("{0}/{1}".format(training_dir, treename))

    # Draw and save contour plot
    ttree.Draw("{0}_true:{0}_fitted".format(attribute_name), "", "colz")
    c1.SaveAs("{0}/ContourPlots/{1}.jpg".format(training_dir, attribute_name))

if __name__=="__main__":
    attributes = ['W_had_px', 'W_had_py', 'W_had_pz', 'W_had_E', 'W_had_m',
    'W_had_pt', 'W_had_y', 'W_had_phi', 'b_had_px', 'b_had_py', 'b_had_pz',
    'b_had_E', 'b_had_m', 'b_had_pt', 'b_had_y', 'b_had_phi', 't_had_px',
    't_had_py', 't_had_pz', 't_had_E', 't_had_m', 't_had_pt', 't_had_y',
    't_had_phi', 'W_lep_px', 'W_lep_py', 'W_lep_pz', 'W_lep_E', 'W_lep_m',
    'W_lep_pt', 'W_lep_y', 'W_lep_phi', 'b_lep_px', 'b_lep_py', 'b_lep_pz',
    'b_lep_E', 'b_lep_m', 'b_lep_pt', 'b_lep_y', 'b_lep_phi', 't_lep_px',
    't_lep_py', 't_lep_pz', 't_lep_E', 't_lep_m', 't_lep_pt', 't_lep_y', 
    't_lep_phi']
    for att in attributes:
        draw_contour(att)
