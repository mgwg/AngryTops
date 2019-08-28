"""This script superimposes the histograms of two other models"""
#!/usr/bin/env python
import os, sys
from ROOT import *
import numpy as np
from AngryTops.features import *
from AngryTops.Plotting.PlottingHelper import *

def DrawRatio( data, prediction, chi_pred, xtitle = "", yrange=[0.4,1.6] ):

    if data.Class() in [ TGraph().Class(), TGraphErrors.Class(), TGraphAsymmErrors().Class() ]:
       n = data.GetN()
       x = Double()
       y = Double()
       data.GetPoint( 0, x, y )
       exl = data.GetErrorXlow( 0 )
       xmin = x - exl
       data.GetPoint( n-1, x, y )
       exh = data.GetErrorXhigh( n-1 )
       xmax = x + exh
    else:
       xmin = data.GetXaxis().GetXmin()
       xmax = data.GetXaxis().GetXmax()

    # tt diffxs 7 TeV: [ 0.4, 1.6 ]
#    frame = gPad.DrawFrame( xmin, 0.7, xmax, 1.3 )
    frame = gPad.DrawFrame( xmin, yrange[0], xmax, yrange[1] ) #2.1
#    frame = gPad.DrawFrame( xmin, 0.3, xmax, 2.2 )

    frame.GetXaxis().SetNdivisions(508)
    frame.GetYaxis().SetNdivisions(504)

    frame.GetXaxis().SetLabelSize( 0.15 )
    frame.GetXaxis().SetTitleSize( 0.15 )
    frame.GetXaxis().SetTitleOffset( 1.2 )

    frame.GetYaxis().SetLabelSize( 0.15 )
    frame.GetYaxis().SetTitle( "#frac{PREDICT}{MC}" )
    frame.GetYaxis().SetTitleSize( 0.15 )
    frame.GetYaxis().SetTitleOffset( 0.5 )

    frame.GetXaxis().SetTitle( xtitle )

    frame.Draw()

    tot_unc  = MakeUncertaintyBand( prediction )
    tot_unc_chi = MakeUncertaintyBand( chi_pred )

    SetTH1FStyle( tot_unc,  color=kGray+1, fillstyle=1001, fillcolor=kGray+1, linewidth=0, markersize=0 )
    SetTH1FStyle( tot_unc_chi,  color=kBlack+1, fillstyle=1001, fillcolor=kBlack+1, linewidth=0, markersize=0 )

    ratio   = MakeRatio( data, prediction, True )
    ratio_chi = MakeRatio(chi_pred, data, True)

    ratio.SetMarkerColor(kBlack)
    ratio_chi.SetMarkerColor(kRed)
    tot_unc.Draw( "e2 same" )
    tot_unc_chi.Draw( "e2 same" )
    ratio.Draw( "p same" )
    ratio_chi.Draw("p same")

    gPad.RedrawAxis()

    return frame, tot_unc, ratio, ratio_chi

# INPUT
training_dir1 = sys.argv[1]
training_dir2 = sys.argv[2]
if len(sys.argv) > 3:
    attributes = attributes_tquark
    corr_2d = corr_2d_tquark

# Style
gStyle.SetPalette(kGreyScale)
gROOT.GetColor(52).InvertPalette()

# Make image file and open trees
os.mkdir('{}/Comparison'.format(training_dir1))
infilename = "{}/histograms.root".format(training_dir1)
chifilename = "{}/histograms.root".format(training_dir2)
infile = TFile.Open(infilename)
chifile = TFile.Open(chifilename)

def plot_observables(obs):
    # Load the histograms
    hname_true = "%s_true" % (obs)
    hame_fitted = "%s_fitted" % (obs)
    hame_chi = "%s_fitted" % (obs)

    # True and fitted leaf
    h_true = infile.Get(hname_true)
    h_fitted = infile.Get(hame_fitted)
    h_chi = chifile.Get(hame_fitted)
    if h_true == None:
        print ("ERROR: invalid histogram for", obs)

    # Axis titles
    xtitle = h_true.GetXaxis().GetTitle()
    ytitle = h_true.GetYaxis().GetTitle()
    if h_true.Class() == TH2F.Class():
        h_true = h_true.ProfileX("pfx")
        h_true.GetYaxis().SetTitle( ytitle )
    else:
        Normalize(h_true)
        Normalize(h_fitted)
        Normalize(h_chi)
        h_true.GetYaxis().SetTitle("A.U.")

    # Set Style
    SetTH1FStyle( h_true,  color=kGray+2, fillstyle=1001, fillcolor=kGray,
                  linewidth=3, markersize=0 )
    SetTH1FStyle( h_fitted, color=kBlack, markersize=0, markerstyle=20,
                  linewidth=3 )
    SetTH1FStyle( h_chi, color=kBlack+1, fillstyle=3351, fillcolor=kGray,
                  markersize=0, markerstyle=20, linewidth=3 )

    c, pad0, pad1 = MakeCanvas()
    pad0.cd()
    gStyle.SetOptTitle(0)

    h_true.Draw("h")
    h_fitted.Draw("h same")
    h_chi.Draw("h same")
    hmax = 1.5 * max( [ h_true.GetMaximum(), h_fitted.GetMaximum(), h_chi.GetMaximum() ] )
    h_fitted.SetMaximum( hmax )
    h_true.SetMaximum( hmax )
    h_chi.SetMaximum( hmax )
    h_fitted.SetMinimum( 0. )
    h_true.SetMinimum( 0. )
    h_chi.SetMinimum( 0. )

    leg = TLegend( 0.20, 0.80, 0.50, 0.90 )
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.SetBorderSize(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.05)
    leg.AddEntry( h_true, "MG5+Py8", "f" )
    leg.AddEntry( h_fitted, "BLSTM Model", "f" )
    leg.AddEntry( h_chi, "Chi2 Model", "f" )
    leg.SetY1( leg.GetY1() - 0.05 * leg.GetNRows() )
    leg.Draw()

    KS1 = h_true.KolmogorovTest( h_fitted )
    X21 = h_true.Chi2Test( h_fitted, "UU NORM CHI2/NDF" ) # UU NORM
    KS2 = h_true.KolmogorovTest( h_chi )
    X22 = h_true.Chi2Test( h_chi, "UU NORM CHI2/NDF" ) # UU NORM
    l = TLatex()
    l.SetNDC()
    l.SetTextFont(42)
    l.SetTextColor(kBlack)
    l.DrawLatex( 0.55, 0.85, "BLSTM KS test: %.2f" % KS1 )
    l.DrawLatex( 0.55, 0.80, "BLSTM #chi^{2}/NDF = %.2f" % X21 )
    l.DrawLatex( 0.55, 0.75, "CHI2 KS test: %.2f" % KS2 )
    l.DrawLatex( 0.55, 0.70, "CHI2 #chi^{2}/NDF = %.2f" % X22 )

    gPad.RedrawAxis()

    pad1.cd()

    yrange = [0.4, 1.6]
    frame, tot_unc, ratio, ratio_chi = DrawRatio(h_fitted, h_true, h_chi, xtitle, yrange)

    gPad.RedrawAxis()

    c.cd()

    c.SaveAs("{0}/Comparison/{1}.png".format(training_dir1, obs))
    pad0.Close()
    pad1.Close()
    c.Close()

################################################################################
if __name__==   "__main__":
    # Make a plot for each observable
    for obs in attributes:
        plot_observables(obs)
