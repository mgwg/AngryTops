import os, sys
from ROOT import *
import numpy as np
import pickle
from AngryTops.features import *

################################################################################
# Set style of plots
gROOT.LoadMacro("../AtlasDocs/AtlasStyle.C")
gROOT.LoadMacro( "../AtlasDocs/AtlasUtils.C" )
SetAtlasStyle()

gStyle.SetOptStat(0)
gROOT.SetBatch(1)

################################################################################
# HELPER FUNCTIONS
def Normalize( h, sf=1.0 ):
  if h == None: return
  A = h.Integral()
  if A == 0.: return
  h.Scale( sf / A )


def SetTH1FStyle(h, color=kBlack, linewidth=1, fillcolor=0, fillstyle=0,
                 markerstyle=21, markersize=1.3, fill_alpha=0):
    '''Set the style with a long list of parameters'''
    h.SetLineColor(color)
    h.SetLineWidth(linewidth)
    h.SetFillColor(fillcolor)
    h.SetFillStyle(fillstyle)
    h.SetMarkerStyle(markerstyle)
    h.SetMarkerColor(h.GetLineColor())
    h.SetMarkerSize(markersize)
    if fill_alpha > 0:
       h.SetFillColorAlpha( color, fill_alpha )

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def MakeCanvas(npads = 1, side = 800, split = 0.25, padding = 0.00):
    # assume that if pads=1 => square plot
    # padding used to be 0.05
    y_plot    = side * ( 1. - ( split + padding ) )
    y_ratio   = side * split
    y_padding = side * padding

    height_tot = y_plot + npads * ( y_ratio + y_padding )
    height_tot = int(height_tot)

    c = TCanvas( "PredictionData", "PREDICT/MC", side, height_tot )
    c.SetFrameFillStyle(4000)
    c.SetFillColor(0)

    pad0 = TPad( "pad0","pad0",0, split+padding,1,1,0,0,0 )
    pad0.SetLeftMargin( 0.18 ) #0.16
    pad0.SetRightMargin( 0.05 )
    pad0.SetBottomMargin( 0.01 )
    #pad0.SetTopMargin( 0.14 )
    pad0.SetTopMargin( 0.07 ) #0.05
    pad0.SetFillColor(0)
    pad0.SetFillStyle(4000)
    pad0.Draw()

    pad1 = TPad( "pad1","pad1",0,0,1, split,0,0,0 )
    pad1.SetLeftMargin( 0.18 ) #0.16
    pad1.SetRightMargin( 0.05 )
    pad1.SetTopMargin( 0. )
#    pad1.SetBottomMargin( 0. )
    pad1.SetGridy(1)
    pad1.SetTopMargin(0)
    pad1.SetBottomMargin(0.5) #0.4
    pad1.Draw()
    pad1.SetFillColor(0)
    pad1.SetFillStyle(4000)

    pad0.cd()
    return c, pad0, pad1

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def MakeUncertaintyBand( prediction ):
    unc = TGraphAsymmErrors()

    i = 0

    if prediction.Class() in [ TGraph().Class(), TGraphErrors.Class(), TGraphAsymmErrors().Class() ]:
       Npoints = prediction.GetN()
    else:
       Npoints = prediction.GetNbinsX()

    for n in range( Npoints ):
       if prediction.Class() in [ TGraph().Class(), TGraphErrors.Class(), TGraphAsymmErrors().Class() ]:
          x_mc = Double()
          y_mc = Double()
          prediction.GetPoint( n, x_mc, y_mc )
       else:
          x_mc = prediction.GetBinCenter(n+1)
          y_mc = prediction.GetBinContent(n+1)

       if y_mc == 0: continue

       unc.SetPoint( i, x_mc, 1.0 )

       if prediction.Class() in [ TGraph().Class(), TGraphErrors.Class(), TGraphAsymmErrors().Class() ]:
          bw_l = prediction.GetErrorXlow( n )
          bw_h = prediction.GetErrorXhigh( n )
          err_y_lo = prediction.GetErrorYlow(n) / y_mc
          err_y_hi = prediction.GetErrorYhigh(n) / y_mc
       else:
          bw_l = prediction.GetBinWidth( n+1 ) / 2.
          bw_h = prediction.GetBinWidth( n+1 ) / 2.
          err_y_lo = prediction.GetBinError( n+1 ) / y_mc
          err_y_hi = prediction.GetBinError( n+1 ) / y_mc

       unc.SetPointError( i, bw_l, bw_h, err_y_lo, err_y_hi )

       i += 1

    return unc

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def MakeRatio( data, prediction):
    ratio = TGraphAsymmErrors()
    SetTH1FStyle( ratio, color=data.GetMarkerColor(), markerstyle=data.GetMarkerStyle() )
    nbins = data.GetNbinsX()
    i = 0
    for n in range( nbins ):
        x_mc = Double()
        y_mc = Double()
        x_data = Double()
        y_data = Double()

        x_mc = data.GetBinCenter( n+1 )
        y_mc = data.GetBinContent( n+1 )

        if y_mc == 0.: continue

        x_data = prediction.GetBinCenter( n+1 )
        y_data = prediction.GetBinContent( n+1 )
        bw = data.GetBinWidth( n+1 )
        dy_u = data.GetBinError( n+1 )
        dy_d = data.GetBinError( n+1 )

        #print '    setting point %i: %f' % (i,y_data/y_mc,)

        ratio.SetPoint( i, x_data, y_data/y_mc )

        ratio.SetPointError( i, 0., 0., dy_d/y_mc, dy_u/y_mc )

        i += 1
    return ratio

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def DrawRatio( data, prediction, xtitle = "", yrange=[0.4,1.6] ):
    xmin = data.GetXaxis().GetXmin()
    xmax = data.GetXaxis().GetXmax()

    frame = gPad.DrawFrame( xmin, yrange[0], xmax, yrange[1] ) #2.1

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

    SetTH1FStyle( tot_unc,  color=kGray+1, fillstyle=1001, fillcolor=kGray+1, linewidth=0, markersize=0 )

    ratio   = MakeRatio(data, prediction)

    tot_unc.Draw( "e2 same" )
    ratio.Draw( "p same" )

    gPad.RedrawAxis()

    return frame, tot_unc, ratio

def MakeCanvas2( npads = 1, side = 800, padding = 0.00 ):
    # Makes simpler plot
    y_plot    = side * ( 1. - ( + padding ) )
    y_ratio   = side
    y_padding = side * padding

    height_tot = y_plot + npads * ( y_ratio + y_padding )
    height_tot = int(height_tot)

    c = TCanvas( "Histogram","PREDICT/MC", height_tot, height_tot )
    c.SetFrameFillStyle(4000)
    c.SetFillColor(0)

    pad0 = TPad( "pad0","pad0",0, padding,1,1,0,0,0 )
    pad0.SetLeftMargin( 0.18 ) #0.16
    pad0.SetRightMargin( 0.05 )
    #pad0.SetTopMargin( 0.14 )
    pad0.SetTopMargin( 0.07 ) #0.05
    pad0.SetBottomMargin(0.5)
    pad0.SetFillColor(0)
    pad0.SetFillStyle(4000)
    pad0.SetGridy(1)
    pad0.SetGridx(1)
    pad0.Draw()

    pad0.cd()
    return c, pad0

def ChiSquared(h_mc, h_pred):
    """
    Calculates the chi-squared values between hist 0 and hist 1.
    @Parameters:
    h_mc: TH1 Class. The first histogram
    h_pred: TH1 Class. The second histogram
    """
    nbins = h_mc.GetNbinsX()
    chi2 = 0
    for i in range(1, nbins + 1):
        y_mc = h_mc.GetBinContent(i)
        dy_mc = h_mc.GetBinError(i)
        y_pred = h_pred.GetBinContent(i)
        dy_pred = h_pred.GetBinError(i)
        sigma = np.sqrt(dy_mc**2 + dy_pred**2)
        chi = np.power(y_mc - y_pred, 2) / sigma**2
        # Prevents nan values from being added
        if chi == chi:
            chi2 += chi
    return chi2 / (nbins - 1)

def getFwhm(hist):
    """
    Calculates the fwhm of hist.
    @Parameters:
    hist: TH1 Class
    """

    #fwhm
    # half_max = hist.GetMaximum()/2.0
    # nbins = hist.GetNbinsX()
    # bin1 = 0
    # bin2 = nbins
    # for i in range(1, nbins + 1):
    #     y = hist.GetBinContent(i)
    #     if (i < nbins/2 and y <= half_max) and (i > bin1):
    #         bin1 = hist.GetBinCenter(i)
    #     if (i > nbins/2 and y <= half_max) and (i < bin2):
    #         bin2 = hist.GetBinCenter(i)
    # fwhm = bin2 - bin1

    #gaus fit
    hist.Fit('gaus', 'q')
    gausFit = hist.GetListOfFunctions().FindObject('gaus')
    sigma = gausFit.GetParameter(2) # mean is 1
    fwhm = sigma*2.35403

    return fwhm, sigma