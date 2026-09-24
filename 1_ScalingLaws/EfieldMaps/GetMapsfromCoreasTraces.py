#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 02:21:35 2024

@author: chiche
"""

# Modules import
#region Modules 
import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
import glob
import sys
import pickle
from MainModules.ShowerClass import CreateShowerfromHDF5
from MainModules.PlotConfig import MatplotlibConfig
#sys.path.append("/Users/chiche/Desktop/DeepCrSearch/Analysis/")
##from Modules.SimParam.GetLayoutScaling import GetDepthScaling, GetEnergyScaling
from scipy.interpolate import interp1d
import scipy
##from Modules.Fluence.FunctionsGetFluence import  Norm, LoadTraces, GetPeakTraces, Traces_cgs_to_si, GetDepths, CorrectScaling, CombineTraces, CorrectLength, GetIntTraces, GetIntTracesSum, GetRadioExtent
from Modules.FunctionsPlotFluence import EfieldMap, PlotLDF, PlotTraces, plot_polarisation, PlotMaxTraces, PlotAllTraces, PlotLayer, PlotGivenTrace, PlotAllChannels, PlotSurfaceEz
##from CleanCoreasTraces import CleanCoreasTraces
##from Modules.SimParam.PlotRadioSimExtent import PlotFillingFactor, PlotRadioSimExtent, PlotAirIceExtent
from scipy.interpolate import griddata
from datetime import datetime
from scipy.optimize import curve_fit
from Modules.ModuleGetCoreasMaps import GetCoreasTracesfromHDF5, PlotCoreasMaps
from MainModules.FormatFaerieOutput import Traces_cgs_to_si
from matplotlib.colors import PowerNorm
from scipy.interpolate import griddata
from Modules.ModuleGetCoreasMaps import interpolate_rbf, GetFluence, GetRadiationEnergyFromInterpolation, PlotEradvsDepths

##from Modules.Fluence.FunctionsRadiationEnergy import GetFluence, GetRadiationEnergy
#endregion

#region Path definition
SimDir = "FullDenseDeepCr"  #"DeepCrLibV1"
WorkPath = os.getcwd()
BatchID = "CoreasMapOnlyProton"
SimName = "Polar_Proton_0.0316_0_0_1.hdf5"

simpath = "/Users/chiche/Desktop/DeepCrAnalysis/Simulations/" \
+ SimDir + "/" + SimName
#simpath = "/Users/chiche/Desktop/DeepCrAnalysis/Simulations/DeepCrLibV1/Rectangle_Proton_0.316_43_0_1_0.hdf5" 
OutputPath = MatplotlibConfig(WorkPath, SimDir, BatchID)
#endregion
Save = False


Shower = CreateShowerfromHDF5(simpath)
Shower.traces_c = GetCoreasTracesfromHDF5(simpath)
Shower.traces_c = Traces_cgs_to_si(Shower.traces_c)

energy, theta, Nant = Shower.energy, Shower.zenith, Shower.nant
Nlay, Nplane, Depths = Shower.GetDepths()
Filter = False
if(Filter):
    fs, lowcut, highcut = 5e9, 50e6, 1e9
    Shower.traces_c =Shower.filter_all_traces(Shower.traces_c, fs, lowcut, highcut)

ExC_int, EyC_int, EzC_int, EtotC_int, peakTime = Shower.GetIntTraces(Shower.traces_c)
ExC, EyC, EzC, EtotC = Shower.GetPeakTraces(Shower.traces_c)

# Coreas maps
EfieldMap(Shower.pos, Depths, Nplane, np.log10(EtotC), "In-air", \
          Save, energy, theta, OutputPath)

#Depths, EradAllDepths = Shower.GetRadiationEnergyGeneric(Shower.traces_c)
#PlotEradvsDepths(Depths, EradAllDepths)
