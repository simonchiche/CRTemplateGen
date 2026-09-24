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
from scipy.signal import butter, filtfilt
##from Modules.Fluence.FunctionsRadiationEnergy import GetFluence, GetRadiationEnergy
#endregion

#region Path definition
SimDir = "MassComposition/Gammas" #"DeepCrLibV1"  #"InterpSim"
SimName = "Polar_Gamma_0.0316_0_0_1.hdf5"
WorkPath = os.getcwd()
simpath = "/Users/chiche/Desktop/DeepCrAnalysis/Simulations/" \
+ SimDir + "/" + SimName 
BatchID = "TracesExample"
OutputPath = MatplotlibConfig(WorkPath, SimDir, BatchID)
#endregion
Save = False
Shower = CreateShowerfromHDF5(simpath)

# =============================================================================
#                              Load Traces
# =============================================================================

energy, theta, Nant = Shower.energy, Shower.zenith, Shower.nant
Traces_C, Traces_G, Pos = Shower.traces_c, Shower.traces_g, Shower.pos
Nlay, Nplane, Depths = Shower.GetDepths()
#Traces_tot = Shower.CombineTraces()


x, y = Pos[:,0], Pos[:,1]
x0 = x[Pos[:,2]==max(Depths)]
y0 = y[Pos[:,2]==max(Depths)]
x100 = x[Pos[:,2]==min(Depths)]
y100 = y[Pos[:,2]==min(Depths)]
plt.scatter(Pos[:,1], Pos[:,2], s=1)
plt.show()
Nlay =1800

plt.scatter(x100, y100, s=1)
plt.scatter(x0, y0, s=1)
plt.xlim(-250, 250)
plt.ylim(-250, 250)
len(Pos[Pos[:,2]==3116])
plt.show()

sys.exit()
# =============================================================================
#                           Get peak amplitude
# =============================================================================

ExC, EyC, EzC, EtotC = Shower.GetPeakTraces(Traces_C)
ExG, EyG, EzG, EtotG = Shower.GetPeakTraces(Traces_G)
#Extot, Eytot, Eztot, Etot_peak = Shower.GetPeakTraces(Traces_tot)

# =============================================================================
#                                 Get integral
# =============================================================================

ExC_int, EyC_int, EzC_int, EtotC_int, peaktime = Shower.GetIntTraces(Traces_C)
ExG_int, EyG_int, EzG_int, EtotG_int, peaktime = Shower.GetIntTraces(Traces_G)
#Ex_tot_int, Ey_tot_int, Ez_tot_int, Etot_int = Shower.GetIntTraces(Traces_tot)

# =============================================================================
#                             Plot Traces
# =============================================================================

### Plot max traces
# Geant
#PlotMaxTraces(Traces_G, EtotG, 1)
#Coreas
PlotMaxTraces(Traces_C, EtotC, 100)

#PlotMaxTraces(Traces_C, EzC_int, 5)   

# Plot all traces above a given threshold
#PlotAllTraces(Nant, Traces_C, 100, 5)

##PlotGivenTrace(Traces_C, 310, "y")

##PlotAllChannels(Traces_C, 1068)
##PlotAllChannels(Traces_G, 1068)

#PlotSurfaceEz(Nant, 3216, Pos, Traces_C, 1e-5)


# =============================================================================
#                        Traces at different radial distances
# =============================================================================


# We extract traces at a depth of 100 meters
y = abs(Pos[:,1])
x = Pos[:,0]
sel = (Pos[:,2]==min(Depths)) & (y<=10) & (x>=0)

Traces_C100 = {i: Traces_C[i] for i in Traces_C if sel[i]}
Traces_C100 = {i: val for i, val in enumerate(Traces_C100.values())}
Traces_G100 = {i: Traces_G[i] for i in Traces_G if sel[i]}
Traces_G100 = {i: val for i, val in enumerate(Traces_G100.values())}

Pos100 = Pos[sel]
radial_dist = np.sqrt(Pos100[:,0]**2 + Pos100[:,1]**2)

r1, r2, r3, r4, r5 = 5, 25, 45, 65, 85
arg1 = np.argmin(np.abs(radial_dist - r1))
arg2 = np.argmin(np.abs(radial_dist - r2))
arg3 = np.argmin(np.abs(radial_dist - r3))
arg4 = np.argmin(np.abs(radial_dist - r4))
arg5 = np.argmin(np.abs(radial_dist - r5))

plt.plot(Traces_C100[arg1][:,0]*1e9, -Traces_C100[arg1][:,2], label="%.d m" %(r1))
plt.plot(Traces_C100[arg2][:,0]*1e9, -Traces_C100[arg2][:,2], label="%.d m" %(r2))
plt.plot(Traces_C100[arg3][:,0]*1e9, -Traces_C100[arg3][:,2], label="%.d m" %(r3))
plt.plot(Traces_C100[arg4][:,0]*1e9, -Traces_C100[arg4][:,2], label="%.d m" %(r4))
plt.plot(Traces_C100[arg5][:,0]*1e9, -Traces_C100[arg5][:,2], label="%.d m" %(r5))
plt.xlabel("Time [ns]")
plt.ylabel("Efield [$\mu \, V/m$]")
plt.legend()
plt.xlim(480, 700)
plt.title("In-air, Depth = 100 m", fontsize=14)
#plt.savefig(OutputPath + "vs_r_InAir_Depth100m_th34.pdf", bbox_inches='tight')
plt.show()

print(radial_dist[arg1], radial_dist[arg2], radial_dist[arg3], radial_dist[arg4], radial_dist[arg5])

r1, r2, r3, r4, r5 = 125, 135, 145, 155, 175
arg1 = np.argmin(np.abs(radial_dist - r1))
arg2 = np.argmin(np.abs(radial_dist - r2))
arg3 = np.argmin(np.abs(radial_dist - r3))
arg4 = np.argmin(np.abs(radial_dist - r4))
arg5 = np.argmin(np.abs(radial_dist - r5))

plt.plot(Traces_G100[arg1][:,0]*1e9, abs(Traces_G100[arg1][:,2]), label="%.d m" %(r1))
plt.plot(Traces_G100[arg2][:,0]*1e9, abs(Traces_G100[arg2][:,2]), label="%.d m" %(r2))
plt.plot(Traces_G100[arg3][:,0]*1e9, abs(Traces_G100[arg3][:,2]), label="%.d m" %(r3))
plt.plot(Traces_G100[arg4][:,0]*1e9, abs(Traces_G100[arg4][:,2]), label="%.d m" %(r4))
plt.plot(Traces_G100[arg5][:,0]*1e9, abs(Traces_G100[arg5][:,2]), label="%.d m" %(r5))
plt.xlabel("Time [ns]")
plt.ylabel("Efield [$\mu \, V/m$]")
plt.legend()
plt.xlim(800, 1100)
plt.title("In-ice, Depth = 100 m", fontsize=14)
plt.savefig(OutputPath + "vs_r_InIce_Depth100m_th34.pdf", bbox_inches='tight')
plt.show()


sel = (Pos[:,2]==max(Depths)) 
GridPos = Pos[sel]
plt.scatter(GridPos[:,0], GridPos[:,1], s=1)
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.savefig(OutputPath + "AntennaGrid.pdf", bbox_inches='tight')
plt.show()