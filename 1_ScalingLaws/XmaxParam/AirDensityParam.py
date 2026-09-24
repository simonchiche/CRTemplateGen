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
from scipy.interpolate import interp1d
import scipy
from scipy.interpolate import griddata
from datetime import datetime
from scipy.optimize import curve_fit
from Modules.PlotErad import PlotEradThetaScaling, PlotEradDepthScaling, PlotEradEnergyScaling, PlotEradEScalingvsDepth,PlotAirIceEradRatiovsTheta, PlotAirIceEradRatiovsThetavsE, PlotHpoleVpoleEradRatiovsThetavsE, PlotEradtotThetaScaling, GetMeanEradScalingVsE, PlotMeanEradScalingVsE, PlotEradIceEScalingvsDepth, PlotGroundParticleEVsZenith, PlotEradIcevsZenE, PlotEradIcevsEgroundPart, EradicevsZenE, GetEradvsEgroundPart, GetGroundParticleEnergy, GetEgroundPart_E
#endregion

#region Path definition
SimDir = "FAERIEParam" #"DeepCrLibV1"  #"InterpSim"
WorkPath = os.getcwd()
BatchID = "AtmosphericModel" #"DepthScaling" #"ThetaScaling"
OutputPath = MatplotlibConfig(WorkPath, SimDir, BatchID)
#endregion
Save = True
simpath = "/Users/chiche/Desktop/DeepCrAnalysis/Simulations/FullDenseDeepCr"
SimpathAll = glob.glob(simpath + "/*")


import numpy as np

def _getAirDensity(_height, model):

    '''Returns the air density at a specific height, using either an 
    isothermal model or the Linsley atmoshperic model as in ZHAireS

    Parameters:
    ---------
        h: float
            height in meters

    Returns:
    -------
        rho: float
            air density in g/cm3
    '''

    if model == "isothermal":
            #Using isothermal Model
            rho_0 = 1.225*0.001    #kg/m^3
            M = 0.028966    #kg/mol
            g = 9.81        #m.s^-2
            T = 288.        #
            R = 8.32        #J/K/mol , J=kg m2/s2
            rho = rho_0*np.exp(-g*M*_height/(R*T))  # kg/m3

    elif model == "linsley":
        #Using Linsey's Model
        bl = np.array([1222., 1144., 1305.5948, 540.1778,1])*10  # g/cm2  ==> kg/cm3
        cl = np.array([9941.8638, 8781.5355, 6361.4304, 7721.7016, 1e7])  #m
        hl = np.array([4,10,40,100,113])*1e3  #m
        if (_height>=hl[-1]):  # no more air
            rho = 0
        else:
            hlinf = np.array([0, 4,10,40,100])*1e3  #m
            ind = np.logical_and([_height>=hlinf],[_height<hl])[0]
            rho = bl[ind]/cl[ind]*np.exp(-_height/cl[ind])
            rho = rho[0]*0.001
    else:
        print("#### Error in GetDensity: model can only be isothermal or linsley.")
        return 0

    return rho

heights = np.linspace(3000, 15000, 1000)
rho_values_linsley = np.array([_getAirDensity(h, "isothermal") for h in heights])



AtmPath = "/Users/chiche/Desktop/FAERIEparam/1_ScalingLaws/XmaxParam/Data/Atmosphere.dat"

def rho_corsika(h_m, filename="Atmosphere.dat"):
    """
    Returns air density at height h_m (in meters)
    from a CORSIKA Atmosphere.dat file.

    Output: rho in g/cm^3
    """

    # --- read first 4 numerical lines ---
    data = []
    with open(filename, "r") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                vals = np.fromstring(line, sep=" ")
                if len(vals) > 0:
                    data.append(vals)
                if len(data) == 4:
                    break

    ATMLAY, A, B, C = data

    # --- convert height ---
    h_cm = h_m * 100.0

    # --- find layer ---
    i = np.searchsorted(ATMLAY, h_cm) - 1
    i = max(0, min(i, 4))

    # --- compute density ---
    if i < 4:
        rho = (B[i] / C[i]) * np.exp(-h_cm / C[i])
    else:
        rho = B[i] / C[i]  # top layer (constant)

    return rho

heights = np.linspace(3000, 15000, 1000)
rho_values = np.array([rho_corsika(h, AtmPath) for h in heights])

Plot=False
if(Plot):
    plt.plot(rho_values*1e3, heights/1e3) # convert to kg/m^3
    #plt.plot(rho_values_linsley*1e3, heights/1e3) # convert to kg/m^3
    plt.ylabel("Altitude above sea level [km]")
    plt.xlabel(r"Air Density [${\rm kg/m}^3$]")
    plt.grid()
    plt.show()

