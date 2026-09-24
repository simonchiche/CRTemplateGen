#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py
import sys
import numpy as np
from MainModules.FormatFaerieOutput import Traces_cgs_to_si

HDF5folderpath = "/Users/chiche/Desktop/HDF5filesReader/hdf5FAERIE/"
hdf5Filename = "Rectangle_Proton_0.1_20_0_1_0.hdf5"
HDF5filepath = HDF5folderpath + hdf5Filename

### Printng HDF5 structure

def print_hdf5_structure(file_name):
    with h5py.File(file_name, "r") as f:
        def recursively_print(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}, Shape: {obj.shape}, Type: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"Group: {name}")
        f.visititems(recursively_print)
        sys.exit()

# Loading main keys
def PrintMainKeys(HDF5filepath):
    with h5py.File(HDF5filepath, "r") as f:
        print("Keys in the file:", list(f.keys()))

# Loading keys and subkeys
def PrintAllKeys(HDF5filepath):
    with h5py.File(HDF5filepath, "r") as f:
        for key in f.keys():
            print(key, list(f[key].keys()))

# Printing attributes
def PrintAllAttributes(HDF5filepath):
    with h5py.File(HDF5filepath, "r") as f:
        print(dict(f.attrs))
        for key in f.keys():
            print(key, dict(f[key].attrs))
   

#print_hdf5_structure(HDF5filepath)
##PrintMainKeys(HDF5filepath)
##PrintAllKeys(HDF5filepath)
##PrintAllAttributes(HDF5filepath)

 # Extracting the main useful information from the HDF5 file
def GetBfromHdf5(HDF5filepath):
    with h5py.File(HDF5filepath, "r") as f:
        input_attrs = f["inputs"].attrs
        B = input_attrs['MAGNET']
    return B

def GetEnergyFromHdf5(HDF5filepath):
    with h5py.File(HDF5filepath, "r") as f:
        input_attrs = f["inputs"].attrs
        Energy = input_attrs['ERANGE'][0]
    return Energy*1e9/1e18

def GetZenithFromHdf5(HDF5filepath):
    with h5py.File(HDF5filepath, "r") as f:
        input_attrs = f["inputs"].attrs
        zenith = input_attrs['THETAP'][0]
    return zenith

# Should be added in the hdf5 file
def GetAzimuthFromHdf5(HDF5filepath):
    with h5py.File(HDF5filepath, "r") as f:
        input_attrs = f["CoREAS"].attrs
        azimuth = input_attrs['ShowerAzimuthAngle']
    return azimuth

def GetGlevelFromHdf5(HDF5filepath):
    # Ice altitude above sea level in meters
    with h5py.File(HDF5filepath, "r") as f:
        input_attrs = f["inputs"].attrs
        glevel = input_attrs['OBSLEV']/1e2
    return glevel

def GetXmaxFromHdf5(HDF5filepath):
    # Ice altitude above sea level in meters
    with h5py.File(HDF5filepath, "r") as f:
        input_attrs = f["CoREAS"].attrs
        XmaxDepth = input_attrs['DepthOfShowerMaximum']
        XmaxDist = input_attrs['DistanceOfShowerMaximum']
    return XmaxDepth, XmaxDist


def GetPosFromHdf5(HDF5filepath):
    Pos = []
    with h5py.File(HDF5filepath, "r") as f:
        k = 0
        fObs = f["CoREAS"]["observers"]
        for key in fObs.keys():
            attrs = fObs[key].attrs
            antpos = attrs['position']
            Pos.append(antpos)
    
    Pos = np.array(Pos)

    return Pos

def GetPrimaryFromHdf5(HDF5filepath):
    with h5py.File(HDF5filepath, "r") as f:
        input_attrs = f["CoREAS"].attrs
        prim_id = input_attrs['PrimaryParticleType']
        if(prim_id == 14):
            Primary = "Proton"
        else:
            Primary = "Undefined"
    return Primary

def GetTracesfromHDF5(HDF5filepath):
    k =0
    Traces_C, Traces_G = dict(), dict()
    with h5py.File(HDF5filepath, "r") as f:
        observers_coreas = f["CoREAS"]["observers"]  # Navigate to the group
        observers_geant = f["CoREAS"]["observers_geant"]
        keys_coreas = list(observers_coreas.keys())  # List all datasets inside
        keys_geant = list(observers_geant.keys())  # List all datasets inside
        #print("Available keys:", keys)
        for key_c, key_g in zip(keys_coreas, keys_geant):
            #print(key_c, key_g)
            Traces_C[k] = observers_coreas[key_c][()]
            Traces_G[k] = observers_geant[key_g][()]
            k = k +1
        
        return Traces_C, Traces_G
    

def LoadHDF5file(HDF5filepath):
    print(HDF5filepath)
    Primary = GetPrimaryFromHdf5(HDF5filepath)
    E0 = GetEnergyFromHdf5(HDF5filepath)
    zenith = GetZenithFromHdf5(HDF5filepath)
    azimuth =  GetAzimuthFromHdf5(HDF5filepath)
    Bgeo = GetBfromHdf5(HDF5filepath)
    glevel = GetGlevelFromHdf5(HDF5filepath)
    XmaxParam = GetXmaxFromHdf5(HDF5filepath)
    Pos = GetPosFromHdf5(HDF5filepath)
    Traces_C, Traces_G = GetTracesfromHDF5(HDF5filepath)
    Traces_C = Traces_cgs_to_si(Traces_C)
    Traces_G = Traces_cgs_to_si(Traces_G)
    Nant = len(Pos)


    return Primary, E0, zenith, azimuth, Bgeo, glevel, XmaxParam, Pos, Nant, Traces_C, Traces_G

#HDF5filepath = "/Users/chiche/Desktop/Circle_Proton_0.0316_0_0_1.hdf5"
#Primary, E0, zenith, azimuth, Bgeo, glevel, XmaxParam, Pos, Nant, Traces_C, Traces_G = LoadHDF5file(HDF5filepath)
#print(E0)

