#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np
import glob
from MainModules.ShowerClass import CreateShowerfromHDF5
from XmaxParam.GetAirXmaxPos import getXmaxPosition, Xmax_param_QGSJETII
from XmaxParam.AirDensityParam import rho_corsika
import sys
from copy import deepcopy


# Configuration, separate from the target shower parameters.
LIBRARY_PATH = Path("/Users/chiche/Desktop/DeepCrAnalysis/Simulations/FullDenseDeepCr")
OUTPUT_PATH = Path("/Users/chiche/Desktop/CRTemplateGen/2_TemplateGeneration/CRTemplates")


def GetSimParameters(SimPath):

    Simfiles = glob.glob(str(Path(SimPath) / "*.hdf5"))
    Nsim = len(Simfiles)
    E_sim = np.zeros(Nsim)
    theta_sim = np.zeros(Nsim)
    phi_sim = np.zeros(Nsim)
    
    
    for i in range(Nsim):
        
        splitfile = Path(Simfiles[i]).stem.split("_")
        E_sim[i] = splitfile[2]
        theta_sim[i] = splitfile[3]
        phi_sim[i] = splitfile[4] 
    
    return E_sim, theta_sim, phi_sim, Simfiles

def find_closest_simulation(primary, energy, theta, phi, LIBRARY_PATH):
    
    E_sim, theta_sim, phi_sim, Simfiles = GetSimParameters(LIBRARY_PATH)

    # 1. Closest available zenith
    theta_sel = theta_sim[np.argmin(np.abs(theta_sim - theta))]

    # 2. Indices of simulations at that zenith
    selzen_sims = np.flatnonzero(theta_sim == theta_sel)

    # 3. Closest energy among those simulations
    idx_Esel = selzen_sims[np.argmin(np.abs(E_sim[selzen_sims] - energy))]

    # 4. Return the actual file, preserving its full path
    return Path(Simfiles[idx_Esel])


def GeomagneticFactor(zenith, azimuth):

    azimuth = 0*np.pi/180
    zenith = zenith*np.pi/180
    Bgeo = np.array([7.705, 0, -54.111])

    uB = Bgeo/np.linalg.norm(Bgeo)
    uv = np.array([np.sin(zenith)*np.cos(azimuth), np.sin(zenith)*np.sin(azimuth), -np.cos(zenith)])
    
    sin_alpha = np.sin(np.arccos(np.dot(uv, uB)))

    return sin_alpha

def GetGeomagneticScale(RefSim, target_zenith, target_azimuth):

    sin_target = GeomagneticFactor(target_zenith, target_azimuth)
    sin_reference = GeomagneticFactor(RefSim.zenith, RefSim.azimuth)

    #print(np.degrees(np.arcsin(sin_target)), "target geomagnetic angle")
    #print(np.degrees(np.arcsin(sin_reference)), "reference geomagnetic angle")

    return (sin_target / sin_reference)**2



def GetXmaxParams(RefSim, TargetSim, MassNumber, target_energy, target_zenith, target_azimuth):

    # Target shower Xmax air density and distance from shower core
    Xmax_target = Xmax_param_QGSJETII(target_energy*1e18, MassNumber, fluctuations=True)[0]
    XmaxPos_target, Dxmax_target = getXmaxPosition(target_azimuth, target_zenith, \
                                                   glevel=3216, injection=1e6, showerDistance=0, Xmax_primary =Xmax_target)
    rhoXmax_target = rho_corsika(XmaxPos_target[2])

    TargetSim.xmax = Xmax_target
    TargetSim.xmaxdist = Dxmax_target

    # Reference shower Xmax air density and distance from shower core
    XmaxPos_ref = getXmaxPosition(RefSim.azimuth, RefSim.zenith, \
                                                   glevel=3216, injection=1e6, showerDistance=0, Xmax_primary=RefSim.xmax)[0]
    Dxmax_ref = RefSim.xmaxdist
    rhoXmax_ref = rho_corsika(XmaxPos_ref[2])

    return rhoXmax_ref, rhoXmax_target, Dxmax_ref, Dxmax_target

 #(rhoXmax_ref/ rhoXmax_target)**2

def GetIceXmaxDistScale(Dxmax_ref, Dxmax_target):
    """Return the scaling factor for the ice component based on Xmax distances.

    """
    Lambda = 1680 # in meters, fitted attenuation legnth due to propgatin in the atmosphere

    return np.exp(-2*Dxmax_target/Lambda)/np.exp(-2*Dxmax_ref/Lambda)

def get_scaling_factors(RefSim, TargetSim, MassNumber, target_energy, target_zenith, target_azimuth):

    rhoXmax_ref, rhoXmax_target, Dxmax_ref, Dxmax_target = \
        GetXmaxParams(RefSim, TargetSim, MassNumber, target_energy, target_zenith, target_azimuth)

    EnergyScale = (target_energy / RefSim.energy)**2

    GeoScale = GetGeomagneticScale(RefSim, target_zenith, target_azimuth)

    AirDensityScale = (rhoXmax_ref / rhoXmax_target) ** 2

    IceXmaxDistScale = GetIceXmaxDistScale(Dxmax_ref, Dxmax_target)

    scaling_factors = {
        "energy": EnergyScale,
        "geomagnetic": GeoScale,
        "air_density": AirDensityScale,
        "XmaxDist": IceXmaxDistScale
    }

    return scaling_factors



def scale_electric_field(RefSim, TargetSim, scaling_factors, primary, energy, zenith, azimuth):
    """Return antenna_positions, traces_air and traces_ice for the target.

    Each returned collection must retain consistent antenna IDs. Reference
    arrays must remain unchanged so they can be reused for another event.
    """
    # TODO: Define any coordinate/polarization rotations, footprint transformations
    # and timing changes required by the target geometry.
    # TODO: Apply the air and ice amplitude factors to electric-field components.
    # TODO: Return the target positions and the two trace dictionaries.

    Escale, GeoScale, AirDensityScale, IceXmaxDistScale = (
        scaling_factors["energy"],
        scaling_factors["geomagnetic"],
        scaling_factors["air_density"],
        scaling_factors["XmaxDist"],
    )

    AirScale = np.sqrt(Escale * GeoScale * AirDensityScale)
    IceScale = np.sqrt(Escale * IceXmaxDistScale)

    TargetSim.traces_C = {ant_id: AirScale * RefSim.traces_C[ant_id] for ant_id in RefSim.traces_C}
    TargetSim.traces_G = {ant_id: IceScale * RefSim.traces_G[ant_id] for ant_id in RefSim.traces_G}


    raise NotImplementedError("Electric-field template scaling is TODO.")























def save_template(template, output_path):
    """Save traces, positions, units, target parameters and reference provenance."""
    # TODO: Choose the output format, naming scheme and overwrite policy.
    # TODO: Create the output directory and serialize the complete template.
    raise NotImplementedError("Template serialization is TODO.")


def generate_template(primary, energy, zenith, azimuth, Save=False):
    """Select a reference, scale its traces and optionally save the result.

    Parameters
    ----------
    primary : str
        Target primary particle (for example, 'Proton' or 'Iron').
    energy : float
        Target primary energy in EeV.
    zenith, azimuth : float
        Target shower angles in degrees.
    Save : bool
        Whether to write the result to OUTPUT_PATH. The result is returned
        regardless of this flag once the TODO functions are implemented.
    """
    simulation_path = find_closest_simulation(
        primary, energy, zenith, azimuth, LIBRARY_PATH
    )
    RefSim = CreateShowerfromHDF5(simulation_path)
    
    TargetSim = deepcopy(RefSim) 
    TargetSim.energy, TargetSim.zenith, TargetSim.azimuth = energy, zenith, azimuth

    scaling_factors = get_scaling_factors(RefSim, TargetSim, primary, energy, zenith, azimuth)

    positions, traces_air, traces_ice = scale_electric_field(
        RefSim, TargetSim, scaling_factors, primary, energy, zenith, azimuth
    )

    sys.exit()


    template = {
        "primary": primary,
        "energy": energy,
        "zenith": zenith,
        "azimuth": azimuth,
        "reference_simulation": str(simulation_path),
        "scaling_factors": scaling_factors,
        "antenna_positions": positions,
        "traces_air": traces_air,
        "traces_ice": traces_ice,
        "units": {
            "energy": "EeV", "angles": "deg", "positions": "m",
            "time": "s", "electric_field": "microvolt/m",
        },
    }
    if Save:
        save_template(template, OUTPUT_PATH)
    return template


generate_template(1, 0.316, 28, 0, Save=False)


'''
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--energy", type=float, required=True, help="Energy in EeV")
    parser.add_argument("--zenith", type=float, required=True, help="Zenith in degrees")
    parser.add_argument("--azimuth", type=float, required=True, help="Azimuth in degrees")
    parser.add_argument("--save", action="store_true", help="Save the generated template")
    args = parser.parse_args()
    return generate_template(
        args.primary, args.energy, args.zenith, args.azimuth, Save=args.save
    )


if __name__ == "__main__":
    main()
'''
