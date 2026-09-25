#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np
import glob


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


    E_sel = E_sim[np.argmin(abs(E_sim-energy))]


    # 1. Closest available zenith
    theta_sel = theta_sim[np.argmin(np.abs(theta_sim - theta))]

    # 2. Indices of simulations at that zenith
    selzen_sims = np.flatnonzero(theta_sim == theta_sel)

    # 3. Closest energy among those simulations
    idx_Esel = selzen_sims[np.argmin(np.abs(E_sim[selzen_sims] - E))]

    # 4. Return the actual file, preserving its full path
    return Path(Simfiles[idx_Esel])


def load_reference_simulation(simulation_path):
    """Load reference metadata, antenna positions and both sets of traces.

    Expected result: a dict with primary, energy, zenith, azimuth,
    antenna_positions, traces_air and traces_ice. Each trace collection maps
    antenna IDs to arrays with columns [time, Ex, Ey, Ez]. Positions in metres
    must use the same antenna IDs. Additional physical metadata may be needed.
    """
    # TODO: Connect the existing HDF5 reader / Shower implementation.
    # TODO: Preserve antenna identity and check units and coordinate conventions.
    raise NotImplementedError("Reference simulation loading is TODO.")


def get_scaling_factors(reference, primary, energy, zenith, azimuth):
    """Return separate electric-field amplitude factors for air and ice.

    Expected result: a dict with 'air' and 'ice' entries.
    """
    # TODO: Compute the required target/reference shower properties (Xmax,
    # distance to Xmax, air density and geomagnetic angle).
    # TODO: Evaluate the radiation-energy scaling laws with calibrated parameters.
    # TODO: Derive amplitude factors from the square root of the target/reference
    # radiation-energy ratios, checking the assumptions about footprint geometry.
    raise NotImplementedError("Electric-field scaling factors are TODO.")


def scale_electric_field(reference, scaling_factors, primary, energy, zenith, azimuth):
    """Return antenna_positions, traces_air and traces_ice for the target.

    Each returned collection must retain consistent antenna IDs. Reference
    arrays must remain unchanged so they can be reused for another event.
    """
    # TODO: Define any coordinate/polarization rotations, footprint transformations
    # and timing changes required by the target geometry.
    # TODO: Apply the air and ice amplitude factors to electric-field components.
    # TODO: Return the target positions and the two trace dictionaries.
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
    reference = load_reference_simulation(simulation_path)
    scaling_factors = get_scaling_factors(reference, primary, energy, zenith, azimuth)
    positions, traces_air, traces_ice = scale_electric_field(
        reference, scaling_factors, primary, energy, zenith, azimuth
    )

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
