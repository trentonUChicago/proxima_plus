import sys
import os

module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'proxima_plus'))
sys.path.append(module_dir)

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

import time
import json
from datetime import datetime
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from scipy.stats import bayes_mvs

import psi4
from ase.calculators.psi4 import Psi4
from sklearn.linear_model import BayesianRidge


# Silence all output from Psi4.
psi4.core.be_quiet()

# For reading data
from ase.io import read
from io import StringIO
import requests

# From my files
from utils import make_data_pipeline, radius_of_gyration
from ensemble import DeepEnsembleSurrogate
from control import ControlWrapper
from visualize import create_error_plot

# Set the random seed
rng = np.random.RandomState()

PERTURB = 0.003

_fidelity = {
    'low': {'method': 'HF', 'basis': 'sto-3g'},
    'medium': {'method': 'b3lyp', 'basis': '6-311g_d_p_'},
    'high': {'method': 'ccsd(t)', 'basis': 'cc-pVTZ'}
}


## For bond rotations
from bond_rotation import get_initial_structure, detect_dihedrals
SMILES_MAP = {
    "methane": "C",
    "ethanol": "CCO",
    "butane":  "CCCC",
    "hexane":  "CCCCCC",
}
def init_atoms_and_dihedrals(molecule_name):
    """
    Return (atoms, dihedrals_list) for the given molecule
    """

    if molecule_name.lower() in SMILES_MAP:
        smiles = SMILES_MAP[molecule_name.lower()]
        atoms, mol = get_initial_structure(smiles)
        dihedrals = detect_dihedrals(mol)
    else:
        atoms = get_pubchem_molecule(molecule_name)
        dihedrals = []

    return atoms, dihedrals
def propose_move(atoms, dihedrals, rng, perturb_stdev, rotation_prob=0.5, torsion_step=30.0):
    """
    Propose a new configuration:
      - With probability rotation_prob: random dihedral rotation
      - Otherwise: small Cartesian rattle (your old behavior)

    dihedrals: list of DihedralInfo from setup.detect_dihedrals
    rng: numpy RandomState
    """
    new_atoms = atoms.copy()

    # If we don't have any dihedrals (e.g., methane), always do rattle
    if not dihedrals or rng.rand() > rotation_prob:
        new_atoms.rattle(stdev=perturb_stdev, rng=rng)
        return new_atoms

    # Pick a random dihedral
    dih = dihedrals[rng.randint(len(dihedrals))]

    # Current angle (degrees)
    current_angle = dih.get_angle(new_atoms)  # uses Atoms.get_dihedral under the hood

    # Propose new angle: random increment around current
    delta = rng.normal(loc=0.0, scale=torsion_step)  # e.g. ~30° changes
    new_angle = current_angle + delta

    # Build mask of atoms that should rotate
    mask = [i in dih.group for i in range(len(new_atoms))]

    # Apply rotation; ASE will rotate that whole group about the dihedral
    # a1, a2, a3, a4 = dih.chain
    new_atoms.set_dihedral(*dih.chain, new_angle, mask=mask)

    return new_atoms
## For bond rotations




def get_pubchem_molecule(name, max_retries=100):
    """Download 3D coordinates of molecule from PubChem"""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/SDF?record_type=3d"
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            # Convert to ASE atoms
            atoms = read(StringIO(response.text), format='sdf')
            return atoms
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(10) # Wait before retrying


def run_mc(molecule_name, num_steps, temp, acceptable_error, target_model, run_name, trial_num, adaptive_threshold, rotation_prob, control_params = {}, surrogate_params = {}, verbose = False):

    # Initialize results
    r_g = []
    energies = []
    new_energies = []
    accept_probs = []
    config_accepts = []
    surrogate_accepts = []

    # Start timer
    start_time = time.time()
    
    # Get atoms
    # atoms = get_pubchem_molecule(molecule_name)
    atoms, dihedrals = init_atoms_and_dihedrals(molecule_name)

    # Set kT
    kT = 3.1668115635e-6 * temp

    # Create data pipeline, surrogate model, and control model
    data_pipeline = make_data_pipeline(soap_kwargs={'species': frozenset(atoms.get_chemical_symbols())})
    surrogate = DeepEnsembleSurrogate(BayesianRidge, data_pipeline, **surrogate_params)

    # Get initial energy
    energy = calc.get_potential_energy(atoms)

    # Create Control model
    target_model.get_potential_energy = ControlWrapper(target_model.get_potential_energy, surrogate, acceptable_error, **control_params)


    for step in range(num_steps):
        # new_atoms = atoms.copy()
        # new_atoms.rattle(stdev=PERTURB, rng=rng)
        new_atoms = propose_move(atoms, dihedrals, rng, perturb_stdev=PERTURB)
        

        # Calculate energy
        target_model.reset()
        new_energy, is_surrogate = target_model.get_potential_energy(new_atoms)

        # Determine if we should reject or accept
        delta_E = new_energy - energy
        prob = np.exp(-delta_E / kT)
        accept = np.random.random() < prob

        # Set new configuration if accepting
        if accept:
            energy = new_energy
            atoms = new_atoms

        # Compute radius of gyration
        r_g.append(radius_of_gyration(atoms))
        energies.append(energy)
        config_accepts.append(accept)
        surrogate_accepts.append(is_surrogate)

        new_energies.append(new_energy)
        accept_probs.append(prob)

        if adaptive_threshold is True:
            if len(r_g) > 2:
                max_rog_change = np.max(np.abs(np.diff(r_g)))
                target_model.get_potential_energy.update_threshold(kT, acceptable_error, max_rog_change)


        if verbose:
            print(f"{step}: we got an energy of {energy} and RoG of {r_g[-1]} after accept({accept}) and using surrogate: {is_surrogate}")

    # Get time and notify of finished run
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Just did a run for {run_name} at temp: {temp}")

    r_g_last_half = r_g[len(r_g) // 2:]
    
    if len(set(r_g_last_half)) == 1:
        stats = {
                 "statistic": r_g_last_half[0],
                 "minmax": [r_g_last_half[0], r_g_last_half[0]]
                }
    else:
        stats = bayes_mvs(r_g_last_half)[0]._asdict()

    json_results = {"r_g_stats": stats,
                'acceptable_error': acceptable_error,
                'temp': temp,
                'time': elapsed_time
                }
    
    csv_results = pd.DataFrame({'Energy': energies,
                                'New Energy': new_energies,
                                'ROG': r_g, 
                                'Acceptance Threshold': accept_probs,
                                'config_accepts': config_accepts,
                                'surrogate_accepts': surrogate_accepts,
                                'surrogate_energy': target_model.get_potential_energy.results["y_prediction"],
                                'target_energy': target_model.get_potential_energy.results["y_target"], 
                                'epistemic_uncertainty': target_model.get_potential_energy.results["epistemic_uncertainty"], 
                                'aleatory_uncertainty': target_model.get_potential_energy.results["aleatory_uncertainty"], 
                                'error_uncertainty_correlation': target_model.get_potential_energy.results["coefficient_of_determination"],
                                'surrogate_error_prediction': target_model.get_potential_energy.results["surrogate_error_prediction"],
                                'acceptable_surrogate_error': target_model.get_potential_energy.results["acceptable_surrogate_error"]
                               })


    # Save extra JSON results
    json_results['surrogate_info'] = {'ensemble_size': surrogate.ensemble_size, 'max_training_size': surrogate.max_data}
    json_results['control_model_info'] = {'initial_surrogate_data': target_model.get_potential_energy.initial_surrogate_data, 
                                          'initial_error_data': target_model.get_potential_energy.initial_error_data,
                                          'retrain_interval': target_model.get_potential_energy.retrain_interval,
                                          'error_prediction_type': target_model.get_potential_energy.error_prediction_type,
                                          'sliding_window_size': target_model.get_potential_energy.sliding_window_size
                                         }
    json_results['retraining_times'] = target_model.get_potential_energy.retraining_times


    # Create Directory
    # current_time = datetime.utcnow().strftime("%d%b%y-%H%M%S")
    directory_path = f"{script_dir}/output/runs/{run_name}_{molecule_name}/{run_name}_{molecule_name}_temp{int(temp)}_trial{trial_num}/"
    os.makedirs(directory_path, exist_ok=True)

    # Add JSON results to directory
    with open(directory_path + "results.json", 'w') as fp:
        json.dump(json_results, fp, indent=2)
    
    # Add CSV results to directory
    csv_results.to_csv(directory_path + "results.csv", index=False)


    
    # Get errors
    surrogate_error = np.array(target_model.get_potential_energy.error_prediction_vals['surrogate_error'])
    epistemic_uncertainty = np.array(target_model.get_potential_energy.error_prediction_vals['epistemic_uncertainty'])
    aleatory_uncertainty = np.array(target_model.get_potential_energy.error_prediction_vals['aleatory_uncertainty'])
    # Create and save error plots
    if epistemic_uncertainty.size != 0:
        create_error_plot(np.log10(epistemic_uncertainty), surrogate_error, f"{directory_path}epistemic_error_plot")
        create_error_plot(np.log10(aleatory_uncertainty), surrogate_error, f"{directory_path}aleatory_error_plot")
    

if __name__ == "__main__":
    # Parse the arguments
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--run-name', '-n', help='Name of run and determines the save directory',
                            default='testing', type=str)
    arg_parser.add_argument('--trial', '-i', help='Trial Number',
                            default=1, type=int)
    arg_parser.add_argument('--molecule', '-m', help='Molecule to simulate',
                            default="methane", type=str)
    arg_parser.add_argument('--temp', '-T', help='Temperature at which to sample (K).'
                                                 'Default is 298 (room temperature)', default=298, type=float)
    arg_parser.add_argument('--nsteps', '-t', help='Number of Monte Carlo steps', default=5000, type=int)
    arg_parser.add_argument('--acceptable-error', '-a', help='Threshold for Acceptable Surrogate Error or Final Error if using adaptive threshold',
                            default=0.002, type=float)
    arg_parser.add_argument('--epsilon', '-e', help='Chance for the control model to run the target model anyways',
                            default=0.1, type=float)
    arg_parser.add_argument('--rotation-prob', '-R', help='Probability of proposing a diheral rotation instead of a cartesian rattle.', default=0.5, type=float)
    arg_parser.add_argument('--retrain-interval', '-r', help='Retraining interval for surrogate model',
                            default=None, type=int)
    arg_parser.add_argument('--prediction-window-size', '-p', help='Size of the prediction window that takes the most recent results to train surrogate error predictor',
                            default=0, type=int)
    arg_parser.add_argument('--max-surrogate-training-size', '-u', help='Maximum amount of training data for surrogate to use',
                            default=None, type=int)
    arg_parser.add_argument('--adaptive-threshold', '-s', action='store_true', help='Decides whether to adaptively set surrogate error threshold or not',
                            default=False)
    arg_parser.add_argument('--fidelity', '-f', help='Controls the accuracy/cost of the quantum chemistry code',
                            default='low', choices=['low', 'medium', 'high'], type=str)                      

    
    args = arg_parser.parse_args()

    # Create DFT target model
    calc = Psi4(memory="500MB", PSI_SCRATCH="{script_dir}/output/tmp/", **_fidelity[args.fidelity])

    run_mc(args.molecule, args.nsteps, args.temp, args.acceptable_error, calc, args.run_name, args.trial, args.adaptive_threshold, rotation_prob=rgs.rotation_prob,
           control_params = {'retrain_interval': args.retrain_interval, 'prediction_window_size': args.prediction_window_size, 'epsilon': args.epsilon}, 
           surrogate_params = {'max_data': args.max_surrogate_training_size}, verbose = False)