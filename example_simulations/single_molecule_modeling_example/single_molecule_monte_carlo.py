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
def propose_move(atoms, dihedrals, rng, perturb_stdev, rotation_prob=0.5, torsion_step=15.0):
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


# For adaptive acceptable error
def tau_int_initial_positive_sequence(x, max_lag=256):
    """Simple IPS estimator of integrated autocorrelation time for a 1D series."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 4:
        return 1.0
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom <= 0:
        return 1.0
    # normalized autocovariances up to max_lag
    ac = np.correlate(x, x, mode='full')[n-1:n+min(max_lag, n-1)] / denom
    s = 1.0
    for k in range(1, len(ac)):
        if ac[k] <= 0:
            break
        s += 2.0 * ac[k]
    return max(1.0, s)




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


def run_mc(molecule_name, num_steps, temp, acceptable_error, target_model, run_name, trial_num, rotation_prob, fidelity,
           control_params = {}, surrogate_params = {}, verbose = False):

    # For adaptive acceptable error
    W = 50 # window size for ROG stats
    rog_deltas_accept = []
    acceptable_error_list = []

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
    energy = target_model.get_potential_energy(atoms)

    # Create Control model
    target_model.get_potential_energy = ControlWrapper(target_model.get_potential_energy, surrogate, acceptable_error, **control_params)


    for step in range(num_steps):
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
        if target_model.get_potential_energy.adaptive_acceptable_error:
            acceptable_error_list.append(target_model.get_potential_energy.acceptable_error)





        # Track |ΔROG| only for accepted moves
        if accept and len(r_g) >= 2:
            rog_deltas_accept.append(abs(r_g[-1] - r_g[-2]))
            if len(rog_deltas_accept) > W:
                rog_deltas_accept = rog_deltas_accept[-W:]

        # Optionally adapt acceptable_error from final ROG bound
        if target_model.get_potential_energy.adaptive_acceptable_error:
            if len(rog_deltas_accept) >= 5 and len(r_g) >= 10 and len(accept_probs) >= 10:
                # robust Δg: 95% quantile of recent |ΔROG|
                delta_g_q = np.quantile(rog_deltas_accept, 0.5)
                # τ_int from a trailing window of ROG
                tau_hat   = tau_int_initial_positive_sequence(r_g[-min(len(r_g), 4*W):])
                # mean acceptance over recent window
                p_avg     = float(np.mean(accept_probs[-min(len(accept_probs), W):]))
                # Treat 'acceptable_error' argument as desired final ROG bound B
                target_model.get_potential_energy.update_acceptable_error(
                    kT,
                    final_error_bound=acceptable_error,
                    delta_g_q=delta_g_q,
                    tau_int=tau_hat,
                    p_avg=p_avg,
                    safety=1,
                    max_eps_ratio=0.7,
                )


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

    json_results = {"run_info": {"molecule": molecule_name, "temp": temp, "target_model_fidelity": fidelity},
                    "r_g_stats": stats,
                    'acceptable_error': acceptable_error,
                    'time': elapsed_time
                   }

    csv_dict = {'Energy': energies,
                'New Energy': new_energies,
                'ROG': r_g, 
                'Acceptance Threshold': accept_probs,
                'config_accepts': config_accepts,
                'surrogate_accepts': surrogate_accepts,
               }

    ctrl = target_model.get_potential_energy
            
    for key, values in ctrl.results.items():
        if len(values) != 0:
            csv_dict[key] = values

    if ctrl.adaptive_acceptable_error:
            csv_dict["acceptable_error"] = acceptable_error_list

    csv_results = pd.DataFrame(csv_dict)

    # Save extra JSON results
    json_results['surrogate_info'] = {'ensemble_size': surrogate.ensemble_size, 'max_training_size': surrogate.max_data}
    json_results['control_model_info'] = {'initial_surrogate_data': ctrl.initial_surrogate_data, 
                                          'initial_error_data': ctrl.initial_error_data,
                                          'retrain_interval': ctrl.retrain_interval,
                                          'error_prediction_type': ctrl.error_prediction_type,
                                          'sliding_window_size': ctrl.sliding_window_size,
                                          'adaptive_acceptable_error': ctrl.adaptive_acceptable_error
                                         }
    json_results['control_model_info'].update({
        'threshold_type': ctrl.threshold_type,
        'threshold_window': ctrl.threshold_window,
        'epsilon': ctrl.epsilon,
        'prediction_window_size': ctrl.prediction_window_size,
    })
    json_results['retraining_times'] = ctrl.retraining_times
    surrogate_used = np.sum(surrogate_accepts)

    # Make sure that there are no nans
    errs = np.array(ctrl.results["surrogate_error"], dtype=np.float64)
    finite_mask = np.isfinite(errs)
    if finite_mask.any():
        mean_err = float(errs[finite_mask].mean())
        median_err = float(np.median(errs[finite_mask]))
    else:
        mean_err = None
        median_err = None
    json_results['summary'] = {
        'num_steps': num_steps,
        'num_surrogate_steps': int(surrogate_used),
        'surrogate_fraction': float(surrogate_used) / num_steps,
        'mean_surrogate_error': mean_err,
        'median_surrogate_error': median_err,
        'acceptance_rate': float(np.mean(config_accepts)),
    }

    if ctrl.adaptive_acceptable_error:
        json_results['summary']['final_acceptable_error'] = ctrl.acceptable_error


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
    surrogate_error = np.array(ctrl.error_prediction_vals['surrogate_error'])
    epistemic_uncertainty = np.array(ctrl.error_prediction_vals['epistemic_uncertainty'])
    aleatory_uncertainty = np.array(ctrl.error_prediction_vals['aleatory_uncertainty'])
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
    arg_parser.add_argument('--nsteps', '-t', help='Number of Monte Carlo steps', default=1000, type=int)
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
    arg_parser.add_argument('--threshold-type', '-s', help='Decides what type of threshold system to use',
                            default="error_pred_fixed_threshold", type=str)
    arg_parser.add_argument('--adaptive-acceptable-error', '-A', help="If set, treat acceptable_error as final ROG bias bound and adapt surrogate error bound online.", action='store_true')
    arg_parser.add_argument('--fidelity', '-f', help='Controls the accuracy/cost of the quantum chemistry code',
                            default='low', choices=['low', 'medium', 'high'], type=str)
    arg_parser.add_argument('--control-mode', '-C', default='original', choices = ['original', 'original_audit', 'tm', 'tm_audit'], type=str)                      

    
    args = arg_parser.parse_args()

    # Create DFT target model
    calc = Psi4(memory="500MB", PSI_SCRATCH=f"{script_dir}/output/tmp/", **_fidelity[args.fidelity])

    run_mc(args.molecule, args.nsteps, args.temp, args.acceptable_error, calc, args.run_name, args.trial, rotation_prob=args.rotation_prob, fidelity=args.fidelity,
           control_params = {'threshold_type': args.threshold_type, 'retrain_interval': args.retrain_interval, 
                             'prediction_window_size': args.prediction_window_size, 'epsilon': args.epsilon, 
                             'adaptive_acceptable_error': args.adaptive_acceptable_error, 'control_mode': args.control_mode}, 
           surrogate_params = {'max_data': args.max_surrogate_training_size}, verbose = False)