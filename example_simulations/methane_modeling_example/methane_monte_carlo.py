import sys
import os

module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'proxima_plus'))
sys.path.append(module_dir)

import time
import json
from datetime import datetime
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from scipy.stats import bayes_mvs

from ase.calculators.psi4 import Psi4
from sklearn.linear_model import BayesianRidge

# From my files
from utils import create_atoms, make_data_pipeline, radius_of_gyration
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


def run_mc(num_steps, temp, acceptable_error, target_model, run_name, control_params = {}, surrogate_params = {}, verbose = False):

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
    atoms = create_atoms()

    # Set kT
    kT = 3.1668115635e-6 * temp

    # Create data pipeline, surrogate model, and control model
    data_pipeline = make_data_pipeline()
    surrogate = DeepEnsembleSurrogate(BayesianRidge, data_pipeline, **surrogate_params)

    # Get initial energy
    energy = calc.get_potential_energy(atoms)

    # Create Control model
    target_model.get_potential_energy = ControlWrapper(target_model.get_potential_energy, surrogate, acceptable_error, **control_params)


    for step in range(num_steps):
        new_atoms = atoms.copy()
        new_atoms.rattle(stdev=PERTURB, rng=rng)
        

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
                                'error_uncertainty_correlation': target_model.get_potential_energy.results["coefficient_of_determination"]
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
    current_time = datetime.utcnow().strftime("%d%b%y-%H%M%S")
    directory_path = f"../output/runs/{run_name}/{run_name}_temp{temp}_{current_time}/"
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
    create_error_plot(np.log10(epistemic_uncertainty), surrogate_error, f"{directory_path}epistemic_error_plot")
    create_error_plot(np.log10(aleatory_uncertainty), surrogate_error, f"{directory_path}aleatory_error_plot")
    

if __name__ == "__main__":
    # Parse the arguments
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--run-name', '-n', help='Name of run and determines the save directory',
                            default='testing', type=str)
    arg_parser.add_argument('--temp', '-T', help='Temperature at which to sample (K).'
                                                 'Default is 298 (room temperature)', default=298, type=float)
    arg_parser.add_argument('--nsteps', '-t', help='Number of Monte Carlo steps', default=1000, type=int)
    arg_parser.add_argument('--acceptable-error', '-a', help='Threshold for Acceptable Surrogate Error',
                            default=0.002, type=float)
    arg_parser.add_argument('--fidelity', '-f', help='Controls the accuracy/cost of the quantum chemistry code',
                            default='low', choices=['low', 'medium', 'high'], type=str)

    
    args = arg_parser.parse_args()

    # Create DFT target model
    calc = Psi4(memory="500MB", PSI_SCRATCH="../output/tmp/", **_fidelity[args.fidelity])


    run_mc(args.nsteps, args.temp, args.acceptable_error, calc, args.run_name, control_params = {}, surrogate_params = {}, verbose = False)