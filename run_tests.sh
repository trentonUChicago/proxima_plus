

# # Methane
# # Run baseline
# snakemake --cores 10 --config run_name=baseline molecule=methane

# # Run with control
# snakemake --cores 10 --config run_name=control_2 molecule=methane
# snakemake --cores 10 --config run_name=full molecule=methane
# snakemake --cores 10 --config run_name=pred_window_100 molecule=methane prediction_window_size=100
# snakemake --cores 10 --config run_name=pred_window_200 molecule=methane prediction_window_size=200
# snakemake --cores 10 --config run_name=pred_window_300 molecule=methane prediction_window_size=300

# snakemake --cores 10 --config run_name=max_train_100 molecule=methane max_surrogate_training_size=100
# snakemake --cores 10 --config run_name=max_train_200 molecule=methane max_surrogate_training_size=200
# snakemake --cores 10 --config run_name=max_train_300 molecule=methane max_surrogate_training_size=300




# python example_simulations/single_molecule_modeling_example/single_molecule_monte_carlo.py --run-name butane_big_attempt_5000 --trial 1 --molecule butane --temp 100 --nsteps 5000 --acceptable-error 0
# python example_simulations/single_molecule_modeling_example/single_molecule_monte_carlo.py --run-name butane_big_attempt_5000 --trial 1 --molecule butane --temp 1000 --nsteps 5000 --acceptable-error 0

# python example_simulations/single_molecule_modeling_example/single_molecule_monte_carlo.py --run-name butane_big_attempt_10000 --trial 1 --molecule butane --temp 100 --nsteps 10000 --acceptable-error 0
# python example_simulations/single_molecule_modeling_example/single_molecule_monte_carlo.py --run-name butane_big_attempt_10000 --trial 1 --molecule butane --temp 1000 --nsteps 10000 --acceptable-error 0


# python example_simulations/single_molecule_modeling_example/single_molecule_monte_carlo.py --run-name butane_big_attempt_20000 --trial 1 --molecule butane --temp 100 --nsteps 20000 --acceptable-error 0
# python example_simulations/single_molecule_modeling_example/single_molecule_monte_carlo.py --run-name butane_big_attempt_20000 --trial 1 --molecule butane --temp 1000 --nsteps 20000 --acceptable-error 0



# ## Ethanol
# # Run baseline
# snakemake --cores 10 --config run_name=big_baseline run_type=baseline molecule=ethanol

# Run with control
# snakemake --cores 10 --config run_name=control molecule=ethanol
# snakemake --cores 10 --config run_name=epsilon_10 molecule=ethanol


## Butane
# Run baseline
# snakemake --cores 10 --config run_name=baseline molecule=butane

# Run with control
# snakemake --cores 10 --config run_name=control molecule=butane
# snakemake --cores 10 --config run_name=epsilon_10 molecule=butane




# snakemake --cores 10 --config run_name=baseline molecule=hexane
# snakemake --cores 10 --config run_name=epsilon_10 molecule=hexane



## Methane
# Run baseline
# snakemake --cores 10 --config run_name=baseline molecule=methane





# Run with control
# snakemake --cores 10 --config run_name=control molecule=methane
# snakemake --cores 10 --config run_name=control_pred_window_50 molecule=methane prediction_window_size=50
# snakemake --cores 10 --config run_name=control_pred_window_100 molecule=methane prediction_window_size=100
# snakemake --cores 10 --config run_name=control_pred_window_200 molecule=methane prediction_window_size=200


# snakemake --cores 10 --config run_name=control_adaptive_threshold molecule=methane adaptive_threshold=True
# snakemake --cores 10 --config run_name=control_adaptive_threshold molecule=ethanol adaptive_threshold=True
# snakemake --cores 10 --config run_name=control_adaptive_threshold molecule=butane adaptive_threshold=True

# snakemake --cores 10 --config run_name=baseline_5000 run_type=baseline molecule=ethanol
# snakemake --cores 10 --config run_name=baseline_5000 run_type=baseline molecule=butane
# snakemake --cores 10 --config run_name=baseline_5000 run_type=baseline molecule=hexane
# snakemake --cores 10 --config run_name=control_5000 run_type=normal molecule=ethanol prediction_window_size=300 max_surrogate_training_size=300
# snakemake --cores 10 --config run_name=control_5000 run_type=normal molecule=butane prediction_window_size=300 max_surrogate_training_size=300
# snakemake --cores 10 --config run_name=control_5000 run_type=normal molecule=hexane prediction_window_size=300 max_surrogate_training_size=300

python example_simulations/single_molecule_modeling_example/single_molecule_monte_carlo.py 