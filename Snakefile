##### Runs a Monte Carlo Simulation Example #####


# Run Parameters
temps = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
n_trials = 20
run_name = config["run_name"]
run_type = config["run_type"]
molecule_name = config["molecule"]
if "adaptive_threshold" in config:
    adaptive_threshold = True
else:
    adaptive_threshold = False


if "epsilon" in config:
    epsilon = float(config["epsilon"])
else:
    epsilon = 0.1


if "retrain_interval" in config:
    retrain_interval = int(config["retrain_interval"])
else:
    retrain_interval = False

if "prediction_window_size" in config: 
    prediction_window_size = int(config["prediction_window_size"])
else:
    prediction_window_size = 0

if "max_surrogate_training_size" in config:
    max_surrogate_training_size = int(config["max_surrogate_training_size"])
else:
    max_surrogate_training_size = False


if "rotation_prob" in config:
    rotation_prob = config["rotation_prob"]
else:
    rotation_prob = 0



# Runs all simulations
rule all:
    input:
        expand("example_simulations/single_molecule_modeling_example/output/runs/{run_name}_{molecule_name}/{run_name}_{molecule_name}_temp{temp}_trial{trial}/results.json",
               run_name=run_name,
               molecule_name=molecule_name,
               temp=temps,
               trial=(range(1, n_trials+1))
              )
            
# Runs monte carlo at a specific temperature
rule monte_carlo:
    output:
        json="example_simulations/single_molecule_modeling_example/output/runs/{run_name}_{molecule_name}/{run_name}_{molecule_name}_temp{temp}_trial{trial}/results.json"
    params:
        scratch="example_simulations/single_molecule_modeling_example/output/tmp/{run_name}_T{temp}_trial{trial}",
        run_name="{run_name}",
        molecule_name="{molecule_name}",
        temp="{temp}",
        trial="{trial}",
        retrain_interval=lambda wildcards: f"--retrain-interval {retrain_interval}" if retrain_interval else "",
        prediction_window_size=prediction_window_size,
        max_surrogate_training_size=lambda wildcards: f"--max-surrogate-training-size {max_surrogate_training_size}" if max_surrogate_training_size else "",
        adaptive_threshold=lambda wildcards: "--adaptive-threshold" if adaptive_threshold else "",
        acceptable_error = 0 if run_type == "baseline" else 0.002,
        epsilon = epsilon
        rotation_prob = rotation_prob
    threads: 1
    shell:
        """
        mkdir -p {params.scratch}
        PSI_SCRATCH={params.scratch} \
        python example_simulations/single_molecule_modeling_example/single_molecule_monte_carlo.py \
            --run-name {params.run_name} \
            --trial {params.trial} \
            --molecule {params.molecule_name} \
            --temp {params.temp} \
            --acceptable-error {params.acceptable_error} \
            --epsilon {params.epsilon} \
            --prediction-window-size {params.prediction_window_size} \
            --rotation-prob {params.rotation_prob} \
            {params.adaptive_threshold} \
            {params.retrain_interval} \
            {params.max_surrogate_training_size}
        rmdir {params.scratch}
        """
