##### Runs a Monte Carlo Simulation Example #####


# Run Parameters
temps = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
n_trials = 20
run_name = config["run_name"]
molecule_name = config["molecule"]
if "adaptive_threshold" in config:
    adaptive_threshold = config["adaptive_threshold"] == "True"
else:
    adaptive_threshold = False

if "retrain_interval" in config:
    retrain_interval = int(config["retrain_interval"])
else:
    retrain_interval = None

if "prediction_window_size" in config: 
    prediction_window_size = int(config["prediction_window_size"])
else:
    prediction_window_size = 0

if "max_surrogate_training_size" in config:
    max_surrogate_training_size = int(config["max_surrogate_training_size"])
else:
    max_surrogate_training_size = None



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
        trial="{trial}",
        molecule_name="{molecule_name}",
        temp="{temp}",
        retrain_interval="{retrain_interval}",
        prediction_window_size="{prediction_window_size}",
        max_surrogate_training_size="{max_surrogate_training_size}",
        adaptive_threshold="{adaptive_threshold}",
        acceptable_error = 0 if run_name == "baseline" else 0.002
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
            --retrain-interval {params.retrain_interval} \
            --prediction-window-size {params.prediction_window_size} \
            --max-surrogate-training-size {params.max_surrogate_training_size} \
            --adaptive_threshold {params.adaptive_threshold}
        rmdir {params.scratch}
        """
