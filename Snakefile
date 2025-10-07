##### Runs the Methane Monte Carlo Simulation Example #####


# Run Parameters
temps = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
n_trials = 20
run_name = config["run_name"]
molecule_name = config["molecule"]


# Runs all simulations
rule all:
    input:
        expand("example_simulations/single_molecule_modeling_example/output/runs/{run_name}_{molecule_name}/{run_name}_{molecule_name}_temp{temp}_trial{trial}/results.json",
               run_name=run_name,
               molecule_name=molecule_name,
               temp=temps,
               trial=(range(1, n_trials+1))
              )
            
# TODO: Remove --acceptable-error 0 so that it is no longer running baseline
# Runs monte carlo at a specific temperature
rule monte_carlo:
    output:
        json="example_simulations/single_molecule_modeling_example/output/runs/{run_name}_{molecule_name}/{run_name}_{molecule_name}_temp{temp}_trial{trial}/results.json"
    params:
        scratch="example_simulations/single_molecule_modeling_example/output/tmp/{run_name}_T{temp}_trial{trial}",
        run_name="{run_name}",
        temp="{temp}",
        trial="{trial}",
        acceptable_error = 0 if run_name == "baseline" else 0.002
    threads: 1
    shell:
        """
        mkdir -p {params.scratch}
        PSI_SCRATCH={params.scratch} \
        python example_simulations/single_molecule_modeling_example/single_molecule_monte_carlo.py \
            --run-name {params.run_name} \
            --trial {params.trial} \
            --temp {params.temp} \
            --acceptable-error {params.acceptable_error}
        rmdir {params.scratch}
        """
