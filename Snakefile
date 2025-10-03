##### Runs the Methane Monte Carlo Simulation Example #####


# Run Parameters
temps = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
n_trials = 20
run_name = "test_run_2"


# Runs all simulations
rule all:
    input:
        expand("example_simulations/methane_modeling_example/output/runs/{run_name}/{run_name}_temp{temp}_trial{trial}/results.json",
               run_name=run_name,
               temp = temps,
               trial=(range(1, n_trials+1))
              )
            

# Runs monte carlo at a specific temperature
rule monte_carlo:
    output:
        json="example_simulations/methane_modeling_example/output/runs/{run_name}/{run_name}_temp{temp}_trial{trial}/results.json"
    params:
        scratch="example_simulations/methane_modeling_example/output/tmp/{run_name}_T{temp}_trial{trial}",
        run_name="{run_name}",
        temp="{temp}",
        trial="{trial}"
    threads: 1
    shell:
        """
        mkdir -p {params.scratch}
        PSI_SCRATCH={params.scratch} \
        python example_simulations/methane_modeling_example/methane_monte_carlo.py \
            --run-name {params.run_name} \
            --trial {params.trial} \
            --temp {params.temp}
        rmdir {params.scratch}
        """
