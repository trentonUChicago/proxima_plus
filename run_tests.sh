






# ## Butane
# # Run baseline
# snakemake --cores 10 --config run_name=baseline molecule=butane

# # Run with control
# snakemake --cores 10 --config run_name=control molecule=butane


## Methane
# Run baseline
snakemake --cores 10 --config run_name=baseline molecule=methane

# Run with control
snakemake --cores 10 --config run_name=control molecule=methane
snakemake --cores 10 --config run_name=control_pred_window_50 molecule=methane prediction_window_size=50
snakemake --cores 10 --config run_name=control_pred_window_100 molecule=methane prediction_window_size=100
snakemake --cores 10 --config run_name=control_pred_window_200 molecule=methane prediction_window_size=200


snakemake --cores 10 --config run_name=control_adaptive_threshold molecule=methane adaptive_threshold=True

# ## Ethanol
# # Run baseline
# snakemake --cores 10 --config run_name=baseline molecule=ethanol

# # Run with control
# snakemake --cores 10 --config run_name=control molecule=ethanol
