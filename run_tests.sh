






## Butane
# Run baseline
snakemake --cores 10 --config run_name=baseline molecule=butane

# Run with control
snakemake --cores 10 --config run_name=control molecule=butane


## Methane
# Run baseline
snakemake --cores 10 --config run_name=baseline molecule=methane

# Run with control
snakemake --cores 10 --config run_name=control molecule=methane


## Ethanol
# Run baseline
snakemake --cores 10 --config run_name=baseline molecule=ethanol

# Run with control
snakemake --cores 10 --config run_name=control molecule=ethanol
