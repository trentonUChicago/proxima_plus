# Snakefile

import os

# ---- Global config ----
base_run_name = config.get("run_name", "testing")
molecules = config.get("molecules", ["methane", "ethanol", "butane", "hexane"])
control_modes = config.get("control_modes", ["original", "original_audit", "tm", "tm_audit"])
temps     = config.get("temps", [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
n_trials  = int(config.get("n_trials", 20))
threshold_types = config.get(
    "threshold_types",
    ["error_pred_fixed_threshold", "proxima", "epistemic_aleatory"]
)

epsilon = float(config.get("epsilon", 0.1))
prediction_window_size = int(config.get("prediction_window_size", 0))
max_surrogate_training_size = config.get("max_surrogate_training_size", None)
retrain_interval = config.get("retrain_interval", None)
adaptive_acceptable_error = bool(config.get("adaptive_acceptable_error", False))
fidelity = config.get("fidelity", "medium")

# Per-molecule steps and rotation prob.
# You can either give a global default or a per-molecule dict in config.yaml.
def steps_for(mol):
    steps_cfg = config.get("nsteps", {})
    default_nsteps = int(config.get("default_nsteps", 5000))
    return int(steps_cfg.get(mol, default_nsteps))

def rotation_prob_for(mol):
    rot_cfg = config.get("rotation_prob", {})
    default_rot = float(config.get("default_rotation_prob", 0.3))
    return float(rot_cfg.get(mol, default_rot))

# Acceptable error can be a single number, or a dict keyed by threshold_type
def acceptable_error_for(mol, threshold):
    """
    Flexible lookup for acceptable_error.

    Supported config patterns:

    1) Single scalar (same everywhere):
       acceptable_error: 0.002

    2) Per-molecule:
       acceptable_error:
         methane: 0.002
         ethanol: 0.002
         butane: 0.01
         hexane: 0.01

    3) Per-threshold (old behavior):
       acceptable_error:
         error_pred_fixed_threshold: 0.005
         proxima: 0.01
         epistemic_aleatory: 0.01
         default: 0.002

    4) Per-molecule AND per-threshold:
       acceptable_error:
         methane:
           proxima: 0.005
           default: 0.002
         butane:
           proxima: 0.02
           epistemic_aleatory: 0.02
         default:
           proxima: 0.01
           epistemic_aleatory: 0.01
           default: 0.002
    """
    ae_cfg = config.get("acceptable_error", 0.002)

    # Simple scalar
    if isinstance(ae_cfg, (int, float)):
        return float(ae_cfg)

    if not isinstance(ae_cfg, dict):
        return 0.002

    # 1) Check molecule-specific entry
    mol_cfg = ae_cfg.get(mol)

    #   1a) Molecule -> scalar
    if isinstance(mol_cfg, (int, float)):
        return float(mol_cfg)

    #   1b) Molecule -> dict (per-threshold)
    if isinstance(mol_cfg, dict):
        if threshold in mol_cfg and isinstance(mol_cfg[threshold], (int, float)):
            return float(mol_cfg[threshold])
        if "default" in mol_cfg and isinstance(mol_cfg["default"], (int, float)):
            return float(mol_cfg["default"])

    # 2) Fall back to top-level threshold-based dict
    if threshold in ae_cfg and isinstance(ae_cfg[threshold], (int, float)):
        return float(ae_cfg[threshold])

    # 3) Global default at top level
    if "default" in ae_cfg and isinstance(ae_cfg["default"], (int, float)):
        return float(ae_cfg["default"])

    # Final hard-coded fallback
    return 0.002


# String pattern for results (no Python functions here)
results_pattern = (
    f"example_simulations/single_molecule_modeling_example/output/runs/{base_run_name}" +
    "_{threshold}_{control_mode}_{molecule}/" +
    f"{base_run_name}" +
    "_{threshold}_{control_mode}_{molecule}_temp{temp}_trial{trial}/results.json"
)

# ---- Targets ----
rule all:
    input:
        expand(
            results_pattern,
            threshold=threshold_types,
            molecule=molecules,
            control_mode=control_modes,
            temp=temps,
            trial=range(1, n_trials + 1),
        )



# ---- Main MC rule ----

rule run_mc:
    """Run a single Monte Carlo trajectory."""
    output:
        results_pattern
    params:
        molecule=lambda wc: wc.molecule,
        temp=lambda wc: wc.temp,
        trial=lambda wc: wc.trial,
        threshold=lambda wc: wc.threshold,
        run_name=lambda wc: f"{base_run_name}_{wc.threshold}_{wc.control_mode}",
        nsteps=lambda wc: steps_for(wc.molecule),
        rotation_prob=lambda wc: rotation_prob_for(wc.molecule),
        acceptable_error=lambda wc: acceptable_error_for(wc.molecule, wc.threshold),
        epsilon=epsilon,
        prediction_window_size=prediction_window_size,
        max_surrogate_training_size=max_surrogate_training_size,
        retrain_interval=retrain_interval,
        adaptive_flag=(
            "--adaptive-acceptable-error"
            if adaptive_acceptable_error else ""
        ),
        # NEW: pre-format the optional CLI args here
        retrain_arg=lambda wc: (
            f"--retrain-interval {retrain_interval}"
            if retrain_interval is not None else ""
        ),
        max_data_arg=lambda wc: (
            f"--max-surrogate-training-size {max_surrogate_training_size}"
            if max_surrogate_training_size is not None else ""
        ),
        fidelity=fidelity,
        control_mode=lambda wc: wc.control_mode,
    threads: 1
    shell:
        r"""
        python example_simulations/single_molecule_modeling_example/single_molecule_monte_carlo.py \
            --run-name {params.run_name} \
            --trial {params.trial} \
            --molecule {params.molecule} \
            --temp {params.temp} \
            --nsteps {params.nsteps} \
            --acceptable-error {params.acceptable_error} \
            --epsilon {params.epsilon} \
            --prediction-window-size {params.prediction_window_size} \
            --rotation-prob {params.rotation_prob} \
            --threshold-type {params.threshold} \
            {params.adaptive_flag} \
            {params.retrain_arg} \
            {params.max_data_arg} \
            --fidelity {params.fidelity} \
            --control-mode {params.control_mode}
        """

