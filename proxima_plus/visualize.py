import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress
import os, re, json

# Get path of this script
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

# Set presentation style for graphs
def set_presentation_style():
    """Set a clean, consistent style for paper / talk figures."""
    mpl.rcParams.update({
        "figure.figsize": (8, 6),
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
    })

DEFAULT_COLORS = {
    "Baseline": "#1f77b4",                            # blue
    "Proxima": "#ff7f0e",                             # orange
    "Distance Uncertainty": "#ff7f0e",                # orange
    "Epistemic Uncertainty": "#2ca02c",               # green
    "Epistemic and Aleatory Uncertainty": "#d62728",  # red
    "DELTA": "#9467bd",                               # purple
}




def get_rog_data_from_dir(directory_path, outlier_threshold=None):
    file_names = os.listdir(directory_path)

    rog_data = {}
    num_outliers_removed = 0

    for file_name in file_names:
        match = re.search(r'_temp(\d+)', file_name)
        if not match:
            continue
        temp = int(match.group(1))

        try:
            # ----- My runs -----
            json_filename = os.path.join(directory_path, file_name, "results.json")
            with open(json_filename, "r") as f:
                data = json.load(f)

            rog = data["r_g_stats"]["statistic"]
            run_time = data["time"]

            csv_filename = os.path.join(directory_path, file_name, "results.csv")
            df = pd.read_csv(csv_filename)

            num_target_run = (df["surrogate_accepts"] == False).sum()
            n_steps = len(df)
            target_fraction = num_target_run / n_steps
            surrogate_fraction = 1.0 - target_fraction

        except Exception:
            # ----- Proxima or other formats (fallback) -----
            try:
                result_json_filename = os.path.join(directory_path, file_name, "result.json")
                stats_json_filename = os.path.join(directory_path, file_name, "lfa_stats.json")
                rog_json_filename = os.path.join(directory_path, file_name, "r_g.json")

                with open(result_json_filename, "r") as f:
                    data = json.load(f)
                with open(stats_json_filename, "r") as f:
                    stats_data = json.load(f)
                with open(rog_json_filename, "r") as f:
                    rog_history_data = json.load(f)

                rog = float(np.mean(rog_history_data[-10:]))
                run_time = (
                    stats_data["lfa_time"]
                    + stats_data["uq_time"]
                    + stats_data["train_time"]
                    + stats_data["target_time"]
                )
                num_target_run = stats_data["target_runs"]
                # If you know n_steps here, set it; otherwise skip fractions
                target_fraction = None
                surrogate_fraction = None
            except Exception:
                continue

        if outlier_threshold is not None and (
            rog <= outlier_threshold[0] or rog >= outlier_threshold[1]
        ):
            print("Outlier Removed. File name:", file_name)
            num_outliers_removed += 1
            continue

        bucket = rog_data.setdefault(
            temp,
            {
                "rog": [],
                "num_target_run": [],
                "time": [],
                "target_fraction": [],
                "surrogate_fraction": [],
            },
        )
        bucket["rog"].append(rog)
        bucket["num_target_run"].append(num_target_run)
        bucket["time"].append(run_time)
        if target_fraction is not None:
            bucket["target_fraction"].append(target_fraction)
            bucket["surrogate_fraction"].append(surrogate_fraction)

    avg_rog_data = {
        "rog_means": [],
        "rog_sems": [],
        "num_target_run_means": [],
        "num_target_run_sems": [],
        "time_means": [],
        "time_sems": [],
        "target_frac_means": [],
        "target_frac_sems": [],
        "surrogate_frac_means": [],
        "surrogate_frac_sems": [],
        "temps": [],
    }

    for temp in sorted(rog_data.keys()):
        bucket = rog_data[temp]
        n = len(bucket["rog"])

        avg_rog_data["temps"].append(temp)

        # ROG statistics
        avg_rog_data["rog_means"].append(np.mean(bucket["rog"]))
        avg_rog_data["rog_sems"].append(np.std(bucket["rog"], ddof=1) / np.sqrt(n))

        # Target counts
        avg_rog_data["num_target_run_means"].append(np.mean(bucket["num_target_run"]))
        avg_rog_data["num_target_run_sems"].append(
            np.std(bucket["num_target_run"], ddof=1) / np.sqrt(n)
        )

        # Run times
        avg_rog_data["time_means"].append(np.mean(bucket["time"]))
        avg_rog_data["time_sems"].append(
            np.std(bucket["time"], ddof=1) / np.sqrt(n)
        )

        # Fractions (only if we have them)
        if bucket["target_fraction"]:
            avg_rog_data["target_frac_means"].append(
                np.mean(bucket["target_fraction"])
            )
            avg_rog_data["target_frac_sems"].append(
                np.std(bucket["target_fraction"], ddof=1) / np.sqrt(len(bucket["target_fraction"]))
            )
            avg_rog_data["surrogate_frac_means"].append(
                np.mean(bucket["surrogate_fraction"])
            )
            avg_rog_data["surrogate_frac_sems"].append(
                np.std(bucket["surrogate_fraction"], ddof=1) / np.sqrt(len(bucket["surrogate_fraction"]))
            )
        else:
            avg_rog_data["target_frac_means"].append(np.nan)
            avg_rog_data["target_frac_sems"].append(np.nan)
            avg_rog_data["surrogate_frac_means"].append(np.nan)
            avg_rog_data["surrogate_frac_sems"].append(np.nan)

    print("Number of outliers removed:", num_outliers_removed)
    return avg_rog_data


def compute_summary_metrics(run_data, baseline_name):
    """
    Compute summary numbers vs baseline for each run type:

    - rmse_rog:      RMSE of mean ROG vs baseline
    - mean_abs_rog_diff
    - mean_rel_rog_diff_percent
    - avg_speedup:   average baseline_time / method_time
    - median_speedup
    - avg_target_fraction / avg_surrogate_fraction
    """
    metrics = {}

    base = run_data[baseline_name]
    base_temps = np.array(base["temps"])
    base_rog = np.array(base["rog_means"])
    base_time = np.array(base["time_means"])

    for name, data in run_data.items():
        if name == baseline_name:
            continue

        temps = np.array(data["temps"])
        # Align by temperature (just in case)
        common, idx_b, idx_m = np.intersect1d(base_temps, temps, return_indices=True)
        if len(common) == 0:
            continue

        rog_m = np.array(data["rog_means"])[idx_m]
        rog_b = base_rog[idx_b]
        time_m = np.array(data["time_means"])[idx_m]
        time_b = base_time[idx_b]

        # Accuracy metrics
        rog_diff = rog_m - rog_b
        abs_diff = np.abs(rog_diff)
        rel_diff = abs_diff / rog_b

        rmse = float(np.sqrt(np.mean(rog_diff**2)))
        mean_abs = float(np.mean(abs_diff))
        mean_rel_pct = float(np.mean(rel_diff) * 100.0)

        # Speedup metrics
        speedup = time_b / time_m
        avg_speedup = float(np.mean(speedup))
        median_speedup = float(np.median(speedup))

        # Usage metrics
        target_frac = np.array(data["target_frac_means"])[idx_m]
        surrogate_frac = np.array(data["surrogate_frac_means"])[idx_m]

        avg_target_frac = float(np.nanmean(target_frac))
        avg_surrogate_frac = float(np.nanmean(surrogate_frac))

        metrics[name] = {
            "rmse_rog": rmse,
            "mean_abs_rog_diff": mean_abs,
            "mean_rel_rog_diff_percent": mean_rel_pct,
            "avg_speedup": avg_speedup,
            "median_speedup": median_speedup,
            "avg_target_fraction": avg_target_frac,
            "avg_surrogate_fraction": avg_surrogate_frac,
        }

    return metrics



def create_error_plot(pred_error, errors, save_path):
    plt.figure(figsize=(8,6))
    # Calculate slope of best fit line
    slope, intercept, r, p, se = linregress(pred_error, errors)
    # Add data points
    plt.plot(pred_error, errors, 'o', label='original data')
    # Add best fit line
    plt.plot(pred_error, intercept + slope*np.array(pred_error), 'r', label='fitted line')
    # Add legend
    plt.legend()
    # Save plot
    plt.savefig(save_path)
    plt.close()



def create_rog_over_time_plot(rog_values):
    x = np.arange(1, len(rog_values) + 1)

    plt.plot(x, rog_values, marker='o', linestyle='-', color='blue')
    plt.title('Radius of Gyration values over time')
    plt.xlabel('Timestep')
    plt.ylabel('Radius of Gyration (ROG)')
    plt.grid(True)
    plt.show()

def create_rog_over_temp_plot(run_data, y_lim=None, title="Final ROG", colors=None, show=True):
    """
    Plot final ROG vs temperature for multiple run types.

    colors: dict mapping run_name -> color (e.g., DEFAULT_COLORS)
    show:   whether to call plt.show() (set False when saving in batch)
    """
    set_presentation_style()
    fig, ax = plt.subplots()

    for run_name, data in run_data.items():
        temps = np.array(data["temps"])
        means = np.array(data["rog_means"])
        sems = np.array(data["rog_sems"])

        color = colors.get(run_name) if colors is not None else None

        ax.errorbar(
            temps,
            means,
            yerr=sems,
            fmt="-o",
            capsize=4,
            label=run_name,
            color=color,
        )

    if y_lim is not None:
        ax.set_ylim(*y_lim)

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Radius of Gyration (ROG)")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()

    if show:
        plt.show()
    return fig, ax


def create_bar_plot(run_data, data_type='num_target_run'):
    fig, ax = plt.subplots(figsize=(8,6))

    # Figure out size of bars based on number of bars to show
    max_width = 90
    width = max_width / len(run_data)
    offset = -max_width/2 + width/2

    for run_name in run_data:
        means = run_data[run_name][f'{data_type}_means']
        sems = run_data[run_name][f'{data_type}_sems']
        temps = np.array(run_data[run_name]['temps'])

        ax.bar(temps + offset, means, width, yerr=sems, label=run_name)

        offset += width


    ax.set_ylabel(data_type)
    # ax.set_ylim(-1081,-1081.04)
    ax.set_xlabel("Temperature (K)")
    ax.set_xticks(temps)
    ax.set_xticklabels(temps)
    ax.legend()

    plt.show()


def create_usage_fraction_plot(run_data, use_surrogate=True, title=None, colors=None, show=True):
    """
    Plot fraction of surrogate or target calls vs temperature (as percentages).

    use_surrogate = True  -> plot surrogate usage (%)
                   False -> plot target usage (%)
    colors: dict mapping run_name -> color
    """
    set_presentation_style()
    fig, ax = plt.subplots()

    ylabel = "Surrogate usage (%)" if use_surrogate else "Target usage (%)"
    if title is None:
        title = ylabel + " vs Temperature"

    for run_name, data in run_data.items():
        temps = np.array(data["temps"])

        if use_surrogate:
            means = np.array(data["surrogate_frac_means"]) * 100.0
            sems = np.array(data["surrogate_frac_sems"]) * 100.0
        else:
            means = np.array(data["target_frac_means"]) * 100.0
            sems = np.array(data["target_frac_sems"]) * 100.0

        color = colors.get(run_name) if colors is not None else None

        ax.errorbar(
            temps,
            means,
            yerr=sems,
            fmt="-o",
            capsize=4,
            label=run_name,
            color=color,
        )

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 100.0)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()

    if show:
        plt.show()
    return fig, ax


def create_target_usage_bar_plot(run_data, title="Target usage (%)", colors=None, show=True):
    """
    Bar plot of target usage percentage vs temperature, grouped by run type.
    Similar layout to create_bar_plot, but uses target_frac_* and percentages.
    """
    set_presentation_style()
    fig, ax = plt.subplots()

    # Use temps from first run as x positions
    first_key = next(iter(run_data))
    temps = np.array(run_data[first_key]["temps"])

    max_width = 90.0  # total width reserved around each temp
    n_series = len(run_data)
    width = max_width / n_series
    offset = -max_width / 2.0 + width / 2.0

    for run_name, data in run_data.items():
        means = np.array(data["target_frac_means"]) * 100.0
        sems = np.array(data["target_frac_sems"]) * 100.0

        color = colors.get(run_name) if colors is not None else None

        ax.bar(
            temps + offset,
            means,
            width,
            yerr=sems,
            label=run_name,
            color=color,
        )
        offset += width

    ax.set_ylabel("Target usage (%)")
    ax.set_xlabel("Temperature (K)")
    ax.set_xticks(temps)
    ax.set_xticklabels(temps)
    ax.set_ylim(0.0, 100.0)
    ax.set_title(title)
    ax.legend(frameon=False)

    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_target_usage_bar_plot(directory, run_names, outlier_threshold=None, colors=None, show=True):
    """
    Wrapper that builds run_data and calls create_target_usage_bar_plot.

    run_names: list of (directory_type, file_name, run_label)
    """
    run_data = {}
    for directory_type, file_name, run_label in run_names:
        path = os.path.join(directory[directory_type], file_name)
        run_data[run_label] = get_rog_data_from_dir(path, outlier_threshold=outlier_threshold)

    return create_target_usage_bar_plot(
        run_data,
        title="Target usage (%) vs Temperature",
        colors=colors,
        show=show,
    )


def create_speedup_plot(
    run_data,
    baseline_name,
    methods=None,
    title="Speedup vs baseline",
    colors=None,
    show=True,
):
    """
    Plot baseline_time / method_time vs T.

    baseline_name: key in run_data to use as reference.
    methods: list of other keys to plot; if None, use all except baseline.
    colors: dict mapping run_name -> color
    """
    set_presentation_style()
    fig, ax = plt.subplots()

    temps = np.array(run_data[baseline_name]["temps"])
    base_times = np.array(run_data[baseline_name]["time_means"])

    if methods is None:
        methods = [k for k in run_data.keys() if k != baseline_name]

    for run_name in methods:
        times = np.array(run_data[run_name]["time_means"])
        speedup = base_times / times  # >1 means faster than baseline

        color = colors.get(run_name) if colors is not None else None

        ax.plot(temps, speedup, "-o", label=run_name, color=color)

    # Draw baseline as a horizontal line
    baseline_color = (
        colors.get(baseline_name) if colors is not None else "gray"
    )
    ax.axhline(1.0, color=baseline_color, linestyle="--", linewidth=1)

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Speedup factor (baseline / method)")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()

    if show:
        plt.show()
    return fig, ax


def plot_speedup_plot(directory, run_names, baseline_name, outlier_threshold=None, colors=None):
    run_data = {}
    for directory_type, file_name, run_label in run_names:
        path = directory[directory_type] + "/" + file_name
        run_data[run_label] = get_rog_data_from_dir(path, outlier_threshold=outlier_threshold)

    create_speedup_plot(run_data, baseline_name, colors=colors)





def plot_fraction_plot(directory, run_names, outlier_threshold=None):
    run_data = {}
    for directory_type, file_name, run_name in run_names:
        run_data[run_name] = get_rog_data_from_dir(directory[directory_type] + "/" + file_name, outlier_threshold=outlier_threshold)

    create_usage_fraction_plot(run_data)


def plot_rog(directory, run_names, y_lim = (.272, .28), outlier_threshold=None):
    run_data = {}
    for directory_type, file_name, run_name in run_names:
        run_data[run_name] = get_rog_data_from_dir(directory[directory_type] + "/" + file_name, outlier_threshold=outlier_threshold)
    
    create_rog_over_temp_plot(run_data, y_lim = y_lim)


def plot_bar_plot(directory, run_names, data_type='num_target_run'):
    run_data = {}
    for directory_type, file_name, run_name in run_names:
        run_data[run_name] = get_rog_data_from_dir(directory[directory_type] + "/" + file_name)
    
    create_bar_plot(run_data, data_type)




def generate_all_plots(molecule, directory, run_names, baseline_label, outdir=None, y_lim=None, outlier_threshold=None, colors=None, show=False):
    """
    Generate ROG, usage, and speedup plots for a molecule and save them.

    run_names: list of (directory_type, file_name, run_label)
               run_label is what appears in legends and in DEFAULT_COLORS.
    baseline_label: run_label for the baseline (e.g. "Baseline").
    outdir: folder to save figures; defaults to <script_dir>/figures/<molecule>
    colors: dict mapping run_label -> color (if None, uses DEFAULT_COLORS)
    """
    if colors is None:
        colors = DEFAULT_COLORS

    if outdir is None:
        outdir = os.path.join(script_dir, "figures", molecule)
    os.makedirs(outdir, exist_ok=True)

    # 1) Load all run data
    run_data = {}
    for directory_type, file_name, run_label in run_names:
        path = os.path.join(directory[directory_type], file_name)
        run_data[run_label] = get_rog_data_from_dir(path, outlier_threshold=outlier_threshold)

    # 2) ROG vs T
    fig, ax = create_rog_over_temp_plot(
        run_data,
        y_lim=y_lim,
        title=f"{molecule.capitalize()} – Final ROG",
        colors=colors,
        show=show,
    )
    fig.savefig(
        os.path.join(outdir, f"{molecule}_rog_vs_temp.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 3) Surrogate usage vs T (percent)
    fig, ax = create_usage_fraction_plot(
        run_data,
        use_surrogate=True,
        title=f"{molecule.capitalize()} – Surrogate usage",
        colors=colors,
        show=show,
    )
    fig.savefig(
        os.path.join(outdir, f"{molecule}_surrogate_usage.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 4) Speedup vs baseline
    fig, ax = create_speedup_plot(
        run_data,
        baseline_name=baseline_label,
        methods=None,
        title=f"{molecule.capitalize()} – Speedup vs baseline",
        colors=colors,
        show=show,
    )
    fig.savefig(
        os.path.join(outdir, f"{molecule}_speedup_vs_baseline.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 5) Compute and save summary metrics
    metrics = compute_summary_metrics(run_data, baseline_label)
    metrics_path = os.path.join(outdir, f"{molecule}_summary_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return run_data, metrics



def compute_uncertainty_r2_for_directory(directory_path, min_points=20):
    """
    For every run folder under directory_path that has a results.csv,
    compute R^2 between surrogate_error and various uncertainty signals.

    Returns: dict[label] -> list of R^2 over runs.
    """
    metrics = {
        "Distance": [],
        "Epistemic": [],
        "Aleatory": [],
        "Log(Epistemic)": [],
        "Log(Aleatory)": [],
        "Epistemic+Aleatory": [],
        "Log(Epistemic+Aleatory)": [],
    }

    for run_name in os.listdir(directory_path):
        csv_path = os.path.join(directory_path, run_name, "results.csv")
        if not os.path.isfile(csv_path):
            continue

        df = pd.read_csv(csv_path)
        if "surrogate_error" not in df.columns:
            continue

        num_valid_values = np.count_nonzero(~np.isnan(df["surrogate_error"]))
        if num_valid_values < 100:
            continue

        err = df["surrogate_error"].values
        # Only rows where we have an actual surrogate error
        mask = np.isfinite(err)

        def add_metric(label, x):
            x = np.asarray(x, float)
            m = mask & np.isfinite(x)
            if np.sum(m) < min_points:
                return
            slope, intercept, r, p, se = linregress(x[m], err[m])
            metrics[label].append(r**2)

        # Distance
        if "distance" in df.columns:
            add_metric("Distance", df["distance"].values)

        # Epistemic / Aleatory and logs
        if "epistemic_uncertainty" in df.columns:
            e = df["epistemic_uncertainty"].values
            add_metric("Epistemic", e)
            add_metric("Log(Epistemic)", np.log(np.clip(e, 1e-12, None)))
        else:
            e = None

        if "aleatory_uncertainty" in df.columns:
            a = df["aleatory_uncertainty"].values
            add_metric("Aleatory", a)
            add_metric("Log(Aleatory)", np.log(np.clip(a, 1e-12, None)))
        else:
            a = None

        if e is not None and a is not None:
            s = e + a
            add_metric("Epistemic+Aleatory", s)
            add_metric("Log(Epistemic+Aleatory)", np.log(np.clip(s, 1e-12, None)))

    return metrics


def plot_uncertainty_r2_violin(r2_dict, title="Correlation between uncertainty and surrogate error", save_path=None, show=True):
    """
    Make a violin plot of R^2 values across runs for each uncertainty signal.
    """
    set_presentation_style()

    labels = [label for label, vals in r2_dict.items() if len(vals) > 0]
    data = [r2_dict[label] for label in labels]

    fig, ax = plt.subplots()
    vp = ax.violinplot(data, showmeans=True, showextrema=False)

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(r"$R^2$ of linear fit")
    ax.set_title(title)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax

def _find_best_threshold(x, y):
    """
    Given continuous x and boolean labels y (True = high error),
    scan possible thresholds and return the one that maximizes F1 score.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, bool)
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    best_t = None
    best_f1 = -np.inf
    best_counts = (0, 0, 0, 0)  # TP, FP, TN, FN

    # Candidate thresholds: midpoints between unique values
    unique_vals = np.unique(x_sorted)
    if len(unique_vals) == 1:
        # degenerate case
        t = unique_vals[0]
        y_pred = x >= t
        TP = np.sum(y & y_pred)
        FP = np.sum(~y & y_pred)
        TN = np.sum(~y & ~y_pred)
        FN = np.sum(y & ~y_pred)
        return t, (TP, FP, TN, FN)

    thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

    for t in thresholds:
        y_pred = x >= t
        TP = np.sum(y & y_pred)
        FP = np.sum(~y & y_pred)
        TN = np.sum(~y & ~y_pred)
        FN = np.sum(y & ~y_pred)
        if TP + FP + FN == 0:
            continue
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
            best_counts = (TP, FP, TN, FN)

    return best_t, best_counts


def compute_confusion_for_metric_directory(directory_path, metric_label, metric_func, error_bound, min_points=20):
    """
    Aggregate confusion matrix for a single uncertainty metric across all runs in a directory.

    metric_label: name for reporting / plotting
    metric_func:  function(df) -> numpy array of metric values
    error_bound:  scalar surrogate_error threshold for "high error"
    """
    TP_tot = FP_tot = TN_tot = FN_tot = 0

    for run_name in os.listdir(directory_path):
        csv_path = os.path.join(directory_path, run_name, "results.csv")
        if not os.path.isfile(csv_path):
            continue

        df = pd.read_csv(csv_path)
        if "surrogate_error" not in df.columns:
            continue

        err = df["surrogate_error"].values
        metric = metric_func(df)

        mask = np.isfinite(err) & np.isfinite(metric)
        if np.sum(mask) < min_points:
            continue

        e = err[mask]
        m = metric[mask]
        y = e > error_bound

        t, (TP, FP, TN, FN) = _find_best_threshold(m, y)
        TP_tot += TP
        FP_tot += FP
        TN_tot += TN
        FN_tot += FN

    return metric_label, np.array([[TP_tot, FP_tot], [FN_tot, TN_tot]])

def build_uncertainty_metric_functions():
    return {
        "Distance": lambda df: df["distance"].values,
        "Epistemic": lambda df: df["epistemic_uncertainty"].values,
        "Aleatory": lambda df: df["aleatory_uncertainty"].values,
        "Log(Epistemic)": lambda df: np.log(np.clip(df["epistemic_uncertainty"].values, 1e-12, None)),
        "Log(Aleatory)": lambda df: np.log(np.clip(df["aleatory_uncertainty"].values, 1e-12, None)),
        "Epistemic+Aleatory": lambda df: (df["epistemic_uncertainty"].values + df["aleatory_uncertainty"].values),
        "Log(Epistemic+Aleatory)": lambda df: np.log(
            np.clip(
                df["epistemic_uncertainty"].values + df["aleatory_uncertainty"].values,
                1e-12,
                None,
            )
        ),
    }

def plot_confusion_matrices_for_uncertainties(
    directory_path,
    error_bound,
    save_dir=None,
    show=True,
):
    """
    For each uncertainty metric, compute an aggregated confusion matrix and plot it.
    Produces one figure per metric (2x2 heatmap with counts).
    """
    set_presentation_style()
    metric_funcs = build_uncertainty_metric_functions()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    figures = {}

    for label, func in metric_funcs.items():
        metric_label, cm = compute_confusion_for_metric_directory(
            directory_path, label, func, error_bound
        )

        # Skip if no data
        if cm.sum() == 0:
            continue

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred High", "Pred Low"])
        ax.set_yticklabels(["True High", "True Low"])
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Ground truth")
        ax.set_title(f"{metric_label} – confusion matrix\n(error_bound = {error_bound})")

        # Annotate counts
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    int(cm[i, j]),
                    ha="center",
                    va="center",
                    color="black",
                )

        fig.tight_layout()

        if save_dir is not None:
            fig.savefig(
                os.path.join(save_dir, f"confusion_{metric_label.replace(' ', '_')}.png"),
                dpi=300,
                bbox_inches="tight",
            )
        if show:
            plt.show()
        else:
            plt.close(fig)

        figures[label] = (fig, ax)

    return figures



def plot_error_vs_speedup(
    run_data,
    baseline_label,
    methods,
    title="Error vs speedup",
    colors=None,
    annotate=True,
    show=True,
    save_path=None,
):
    """
    Scatter plot of mean_abs_rog_diff vs avg_speedup for selected methods.

    run_data:     dict[label] -> aggregated stats (from get_rog_data_from_dir)
    baseline_label: label of baseline run
    methods:      list of labels to include in the scatter
    """
    if colors is None:
        colors = DEFAULT_COLORS

    metrics = compute_summary_metrics(run_data, baseline_label)

    set_presentation_style()
    fig, ax = plt.subplots()

    for label in methods:
        if label not in metrics:
            continue
        m = metrics[label]
        err = m["mean_abs_rog_diff"]
        speed = m["avg_speedup"]

        color = colors.get(label) if colors is not None else None
        ax.scatter(speed, err, label=label, color=color)

        if annotate:
            ax.annotate(
                label,
                (speed, err),
                textcoords="offset points",
                xytext=(5, 5),
                ha="left",
            )

    # Baseline reference: speedup=1, error=0
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)

    ax.set_xlabel("Average speedup (baseline / method)")
    ax.set_ylabel("Mean |ROG(method) - ROG(baseline)|")
    ax.set_title(title)
    # ax.legend(frameon=False)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax, metrics


def plot_delta_adaptive_tradeoff(
    molecule,
    directory,
    baseline_tuple,
    adaptive_nums,
    reference_tuples=None,
    outdir=None,
    outlier_threshold=None,
    colors=None,
    show=False,
):
    """
    Build and plot error-vs-speedup tradeoff for DELTA adaptive thresholds.

    baseline_tuple: (dir_key, file_name, label) for baseline.
    adaptive_nums:  list like [1, 2, 3, 4] corresponding to the NUM in
                    final_adaptive_[NUM]_error_pred_fixed_threshold_{molecule}
    reference_tuples: optional list of (dir_key, file_name, label) for Proxima etc.
    """
    if colors is None:
        colors = DEFAULT_COLORS

    if outdir is None:
        outdir = os.path.join(script_dir, "figures", molecule, "adaptive")
    os.makedirs(outdir, exist_ok=True)

    # Build run_names
    run_names = [baseline_tuple]

    # Adaptive DELTA runs
    for num in adaptive_nums:
        file_name = f"final_adaptive_{num}_error_pred_fixed_threshold_{molecule}"
        label = f"DELTA (leniency={num})"
        run_names.append(("fin", file_name, label))

    # Reference methods (Proxima, Epistemic, etc.)
    if reference_tuples is not None:
        run_names.extend(reference_tuples)

    # Load data
    run_data = {}
    for dir_key, file_name, label in run_names:
        path = os.path.join(directory[dir_key], file_name)
        run_data[label] = get_rog_data_from_dir(path, outlier_threshold=outlier_threshold)

    baseline_label = baseline_tuple[2]
    method_labels = [label for _, _, label in run_names if label != baseline_label]

    save_path = os.path.join(
        outdir,
        f"{molecule}_delta_error_vs_speedup.png",
    )

    fig, ax, metrics = plot_error_vs_speedup(
        run_data,
        baseline_label=baseline_label,
        methods=method_labels,
        title=f"{molecule.capitalize()} – DELTA error vs speedup",
        colors=colors,
        annotate=True,
        show=show,
        save_path=save_path,
    )

    # Save metrics for later (so you can quote them on slides)
    metrics_path = os.path.join(outdir, f"{molecule}_delta_tradeoff_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return run_data, metrics








if __name__ == "__main__":
    save_path = "/home/trentonjw/Documents/Project/delta_graphs"

    molecule = "hexane"
    directory = {"mine": f"/home/trentonjw/Documents/Project/delta_runs/{molecule}_runs",
             "mine_2": f"/home/trentonjw/Documents/Project/delta_runs/low_{molecule}_runs",
             "proxima": "/home/trentonjw/Documents/Project/temp_proxima/proxima_cc/examples/molecule-sampling/runs",
             "proxima_runs": "/home/trentonjw/Documents/Project/proxima_stuff/proxima_runs",
             "pre_final": "/home/trentonjw/Documents/Project/delta_runs/pre_final_runs",
             "final": "/home/trentonjw/Documents/Project/delta_runs/final_runs",
             "fin": "/home/trentonjw/Documents/masters_presentation/final_runs/runs",
             "new_test_runs_1": "/home/trentonjw/Documents/Project/delta_runs/new_test_runs_1",
             "new_test_runs_2": "/home/trentonjw/Documents/Project/delta_runs/new_test_runs_2",
             "new_test_runs_3": "/home/trentonjw/Documents/Project/delta_runs/new_test_runs_3",
             "new_test_runs_4": "/home/trentonjw/Documents/Project/delta_runs/new_test_runs_4",
             "new_test_runs_5": "/home/trentonjw/Documents/Project/delta_runs/new_test_runs_5",
             "new_test_runs_6": "/home/trentonjw/Documents/Project/delta_runs/new_test_runs_6",
            }

    # run_names = [("mine", f"mid_finals_baseline_{molecule}", "Baseline"),
    #          ("mine", f"mid_finals_epistemic_{molecule}", "Proxima"),
    #          ("mine", f"mid_finals_proxima_{molecule}", "Epistemic Uncertainty"),
    #          ("mine", f"mid_finals_epistemic_aleatory_{molecule}", "Epistemic and Aleatory Uncertainty"),
    #         ]


    run_names = [
            #  ("mine", f"mid_finals_baseline_{molecule}", "Baseline"),
             ("fin", f"final_baseline_{molecule}", "Baseline"),
             ("new_test_runs_1", f"new_changes_proxima_{molecule}", "Distance Uncertainty"),
             ("new_test_runs_2", f"new_changes_epistemic_{molecule}", "Epistemic Uncertainty"),
            ]

    y_lim = {"methane": (.272, .28), "ethanol": (1.125,1.150), "butane": (1.485,1.51), "hexane": (2.075, 2.1)}

    run_data, metrics = generate_all_plots(
        molecule,
        directory,
        run_names,
        baseline_label="Baseline",
        outdir=save_path,
        y_lim=y_lim[molecule],
        # outlier_threshold= (1.485,1.51),
        colors=DEFAULT_COLORS,
        show=False,  # don’t pop windows, just save files
    )

    # proxima_dir ="/home/trentonjw/Documents/Project/delta_runs/methane_runs/mid_finals_proxima_methane"
    # proxima_dir = "/home/trentonjw/Documents/Project/delta_runs/low_methane_runs/low_test_proxima_methane"
    # r2_dict = compute_uncertainty_r2_for_directory(proxima_dir)
    

    # labels = [label for label, vals in r2_dict.items() if len(vals) > 0]
    # data = [r2_dict[label] for label in labels]

    # for label in labels:
    #     print(f"{label} mean:", np.mean(r2_dict[label]))
    #     print(f"{label} median:", np.median(r2_dict[label]))


    # plot_uncertainty_r2_violin(r2_dict, save_path=save_path)

    # error_bound = 0.002
    # plot_confusion_matrices_for_uncertainties(
    #     proxima_dir,
    #     error_bound=error_bound,
    #     save_dir=save_path,
    #     show=False,
    # )

    
#     print(json.dumps(metrics, indent=2))


# if __name__ == "__main__":
#     molecule = "hexane"
#     directory = {
#         "mine": f"/home/trentonjw/Documents/Project/delta_runs/{molecule}_runs",
#         "final": "/home/trentonjw/Documents/Project/delta_runs/final_runs",
#         "fin": "/home/trentonjw/Documents/masters_presentation/final_runs/runs",
#         "new_tests": "/home/trentonjw/Documents/Project/proxima_plus/example_simulations/single_molecule_modeling_example/output/runs"
#     }

#     baseline_tuple = ("fin", f"final_baseline_{molecule}", "Baseline")
#     adaptive_nums = [1, 2, 4, 8, 20]

#     reference_tuples = [
#         ("fin", f"final_epistemic_{molecule}", "Epistemic Uncertainty"),
#     ]

#     run_data, metrics = plot_delta_adaptive_tradeoff(
#         molecule,
#         directory,
#         baseline_tuple=baseline_tuple,
#         adaptive_nums=adaptive_nums,
#         reference_tuples=reference_tuples,
#         outdir="/home/trentonjw/Documents/masters_presentation/auto_graphs",
#         outlier_threshold=None,
#         colors=DEFAULT_COLORS,
#         show=False,
#     )

#     print(json.dumps(metrics, indent=2))










# molecule = "hexane"
# directory = {"mine": f"/home/trentonjw/Documents/Project/delta_runs/{molecule}_runs",
#              "mine_2": f"/home/trentonjw/Documents/Project/delta_runs/low_{molecule}_runs",
#              "proxima": "/home/trentonjw/Documents/Project/temp_proxima/proxima_cc/examples/molecule-sampling/runs",
#              "proxima_runs": "/home/trentonjw/Documents/Project/proxima_stuff/proxima_runs",
#              "pre_final": "/home/trentonjw/Documents/Project/delta_runs/pre_final_runs",
#              "final": "/home/trentonjw/Documents/Project/delta_runs/final_runs",
#              "new_tests": "/home/trentonjw/Documents/Project/proxima_plus/example_simulations/single_molecule_modeling_example/output/runs",
#              "fin": "/home/trentonjw/Documents/masters_presentation/final_runs/runs",
#             }
# y_lim = {"methane": (.272, .28), "ethanol": (1.125,1.150), "butane": (1.485,1.51), "hexane": (2.075, 2.1)}


# # # run_names = [("mine", f"baseline_{molecule}"), ("mine", f"control_{molecule}"), ("mine", f"epsilon_{molecule}"), ("mine", f"epsilon_10_{molecule}"),
# # #              ("proxima", "methane_mol1")]
# # # run_names = [("proxima_runs", "no_rotation_molbutane"), ("proxima_runs", "baseline_molbutane"), ("proxima_runs", "proxima_molbutane")]
# # # run_names = [("proxima_runs", "baseline_molbutane"), ("proxima_runs", "proxima_molbutane")]
# # # run_names = [("mine", f"mid_finals_baseline_{molecule}"), ("mine", f"mid_finals_proxima_{molecule}"),
# # #              ("mine", f"mid_finals_epistemic_{molecule}"), ("mine", f"mid_finals_epistemic_aleatory_{molecule}"),
# # #              ("mine", f"mid_finals_error-pred_fixed_threshold_{molecule}")
# # #             ]
# # run_names = [("mine", f"mid_finals_baseline_{molecule}", "Baseline"),
# #              ("mine", f"mid_finals_epistemic_{molecule}", "Proxima"),
# #              ("mine", f"mid_finals_proxima_{molecule}", "Epistemic Uncertainty"),
# #              ("mine", f"mid_finals_epistemic_aleatory_{molecule}", "Epistemic and Aleatory Uncertainty"),
# #             ]
# # run_names = [("mine", f"mid_finals_baseline_{molecule}", "Baseline"),
# #             #  ("pre_final", f"low_fid_big_mol_proxima_{molecule}", "Proxima"),
# #              ("mine", f"mid_finals_epistemic_{molecule}", "Epistemic Uncertainty"),
# #             #  ("pre_final", f"low_fid_big_mol_error_pred_fixed_threshold_{molecule}", "Epistemic and Aleatory Uncertainty with Predictor"),
# #             #  ("pre_final", f"low_fid_adaptive_test_error_pred_fixed_threshold_{molecule}", "Auto Threshold"),
# #             ("mine", f"mid_finals_proxima_{molecule}", "Distance Uncertainty"),
# #             ]

# run_names = [("fin", f"final_baseline_{molecule}", "Baseline"),
#              ("fin", f"final_epistemic_{molecule}", "Epistemic Uncertainty"),
#              ("fin", f"final_proxima_{molecule}", "Distance Uncertainty"),
#             ]


# # # run_names = [("pre_final", f"low_fidelity_test_baseline_{molecule}"),
# # #              ("pre_final", f"low_fidelity_test_proxima_{molecule}"),
# # #              ("pre_final", f"low_fidelity_test_epistemic_{molecule}"),
# # #              ("pre_final", f"low_fidelity_test_error_pred_fixed_threshold_{molecule}"),
# # #             ]

# # # run_names = [
# # #             #  ("pre_final", f"low_fidelity_test_baseline_{molecule}"),
# # #              ("final", f"final_proxima_{molecule}", "Proxima"),
# # #              ("final", f"final_epistemic_{molecule}", ""),
# # #              ("final", f"final_epistemic_aleatory_{molecule}"),
# # #              ("final", f"final_error_pred_fixed_threshold_{molecule}"),
# # #             ]
# # set_presentation_style()
# # # plot_rog(directory, run_names, y_lim = y_lim[molecule], outlier_threshold=(.272, .28))
# plot_rog(directory, run_names, y_lim = y_lim[molecule])
# # plot_bar_plot(directory, run_names)
# # plot_bar_plot(directory, run_names, "time")
# # # plot_speedup_plot(directory, run_names, f"low_fid_big_mol_baseline_{molecule}")
# plot_speedup_plot(directory, run_names, "Baseline")
