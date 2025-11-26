import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress
import os
import re
import json

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)


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



def get_rog_data_from_dir(directory_path, outlier_threshold=None):
    file_names = os.listdir(directory_path)

    rog_data = {}
    num_outliers_removed = 0

    for file_name in file_names:
        match = re.search(r'_temp(\d+)', file_name)
        temp = int(match.group(1))

        # Run if looking at my runs
        try:
            # Read json results file
            json_filename = directory_path + "/" + file_name + "/results.json"
            with open(json_filename, 'r') as f:
                data = json.load(f)

            rog = data["r_g_stats"]["statistic"]
            run_time = data["time"]
            
            # Read csv results file
            df = pd.read_csv(directory_path + "/" + file_name + "/results.csv")
            # Get number of target runs
            num_target_run = (df['surrogate_accepts'] == False).sum()
        # Run if looking at proxima runs
        except:
            try:
                # Read json results file
                rog_json_filename = directory_path + "/" + file_name + "/result.json"
                stats_json_filename = directory_path + "/" + file_name + "/lfa_stats.json"
                with open(rog_json_filename, 'r') as f:
                    data = json.load(f)
                with open(stats_json_filename, 'r') as f:
                    stats_data = json.load(f)
                
                rog = data["r_g"]["statistic"]
                run_time = stats_data["lfa_time"] + stats_data["uq_time"] + stats_data["train_time"] + stats_data["target_time"]
                num_target_run = stats_data["target_runs"]
            except:
                continue


        
        if (outlier_threshold is not None) and (rog <= outlier_threshold[0] or rog >= outlier_threshold[1]):
            print("Outlier Removed. File name:", file_name)
            num_outliers_removed += 1
        else:
            if temp in rog_data:
                rog_data[temp]['rog'].append(rog)
                rog_data[temp]['num_target_run'].append(num_target_run)
                rog_data[temp]['time'].append(run_time)
            else:
                rog_data[temp] = {'rog': [rog],
                                'num_target_run': [num_target_run],
                                'time': [run_time]}
    
    avg_rog_data = {'rog_means': [], 'rog_sems': [], 'num_target_run_means': [], 'num_target_run_sems': [], 'time_means': [], 'time_sems': [], 'temps': []}
    for temp in sorted(rog_data.keys()):
        avg_rog_data['rog_means'].append(np.mean(rog_data[temp]['rog']))
        avg_rog_data['rog_sems'].append(np.std(rog_data[temp]['rog']) / np.sqrt(len(rog_data[temp]['rog'])))
        avg_rog_data['num_target_run_means'].append(np.mean(rog_data[temp]['num_target_run']))
        avg_rog_data['num_target_run_sems'].append(np.std(rog_data[temp]['num_target_run']) / np.sqrt(len(rog_data[temp]['num_target_run'])))
        avg_rog_data['time_means'].append(np.mean(rog_data[temp]['time']))
        avg_rog_data['time_sems'].append(np.std(rog_data[temp]['time']) / np.sqrt(len(rog_data[temp]['time'])))
        avg_rog_data['temps'].append(temp)
    
    print("Number of outliers removed:", num_outliers_removed)
    return avg_rog_data


def create_rog_over_temp_plot(run_data, y_lim = (.272, .28)):
    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot lines
    for run_name in run_data:
        means = run_data[run_name]['rog_means']
        sems = run_data[run_name]['rog_sems']
        temps = run_data[run_name]['temps']

        plt.errorbar(temps, means, yerr=sems, fmt='-o', capsize=5, label=run_name)
    
    plt.ylim(*y_lim)
    plt.xlabel('Temperature (K)')
    plt.ylabel('Radius of Gyration (ROG)')
    plt.title('Final ROG')
    plt.legend()
    plt.grid(True)
    plt.show()


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


def plot_rog(directory, run_names, y_lim = (.272, .28), outlier_threshold=None):
    run_data = {}
    for directory_type, run_name in run_names:
        run_data[run_name] = get_rog_data_from_dir(directory[directory_type] + "/" + run_name, outlier_threshold=outlier_threshold)
    
    create_rog_over_temp_plot(run_data, y_lim = y_lim)


def plot_bar_plot(directory, run_names, data_type='num_target_run'):
    run_data = {}
    for directory_type, run_name in run_names:
        run_data[run_name] = get_rog_data_from_dir(directory[directory_type] + "/" + run_name)
    
    create_bar_plot(run_data, data_type)

directory = {"mine": "/home/trentonjw/Documents/Project/proxima_plus/example_simulations/single_molecule_modeling_example/output/runs",
             "proxima": "/home/trentonjw/Documents/Project/temp_proxima/proxima_cc/examples/molecule-sampling/runs",
             "proxima_runs": "/home/trentonjw/Documents/Project/temp_proxima/proxima_runs"}
molecule = "butane"
y_lim = {"methane": (.272, .28), "ethanol": (1.125,1.150), "butane": (1.485,1.51)}
# run_names = [("mine", f"baseline_{molecule}"), ("mine", f"control_{molecule}"), ("mine", f"epsilon_{molecule}"), ("mine", f"epsilon_10_{molecule}"),
#              ("proxima", "methane_mol1")]
run_names = [("proxima_runs", "no_rotation_molbutane"), ("proxima_runs", "baseline_molbutane")]

plot_rog(directory, run_names, y_lim = y_lim[molecule])