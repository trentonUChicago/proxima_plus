import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

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