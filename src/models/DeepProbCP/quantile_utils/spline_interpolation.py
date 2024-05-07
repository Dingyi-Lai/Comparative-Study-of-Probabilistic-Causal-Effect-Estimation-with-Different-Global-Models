import numpy as np
from scipy.interpolate import CubicSpline
from quantile_utils.CRPS_QL import mean_weighted_quantile_loss
import csv
import pandas as pd

def spline_interpolation_from_forecasts(forecasts, model_identifier, output_path, error_path, actual_path):
    """
    Modeling quantile distribution via spline interpolation
    
    Args:
    forecasts: a dict with quantile being the key and a matrix for multiple ts being the value 
    
    Returns:
    quantile_distr: a dict with quantile being the key and params for quantile distributions being the value 
    """
    quantile_distr = {}
    crps = []
    num_q = len(forecasts.keys())
    num_len = forecasts[0.5].shape[0]
    num_horizon = forecasts[0.5].shape[1]
    crps_y_pred = []
    pd.read_csv(actual_path, sep=';', header=None)
    crps_y_true = []
    for k, v in forecasts.items():

        # Perform spline regression for each time series
        num_time_series = v.shape[1]
        smoothed_data = np.zeros_like(v)
        for i in range(num_time_series):
            ts = v.iloc[:, i]
            x = np.arange(len(ts))
            cs = CubicSpline(x, ts, bc_type='natural')
            smoothed_data[:, i] = cs(x)
        
        # Store the results
        quantile_distr[k] = smoothed_data
        # print(crps_y_pred, v)
        crps_y_pred.extend(v.values)
        crps_y_true.extend(smoothed_data)




        # write the ensembled forecasts to a file
        output_file = output_path + model_identifier + "_cs_" + str(k) +".txt"
        np.savetxt(output_file, quantile_distr[k], delimiter = ',')
    crps = mean_weighted_quantile_loss(np.array(crps_y_pred), np.array(crps_y_true), forecasts.keys(),num_len, num_horizon)
    print(crps)
    data = list(zip(forecasts.keys(), crps))
    # Write the quantile and crps to a CSV file
    with open(error_path+'quantiles_crps.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    return quantile_distr, crps

