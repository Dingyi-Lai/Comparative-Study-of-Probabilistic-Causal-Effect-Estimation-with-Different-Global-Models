import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from quantile_utils.CRPS_QL import mean_weighted_quantile_loss
import pickle

# def custom_smape(forecasts, actual, epsilon=0.1):
#     comparator = 0.5 + epsilon
#     sum_term = np.maximum(comparator, (np.abs(forecasts) + np.abs(actual) + epsilon))
#     return 2 * np.abs(forecasts - actual) / sum_term

def mase_greybox(holdout, forecast, scale):
    """
    Calculates Mean Absolute Scaled Error as in Hyndman & Koehler, 2006.
    
    Reference: https://github.com/config-i1/greybox/blob/6c84c729786f33a474ef833a13b7715831bd29e6/R/error-measures.R#L267

    Parameters:
        holdout (list or numpy array): Holdout values.
        forecast (list or numpy array): Forecasted values.
        scale (float): The measure to scale errors with. Usually - MAE of in-sample.
        na_rm (bool, optional): Whether to remove NA values from calculations.
                                Default is True.

    Returns:
        float: Mean Absolute Scaled Error.
    """
    if len(holdout) != len(forecast):
        print("The length of the provided data differs.")
        print(f"Length of holdout: {len(holdout)}")
        print(f"Length of forecast: {len(forecast)}")
        raise ValueError("Cannot proceed.")
    else:
        return np.mean(np.abs(np.array(holdout) - np.array(forecast)) / scale)

def evaluate(evaluate_args, ensembled_forecasts):

    # Example values for testing
    rnn_forecast_file_path = evaluate_args[0]
    errors_directory = evaluate_args[1]
    processed_forecasts_directory = evaluate_args[2]
    txt_test_file_name = evaluate_args[4]
    actual_results_file_name = evaluate_args[5]
    original_data_file_name = evaluate_args[6]
    input_size = evaluate_args[7]
    output_size = evaluate_args[8]
    contain_zero_values = evaluate_args[9]
    address_near_zero_instability = evaluate_args[10]
    integer_conversion = evaluate_args[11]
    seasonality_period = evaluate_args[12]
    without_stl_decomposition = evaluate_args[13]
    # root_directory = '/path/to/root/directory/'

    # Errors file names
    errors_directory = errors_directory + '/'
    errors_file_name = evaluate_args[3]
    errors_file_name_mean_median = 'mean_median_' + errors_file_name
    SMAPE_file_name_all_errors = 'all_smape_errors_' + errors_file_name
    MASE_file_name_all_errors = 'all_mase_errors_' + errors_file_name
    CRPS_file_name_cs = errors_file_name+ '_cs'
    errors_file_full_name_mean_median = errors_directory + errors_file_name_mean_median+'.txt'
    SMAPE_file_full_name_all_errors = errors_directory + SMAPE_file_name_all_errors
    MASE_file_full_name_all_errors = errors_directory + MASE_file_name_all_errors
    CRPS_file_cs = processed_forecasts_directory + CRPS_file_name_cs

    # Actual results file name
    actual_results = pd.read_csv(actual_results_file_name, sep=';', header=None)

    # Text test data file name
    txt_test_df = pd.read_csv(txt_test_file_name, sep=' ', header=None)

    # RNN forecasts file name as one of the argument which is ensembled_forecasts

    # Reading the original data to calculate the MASE errors
    with open(original_data_file_name, 'r') as file:
        original_dataset = [line.strip().split(',') for line in file]

    # Persisting the final forecasts
    processed_forecasts_file = processed_forecasts_directory + errors_file_name

    actual_results = actual_results.drop(columns=0)

    # take the uniqueindexes
    value = list(txt_test_df[0])
    uniqueindexes = [i-2 for i, val in enumerate(value, start=1) if val != value[i-2] and i!= 1]
    uniqueindexes.append(len(value)-1)

    actual_results_df = actual_results.dropna()

    # initialize
    converted_forecasts_matrix = np.zeros((len(ensembled_forecasts[0.5]), output_size))
    mase_vector = []
    crps_vector = []
    # lambda_val = -0.7 useless

    for k, v in ensembled_forecasts.items():
        # Perform spline regression for each time series
        num_time_series = v.shape[0]
        for i in range(num_time_series):

            # post-processing
            one_ts_forecasts = v.iloc[i].values
            finalindex = uniqueindexes[i]
            one_line_test_data = txt_test_df.iloc[finalindex].values
            mean_value = one_line_test_data[input_size + 2]
            level_value = one_line_test_data[input_size + 3]
            if without_stl_decomposition:
                converted_forecasts_df = np.exp(one_ts_forecasts + level_value)
            else:
                seasonal_values = one_line_test_data[(input_size + 4):(3 + input_size + output_size)]
                converted_forecasts_df = np.exp(one_ts_forecasts + level_value + seasonal_values)
            
            if contain_zero_values:
                converted_forecasts_df = converted_forecasts_df - 1

            converted_forecasts_df = mean_value * converted_forecasts_df
            converted_forecasts_df[converted_forecasts_df < 0] = 0

            if integer_conversion:
                converted_forecasts_df = np.round(converted_forecasts_df)

            converted_forecasts_matrix[i, :] = converted_forecasts_df # one_ts_forecasts
            
            if k == 0.5:
                # np.diff(np.array(original_dataset[i]), lag=seasonality_period, differences=1))
                original_values = list(map(float, original_dataset[i]))
                lagged_diff = [original_values[i] - original_values[i - seasonality_period] for i in range(seasonality_period, len(original_values))]
                mase_vector.append(mase_greybox(np.array(actual_results_df.iloc[i]), converted_forecasts_df, np.mean(np.abs(lagged_diff))))

        if k == 0.5:
            converted_forecasts_smape = converted_forecasts_matrix
        crps_vector.append(converted_forecasts_matrix)
        # Persisting the converted forecasts
        np.savetxt(processed_forecasts_file+'_'+str(k)+'.txt', converted_forecasts_matrix, delimiter=",")

    cs = CubicSpline(list(ensembled_forecasts.keys()), crps_vector, bc_type='natural')
    crps_y_pred = np.transpose(cs(list(ensembled_forecasts.keys())), (1, 0, 2))
    
    # Calculating the CRPS
    crps_qs = mean_weighted_quantile_loss(crps_y_pred, np.array(actual_results_df), ensembled_forecasts.keys())

    mean_CRPS = np.mean(crps_qs)

    mean_CRPS_str = f"mean_CRPS:{mean_CRPS}"
    all_CRPS_qs = f"CRPS for different quantiles:{crps_qs}"
    # std_CRPS_str = f"std_CRPS:{std_CRPS}"

    print(mean_CRPS_str)
    print(all_CRPS_qs)
    # print(std_CRPS_str)

    # Calculating the SMAPE
    if address_near_zero_instability:
        epsilon = 0.1
        comparator = 0.5 + epsilon
        sum_term = np.maximum(comparator, (np.abs(converted_forecasts_smape) + np.abs(np.array(actual_results_df)) + epsilon))
        time_series_wise_SMAPE = 2 * np.abs(converted_forecasts_smape - np.array(actual_results_df)) / sum_term
    else:
        time_series_wise_SMAPE = 2 * np.abs(converted_forecasts_smape - np.array(actual_results_df)) / (np.abs(converted_forecasts_smape) + np.abs(np.array(actual_results_df)))

    SMAPEPerSeries = np.mean(time_series_wise_SMAPE, axis=1)

    mean_SMAPE = np.mean(SMAPEPerSeries)
    # median_SMAPE = np.median(SMAPEPerSeries)
    # std_SMAPE = np.std(SMAPEPerSeries)

    mean_SMAPE_str = f"mean_SMAPE:{mean_SMAPE}"
    # median_SMAPE_str = f"median_SMAPE:{median_SMAPE}"
    # std_SMAPE_str = f"std_SMAPE:{std_SMAPE}"

    print(mean_SMAPE_str)
    # print(median_SMAPE_str)
    # print(std_SMAPE_str)

    # MASE
    mean_MASE = np.mean(mase_vector)
    # median_MASE = np.median(mase_vector)
    # std_MASE = np.std(mase_vector)

    mean_MASE_str = f"mean_MASE:{mean_MASE}"
    # median_MASE_str = f"median_MASE:{median_MASE}"
    # std_MASE_str = f"std_MASE:{std_MASE}"

    print(mean_MASE_str)
    # print(median_MASE_str)
    # print(std_MASE_str)


    # Writing the SMAPE results to file
    with open(errors_file_full_name_mean_median, 'w') as f:
        # f.write('\n'.join([mean_SMAPE_str, median_SMAPE_str, std_SMAPE_str]))
        f.write('\n'.join([mean_SMAPE_str]))

    np.savetxt(SMAPE_file_full_name_all_errors+'.txt', SMAPEPerSeries, delimiter=",", fmt='%f')

    # Writing the MASE results to file
    with open(errors_file_full_name_mean_median, 'a') as f:
        # f.write('\n'.join([mean_MASE_str, median_MASE_str, std_MASE_str]))
        f.write('\n'.join([mean_MASE_str]))

    np.savetxt(MASE_file_full_name_all_errors+'.txt', mase_vector, delimiter=",", fmt='%f')

    # Writing the CRPS results to file
    with open(errors_file_full_name_mean_median, 'a') as f:
        # f.write('\n'.join([mean_CRPS_str, median_CRPS_str, std_CRPS_str]))
        f.write('\n'.join([mean_CRPS_str]))
    with open(CRPS_file_cs+'.pickle', 'wb') as f:
        pickle.dump(crps_vector, f)
    # np.savetxt(CRPS_file_cs, crps_vector, delimiter=",", fmt='%f')
