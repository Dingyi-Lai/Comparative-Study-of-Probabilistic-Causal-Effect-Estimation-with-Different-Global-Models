## This file concatenates the results across all the models for the same dataset, takes the median of errors for different seeds and writes the mean SMAPE, median SMAPE, ranked SMAPE,
# mean MASE, median MASE and ranked MASE to a csv file

import glob
import pandas as pd
import numpy as np
from configs.global_configs import model_testing_configs

def ensembling_forecasts(model_identifier, input_path, output_path, qr):
    ensembled_forecasts = {}
    for q in qr:
        all_forecast_files = [filename for filename in glob.iglob(input_path +
        model_identifier + "_*") if "_"+str(q) in filename]
        print(all_forecast_files)
        all_seeds_forecasts = []

        # iterate the forecasts file list
        for forecast_file in all_forecast_files:
            all_seeds_forecasts.append(pd.read_csv(forecast_file, header=None, dtype=np.float64))

        # convert the dataframe list to an array
        forecasts_array = np.stack(all_seeds_forecasts)

        # take the mean of forecasts across seeds for ensembling
        ensembled_forecasts[q] = np.nanmedian(forecasts_array, axis=0)

        # write the ensembled forecasts to a file
        output_file = output_path + model_identifier + "_" + str(q) +".txt"
        np.savetxt(output_file, ensembled_forecasts[q], delimiter = ',')
        ensembled_forecasts[q] = pd.DataFrame(ensembled_forecasts[q])
    # print(ensembled_forecasts)
    return ensembled_forecasts


