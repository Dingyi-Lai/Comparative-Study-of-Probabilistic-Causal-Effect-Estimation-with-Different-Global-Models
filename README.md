## Software Requirements ##

### Python Packages ###
| Software  | Version |
| ------------- | ------------- |
| `Python`  |  `>=3.6`  |
| `Tensorflow`  | `2.0.0`  |
| `smac`  | `0.11.1` |

### R Packages ###
* smooth
* MASS
* forecast
* stringr, dplyr, ggplot2, tidyverse, hrbrthemes, rstatix, ggpubr
* "readxl", seasonal, fpp, fpp3, plotly

## Path Variables ##

Set the `PYTHONPATH` env variable of the system. Append absolute paths of both the project root directory and the directory of the `external_packages/cocob_optimizer` into the `PYTHONPATH`  

For R scripts, make sure to set the working directory to the project root folder.

## Preprocessing the Data ##

We have used two datasets for the experiments of this paper:

1) The National Ambulance Surveillance System (NASS) dataset --> The national dataset of coded ambulance clinical records held by Turning Point, an Australian addiction research and education centre. This data is not an open dataset, therefore it was not possible to share this data here. Then, for the script codes that deal with this data, it was only possible to share the codes, to observe how was done the pre-processing and forecasting proceedings.

2) The 911 Emergency Calls Dataset for Montgomery County --> an opened dataset from Kaggle containing the 911 emergency calls from Montgomery County (Pennsylvania-USA) from December-2015 to July-2020, where the raw data can be retrieved from this [link](https://www.kaggle.com/mchirico/montcoalert).

In the `preprocess_scripts/EMS-MC` directory, there are some wrangling and pre-processing script codes to adjust this Dataset 2. First, the `calls911_wrangling_code.R` file wrangles the raw data retrived from the link mentioned previously. Then, the `calls911_to_forecasting_code.R` file adjusts the time series from this dataset for the forecasting task. Finally, the files `adjustOrigTestDtSet.R` and `adjustOrigTraintDtSet.R`, execute some additional steps to format the training and testing datasets to be used for the DeepCPNet framework modelling, in the next steps that will be described in the sequence.

In the `preprocess_scripts/EMS` directory we have the same files `adjustOrigTestDtSet.R` and `adjustOrigTraintDtSet.R` to adjust Dataset 1 for the next preprocessing steps as described in the next section.

#### Create the text files of the data ####

From the data reserved to be trained, three files need to be created for every model, one per training, validation and testing. Example preprocessing R scripts to create text data files (`mean_stl_train_validation_....R` and `mean_stl_test_....R` files) are in `preprocess_scripts/EMS (or EMS-MC)/moving_window/without_stl_decomposition` directories.

Sample Record of validation file in moving window format without STL Decomposition: (in `datasets/text_data/EMS-MC/moving_window/without_stl_decomposition` directory):

`1|i 0.0353890753105493 -0.0169499558775499 -0.0447295199846256 0.0870397576464977 -0.00334430382177138 -0.282057706290792 -0.0447295199846256 0.123407401817373 -0.0307432780098858 0.169927417452265 -0.0589141549765821 -0.117754654999516 -0.164274670634408 0.0746172376479405 0.2464674945746 0.0620384554410805 |o -0.164274670634408 0.214379180023099 -0.00334430382177138 0.192400273304324 -0.21306483480384 0.0492994296636507 0.225190096127315 |# 80.3469387755102 -0.0789445958071699`

`input_size = 15`\
`max_forecast_horizon = 12`\
`seasonality_period = 12`

#### Create the tfrecord files of the data ####

For faster execution, the text data files are converted to tfrecord binary format. The `tfrecords_handler` module converts the text data into tfrecords format (using `tfrecord_writer.py`) as well as reads in tfrecord data (using `tfrecord_reader.py`) during execution. Example scripts to convert text data into tfrecords format (`create_tfrecords.py` files) can be found in the `preprocess_scripts` directory.

## Execution Instructions ##

Example bash scripts are in the directory `utility_scripts/execution_scripts`. 

#### External Arguments ####
The model expects a number of arguments.
1. dataset_name - Any unique string for the name of the dataset
2. contain_zero_values - Whether the dataset contains zero values(0/1)
3. address_near_zero_instability - Whether the dataset contains zero values(0/1) - Whether to use a custom SMAPE function to address near zero instability(0/1). Default value is 0
4. integer_conversion - Whether to convert the final forecasts to integers(0/1). Default is 0
5. initial_hyperparameter_values_file - The file for the initial hyperparameter range configurations
6. binary_train_file_train_mode - The tfrecords file for train dataset in the training mode
7. binary_valid_file_train_mode - The tfrecords file for validation dataset in the training mode
8. binary_train_file_test_mode - The tfrecords file for train dataset in the testing mode
9. binary_test_file_test_mode - The tfrecords file for test dataset in the testing mode
10. txt_test_file - The text file for test dataset
11. actual_results_file - The text file of the actual results
12. original_data_file - The text file of the original dataset with all the given data points
13. cell_type - The cell type of the RNN(LSTM/GRU/RNN). Default is LSTM
14. input_size - The input size of the moving window. Default is 0 in the case of non moving window format
15. seasonality_period - The seasonality period of the time series
16. forecast_horizon - The forecast horizon of the dataset
17. optimizer - The type of the optimizer(cocob/adam/adagrad)
18. without_stl_decomposition - Whether not to use stl decomposition(0/1). Default is 0
19. no_of_series - The number of series of the dataset.

#### Execution Flow ####

##### 1.Invoking the Script: #####
The first point of invoking the models is the `generic_model_handler.py`. The `generic_model_handler.py` parses the external arguments and identifies the required type of optimizer, cell etc... The actual stacking model is inside the directory `rnn_architectures`. 
First, the hyperparameter tuning is carried out using the validation errors of the stacking model. Example initial hyperparameter ranges can be found inside the directory `configs/initial_hyperparameter_values`. The found optimal hyperparameter combination is written to a file in the directory `results/nn_model_results/rnn/optimized_configurations`. 
Then the found optimal hyperparameter combination is used on the respective model to generate the final forecasts. Every model is run on 10 Tensorflow graph seeds (from 1 to 10). The forecasts are written to 10 files inside the directory `results/nn_model_results/rnn/forecasts`.

##### 2.Ensembling Forecasts: #####
The forecasts from the 10 seeds are ensembled by taking the median. The `utility_scripts/ensembling_forecasts.py` script does this. This script is invoked implicitly inside the `generic_model_handler.py`. The ensembled forecasts are written to the directory `results/nn_model_results/rnn/ensemble_forecasts`. 

##### 3.Error Calculation: #####
The SMAPE and MASE errors are calculated per each series for each model using the error calculation scripts in the directory `error_calculator`. The name of the script is `final_evaluation.R`. This script is also implicitly invoked inside the `generic_model_handler.py`. The script perform the post processing of the forecasts to reverse initial preprocessing. The errors of the ensembles are written to the directory `results/nn_model_results/rnn/ensemble_errors`. And the processed ensembled forecasts after the post processing are written to the directory `results/nn_model_results/rnn/processed_ensemble_forecasts`. 

##### 4.Benchmark Forecasts - ARIMA and ETS: #####
Inside the directory `stastical_bench_methods` there are the Arima and ETS forecasts codes files performed for both datasets. The results of these forecasting are written to the directory `results/arima_forecasts/` or `results/ets_forecasts/`. Inside these directories we also have examples for the scritps used to calculate the errors.

#### Support code scripts to build the graphs, to perform the statistical significance tests, and to execute STL exploratory analysis for ALI variable:  ####
Inside the directory `support_code_scripts` there are 3 support code files:

* `graphs_and_meanTests_NASSdataset.R` --> code to adjust the data for plotting the Figure 2A concerning to the Dataset 1
* `graphs_and_meanTests_MONTdataset.R` --> code to adjust the data for plotting the Figure 2B concerning to the Dataset 2
* `licenses_STL_analysis_code.R` --> code to perfom the STL decomposition exploratory analysis for the ALI variable, and to plot Figure 1.
# master_thesis
