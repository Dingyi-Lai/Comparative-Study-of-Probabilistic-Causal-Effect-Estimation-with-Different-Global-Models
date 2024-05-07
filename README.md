# Title

**Type:** Master's Thesis

**Author:** Dingyi Lai

**1st Examiner:** Prof. Dr. Stefan Lessmann

**2nd Examiner:** Prof. Dr. Sonja Greven

[Insert here a figure explaining your approach or main results]

## Table of Content

- [Summary](#summary)
- [Working with the repo](#Working-with-the-repo)
    - [Dependencies](#Dependencies)
    - [Setup](#Setup)
- [Reproducing results](#Reproducing-results)
    - [Training code](#Training-code)
    - [Evaluation code](#Evaluation-code)
    - [Pretrained models](#Pretrained-models)
- [Results](#Results)
- [Project structure](-Project-structure)

## Summary

The estimation of causal effects becomes increasingly complex when interventions exert diverse influences across quantiles. Addressing this challenge, we introduce a novel global framework that seamlessly integrates causal analysis with prediction algorithms. Despite remaining theoretical gaps, we propose a standardized approach to answer this research question. This involves defining causal mechanisms via directed acyclic graphs, elucidating theoretical assumptions, conducting placebo tests to identify causal effects, and estimating probabilistic causal effects within the global causal framework. Through comparative analysis utilizing synthetic and real-world datasets, we demonstrate the potential of this framework to estimate varying causal effects across quantiles over time. While promising, ongoing refinement is necessary to enhance the framework's consistency and robustness.

**Keywords**: Causal Inference \sep Time-varying Effect \sep Probabilistic Prediction

## Working with the repo

### Dependencies
`Python >= 3.6`

### Setup

## Software Requirements ##

1. Clone this repository

2. Create an virtual environment and activate it
```bash
python -m venv thesis-env
source thesis-env/bin/activate
```

3. Install requirements
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Reproducing results

### Data preprocessing code
We have used two datasets for the experiments of this paper:
1) **Synthetic Dataset:**

The synthetic dataset comprises 24 scenarios for generation, varying from short time series (with a length of 60) to long time series (with a length of 222). These scenarios range from homogeneous intervention over 0.9 quantiles (adding one unit standard deviation of treated units before intervention) to heterogeneous intervention over 0.9 quantiles (adding a random number between 0.7 to 1.5, multiplied by one unit standard deviation of treated units before intervention).

Furthermore, the dataset spans from a few time series (10 in total), medium time series (101 in total) to many time series (500 in total), and from a linear data generation process (autoregressive regression, i.e., AR) to a nonlinear structure (self-exciting threshold autoregressive, i.e., SETAR).

The codes for simulation and exploratory data analysis are located in `./src/prepare_source_data/sim/simulation.rmd`, with its HTML version available at `simulation.html`. The original data for simulation is situated in `./data/text_data/sim/unrate.txt`, accessible from Data_USEconModel in MATLAB.

For DeepProbCP, the dataset undergoes preprocessing using a window moving strategy, implemented in `./src/prepare_source_data/sim/preprocessing_layer.R`. This aligns with the original codes from Grecov et al. for DeepCPNet. Subsequently, `./src/prepare_source_data/sim/create_tfrecords.py` facilitates the creation of data in the tfrecords format to enhance computational efficiency, a method also employed in the previous codes for DeepCPNet. The customized module "tfrecords_handler" is also available in `./src/models/DeepProbCP`.

2) **The 911 Emergency Calls Dataset for Montgomery County**

This dataset, sourced from Kaggle, contains records of 911 emergency calls from Montgomery County, Pennsylvania, USA, spanning from December 2015 to July 2020. The raw data file, named "911.csv," is available for download from the following https://www.kaggle.com/mchirico/montcoalert.

**Exploratory Data Analysis:**
- The exploratory data analysis is documented in `./src/prepare_source_data/calls911/EDA.ipynb`.

**Data Wrangling:**
- To process the data, the script `./src/prepare_source_data/calls911/calls911_wrangling_code.R` is utilized for tasks such as splitting, formatting, and cleaning.
- The resulting output data should ideally be stored in `./data/text_data/calls911/calls3.csv`. However, due to size limitations on GitHub, this file has been removed from the current repository.

**Preprocessing:**
- Further preprocessing steps are conducted using `./src/prepare_source_data/calls911/calls911_to_forecasting_code.R`.
- The full monthly dataset is available in `./data/text_data/calls911/calls911_month_full.txt`, while the training dataset spans from December 2015 to December 2019 and is stored in `./data/text_data/calls911/calls911_month_train.txt`. The testing dataset spans from January 2020 to July 2020 and is stored in `./data/text_data/calls911/calls911_month_test.txt`.
- To address the training and testing datasets in subsequent steps, scripts `./src/prepare_source_data/calls911/adjustOrigTrainDtSet.R` and `./src/prepare_source_data/calls911/adjustOrigTestDtSet.R` are utilized. The resulting output files are named `callsMT2_train.csv` and `callsMT2_test_actual.csv`, serving as the training and testing datasets, respectively.

**For DeepProbCP:**
- The dataset undergoes preprocessing using a window moving strategy after being transformed to tfrecords by `./src/prepare_source_data/calls911/create_tfrecords.py`.
- These preprocessing steps are implemented in `./src/prepare_source_data/calls911/mean_stl_train_validation_withoutEXvar.R` and `./src/prepare_source_data/calls911/mean_stl_test_withoutEXvar.R`.

**For Other Models:**
- The script `./src/prepare_source_data/calls911/benchmarks.py` is utilized to convert `calls911_month_full.txt` to a standardized input format, resulting in `./data/text_data/calls911/calls911_benchmarks.csv`.

### Training code

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

### Evaluation code

Does a repository contain a script to calculate the performance of the trained model(s) or run experiments on models?

### Pretrained models

Does a repository provide free access to pretrained model weights?

## Results

Does a repository contain a table/plot of main results and a script to reproduce those results?

## Path Variables ##

Set the `PYTHONPATH` env variable of the system. Append absolute paths of both the project root directory and the directory of the `external_packages/cocob_optimizer` into the `PYTHONPATH`  

For R scripts, make sure to set the working directory to the project root folder.

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

## Project structure

(Here is an example from SMART_HOME_N_ENERGY, [Appliance Level Load Prediction](https://github.com/Humboldt-WI/dissertations/tree/main/SMART_HOME_N_ENERGY/Appliance%20Level%20Load%20Prediction) dissertation)

```bash
├── README.md
├── requirements.txt                                -- required libraries
├── data                                            -- stores csv file 
├── plots                                           -- stores image files
└── src
    ├── prepare_source_data.ipynb                   -- preprocesses data
    ├── data_preparation.ipynb                      -- preparing datasets
    ├── model_tuning.ipynb                          -- tuning functions
    └── run_experiment.ipynb                        -- run experiments 
    └── plots                                       -- plotting functions                 
```