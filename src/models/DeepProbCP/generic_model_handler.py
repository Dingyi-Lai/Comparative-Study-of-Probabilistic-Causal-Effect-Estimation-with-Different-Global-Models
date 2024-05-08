# Record the running time
print('begin')
import time
T1 = time.time()

# Inbuilt or External Modules
import argparse # customized arguments in .bash
import csv # input and output .csv data
import glob # file matching using wildcards
import numpy as np # for numerical computing
import os # OS routines
# import pdb # debugger, but it doesn't work remotely
import pandas as pd
import pickle
# Customized Modules
from configs.global_configs import hyperparameter_tuning_configs
from configs.global_configs import model_testing_configs
from error_calculator.final_evaluation import evaluate
from models.DeepProbCP.ensembling_forecasts import ensembling_forecasts
# from utility_scripts.invoke_final_evaluation import invoke_script # for invoking R
from models.DeepProbCP.hyperparameter_config_reader import read_initial_hyperparameter_values, read_optimal_hyperparameter_values
from models.DeepProbCP.persist_optimized_config_results import persist_results
# import SMAC utilities
# import the config space and the different types of parameters
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter
from smac.configspace import ConfigurationSpace
from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO
# stacking model
from rnn_architectures.stacking_model_p import StackingModel


LSTM_USE_PEEPHOLES = True # LSTM with “peephole connections"
BIAS = False # in tf.keras.layers.dense
# In TensorFlow's Keras API, the use_bias parameter in the tf.keras.layers.Dense layer is a Boolean 
# (True/False) parameter that determines whether the layer should include biases or not during the
# computation. A bias term is an additional parameter in a neural network layer that allows the model
# to learn an offset that is added to the weighted sum of the inputs.

# Here's a breakdown:

# use_bias=True: If you set use_bias=True, the dense layer will include a bias term in its computation.
# This means that, for each neuron in the layer, not only are weights applied to the input features, but
# there is also a learnable bias term added.

# use_bias=False: If you set use_bias=False, the dense layer will not include a bias term. In this case,
#  only the weighted sum of the input features is considered without an additional bias term.

# Choosing whether to include biases often depends on the specific characteristics of the problem you're
# working on. In some cases, biases help the model better fit the data by allowing it to learn an offset.
# In other cases, especially when the data is naturally centered, biases may not be necessary. It's often
# a matter of experimentation to determine the best configuration for your particular task.


# final execution with the optimized config
def train_model(configs):
    print(configs)

    hyperparameter_values = {
        "num_hidden_layers": configs["num_hidden_layers"],
        "cell_dimension": configs["cell_dimension"],
        "minibatch_size": configs["minibatch_size"],
        "max_epoch_size": configs["max_epoch_size"],
        "max_num_epochs": configs["max_num_epochs"],
        "l2_regularization": configs["l2_regularization"],
        "gaussian_noise_stdev": configs["gaussian_noise_stdev"],
        "random_normal_initializer_stdev": configs["random_normal_initializer_stdev"],
    }

    if optimizer != "cocob":
        hyperparameter_values["initial_learning_rate"] = configs["initial_learning_rate"]

    error = model.tune_hyperparameters(**hyperparameter_values)

    print(model_identifier)
    print(error)
    return error.item()

def smac():
    # Build Configuration Space which defines all parameters and their ranges
    configuration_space = ConfigurationSpace()

    initial_learning_rate = UniformFloatHyperparameter("initial_learning_rate", hyperparameter_values_dic['initial_learning_rate'][0],
                                                  hyperparameter_values_dic['initial_learning_rate'][1],
                                                  default_value=hyperparameter_values_dic['initial_learning_rate'][0])
    cell_dimension = UniformIntegerHyperparameter("cell_dimension",
                                                  hyperparameter_values_dic['cell_dimension'][0],
                                                  hyperparameter_values_dic['cell_dimension'][1],
                                                  default_value=hyperparameter_values_dic['cell_dimension'][
                                                      0])
    no_hidden_layers = UniformIntegerHyperparameter("num_hidden_layers",
                                                    hyperparameter_values_dic['num_hidden_layers'][0],
                                                    hyperparameter_values_dic['num_hidden_layers'][1],
                                                    default_value=hyperparameter_values_dic['num_hidden_layers'][0])
    minibatch_size = UniformIntegerHyperparameter("minibatch_size", hyperparameter_values_dic['minibatch_size'][0],
                                                  hyperparameter_values_dic['minibatch_size'][1],
                                                  default_value=hyperparameter_values_dic['minibatch_size'][0])
    max_epoch_size = UniformIntegerHyperparameter("max_epoch_size", hyperparameter_values_dic['max_epoch_size'][0],
                                                  hyperparameter_values_dic['max_epoch_size'][1],
                                                  default_value=hyperparameter_values_dic['max_epoch_size'][0])
    max_num_of_epochs = UniformIntegerHyperparameter("max_num_epochs", hyperparameter_values_dic['max_num_epochs'][0],
                                                     hyperparameter_values_dic['max_num_epochs'][1],
                                                     default_value=hyperparameter_values_dic['max_num_epochs'][0])
    l2_regularization = UniformFloatHyperparameter("l2_regularization",
                                                   hyperparameter_values_dic['l2_regularization'][0],
                                                   hyperparameter_values_dic['l2_regularization'][1],
                                                   default_value=hyperparameter_values_dic['l2_regularization'][0])
    gaussian_noise_stdev = UniformFloatHyperparameter("gaussian_noise_stdev",
                                                      hyperparameter_values_dic['gaussian_noise_stdev'][0],
                                                      hyperparameter_values_dic['gaussian_noise_stdev'][1],
                                                      default_value=hyperparameter_values_dic['gaussian_noise_stdev'][
                                                          0])
    random_normal_initializer_stdev = UniformFloatHyperparameter("random_normal_initializer_stdev",
                                                                 hyperparameter_values_dic[
                                                                     'random_normal_initializer_stdev'][0],
                                                                 hyperparameter_values_dic[
                                                                     'random_normal_initializer_stdev'][1],
                                                                 default_value=hyperparameter_values_dic[
                                                                     'random_normal_initializer_stdev'][
                                                                     0])

    # add the hyperparameter for learning rate only if the  optimization is not cocob
    if optimizer == "cocob":
        configuration_space.add_hyperparameters(
            [cell_dimension, no_hidden_layers, minibatch_size, max_epoch_size, max_num_of_epochs,
             l2_regularization, gaussian_noise_stdev, random_normal_initializer_stdev])
    else:

        configuration_space.add_hyperparameters(
            [initial_learning_rate, cell_dimension, minibatch_size, max_epoch_size,
             max_num_of_epochs, no_hidden_layers,
             l2_regularization, gaussian_noise_stdev, random_normal_initializer_stdev])

    # creating the scenario object
    scenario = Scenario({
        "run_obj": "quality",
        "runcount-limit": hyperparameter_tuning_configs.SMAC_RUNCOUNT_LIMIT,
        "cs": configuration_space,
        "deterministic": "true",
        "abort_on_first_run_crash": "false"
    })

    # optimize using an SMAC object
    smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(seed), tae_runner=train_model)

    incumbent = smac.optimize()
    return incumbent.get_dictionary()


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser("Train different forecasting models")
    argument_parser.add_argument('--dataset_type', required=True,
                                 help='calls911/sim/...')
    argument_parser.add_argument('--dataset_name', required=True, help='Unique string for the name of the dataset')
    argument_parser.add_argument('--contain_zero_values', required=False,
                                 help='Whether the dataset contains zero values(0/1). Default is 0')
    argument_parser.add_argument('--address_near_zero_instability', required=False,
                                 help='Whether to use a custom SMAPE function to address near zero instability(0/1). Default is 0')
    argument_parser.add_argument('--integer_conversion', required=False,
                                 help='Whether to convert the final forecasts to integers(0/1). Default is 0')
    argument_parser.add_argument('--no_of_series', required=True,
                                 help='The number of series in the dataset')
    # argument_parser.add_argument('--initial_hyperparameter_values_file', required=True,
    #                              help='The file for the initial hyperparameter configurations')
    # argument_parser.add_argument('--binary_train_file_train_mode', required=True,
    #                              help='The tfrecords file for train dataset in the training mode')
    # argument_parser.add_argument('--binary_valid_file_train_mode', required=True,
    #                              help='The tfrecords file for validation dataset in the training mode')
    # argument_parser.add_argument('--binary_train_file_test_mode', required=True,
    #                              help='The tfrecords file for train dataset in the testing mode')
    # argument_parser.add_argument('--binary_test_file_test_mode', required=True,
    #                              help='The tfrecords file for test dataset in the testing mode')
    # argument_parser.add_argument('--txt_test_file', required=True, help='The txt file for test dataset')
    # argument_parser.add_argument('--actual_results_file', required=True, help='The txt file of the actual results')
    # argument_parser.add_argument('--original_data_file', required=True, help='The txt file of the original dataset')
    argument_parser.add_argument('--cell_type', required=False,
                                 help='The cell type of the RNN(LSTM/GRU/RNN). Default is LSTM')
    # argument_parser.add_argument('--input_size', required=False,
    #                              help='The input size of the moving window. Default is 0')
    argument_parser.add_argument('--seasonality_period', required=True, help='The seasonality period of the time series')
    argument_parser.add_argument('--forecast_horizon', required=True, help='The forecast horizon of the dataset')
    argument_parser.add_argument('--optimizer', required=False, help='The type of the optimizer(cocob/adam/adagrad...). Default is cocob')
    argument_parser.add_argument('--quantile_range', required=False, 
                                 help='The range of the quantile for quantile forecasting. Default is np.linspace(0, 1, 21)')
    argument_parser.add_argument('--evaluation_metric', required=False, 
                                 help='The evaluation metric like sMAPE. Default is CRPS')
    argument_parser.add_argument('--without_stl_decomposition', required=False,
                                 help='Whether not to use stl decomposition(0/1). Default is 0')

    # parse the user arguments
    args = argument_parser.parse_args()

    # arguments with no default values
    dataset_name = args.dataset_name
    no_of_series = int(args.no_of_series)
    output_size = int(args.forecast_horizon)
    seasonality_period = int(args.seasonality_period)
    seed = 1234

    # arguments with default values
    if args.contain_zero_values:
        contain_zero_values = bool(int(args.contain_zero_values))
    else:
        contain_zero_values = False

    if args.optimizer:
        optimizer = args.optimizer
    else:
        optimizer = "cocob"
    
    if args.quantile_range:
        quantile_range = args.quantile_range
    else:
        # quantile_range = np.linspace(0, 1, 21)
        quantile_range = [0.1,0.5,0.9]
    
    if args.evaluation_metric:
        evaluation_metric = args.evaluation_metric
    else:
        # evaluation_metric = "sMAPE"
        evaluation_metric = "CRPS"

    if args.without_stl_decomposition:
        without_stl_decomposition = bool(int(args.without_stl_decomposition))
    else:
        without_stl_decomposition = False

    if args.cell_type:
        cell_type = args.cell_type
    else:
        cell_type = "LSTM"

    if args.address_near_zero_instability:
        address_near_zero_instability = bool(int(args.address_near_zero_instability))
    else:
        address_near_zero_instability = False

    if args.integer_conversion:
        integer_conversion = bool(int(args.integer_conversion))
    else:
        integer_conversion = False

    if without_stl_decomposition:
        stl_decomposition_identifier = "without_stl_decomposition"
    else:
        stl_decomposition_identifier = "with_stl_decomposition"


    model_identifier = dataset_name + "_" + cell_type + "cell" + "_" +  optimizer + "_" + \
                       stl_decomposition_identifier
    print("Model Training Started for {}".format(model_identifier))
    
    input_size = int(seasonality_period * 1.25) + 1
    initial_hyperparameter_values_file = "configs/initial_hyperparameter_values/" + \
        args.dataset_type + "_" + cell_type + "cell" + "_" +  optimizer
    binary_train_file_path_train_mode = "datasets/binary_data/" + args.dataset_type +  \
        "/moving_window/" + dataset_name + "_train_" + args.forecast_horizon + "_" +  \
            str(input_size-1) + ".tfrecords"
    binary_validation_file_path_train_mode = "datasets/binary_data/" + args.dataset_type +  \
        "/moving_window/" + dataset_name + "_validation_" + args.forecast_horizon + "_" +  \
            str(input_size-1) + ".tfrecords"
    binary_train_file_test_mode = "datasets/binary_data/" + args.dataset_type +  \
        "/moving_window/" + dataset_name + "_validation_" + args.forecast_horizon + "_" +  \
            str(input_size-1) + ".tfrecords"
    binary_test_file_path_test_mode = "datasets/binary_data/" + args.dataset_type +  \
        "/moving_window/" + dataset_name + "_test_" + args.forecast_horizon + "_" +  \
            str(input_size-1) + ".tfrecords"
    txt_test_file_path = "datasets/text_data/" + args.dataset_type +  \
        "/moving_window/" + dataset_name + "_test_" + args.forecast_horizon + "_" +  \
            str(input_size-1) + ".txt"
    # actual_results_file_path = "datasets/text_data/" + args.dataset_type +  \
    #     "/" + dataset_name + "_test_actual.csv"
    actual_results_file_path = "datasets/text_data/" + args.dataset_type +  \
        "/" + dataset_name + "_for_errors.csv"
    # original_data_file_path = "datasets/text_data/" + args.dataset_type +  \
    #     "/" + dataset_name + "_train.csv"
    # define the key word arguments for the different model types
    model_kwargs = {
        'use_bias': BIAS,
        'use_peepholes': LSTM_USE_PEEPHOLES,
        'input_size': input_size,
        'output_size': output_size,
        'optimizer': optimizer,
        'quantile_range': quantile_range,
        'evaluation_metric': evaluation_metric,
        'no_of_series': no_of_series,
        'binary_train_file_path': binary_train_file_path_train_mode,
        'binary_test_file_path': binary_test_file_path_test_mode,
        'binary_validation_file_path': binary_validation_file_path_train_mode,
        'contain_zero_values': contain_zero_values,
        'address_near_zero_instability': address_near_zero_instability,
        'integer_conversion': integer_conversion,
        'seed': seed,
        'cell_type': cell_type,
        'without_stl_decomposition': without_stl_decomposition
    }

    # # select the model type
    # model = StackingModel(**model_kwargs)

    # # delete model if existing
    # for file in glob.glob("./results/DeepProbCP/"+model_identifier+"_model.pkl"):
    #     os.remove(file)

    # # save model
    # with open("./results/DeepProbCP/"+model_identifier+"_model.pkl", "wb") as fout:
    #     pickle.dump(model, fout)
    
    # # # Load the study from the saved file
    # # with open("./results/DeepProbCP/"+model_identifier+"_model.pkl", "rb") as fin:
    # #     model = pickle.load(fin)
    
    # # delete hyperparameter configs files if existing
    # for file in glob.glob(hyperparameter_tuning_configs.OPTIMIZED_CONFIG_DIRECTORY + model_identifier + "*"):
    #     os.remove(file)

    # # read the initial hyperparamter configurations from the file
    # hyperparameter_values_dic = read_initial_hyperparameter_values(initial_hyperparameter_values_file)
    # optimized_configuration = smac()
    # print(optimized_configuration)

    # # persist the optimized configuration to a file
    # persist_results(optimized_configuration, hyperparameter_tuning_configs.OPTIMIZED_CONFIG_DIRECTORY + '/' + model_identifier + '.txt')

    # # # not training again but just read in
    # # optimized_configuration = read_optimal_hyperparameter_values("./results/DeepProbCP/optimized_configurations/" + model_identifier + ".txt")
    # # print(optimized_configuration)

    # # delete the forecast files if existing
    # for file in glob.glob(
    #         model_testing_configs.FORECASTS_DIRECTORY + model_identifier + "*"):
    #     os.remove(file)

    # print("tuning finished")
    # T2 = time.time()
    # print(T2)
    # for seed in range(1, 11):
    #     forecasts = model.test_model(optimized_configuration, seed)

    #     model_identifier_extended = model_identifier + "_" + str(seed)
    #     for k, v in forecasts.items():
    #         rnn_forecasts_file_path = model_testing_configs.FORECASTS_DIRECTORY + model_identifier_extended + 'q_' + str(k) + '.txt'
            
    #         with open(rnn_forecasts_file_path, "w") as output:
    #             writer = csv.writer(output, lineterminator='\n')
    #             writer.writerows(forecasts[k])
    # print("prediction finished")
    # T3 = time.time()
    
    
    # # delete the ensembled forecast files if existing
    # for file in glob.glob(
    #         model_testing_configs.ENSEMBLE_FORECASTS_DIRECTORY + model_identifier + "*"):
    #     os.remove(file)

    # # ensemble the forecasts
    # ensembled_forecasts = ensembling_forecasts(model_identifier, model_testing_configs.FORECASTS_DIRECTORY,
    #                      model_testing_configs.ENSEMBLE_FORECASTS_DIRECTORY,quantile_range)

    # not training again but just read in
    ensembled_forecasts = {}
    for q in quantile_range:
        ensembled_forecasts[q] = pd.read_csv(model_testing_configs.ENSEMBLE_FORECASTS_DIRECTORY +\
                                              model_identifier + "_" + str(q) +".txt",header=None)

    # print("ensembled finished")
    # T4 = time.time()

    evaluate_args = [model_testing_configs.ENSEMBLE_FORECASTS_DIRECTORY,
                   model_testing_configs.ENSEMBLE_ERRORS_DIRECTORY,
                   model_testing_configs.PROCESSED_ENSEMBLE_FORECASTS_DIRECTORY,
                   model_identifier,
                   txt_test_file_path,
                   actual_results_file_path,
                #    original_data_file_path,
                   input_size,
                   output_size,
                   int(contain_zero_values),
                   int(address_near_zero_instability),
                   int(integer_conversion),
                   seasonality_period,
                   int(without_stl_decomposition),
                   args.dataset_type]
    evaluate(evaluate_args, ensembled_forecasts)
    # print("invoking R")
    # # invoke the final error calculation
    # invoke_script([model_testing_configs.ENSEMBLE_FORECASTS_DIRECTORY,
    #                model_testing_configs.ENSEMBLE_ERRORS_DIRECTORY,
    #                model_testing_configs.PROCESSED_ENSEMBLE_FORECASTS_DIRECTORY,
    #                model_identifier,
    #                txt_test_file_path,
    #                actual_results_file_path,
    #                original_data_file_path,
    #                str(input_size),
    #                str(output_size),
    #                str(int(contain_zero_values)),
    #                str(int(address_near_zero_instability)),
    #                str(int(integer_conversion)),
    #                str(seasonality_period),
    #                str(int(without_stl_decomposition))])
    
    # print("R script finished")
    T5 = time.time()

    # print('Running time: %s m' % ((T2 - T1) / 60))
    # print('Running time: %s m' % ((T3 - T2) / 60))
    # print('Running time: %s m' % ((T3 - T1) / 60))
    # print('Running time: %s m' % ((T4 - T2) / 60))
    print('Running time: %s m' % ((T5 - T1) / 60))
