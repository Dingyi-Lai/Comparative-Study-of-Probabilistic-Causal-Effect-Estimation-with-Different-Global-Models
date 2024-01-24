from configs.global_configs import hyperparameter_tuning_configs
import numpy as np
from utility_scripts.hyperparameter_scripts.hyperparameter_config_reader import read_initial_hyperparameter_values, read_optimal_hyperparameter_values
# import SMAC utilities
# import the config space and the different types of parameters
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter
from smac.configspace import ConfigurationSpace
from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO

# final execution with the optimized config
def train_model(model,configs):
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

    error = model.tune_hyperparameters(**hyperparameter_values)

    print(error)
    return error.item()

def smac(hyperparameter_values_dic, optimizer):
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
