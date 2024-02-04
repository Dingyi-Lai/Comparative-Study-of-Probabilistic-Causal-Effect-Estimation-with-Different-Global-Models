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


class StackingModel:

    def __init__(self, **kwargs):
        self.__use_bias = kwargs["use_bias"]
        self.__use_peepholes = kwargs["use_peepholes"]
        self.__input_size = kwargs["input_size"]
        self.__output_size = kwargs["output_size"]
        self.__no_of_series = kwargs["no_of_series"]
        self.__optimizer = kwargs["optimizer"]
        self.__quantile_range = kwargs["quantile_range"]
        self.__evaluation_metric = kwargs["evaluation_metric"]
        self.__seed = kwargs["seed"]
        self.__cell_type = kwargs["cell_type"]
        self.__without_stl_decomposition = kwargs["without_stl_decomposition"]

    def __create_training_validation_datasets(self, gaussian_noise_stdev):
        self.__training_dataset_for_train = tf.data.TFRecordDataset(filenames=[self.__binary_train_file_path], compression_type="ZLIB")
        self.__validation_dataset_for_train = tf.data.TFRecordDataset(filenames=[self.__binary_validation_file_path],
                                                     compression_type="ZLIB")

        # parsing the data

        # define the expected shapes of validation data after padding
        validation_input_padded_shapes = ([None, self.__input_size])
        validation_output_padded_shapes = ([None, self.__output_size])
        validation_metadata_padded_shapes = ([None, self.__meta_data_size])

        self.__training_dataset_for_train_parsed = self.__training_dataset_for_train.map(
            lambda example: self.tfrecord_reader.train_data_parser_for_training(example, gaussian_noise_stdev))
        self.__training_dataset_for_train_parsed = self.__training_dataset_for_train_parsed.shuffle(training_data_configs.SHUFFLE_BUFFER_SIZE)

        self.__validation_dataset_input_parsed = self.__validation_dataset_for_train.map(
            self.tfrecord_reader.validation_data_input_parser)
        self.__validation_dataset_input_padded = self.__validation_dataset_input_parsed.padded_batch(batch_size=int(self.__no_of_series),
                                                                 padded_shapes=validation_input_padded_shapes)

        self.__validation_dataset_output_parsed = self.__validation_dataset_for_train.map(
            self.tfrecord_reader.validation_data_output_parser)
        self.__validation_dataset_output_padded = self.__validation_dataset_output_parsed.padded_batch(
            batch_size=int(self.__no_of_series), padded_shapes=validation_output_padded_shapes)

        self.__validation_dataset_lengths_parsed = self.__validation_dataset_for_train.map(
            self.tfrecord_reader.validation_data_lengths_parser)
        self.__validation_data_lengths = self.__validation_dataset_lengths_parsed.batch(self.__no_of_series)

        self.__validation_dataset_metadata_parsed = self.__validation_dataset_for_train.map(
            self.tfrecord_reader.validation_data_metadata_parser)
        self.__validation_metadata = self.__validation_dataset_metadata_parsed.padded_batch(self.__no_of_series,
                                                                             validation_metadata_padded_shapes)

        for lengths, metadata, actuals in zip(self.__validation_data_lengths, self.__validation_metadata,
                                              self.__validation_dataset_output_padded):
            self.__validation_lengths = lengths.numpy()
            self.__validation_metadata = metadata.numpy()
            self.__validation_dataset_output_padded = actuals.numpy()

        self.__last_indices = self.__validation_lengths - 1
        self.__array_first_dimension = np.array(range(0, self.__no_of_series))

        self.__true_seasonality_values = self.__validation_metadata[self.__array_first_dimension,
                                         self.__last_indices, 1:]

        self.__validation_metadata = self.__validation_metadata[self.__array_first_dimension, self.__last_indices, :]

    def __create_testing_datasets(self, gaussian_noise_stdev):

        self.__training_dataset_for_test = tf.data.TFRecordDataset(filenames=[self.__binary_validation_file_path],
                                                     compression_type="ZLIB")
        self.__testing_dataset_for_test = tf.data.TFRecordDataset(filenames=[self.__binary_test_file_path],
                                               compression_type="ZLIB")

        # define the expected shapes of data after padding
        test_padded_shapes = ([None, self.__input_size])

        self.__training_dataset_for_test_parsed = self.__training_dataset_for_test.map(
            lambda example: self.tfrecord_reader.train_data_parser_for_testing(example, gaussian_noise_stdev))
        self.__training_dataset_for_test_parsed = self.__training_dataset_for_test_parsed.shuffle(
            training_data_configs.SHUFFLE_BUFFER_SIZE)

        self.__testing_dataset_input_parsed = self.__testing_dataset_for_test.map(self.tfrecord_reader.test_data_input_parser)
        self.__testing_dataset_input_padded = self.__testing_dataset_input_parsed.padded_batch(self.__no_of_series, test_padded_shapes)

        self.__testing_dataset_lengths_parsed = self.__testing_dataset_for_test.map(
            self.tfrecord_reader.test_data_lengths_parser)
        self.__testing_dataset_lengths = self.__testing_dataset_lengths_parsed.batch(self.__no_of_series)

        for lengths in self.__testing_dataset_lengths:
            self.__testing_dataset_lengths = lengths.numpy()


    def __get_optimizer(self, initial_learning_rate = 0.0):
        if self.__optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate = initial_learning_rate)
        elif self.__optimizer == "adagrad":
            optimizer = tf.keras.optimizers.Adagrad(learning_rate=initial_learning_rate)
        elif self.__optimizer == "cocob":
            optimizer = cocob_optimizer.COCOB()
        return optimizer

    
    def __build_model(self, random_normal_initializer_stdev, num_hidden_layers, cell_dimension, l2_regularization, q, optimizer):
    # def __build_model(self, random_normal_initializer_stdev, num_hidden_layers, cell_dimension, q, optimizer):
        initializer = tf.keras.initializers.TruncatedNormal(stddev=random_normal_initializer_stdev)

        # model from the functional API
        input = tf.keras.Input(shape=(None, self.__input_size), name='inputs')
        masked_output = tf.keras.layers.Masking(mask_value=0.0)(input)

        # lstm stack
        next_input = masked_output
        for i in range(num_hidden_layers):
            lstm_output = tf.keras.layers.RNN(tf.keras.experimental.PeepholeLSTMCell(cell_dimension, kernel_initializer=initializer), return_sequences=True) (next_input)
            next_input = lstm_output

        dense_layer_output = tf.keras.layers.Dense(self.__output_size, use_bias=self.__use_bias, kernel_initializer=initializer) (masked_output)

        # build the model
        self.__model = tf.keras.Model(inputs=input, outputs=dense_layer_output, name='stacking_model')

        # plot the model to validate
        self.__model.summary()
        
        def custom_mae(y_true, y_pred):
            error = tf.keras.losses.mae(y_true, y_pred)
            l2_loss = 0.0
            for var in self.__model.trainable_variables:
                l2_loss += tf.nn.l2_loss(var)

            l2_loss = tf.math.multiply(l2_regularization, l2_loss)

            total_loss = error + l2_loss
            return total_loss

        def custom_ql(y_true, y_pred,q): # add l2_regularization
            error = pinball_loss(y_true, y_pred, q)
            l2_loss = 0.0
            for var in self.__model.trainable_variables:
                l2_loss += tf.nn.l2_loss(var)

            l2_loss = tf.math.multiply(l2_regularization, l2_loss)

            total_loss = error + l2_loss
            return total_loss

        self.__model.compile(loss=PinballLoss(tau=q),
                      optimizer=optimizer)
        # self.__model.compile(loss=lambda y,f: custom_ql(y,f,q),
        #                      optimizer=optimizer)
        # self.__model.compile(loss=custom_mae,
        #               optimizer=optimizer)

    def tune_hyperparameters(self, **kwargs):
        num_hidden_layers = int(kwargs['num_hidden_layers'])
        cell_dimension = int(kwargs['cell_dimension'])
        minibatch_size = int(kwargs['minibatch_size'])
        max_epoch_size = int(kwargs['max_epoch_size'])
        max_num_epochs = int(kwargs['max_num_epochs'])
        l2_regularization = kwargs['l2_regularization']
        gaussian_noise_stdev = kwargs['gaussian_noise_stdev']
        random_normal_initializer_stdev = kwargs['random_normal_initializer_stdev']

        # clear previous session
        tf.keras.backend.clear_session()
        tf.compat.v1.random.set_random_seed(self.__seed)

        if self.__optimizer != "cocob":
            initial_learning_rate = kwargs['initial_learning_rate']
            optimizer = self.__get_optimizer(initial_learning_rate)
        else:
            optimizer = self.__get_optimizer()

        qr = self.__quantile_range
        self.__create_training_validation_datasets(gaussian_noise_stdev)

        train_padded_shapes = ([None, self.__input_size], [None, self.__output_size])

        # repeating for epoch size and batching
        train_dataset = self.__training_dataset_for_train_parsed.repeat(max_epoch_size)
        train_dataset = train_dataset.padded_batch(batch_size=int(minibatch_size), padded_shapes=train_padded_shapes)
        last_validation_outputs = {} # for different quantiles

        for q in qr:
            print(q)
            self.__build_model(random_normal_initializer_stdev, num_hidden_layers, cell_dimension, l2_regularization, q, optimizer)

            # training
            self.__model.fit(train_dataset, epochs=max_num_epochs, shuffle=True)

            # get the validation predictions
            validation_prediction = self.__model.predict(self.__validation_dataset_input_padded)

            #calculate the validation losses
            last_validation_outputs[q] = validation_prediction[self.__array_first_dimension, self.__last_indices]
            
            actual_values = self.__validation_dataset_output_padded[self.__array_first_dimension, self.__last_indices, :]

            mean_values = self.__validation_metadata[:, 0]
            mean_values = mean_values[:, np.newaxis]

            level_values = self.__validation_metadata[:, 1]
            level_values = level_values[:, np.newaxis]

            converted_validation_output = {} # for different quantiles
            if self.__without_stl_decomposition:

                converted_validation_output[q] = np.exp(last_validation_outputs[q] + level_values)
                converted_actual_values = np.exp(actual_values + level_values)

            else:
                true_seasonality_values = self.__validation_metadata[:, 2:]
                converted_validation_output[q] = np.exp(
                    true_seasonality_values + level_values + last_validation_outputs[q])
                converted_actual_values = np.exp(
                    true_seasonality_values + level_values + actual_values)

            if self.__contain_zero_values:  # to compensate for 0 values in data
                converted_validation_output[q] = converted_validation_output[q] - 1
                converted_actual_values = converted_actual_values - 1

            converted_validation_output[q] = converted_validation_output[q] * mean_values
            converted_actual_values = converted_actual_values * mean_values

            if self.__integer_conversion:
                converted_validation_output[q] = np.round(converted_validation_output[q])
                converted_actual_values = np.round(converted_actual_values)

            converted_validation_output[q][converted_validation_output[q] < 0] = 0
            converted_actual_values[converted_actual_values < 0] = 0

        if self.__evaluation_metric == 'CRPS':
            self.quantile_weights = [1.0 / len(qr)] * len(qr)
            qt_loss = []
            for level, weight, y_pred_q in zip(
                qr, self.quantile_weights, list(converted_validation_output.values())
            ):
                qt_loss.append(
                    weight * pinball_loss(converted_actual_values, y_pred_q, level) * 2
                )
            print(qt_loss)
            # stacked_qt_losses = F.stack(*qt_loss, axis=-1)
            metric = np.mean(qt_loss)

        if self.__evaluation_metric == 'sMAPE':
            smape_values_per_series = []
            for level, y_pred_q in zip(
                qr, list(converted_validation_output.values())
            ):
                
                if self.__address_near_zero_instability:
                    # calculate the smape
                    epsilon = 0.1
                    sum = np.maximum(np.abs(y_pred_q) + np.abs(converted_actual_values) + epsilon,
                                    0.5 + epsilon)
                    smape_values = (np.abs(y_pred_q - converted_actual_values) /
                                    sum) * 2
                    smape_values_per_series.append(np.mean(smape_values, axis=1))
                else:
                    # calculate the smape
                    smape_values = (np.abs(y_pred_q - converted_actual_values) /
                                    (np.abs(y_pred_q) + np.abs(converted_actual_values))) * 2
                    smape_values_per_series.append(np.mean(smape_values, axis=1))
    
            # if self.__address_near_zero_instability:
            #     # calculate the smape
            #     epsilon = 0.1
            #     sum = np.maximum(np.abs(converted_validation_output) + np.abs(converted_actual_values) + epsilon,
            #                     0.5 + epsilon)
            #     smape_values = (np.abs(converted_validation_output - converted_actual_values) /
            #                     sum) * 2
            #     smape_values_per_series = np.mean(smape_values, axis=1)
            # else:
            #     # calculate the smape
            #     smape_values = (np.abs(converted_validation_output - converted_actual_values) /
            #                     (np.abs(converted_validation_output) + np.abs(converted_actual_values))) * 2
            #     smape_values_per_series = np.mean(smape_values, axis=1)

            metric = np.mean(smape_values_per_series)

        return metric

    def test_model(self, kwargs, seed):

        num_hidden_layers = int(kwargs['num_hidden_layers'])
        cell_dimension = int(kwargs['cell_dimension'])
        minibatch_size = int(kwargs['minibatch_size'])
        max_epoch_size = int(kwargs['max_epoch_size'])
        max_num_epochs = int(kwargs['max_num_epochs'])
        l2_regularization = kwargs['l2_regularization']
        gaussian_noise_stdev = kwargs['gaussian_noise_stdev']
        random_normal_initializer_stdev = kwargs['random_normal_initializer_stdev']

        # clear previous session
        tf.keras.backend.clear_session()
        tf.compat.v1.random.set_random_seed(seed)

        if self.__optimizer != "cocob":
            initial_learning_rate = kwargs['initial_learning_rate']
            optimizer = self.__get_optimizer(initial_learning_rate)
        else:
            optimizer = self.__get_optimizer()

        qr = self.__quantile_range
        # prepare the data
        self.__create_testing_datasets(gaussian_noise_stdev)

        train_padded_shapes = ([None, self.__input_size], [None, self.__output_size])
        # train_dataset = self.__training_dataset_for_test_parsed.repeat(max_epoch_size)
        # train_dataset = self.__training_dataset_for_test_parsed.padded_batch(batch_size=minibatch_size,
        #                                                          padded_shapes=train_padded_shapes)
        train_dataset = self.__training_dataset_for_test_parsed.repeat(max_epoch_size)
        train_dataset = train_dataset.padded_batch(batch_size=int(minibatch_size), padded_shapes=train_padded_shapes)
        forecasts = {} # for different quantiles
        # build a model for each quantile
        for q in qr:
            self.__build_model(random_normal_initializer_stdev, num_hidden_layers, cell_dimension, l2_regularization, q, optimizer)

            # training
            self.__model.fit(train_dataset, epochs=max_num_epochs, shuffle=True)

            # testing
            test_prediction = self.__model.predict(self.__testing_dataset_input_padded)

            # extracting the final time step forecast
            last_output_index = self.__testing_dataset_lengths - 1
            array_first_dimension = np.array(range(0, self.__no_of_series))
            forecasts[q] = test_prediction[array_first_dimension, last_output_index]
        return forecasts