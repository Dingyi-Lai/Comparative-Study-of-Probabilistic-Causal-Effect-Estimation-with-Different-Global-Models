# python preprocess_scripts/Simulation/create_tfrecords.py
import sys
sys.path.insert(0, ".")

import tfrecords_handler.moving_window.tfrecord_writer as tw
import os
import itertools

output_path = "./datasets/binary_data/Synthetics/moving_window/without_stl_decomposition/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

if __name__ == '__main__':
    # Define the parameters
    time_series_lengths = [60, 222]
    amount_of_time_series = [10, 101, 500]
    te_intervention = ["homogeneous", "heterogeneous"]
    dgp = ["linear", "nonlinear"]

    # Nested loops
    for length, amount, ln, te in itertools.product(time_series_lengths, amount_of_time_series, dgp, te_intervention):
        tfrecord_writer = tw.TFRecordWriter(
            input_size = 16,
            output_size = 12,
            train_file_path = 'datasets/text_data/Synthetics/moving_window/_sim_'+\
                str(amount) + '_' + str(length) + '_' + ln + '_' + te + '_train_12_15.txt',
            validate_file_path = 'datasets/text_data/Synthetics/moving_window/_sim_'+\
                str(amount) + '_' + str(length) + '_' + ln + '_' + te + '_validation_12_15.txt',
            test_file_path = 'datasets/text_data/Synthetics/moving_window/_sim_'+\
                str(amount) + '_' + str(length) + '_' + ln + '_' + te + '_test_12_15.txt',
            binary_train_file_path = output_path + '_sim_'+\
                str(amount) + '_' + str(length) + '_' + ln + '_' + te + '_train_12_15.tfrecords',
            binary_validation_file_path = output_path + '_sim_'+\
                str(amount) + '_' + str(length) + '_' + ln + '_' + te + '_validation_12_15.tfrecords',
            binary_test_file_path = output_path + '_sim_'+\
                str(amount) + '_' + str(length) + '_' + ln + '_' + te + '_test_12_15.tfrecords'
        )

        tfrecord_writer.read_text_data()
        tfrecord_writer.write_train_data_to_tfrecord_file()
        tfrecord_writer.write_validation_data_to_tfrecord_file()
        tfrecord_writer.write_test_data_to_tfrecord_file()