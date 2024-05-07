import sys
sys.path.insert(0, ".")

import src.tfrecords_handler.moving_window.tfrecord_writer as tw
import os
import itertools

output_path = "./data/binary_data/sim/moving_window/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

if __name__ == '__main__':
    # Define the parameters
    time_series_lengths = [60, 222]
    amount_of_time_series = [10, 101, 500]
    te_intervention = ["ho", "he"]
    dgp = ["l", "nl"]
    input_size = 15
    output_size = 12
    # Nested loops
    for length, amount, ln, te in itertools.product(time_series_lengths, amount_of_time_series, dgp, te_intervention):
        tfrecord_writer = tw.TFRecordWriter(
            input_size = input_size+1,
            output_size = output_size,
            train_file_path = 'data/text_data/sim/moving_window/sim_'+\
                str(amount) + '_' + str(length) + '_' + ln + '_' + te + '_train'+\
                    '_' + str(output_size) + '_' + str(input_size) + '.txt',
            validate_file_path = 'data/text_data/sim/moving_window/sim_'+\
                str(amount) + '_' + str(length) + '_' + ln + '_' + te + '_validation'+\
                    '_' + str(output_size) + '_' + str(input_size) + '.txt',
            test_file_path = 'data/text_data/sim/moving_window/sim_'+\
                str(amount) + '_' + str(length) + '_' + ln + '_' + te + '_test'+\
                    '_' + str(output_size) + '_' + str(input_size) + '.txt',
            binary_train_file_path = output_path + 'sim_'+\
                str(amount) + '_' + str(length) + '_' + ln + '_' + te + '_train'+\
                    '_' + str(output_size) + '_' + str(input_size) + '.tfrecords',
            binary_validation_file_path = output_path + 'sim_'+\
                str(amount) + '_' + str(length) + '_' + ln + '_' + te + '_validation'+\
                    '_' + str(output_size) + '_' + str(input_size) + '.tfrecords',
            binary_test_file_path = output_path + 'sim_'+\
                str(amount) + '_' + str(length) + '_' + ln + '_' + te + '_test'+\
                    '_' + str(output_size) + '_' + str(input_size) + '.tfrecords',
        )

        tfrecord_writer.read_text_data()
        tfrecord_writer.write_train_data_to_tfrecord_file()
        tfrecord_writer.write_validation_data_to_tfrecord_file()
        tfrecord_writer.write_test_data_to_tfrecord_file()