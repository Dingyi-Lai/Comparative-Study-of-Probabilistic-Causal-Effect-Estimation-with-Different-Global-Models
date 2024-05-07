import sys
sys.path.insert(0, ".")

import src.tfrecords_handler.moving_window.tfrecord_writer as tw
import os

output_path = "./data/binary_data/calls911/moving_window/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

if __name__ == '__main__':
    input_size = 15,
    output_size = 7,
    tfrecord_writer = tw.TFRecordWriter(
        input_size = input_size+1,
        output_size = output_size,
        train_file_path = 'data/text_data/calls911/moving_window/callsMT2_train_' + \
            str(output_size) + '_' + str(input_size) + '.txt',
        validate_file_path = 'data/text_data/calls911/moving_window/callsMT2_validation_' + \
            str(output_size) + '_' + str(input_size) + '.txt',
        test_file_path = 'data/text_data/calls911/moving_window/callsMT2_test_' + \
            str(output_size) + '_' + str(input_size) + '.txt',
        binary_train_file_path = output_path + 'callsMT2_train_' + \
            str(output_size) + '_' + str(input_size) + '.tfrecords',
        binary_validation_file_path = output_path + 'callsMT2_validation_' + \
            str(output_size) + '_' + str(input_size) + '.tfrecords',
        binary_test_file_path = output_path + 'callsMT2_test_' + \
            str(output_size) + '_' + str(input_size) + '.tfrecords'
    )

    tfrecord_writer.read_text_data()
    tfrecord_writer.write_train_data_to_tfrecord_file()
    tfrecord_writer.write_validation_data_to_tfrecord_file()
    tfrecord_writer.write_test_data_to_tfrecord_file()