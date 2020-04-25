from datetime import datetime
import os
import json
import pickle


def pickle_save(filename, an_object):

    with open(filename, 'wb') as output_file:
        pickle.dump(an_object, output_file, protocol=4)


def pickle_load(filename):

    with open(filename, 'rb') as input_file:
        an_object = pickle.load(input_file, encoding='bytes')

    return an_object


def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    return data


def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as text_file:
        data = text_file.read()

    return data


def write_text_file(file_path, a_string):
    with open(file_path, 'w', encoding='utf-8') as text_file:
        chars_written_int = text_file.write(a_string)

    return chars_written_int


def write_json_file(file_path, data_dict):
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data_dict, json_file, ensure_ascii=False, sort_keys=True, indent=4)


def create_directory_path(prefix, training_dir_name):

    if not training_dir_name:
        raise Exception('Directory path not defined ! Check training/data directory.')

    sub_dir = os.path.join(prefix, '{}'.format(training_dir_name))

    if os.path.exists(sub_dir):
        files_in_directory = os.listdir(sub_dir)
        # ignore log files and .json files if they already exist
        files_in_directory = [file for file in files_in_directory if not(file.endswith('.log') or file.endswith('.json')) ]
        if files_in_directory:
            raise Exception('Directory {} already exists and not empty. Aborting.'.format(sub_dir))
    elif not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    return sub_dir


def create_directory_path_with_timestamp(destination_dir, dir_prefix=''):

    directory_name = datetime.now().strftime("%Y_%m_%d_T%H_%M_%S")
    if dir_prefix != '':
        directory_name = dir_prefix + directory_name

    sub_dir = create_directory_path(destination_dir, directory_name)

    return sub_dir
