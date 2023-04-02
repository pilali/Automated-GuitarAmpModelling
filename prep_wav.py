# Creating a valid dataset for the trainining script
# using wav files provided by user.
# Example of usage:
# python3 prep_wav.py -f input.wav target.wav -l "RNN-aidadsp-1"
# the files will be splitted 70% 15% 15%
# and used to populate train test val.
# This is done to have different data for training, testing and validation phase
# according with the paper.
# If the user provide multiple wav files pairs e.g. guitar_in.wav guitar_tg.wav bass_in.wav bass_tg.wav
# then 70% of guitar_in.wav is concatenated to 70% of bass_in.wav and so on.
# If the user provide guitar and bass files of the same length, then the same amount
# of guitar and bass recorded material will be used for network training.

import CoreAudioML.miscfuncs as miscfuncs
from CoreAudioML.dataset import audio_converter, audio_splitter
from scipy.io import wavfile
import numpy as np
import argparse
import os
import csv

def save_wav(name, rate, data, flatten=True):
    print("Writing %s with rate: %d length: %d dtype: %s" % (name, rate, data.size, data.dtype))
    if flatten:
        wavfile.write(name, rate, data.flatten().astype(np.float32))
    else:
        wavfile.write(name, rate, data.astype(np.float32))

def parse_csv(path):
    train_bounds = []
    test_bounds = []
    val_bounds = []
    print("Using csv file %s" % path)
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                ref_names = ["#", "Name", "Start", "End", "Length", "Color"]
                if row != ref_names:
                    print("Error: csv file with wrong format")
                    exit(1)
            else:
                if row[5] == "FF0000": # Red means training
                    train_bounds.append([int(row[2]), int(row[3])])
                elif row[5] == "00FF00": # Green means test
                    test_bounds.append([int(row[2]), int(row[3])])
                elif row[5] == "0000FF": # Blue means val
                    val_bounds.append([int(row[2]), int(row[3])])
                elif row[5] == "00FFFF": # Green+Blue means test+val
                    test_bounds.append([int(row[2]), int(row[3])])
                    val_bounds.append([int(row[2]), int(row[3])])
            line_count = line_count + 1

    if len(train_bounds) < 1 or len(test_bounds) < 1 or len(val_bounds) < 1:
        print("Error: csv file is not containing correct RGB codes")
        exit(1)

    return[train_bounds, test_bounds, val_bounds]

def nonConditionedWavParse(args):
    print("Using config file %s" % args.load_config)
    file_name = ""
    configs = miscfuncs.json_load(args.load_config, args.config_location)
    try:
        file_name = configs['file_name']
    except KeyError:
        print("Error: config file doesn't have file_name defined")
        exit(1)

    counter = 0
    main_rate = 0
    train_in = np.ndarray([0], dtype=np.float32)
    train_tg = np.ndarray([0], dtype=np.float32)
    test_in = np.ndarray([0], dtype=np.float32)
    test_tg = np.ndarray([0], dtype=np.float32)
    val_in = np.ndarray([0], dtype=np.float32)
    val_tg = np.ndarray([0], dtype=np.float32)

    for in_file, tg_file in zip(args.files[::2], args.files[1::2]):
        print("Input file name: %s" % in_file)
        in_rate, in_data = wavfile.read(in_file)
        print("Target file name: %s" % tg_file)
        tg_rate, tg_data = wavfile.read(tg_file)

        print("Input rate: %d length: %d [samples]" % (in_rate, in_data.size))
        print("Target rate: %d length: %d [samples]" % (tg_rate, tg_data.size))

        if in_rate != tg_rate:
            print("Error! Sample rate needs to be equal")
            exit(1)
        else:
            rate = in_rate

        # First wav file sets the rate
        if counter == 0:
            main_rate = rate

        if rate != main_rate:
            print("Error: all the wav files needs to have the same format and rate")
            exit(1)

        min_size = in_data.size
        if(in_data.size != tg_data.size):
            min_size = min(in_data.size, tg_data.size)
            print("Warning! Length for audio files\n\r  %s\n\r  %s\n\rdoes not match, setting both to %d [samples]" % (in_file, tg_file, min_size))
            _in_data = np.resize(in_data, min_size)
            _tg_data = np.resize(tg_data, min_size)
            in_data = _in_data
            tg_data = _tg_data
            del _in_data
            del _tg_data

        x_all = audio_converter(in_data)
        y_all = audio_converter(tg_data)

        # @TODO: auto-align code goes here

        # Default to 70% 15% 15% split
        if not args.csv_file:
            splitted_x = audio_splitter(x_all, [0.70, 0.15, 0.15])
            splitted_y = audio_splitter(y_all, [0.70, 0.15, 0.15])
        else:
            # Csv file to be named as in file
            [train_bounds, test_bounds, val_bounds] = parse_csv(os.path.splitext(in_file)[0] + ".csv")
            splitted_x = [np.ndarray([0], dtype=np.float32), np.ndarray([0], dtype=np.float32), np.ndarray([0], dtype=np.float32)]
            splitted_y = [np.ndarray([0], dtype=np.float32), np.ndarray([0], dtype=np.float32), np.ndarray([0], dtype=np.float32)]
            for bounds in train_bounds:
                splitted_x[0] = np.append(splitted_x[0], audio_splitter(x_all, bounds, unit='s'))
                splitted_y[0] = np.append(splitted_y[0], audio_splitter(y_all, bounds, unit='s'))
            for bounds in test_bounds:
                splitted_x[1] = np.append(splitted_x[1], audio_splitter(x_all, bounds, unit='s'))
                splitted_y[1] = np.append(splitted_y[1], audio_splitter(y_all, bounds, unit='s'))
            for bounds in val_bounds:
                splitted_x[2] = np.append(splitted_x[2], audio_splitter(x_all, bounds, unit='s'))
                splitted_y[2] = np.append(splitted_y[2], audio_splitter(y_all, bounds, unit='s'))

        train_in = np.append(train_in, splitted_x[0])
        train_tg = np.append(train_tg, splitted_y[0])
        test_in = np.append(test_in, splitted_x[1])
        test_tg = np.append(test_tg, splitted_y[1])
        val_in = np.append(val_in, splitted_x[2])
        val_tg = np.append(val_tg, splitted_y[2])

        counter = counter + 1

    print("Saving processed wav files into dataset")

    save_wav("Data/train/" + file_name + "-input.wav", rate, train_in)
    save_wav("Data/train/" + file_name + "-target.wav", rate, train_tg)

    save_wav("Data/test/" + file_name + "-input.wav", rate, test_in)
    save_wav("Data/test/" + file_name + "-target.wav", rate, test_tg)

    save_wav("Data/val/" + file_name + "-input.wav", rate, val_in)
    save_wav("Data/val/" + file_name + "-target.wav", rate, val_tg)

def conditionedWavParse(args):
    print("Using config file %s" % args.load_config)
    file_name = ""
    configs = miscfuncs.json_load(args.load_config, args.config_location)
    try:
        file_name = configs['file_name']
    except KeyError:
        print("Error: config file doesn't have file_name defined")
        exit(1)

    params = configs['params']

    counter = 0
    main_rate = 0
    all_train_in = np.array([[]]*(1 + params['n']), dtype=np.float32) # 1 channel for in audio, n channels per parameters
    all_train_tg = np.array([[]], dtype=np.float32) # 1 channels of all (out audio)
    all_test_in = np.array([[]]*(1 + params['n']), dtype=np.float32) # 1 channel for in audio, n channels per parameters
    all_test_tg = np.array([[]], dtype=np.float32) # 1 channels of all (out audio)
    all_val_in = np.array([[]]*(1 + params['n']), dtype=np.float32) # 1 channel for in audio, n channels per parameters
    all_val_tg = np.array([[]], dtype=np.float32) # 1 channels of all (out audio)

    for entry in params['datasets']:
        print("Input file name: %s" % entry['input'])
        in_rate, in_data = wavfile.read(entry['input'])
        print("Target file name: %s" % entry['target'])
        tg_rate, tg_data = wavfile.read(entry['target'])

        print("Input rate: %d length: %d [samples]" % (in_rate, in_data.size))
        print("Target rate: %d length: %d [samples]" % (tg_rate, tg_data.size))

        if in_rate != tg_rate:
            print("Error! Sample rate needs to be equal")
            exit(1)
        else:
            rate = in_rate

        # First wav file sets the rate
        if counter == 0:
            main_rate = rate

        if rate != main_rate:
            print("Error: all the wav files needs to have the same format and rate")
            exit(1)

        min_size = in_data.size
        if(in_data.size != tg_data.size):
            min_size = min(in_data.size, tg_data.size)
            print("Warning! Length for audio files\n\r  %s\n\r  %s\n\rdoes not match, setting both to %d [samples]" % (entry['input'], entry['target'], min_size))
            _in_data = np.resize(in_data, min_size)
            _tg_data = np.resize(tg_data, min_size)
            in_data = _in_data
            tg_data = _tg_data
            del _in_data
            del _tg_data

        x_all = audio_converter(in_data)
        y_all = audio_converter(tg_data)

        # @TODO: auto-align code goes here

        # Default to 70% 15% 15% split
        if not args.csv_file:
            splitted_x = audio_splitter(x_all, [0.70, 0.15, 0.15])
            splitted_y = audio_splitter(y_all, [0.70, 0.15, 0.15])
        else:
            # Csv file to be named as in file
            [train_bounds, test_bounds, val_bounds] = parse_csv(os.path.splitext(entry['input'])[0] + ".csv")
            splitted_x = [np.ndarray([0], dtype=np.float32), np.ndarray([0], dtype=np.float32), np.ndarray([0], dtype=np.float32)]
            splitted_y = [np.ndarray([0], dtype=np.float32), np.ndarray([0], dtype=np.float32), np.ndarray([0], dtype=np.float32)]
            for bounds in train_bounds:
                splitted_x[0] = np.append(splitted_x[0], audio_splitter(x_all, bounds, unit='s'))
                splitted_y[0] = np.append(splitted_y[0], audio_splitter(y_all, bounds, unit='s'))
            for bounds in test_bounds:
                splitted_x[1] = np.append(splitted_x[1], audio_splitter(x_all, bounds, unit='s'))
                splitted_y[1] = np.append(splitted_y[1], audio_splitter(y_all, bounds, unit='s'))
            for bounds in val_bounds:
                splitted_x[2] = np.append(splitted_x[2], audio_splitter(x_all, bounds, unit='s'))
                splitted_y[2] = np.append(splitted_y[2], audio_splitter(y_all, bounds, unit='s'))

        # Initialize lists to handle the number of parameters
        params_train = []
        params_val = []
        params_test = []

        # Create a list of np arrays of the parameter values
        for val in entry["params"]:
            # Create the parameter arrays
            params_train.append(np.array([val]*len(splitted_x[0]), dtype=np.float32))
            params_test.append(np.array([val]*len(splitted_x[1]), dtype=np.float32))
            params_val.append(np.array([val]*len(splitted_x[2]), dtype=np.float32))

        # Convert the lists to numpy arrays
        params_train = np.array(params_train, dtype=np.float32)
        params_val = np.array(params_val, dtype=np.float32)
        params_test = np.array(params_test, dtype=np.float32)

        # Append the audio and paramters to the full data sets
        all_train_in = np.append(all_train_in, np.append([splitted_x[0]], params_train, axis=0), axis = 1)
        all_train_tg = np.append(all_train_tg, splitted_y[0])
        all_test_in = np.append(all_test_in, np.append([splitted_x[1]], params_test, axis=0), axis = 1)
        all_test_tg = np.append(all_test_tg, splitted_y[1])
        all_val_in = np.append(all_val_in, np.append([splitted_x[2]], params_val, axis=0), axis = 1)
        all_val_tg = np.append(all_val_tg, splitted_y[2])

        counter = counter + 1

    # Save the wav files
    save_wav("Data/train/" + file_name + "-input.wav", rate, all_train_in.T, flatten=False)
    save_wav("Data/test/" + file_name + "-input.wav", rate, all_test_in.T, flatten=False)
    save_wav("Data/val/" + file_name + "-input.wav", rate, all_val_in.T, flatten=False)

    save_wav("Data/train/" + file_name + "-target.wav", rate, all_train_tg)
    save_wav("Data/test/" + file_name + "-target.wav", rate, all_test_tg)
    save_wav("Data/val/" + file_name + "-target.wav", rate, all_val_tg)

def main(args):
    if args.files:
        if (len(args.files) % 2) and not args.parameterize:
            print("Error: you should provide arguments in pairs see help")
            exit(1)

    if args.parameterize:
        conditionedWavParse(args)
    else:
        conditionedWavParse(args)

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', '-f', nargs='+', help='provide input target files in pairs e.g. guitar_in.wav guitar_tg.wav bass_in.wav bass_tg.wav')
    parser.add_argument('--load_config', '-l',
                  help="File path, to a JSON config file, arguments listed in the config file will replace the defaults", default='RNN-aidadsp-1')
    parser.add_argument('--csv_file', '-csv', action=argparse.BooleanOptionalAction, default=False, help='Use csv file for split bounds')
    parser.add_argument('--config_location', '-cl', default='Configs', help='Location of the "Configs" directory')
    parser.add_argument('--parameterize', '-p', action=argparse.BooleanOptionalAction, default=False, help='Perform parameterized training')

    args = parser.parse_args()
    main(args)
