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
import CoreAudioML.training as training
import CoreAudioML.dataset as CAMLdataset
import CoreAudioML.networks as networks
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from scipy.io.wavfile import write
from scipy.io import wavfile
import numpy as np
import random
import torch
import time
import os
import csv
import librosa
import json
import argparse


def wav2tensor(filepath):
  aud, sr = librosa.load(filepath, sr=None, mono=True)
  aud = librosa.resample(aud, orig_sr=sr, target_sr=48000)
  return torch.tensor(aud)


def extract_best_esr_model(dirpath):
  stats_file = dirpath + "/training_stats.json"
  with open(stats_file) as json_file:
    stats_data = json.load(json_file)
    test_lossESR_final = stats_data['test_lossESR_final']
    test_lossESR_best = stats_data['test_lossESR_best']
    esr = min(test_lossESR_final, test_lossESR_best)
    if esr == test_lossESR_final:
      model_path = dirpath + "/model.json"
    else:
      model_path = dirpath + "/model_best.json"
  return model_path, esr


def is_ref_input(input_data):
    ref = np.load("input_ref.npz")['ref']
    if (input_data[:48000] - ref).sum()==0:
        return True
    return False


_V1_BLIP_LOCATIONS = 12_000, 36_000
def align_target(tg_data):
    """
    Based on _calibrate_delay_v1 from https://github.com/sdatkinson/neural-amp-modeler/blob/413d031b92e011ec0b3e6ab3b865b8632725a219/nam/train/core.py#L60
    Copyright (c) 2022 Steven Atkinson
    SPDX - License - Identifier: MIT
    """
    lookahead = 1_000
    lookback = 10_000
    safety_factor = 4

    # Calibrate the trigger:
    y = tg_data[:48_000]
    background_level = np.max(np.abs(y[:6_000]))
    background_avg = np.mean(np.abs(y[:6_000]))
    trigger_threshold = max(background_level + 0.01, 1.01 * background_level)

    delays = []
    for blip_index, i in enumerate(_V1_BLIP_LOCATIONS, 1):

        start_looking = i - lookahead
        stop_looking = i + lookback
        y_scan = y[start_looking:stop_looking]
        triggered = np.where(np.abs(y_scan) > trigger_threshold)[0]
        if len(triggered) == 0:
            return None
        else:
            j = triggered[0]
            delays.append(j + start_looking - i)

    delay = int(np.min(delays)) - safety_factor
    if delay<0:
        return np.concatenate((np.zeros(abs(delay)), tg_data))
    return tg_data[delay:]

def init_model(save_path, load_model, unit_type, input_size, hidden_size, output_size, skip_con):
    # Search for an existing model in the save directory
    if miscfuncs.file_check('model.json', save_path) and load_model:
        print('existing model file found, loading network')
        model_data = miscfuncs.json_load('model', save_path)
        # assertions to check that the model.json file is for the right neural network architecture
        try:
            assert model_data['model_data']['unit_type'] == unit_type
            assert model_data['model_data']['input_size'] == input_size
            assert model_data['model_data']['hidden_size'] == hidden_size
            assert model_data['model_data']['output_size'] == output_size
        except AssertionError:
            print("model file found with network structure not matching config file structure")
        network = networks.load_model(model_data)
    # If no existing model is found, create a new one
    else:
        print('no saved model found, creating new network')
        network = networks.SimpleRNN(input_size=input_size, unit_type=unit_type, hidden_size=hidden_size,
                                     output_size=output_size, skip=skip_con)
        network.save_state = False
        network.save_model('model', save_path)
    return network


def save_wav(name, rate, data):
    # print("Writing %s with rate: %d length: %d dtype: %s" % (name, rate, data.size, data.dtype))
    wavfile.write(name, rate, data)

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

def prep_audio(files, file_name, csv_file=False, data_split_ratio=[.7, .15, .15]):

    # configs = miscfuncs.json_load(load_config, config_location)
    # configs['file_name'] = file_name

    counter = 0
    main_rate = 0
    train_in = np.ndarray([0], dtype=np.float32)
    train_tg = np.ndarray([0], dtype=np.float32)
    test_in = np.ndarray([0], dtype=np.float32)
    test_tg = np.ndarray([0], dtype=np.float32)
    val_in = np.ndarray([0], dtype=np.float32)
    val_tg = np.ndarray([0], dtype=np.float32)
    for in_file, tg_file in zip(files[::2], files[1::2]):
        print("Input file name: %s" % in_file)
        in_data, in_rate = librosa.load(in_file, sr=None, mono=True)
        in_file_base = os.path.basename(in_file)
        print("Target file name: %s" % tg_file)
        tg_data, tg_rate = librosa.load(tg_file, sr=None, mono=True)
        tg_file_base = os.path.basename(tg_file)

        print("Input rate: %d length: %d [samples]" % (in_rate, in_data.size))
        print("Target rate: %d length: %d [samples]" % (tg_rate, tg_data.size))

        if in_rate != tg_rate:
            print("Error! Sample rate needs to be equal")
            exit(1)
        else:
            rate = in_rate

        if in_rate != 48000:
            print("Converting audio sample rate to 48kHz.")
            in_data = librosa.resample(in_data, orig_sr=in_rate, target_sr=48000)
            in_rate = 48000
            tg_data = librosa.resample(tg_data, orig_sr=tg_data, target_sr=48000)
            in_rate = 48000

        # First wav file sets the rate
        if counter == 0:
            main_rate = rate

        if rate != main_rate:
            print("Error: all the wav files needs to have the same format and rate")
            exit(1)

        if is_ref_input(in_data):
            tg_data = align_target(tg_data)

        if(in_data.size != tg_data.size):
            min_size = min(in_data.size, tg_data.size)
            print("Warning! Length for audio files\n\r  %s\n\r  %s\n\rdoes not match, setting both to %d [samples]" % (in_file, tg_file, min_size))
            _in_data = np.resize(in_data, min_size)
            _tg_data = np.resize(tg_data, min_size)
            in_data = _in_data
            tg_data = _tg_data
            del _in_data
            del _tg_data

        print("Preprocessing the training data...")

        x_all = audio_converter(in_data)
        y_all = audio_converter(tg_data)

        # Default to 70% 15% 15% split
        if not csv_file:
            splitted_x = audio_splitter(x_all, data_split_ratio)
            splitted_y = audio_splitter(y_all, data_split_ratio)
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

    # print("Saving processed wav files into dataset")

    save_wav("Data/train/" + file_name + "-input.wav", rate, train_in)
    save_wav("Data/train/" + file_name + "-target.wav", rate, train_tg)

    save_wav("Data/test/" + file_name + "-input.wav", rate, test_in)
    save_wav("Data/test/" + file_name + "-target.wav", rate, test_tg)

    save_wav("Data/val/" + file_name + "-input.wav", rate, val_in)
    save_wav("Data/val/" + file_name + "-target.wav", rate, val_tg)

    # print("Done!")



if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--files', '-f', nargs='+', help='provide input target files in pairs e.g. guitar_in.wav guitar_tg.wav bass_in.wav bass_tg.wav')
    # parser.add_argument('--load_config', '-l',
    #               help="File path, to a JSON config file, arguments listed in the config file will replace the defaults", default='RNN-aidadsp-1')
    # parser.add_argument('--csv_file', '-csv', action=argparse.BooleanOptionalAction, default=False, help='Use csv file for split bounds')
    # parser.add_argument('--config_location', '-cl', default='Configs', help='Location of the "Configs" directory')
    prep_audio(["D:\\MOD\\Automated-GuitarAmpModelling\\Data\\alignment\\input.wav", "D:\\MOD\\Automated-GuitarAmpModelling\\Data\\alignment\\mic raw.wav"], "testfile")
    # train_routine(load_config="RNN-aidadsp-1", segment_length=24000, seed=39, )
