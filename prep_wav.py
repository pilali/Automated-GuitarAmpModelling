# Creating a valid dataset for the trainining script
# using wav files provided by user.
# Example of usage:
# python3 prep_wav.py -f input.wav target.wav -l "RNN3-aidadsp-1"
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

def save_wav(name, rate, data):
    print("Writing %s with rate=%d" % (name, rate))
    wavfile.write(name, rate, data)

def main(args):
    if (len(args.files) % 2):
        print("Error: you should provide arguments in pairs see help")
        exit(1)

    print("Using config file %s" % args.load_config)
    file_name = ""
    configs = miscfuncs.json_load(args.load_config, args.config_location)
    try:
        file_name = configs['file_name']
    except KeyError:
        print("Error: config file doesn't have file_name defined")
        exit(1)

    print("Splitting audio as following: 0.70 train 0.15 test 0.15 val")
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

        print("Input rate: %d" % in_rate)
        print("Target rate: %d" % tg_rate)

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

        if(in_data.size != tg_data.size):
            min_size = min(in_data.size, tg_data.size)
            print("Length for audio files %s %s does not match, setting both to size %d" % (in_file, tg_file, min_size))
            _in_data = np.resize(in_data, min_size)
            _tg_data = np.resize(tg_data, min_size)
            in_data = _in_data
            tg_data = _tg_data
            del _in_data
            del _tg_data

        x_all = audio_converter(in_data)
        y_all = audio_converter(tg_data)

        splitted_x = audio_splitter(x_all, [0.70, 0.15, 0.15])
        splitted_y = audio_splitter(y_all, [0.70, 0.15, 0.15])

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

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', '-f', nargs='+', help='provide input target files in pairs e.g. guitar_in.wav guitar_tg.wav bass_in.wav bass_tg.wav')
    parser.add_argument('--load_config', '-l',
                  help="File path, to a JSON config file, arguments listed in the config file will replace the defaults", default='RNN3')
    parser.add_argument('--config_location', '-cl', default='Configs', help='Location of the "Configs" directory')

    args = parser.parse_args()
    main(args)
