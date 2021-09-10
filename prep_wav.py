from scipy.io import wavfile
import argparse
import numpy as np
from CoreAudioML.dataset import audio_converter, audio_splitter

def save_wav(name, data):
    wavfile.write(name, 44100, data.flatten().astype(np.float32))

def normalize(data):
    data_max = max(data)
    data_min = min(data)
    data_norm = max(data_max,abs(data_min))
    return data / data_norm

def main(args):
    # Load and Preprocess Data ###########################################
    in_rate, in_data = wavfile.read(args.in_file)
    out_rate, out_data = wavfile.read(args.out_file)

    print("Input rate: %d" % in_rate)
    print("Output rate: %d" % out_rate)

    x_all = audio_converter(in_data[1])
    print("Applying normalization to input")
    x_all = normalize(x_all).reshape(len(x_all),1)
    y_all = audio_converter(out_data[1])
    print("Applying normalization to output")
    y_all = normalize(y_all).reshape(len(y_all),1)

    print("Splitting audio as following: 0.70 train 0.15 test 0.15 val")
    splitted_x = audio_splitter(x_all, [0.70, 0.15, 0.15])
    splitted_y = audio_splitter(y_all, [0.70, 0.15, 0.15])

    save_wav(args.path + "/train/" + args.name + "-input.wav", splitted_x[0])
    save_wav(args.path + "/train/" + args.name + "-target.wav", splitted_y[0])

    save_wav(args.path + "/test/" + args.name + "-input.wav", splitted_x[1])
    save_wav(args.path + "/test/" + args.name + "-target.wav", splitted_y[1])

    save_wav(args.path + "/val/" + args.name + "-input.wav", splitted_x[2])
    save_wav(args.path + "/val/" + args.name + "-target.wav", splitted_y[2])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file")
    parser.add_argument("out_file")
    parser.add_argument("name")
    parser.add_argument("--path", type=str, default="Data")

    args = parser.parse_args()
    main(args)
