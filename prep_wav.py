from CoreAudioML.dataset import audio_converter, audio_splitter
from scipy.io import wavfile
import argparse

def save_wav(name, rate, data):
    print("Writing %s with rate=%d" % (name, rate))
    wavfile.write(name, rate, data)

def main(args):
    # Load and Preprocess Data ###########################################
    print("Input file name: %s" % args.in_file)
    in_rate, in_data = wavfile.read(args.in_file)
    print("Output file name: %s" % args.out_file)
    out_rate, out_data = wavfile.read(args.out_file)

    if in_rate != out_rate:
        print("Error! Sample rate needs to be equal")
        exit(1)
    else:
        rate = in_rate

    print("Input rate: %d" % in_rate)
    print("Output rate: %d" % out_rate)

    x_all = audio_converter(in_data)
    y_all = audio_converter(out_data)

    print("Splitting audio as following: 0.70 train 0.15 test 0.15 val")
    splitted_x = audio_splitter(x_all, [0.70, 0.15, 0.15])
    splitted_y = audio_splitter(y_all, [0.70, 0.15, 0.15])

    save_wav(args.path + "/train/" + args.name + "-input.wav", rate, splitted_x[0])
    save_wav(args.path + "/train/" + args.name + "-target.wav", rate, splitted_y[0])

    save_wav(args.path + "/test/" + args.name + "-input.wav", rate, splitted_x[1])
    save_wav(args.path + "/test/" + args.name + "-target.wav", rate, splitted_y[1])

    save_wav(args.path + "/val/" + args.name + "-input.wav", rate, splitted_x[2])
    save_wav(args.path + "/val/" + args.name + "-target.wav", rate, splitted_y[2])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file")
    parser.add_argument("out_file")
    parser.add_argument("name")
    parser.add_argument("--path", type=str, default="Data")

    args = parser.parse_args()
    main(args)
