import json
import matplotlib.pyplot as plt
import numpy as np

from scipy.io import wavfile
import sys
from scipy import signal
import argparse
import struct

# @TODO: it seems dc error is not used in CoreAudioML training.py code, investigate
def error_to_signal(y, y_pred, use_filter=1):
    """
    Error to signal ratio with pre-emphasis filter:
    https://www.mdpi.com/2076-3417/10/3/766/htm
    and please check training.py code in CoreAudioML
    """
    epsilon = 0.00001
    if use_filter == 1:
        y, y_pred = pre_emphasis_filter(y), pre_emphasis_filter(y_pred)
    loss = np.mean(np.sum(np.power(y - y_pred, 2))) + epsilon
    energy = np.mean(np.sum(np.power(y, 2))) + epsilon
    return np.divide(loss, energy)

# Filter used in training has transfer function H(z) = 1 - 0.85z^-1
# as reported in the paper
def pre_emphasis_filter(x, coeff=0.85):
    return np.concatenate([x, np.subtract(x, np.multiply(x, coeff))])

def read_wave(wav_file):
    # Extract Audio and framerate from Wav File
    fs, signal = wavfile.read(wav_file)
    return signal, fs

def analyze_pred_vs_actual(input_wav, output_wav, pred_wav, model_name, show_plots):
    """Generate plots to analyze the predicted signal vs the actual
    signal.
    Inputs:
        input_wav : The pre effect signal
        output_wav : The actual signal
        pred_wav : The predicted signal
        model_name : Used to add the model name to the plot .png filename
        show_plots : Default is 1 to show plots, 0 to only generate .png files and suppress plots
    1. Plots the two signals
    2. Calculates Error to signal ratio the same way Pedalnet evauluates the model for training
    3. Plots the absolute value of pred_signal - actual_signal  (to visualize abs error over time)
    4. Plots the spectrogram of (pred_signal - actual signal)
         The idea here is to show problem frequencies from the model training
    """
    path = result_dir

    # Read the input wav file
    signal_in, fs_in = read_wave(input_wav)
    # Read the output wav file
    signal_out, fs_out = read_wave(output_wav)

    Time = np.linspace(0, len(signal_out) / fs_out, num=len(signal_out))
    fig, (ax3, ax1, ax2) = plt.subplots(3, sharex=True, figsize=(13, 8))
    fig.suptitle("Predicted vs Actual Signal")
    ax1.plot(Time, signal_out, label=output_wav, color="red")

    # Read the predicted wav file
    signal_pred, fs_pred = read_wave(pred_wav)

    Time2 = np.linspace(0, len(signal_pred) / fs_pred, num=len(signal_pred))
    ax1.plot(Time2, signal_pred, label=pred_wav, color="green")
    ax1.legend(loc="upper right")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Wav File Comparison")
    ax1.grid("on")

    # Calculate error to signal ratio with pre-emphasis filter as
    #    used to train the model
    e2s = error_to_signal(signal_out, signal_pred)
    e2s_no_filter = error_to_signal(signal_out, signal_pred, use_filter=0)

    print("Error to signal (with pre-emphasis filter): ", e2s)
    print("Error to signal (no pre-emphasis filter): ", e2s_no_filter)
    fig.suptitle("Predicted vs Actual Signal (error to signal: " + str(round(e2s, 4)) + ")")
    # Plot signal difference
    signal_diff = np.absolute(np.subtract(signal_pred, signal_out))
    ax2.plot(Time2, signal_diff, label="signal diff", color="blue")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("abs(pred_signal-actual_signal)")
    ax2.grid("on")

    # Plot the original signal
    Time3 = np.linspace(0, len(signal_in) / fs_in, num=len(signal_in))
    ax3.plot(Time3, signal_in, label=input_wav, color="purple")
    ax3.legend(loc="upper right")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Amplitude")
    ax3.set_title("Original Input")
    ax3.grid("on")

    # Save the plot
    plt.savefig(path+'/'+model_name + "_signal_comparison_e2s_" + str(round(e2s, 4)) + ".png", bbox_inches="tight")

    # Create a zoomed in plot of 0.01 seconds centered at the max input signal value
    sig_temp = signal_out.tolist()
    plt.axis(
        [
            Time3[sig_temp.index((max(sig_temp)))] - 0.005,
            Time3[sig_temp.index((max(sig_temp)))] + 0.005,
            min(signal_pred),
            max(signal_pred),
        ]
    )
    plt.savefig(path+'/'+model_name + "_detail_signal_comparison_e2s_" + str(round(e2s, 4)) + ".png", bbox_inches="tight")

    # Reset the axis
    plt.axis([0, Time3[-1], min(signal_pred), max(signal_pred)])

    # Plot spectrogram difference
    plt.figure(figsize=(12, 8))
    print("Creating spectrogram data..")
    frequencies, times, spectrogram = signal.spectrogram(signal_diff, fs_in)
    plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram))
    plt.colorbar()
    plt.title("Diff Spectrogram")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.savefig(path+'/'+model_name + "_diff_spectrogram.png", bbox_inches="tight")

    if show_plots == 1:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_config', '-l', help="Json config file describing the nn and the dataset", default='RNN-aidadsp-1')
    parser.add_argument('--config_location', '-cl', default='Configs', help='Location of the "Configs" directory')
    parser.add_argument("--show_plots", default=1)
    args = parser.parse_args()

    # Open config file
    config = args.config_location + "/" + args.load_config + ".json"
    with open(config) as json_file:
        config_data = json.load(json_file)
        device = config_data['device']

    result_dir = "Results/" + device + "-" + args.load_config

    show_plots = args.show_plots

    # Create graphs on validation data
    input_wav = "Data/val/" + device + "-input.wav"
    output_wav = "Data/val/" + device + "-target.wav"
    pred_wav = result_dir + "/best_val_out.wav"
    model_name = device + "_validation"
    analyze_pred_vs_actual(input_wav, output_wav, pred_wav, model_name, show_plots)

    # Decide which model to use based on ESR results from
    # training
    stats = result_dir + "/training_stats.json"
    with open(stats) as json_file:
        data = json.load(json_file)
        test_lossESR_final = data['test_lossESR_final']
        test_lossESR_best = data['test_lossESR_best']
        tmp = min(test_lossESR_final, test_lossESR_best)
        if tmp == test_lossESR_final:
            pred_wav = result_dir + "/test_out_final.wav"
        else:
            pred_wav = result_dir + "/test_out_best.wav"

    # Create graphs on test data
    input_wav = "Data/test/" + device + "-input.wav"
    output_wav = "Data/test/" + device + "-target.wav"
    model_name = device + "_test"
    analyze_pred_vs_actual(input_wav, output_wav, pred_wav, model_name, show_plots)
