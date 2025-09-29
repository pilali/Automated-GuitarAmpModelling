import CoreAudioML.miscfuncs as miscfuncs
import CoreAudioML.training as training
import CoreAudioML.dataset as dataset
import CoreAudioML.networks as networks
import argparse
from scipy.io.wavfile import write
import torch
import torch.nn as nn
import time

import json

from auraloss.perceptual import FIRFilter

# Example
# python3 proc_audio.py -l Results/2023-10-05-16:38:04_fnd-twin-rev-aidadsp/model_best.json -i Data/test/aidadsp-auto-input.wav -t Data/test/aidadsp-auto-target.wav -o ./proc.wav -sp

def parse_args():
    parser = argparse.ArgumentParser(
        description='''This script takes an input .wav file, loads it and processes it with a neural network model of a
                    device, i.e guitar amp/pedal, and saves the output as a .wav file. Optionally it can calculate the ESR
                    over a provided target .wav file, so that you can see how good the prediction was.''')
    parser.add_argument('--load_model', '-l', help="Json model file at the end of the training", default='model.json')
    parser.add_argument('--data_location', '-dl', default='./Data', help='Location of the "Data" directory')
    parser.add_argument('--input_file', '-i', default='', help='Location of the input file')
    parser.add_argument('--target_file', '-t', default='', help='Location of the target file')
    parser.add_argument('--output_file', '-o', default='', help='Location of the output file')
    parser.add_argument('--spectrogram', '-sp', action=argparse.BooleanOptionalAction, default=False, help='Create spectrogram')
    parser.add_argument('--start', '-s', type=int, default=-1, help='Start point expressed in samples')
    parser.add_argument('--end', '-e', type=int, default=-1, help='End point expressed in samples')
    parser.add_argument('--filter', '-f', default='', help='Filter type to apply to the output')
    parser.add_argument('--params_values', '-pv', type=str, default=None, help='Comma-separated string of conditioning parameter values (e.g., "0.5,0.25"). Order must match training.')
    return parser.parse_args()


def proc_audio(args):
    print("Using %s file" % args.load_model)

    # Load network model from config file
    network_data = miscfuncs.json_load(args.load_model)
    network = networks.load_model(network_data)

    # Determine model's expected input size
    if hasattr(network, 'input_size'):
        expected_input_channels = network.input_size
    elif 'model_data' in network_data and 'input_size' in network_data['model_data']:
        expected_input_channels = network_data['model_data']['input_size']
    else:
        # Fallback or error if input_size cannot be determined.
        # This might happen with older model files or if the structure is unexpected.
        # For now, let's try to infer from a common attribute if possible, or raise an error.
        try:
            # Attempt to get it from the first linear layer's in_features if it's a common PyTorch pattern
            # This is a guess and might not be universally applicable.
            first_layer = next(network.modules())
            if hasattr(first_layer, 'in_features'):
                 expected_input_channels = first_layer.in_features
            elif hasattr(first_layer, 'in_channels'): # For Conv layers
                 expected_input_channels = first_layer.in_channels
            else:
                raise ValueError("Could not determine model's expected input_size from network object or network_data.")
        except StopIteration: # No modules in network
            raise ValueError("Could not determine model's expected input_size. Network has no modules.")
        print(f"Warning: Could not definitively determine 'input_size' from model file. Inferred {expected_input_channels} channels. If this is incorrect, processing may fail or produce unexpected results.")


    data = dataset.DataSet(data_dir='', extensions='')
    data.create_subset('input')
    data.load_file(args.input_file, set_names='input')

    audio_data_tensor = data.subsets['input'].data['data'][0] # Shape: (sequence_length, 1, num_audio_channels_in_file)
    num_audio_channels_in_file = audio_data_tensor.shape[2]
    final_input_tensor = audio_data_tensor

    if args.params_values:
        try:
            param_floats = [float(p.strip()) for p in args.params_values.split(',')]
        except ValueError as e:
            raise ValueError(f"Error parsing --params_values '{args.params_values}'. Ensure it's a comma-separated list of numbers. Original error: {e}")

        num_provided_params = len(param_floats)
        num_expected_conditioning_params = expected_input_channels - num_audio_channels_in_file

        if num_audio_channels_in_file > 1:
            print(f"Warning: --params_values provided with a multi-channel input file ({num_audio_channels_in_file} channels). Parameters will be appended. If the input file already contains parameters, this may lead to unexpected behavior or errors if the total channel count exceeds the model's input_size ({expected_input_channels}).")

        if num_expected_conditioning_params < 0:
             raise ValueError(f"Model expects {expected_input_channels} total input channels, but the input file '{args.input_file}' already has {num_audio_channels_in_file} channels. Cannot add conditioning parameters.")

        if num_provided_params != num_expected_conditioning_params:
            raise ValueError(f"Parameter mismatch: Model expects {num_expected_conditioning_params} conditioning parameters (total input_size: {expected_input_channels}, audio channels in file: {num_audio_channels_in_file}), but {num_provided_params} were provided via --params_values.")

        param_tensors_to_cat = []
        if num_expected_conditioning_params > 0: # only create tensors if they are expected
            for p_val in param_floats:
                # Shape of audio_data_tensor is (sequence_length, 1, num_audio_channels_in_file)
                # We want to create (sequence_length, 1, 1) for each parameter
                param_tensor = torch.full_like(audio_data_tensor[:, :, 0:1], p_val, device=audio_data_tensor.device, dtype=audio_data_tensor.dtype)
                param_tensors_to_cat.append(param_tensor)

            if param_tensors_to_cat:
                final_input_tensor = torch.cat([audio_data_tensor] + param_tensors_to_cat, dim=2)
    else: # No args.params_values
        if final_input_tensor.shape[2] != expected_input_channels:
            print(f"Warning: Model expects {expected_input_channels} input channels, but the input file has {final_input_tensor.shape[2]} channels and no --params_values were given. Ensure the input file is correctly formatted for the model.")

    # Final check on total channels
    if final_input_tensor.shape[2] != expected_input_channels:
        raise ValueError(f"Mismatch: Final input tensor has {final_input_tensor.shape[2]} channels, but model '{args.load_model}' expects {expected_input_channels} channels. Check your --input_file and --params_values.")

    if args.target_file:
        data.create_subset('target')
        data.load_file(args.target_file, set_names='target')
        lossESR = training.ESRLoss()
        lossDC = training.DCLoss()

    if args.start < 0:
        args.start = 0
    if args.end < 0:
        args.end = int(list(final_input_tensor.size())[0])

    print("Using start = %s, end = %s" % (str(args.start), str(args.end)))

    # Ensure model and data are on the same device
    # Assuming network has its parameters on some device, e.g., network.fc.weight.device
    # A more general way if fc layer isn't guaranteed: next(network.parameters()).device
    try:
        model_device = next(network.parameters()).device
    except StopIteration: # Model has no parameters
        model_device = torch.device("cpu") # Default to CPU if no parameters
        print("Warning: Model has no parameters. Assuming CPU device for input tensor.")

    final_input_tensor = final_input_tensor.to(model_device)

    with torch.no_grad():
        output = network(final_input_tensor[args.start:args.end])

    if args.filter == 'lp':
        a1 = (8.5e+03 * 2 * 3.1416) / data.subsets['input'].fs
        filt = FIRFilter(filter_type='hp', coef=-a1, fs=data.subsets['input'].fs, ntaps=3) # Note: hp with -coef = lp see Auraloss impl.
        input = _ = output.permute(1, 2, 0)
        input, _ = filt(input, _)
        output = input.permute(2, 0, 1).div(1 + a1)

    if args.filter == 'lp':
        a1 = (8.5e+03 * 2 * 3.1416) / data.subsets['input'].fs
        filt = FIRFilter(filter_type='hp', coef=-a1, fs=data.subsets['input'].fs, ntaps=3) # Note: hp with -coef = lp see Auraloss impl.
        input = _ = output.permute(1, 2, 0)
        input, _ = filt(input, _)
        output = input.permute(2, 0, 1).div(1 + a1)

    if args.target_file:
        test_loss_ESR = lossESR(output, data.subsets['target'].data['data'][0][args.start:args.end])
        test_loss_ESR_p = lossESR(output, data.subsets['target'].data['data'][0][args.start:args.end], pooling=True)
        test_loss_DC = lossDC(output, data.subsets['target'].data['data'][0][args.start:args.end])
        write(args.output_file.rsplit('.', 1)[0]+'_ESR.wav', data.subsets['input'].fs, test_loss_ESR_p.cpu().numpy()[:, 0, 0])
        write(args.output_file.rsplit('.', 1)[0]+'_target.wav', data.subsets['input'].fs, data.subsets['target'].data['data'][0][args.start:args.end].cpu().numpy()[:, 0, 0])
        print("test_loss_ESR = %.6f test_loss_DC = %.6f" % (test_loss_ESR.item(), test_loss_DC.item()))

    if args.spectrogram:
        import matplotlib.pyplot as plt
        from colab_functions import smoothed_spectrogram, gen_smoothed_spectrogram_plot
        f, y1, min_, max_ = smoothed_spectrogram(output.numpy()[:, 0, 0], fs=data.subsets['input'].fs, size=4096)
        if args.target_file:
            f, y2, min_, max_ = smoothed_spectrogram(data.subsets['target'].data['data'][0][args.start:args.end].numpy()[:, 0, 0], fs=data.subsets['input'].fs, size=4096)
            gen_smoothed_spectrogram_plot(f, target=y2, predicted=y1, title="Peak Spectrogram").savefig('spectrogram.png')
        else:
            gen_smoothed_spectrogram_plot(f, target=None, predicted=y1, title="Peak Spectrogram").savefig('spectrogram.png')

    # Output is in this format tuple(tensor, tuple(tensor, tensor))
    write(args.output_file, data.subsets['input'].fs, output.cpu().numpy()[:, 0, 0])


def main():
    args = parse_args()
    print(args)
    proc_audio(args)

if __name__ == '__main__':
    main()