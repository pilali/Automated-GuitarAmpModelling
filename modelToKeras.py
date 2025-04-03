import argparse
import json
import numpy as np
import os

def save_model_dict(model_dict, filename, skip=0, input_batch=None, output_batch=None, metadata=None, verbose=True):
    """
    Save the model dictionary in the expected .aidax format.

    Args:
        model_dict (dict): The model dictionary containing layers and weights.
        filename (str): The output file path.
        skip (int): Number of input elements skipped during training.
        input_batch (list): Input batch data.
        output_batch (list): Output batch data.
        metadata (dict): Additional metadata for the model.
        verbose (bool): Whether to print debug information.
    """
    # Add the "in_skip" field if skip is greater than 0
    if skip > 0:
        model_dict["in_skip"] = skip

    # Add input and output batch data if provided
    if input_batch is not None:
        model_dict["input_batch"] = input_batch
    if output_batch is not None:
        model_dict["output_batch"] = output_batch

    # Add metadata if provided
    if metadata is not None:
        model_dict["metadata"] = metadata

    # Ensure each layer has a "shape" field
    for layer in model_dict["layers"]:
        if "weights" in layer:
            if layer["type"] == "dense":
                layer["shape"] = [None, None, len(layer["weights"][1])]
            elif layer["type"] in ["gru", "lstm"]:
                layer["shape"] = [None, None, layer["hidden_size"]]
            else:
                layer["shape"] = [None, None, None]  # Default shape if unknown

        # Add an empty "activation" field if not already present
        if "activation" not in layer:
            layer["activation"] = ""

    # Save the model dictionary to a JSON file
    with open(filename, 'w') as outfile:
        json.dump(model_dict, outfile, indent=4)

    if verbose:
        print(f"Model saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', '-lm', help="Json model as exported from Pytorch", default='')
    parser.add_argument('--load_config', '-l', help="Json config file describing the nn and the dataset", default='LSTM-12')
    parser.add_argument('--results_path', '-rp', help="Directory of the resulting model", default='None')
    parser.add_argument('--config_location', '-cl', help='Location of the "Configs" directory', default='Configs')
    parser.add_argument('--aidax', '-ax', action=argparse.BooleanOptionalAction, help='The output file extension will be .aidax', default=False)
    args = parser.parse_args()

    if not args.load_model:
        # Open config file
        config = args.config_location + "/" + args.load_config + ".json"
        with open(config) as json_file:
            config_data = json.load(json_file)
            device = config_data['device']
            unit_type = config_data['unit_type']
            hidden_size = config_data['hidden_size']
            num_layers = config_data['num_layers']
            skip = config_data['skip_con']
            metadata = config_data['metadata']

        if args.results_path == "None":
            results_path = "Results/" + device
        else:
            results_path = args.results_path

        # Decide which model to use based on ESR results from
        # training, extract input/output batch consequently
        stats = results_path + "/training_stats.json"
        with open(stats) as json_file:
            data = json.load(json_file)
            test_lossESR_final = data['test_lossESR_final']
            test_lossESR_best = data['test_lossESR_best']
            esr = min(test_lossESR_final, test_lossESR_best)
            input_batch = data['input_batch']
            if esr == test_lossESR_final:
                model = results_path + "/model.json"
                output_batch = data['output_batch_final']
            else:
                model = results_path + "/model_best.json"
                output_batch = data['output_batch_best']
    else:
        model = args.load_model
        results_path = os.path.dirname(args.load_model)

    # Open model file and parse only once params
    with open(model) as json_file:
        model_data = json.load(json_file)
        try:
            model_type = model_data['model_data']['model']
            if model_type != "SimpleRNN":
                print("Error! This model type is still unsupported")
                raise KeyError
            input_size = model_data['model_data']['input_size']
            num_layers = model_data['model_data']['num_layers']
            unit_type = model_data['model_data']['unit_type']
            hidden_size = model_data['model_data']['hidden_size']
            skip = int(model_data['model_data']['skip']) # How many input elements are skipped
            output_size = model_data['model_data']['output_size']
            bias_fl = bool(model_data['model_data']['bias_fl'])
            lin_weight = np.array(model_data['state_dict']['lin.weight'])
            lin_bias = np.array(model_data['state_dict']['lin.bias'])
        except KeyError:
            print("Model file %s is corrupted" % (model))
            exit(1)

    # Construct model dictionary
    model_dict = {"in_shape": [None, None, input_size], "layers": []}

    for num in range(0, num_layers):
        try:
            WVals = np.array(model_data['state_dict']['rec.weight_ih_l%d' % num])
            UVals = np.array(model_data['state_dict']['rec.weight_hh_l%d' % num])
            bias_ih_l0 =  np.array(model_data['state_dict']['rec.bias_ih_l%d' % num])
            bias_hh_l0 = np.array(model_data['state_dict']['rec.bias_hh_l%d' % num])
        except KeyError:
            print("Model file %s is corrupted" % (model))
            exit(1)

        if unit_type == "LSTM":
            lstm_layer = {
                "type": "lstm",
                "weights": [np.transpose(WVals), np.transpose(UVals), bias_ih_l0 + bias_hh_l0],
                "hidden_size": hidden_size,
                "use_bias": bias_fl
            }
            model_dict["layers"].append(lstm_layer)
        elif unit_type == "GRU":
            WVals = np.transpose(WVals)
            UVals = np.transpose(UVals)
            for i, row in enumerate(WVals):
                WVals[i] = np.concatenate((row[hidden_size:hidden_size*2], row[0:hidden_size], row[hidden_size*2:]))
            for i, row in enumerate(UVals):
                UVals[i] = np.concatenate((row[hidden_size:hidden_size*2], row[0:hidden_size], row[hidden_size*2:]))
            BVals = np.zeros((2, hidden_size*3))
            BVals[0] = np.concatenate((bias_ih_l0[hidden_size:hidden_size*2], bias_ih_l0[0:hidden_size], bias_ih_l0[hidden_size*2:]))
            BVals[1] = np.concatenate((bias_hh_l0[hidden_size:hidden_size*2], bias_hh_l0[0:hidden_size], bias_hh_l0[hidden_size*2:]))
            gru_layer = {
                "type": "gru",
                "weights": [WVals, UVals, BVals],
                "hidden_size": hidden_size,
                "use_bias": bias_fl
            }
            model_dict["layers"].append(gru_layer)
        else:
            print("Cannot parse unit_type = %s" % unit_type)
            exit(1)

    dense_layer = {
        "type": "dense",
        "weights": [lin_weight.reshape(hidden_size, 1), lin_bias],
        "output_size": 1
    }
    model_dict["layers"].append(dense_layer)

    # Add RTNeural-specific metadata if required
    model_dict["rtneural_metadata"] = {
        "engine_requirements": "Check RTNeural documentation for specific requirements"
    }

    if not args.load_model:
        metadata['esr'] = esr
    else:
        input_batch = None
        output_batch = None
        metadata = None

    if args.aidax:
        output_model_path = results_path + "/model_keras.aidax"
    else:
        output_model_path = results_path + "/model_keras.json"

    save_model_dict(model_dict, output_model_path, skip=skip, input_batch=input_batch, output_batch=output_batch, metadata=metadata, verbose=False)
