import argparse
import os
import json
import numpy as np
from tensorflow import keras
from model_utils import save_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputfile', '-if')
    parser.add_argument('--outputfile', '-of')
    args = parser.parse_args()
    model = args.inputfile
    out_path = args.outputfile
    model_name = os.path.split(model)[-1]

    print("Reading model...")
    # Open model file
    with open(model) as json_file:
        model_data = json.load(json_file)
        try:
            unit_type = model_data['model_data']['unit_type']
            input_size = model_data['model_data']['input_size']
            skip = int(model_data['model_data']['skip']) # How many input elements are skipped
            hidden_size = model_data['model_data']['hidden_size']
            output_size = model_data['model_data']['output_size']
            bias_fl = bool(model_data['model_data']['bias_fl'])
            WVals = np.array(model_data['state_dict']['rec.weight_ih_l0'])
            UVals = np.array(model_data['state_dict']['rec.weight_hh_l0'])
            bias_ih_l0 =  model_data['state_dict']['rec.bias_ih_l0']
            bias_hh_l0 = model_data['state_dict']['rec.bias_hh_l0']
            lin_weight = np.array(model_data['state_dict']['lin.weight'])
            lin_bias = np.array(model_data['state_dict']['lin.bias'])
        except KeyError:
            print("Model file %s is corrupted" % (model))

    # construct TensorFlow model
    print("Reconstructing model...")
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(None, input_size)))

    if unit_type == "LSTM":
        lstm_weights = []
        lstm_weights.append(np.transpose(WVals))
        lstm_weights.append(np.transpose(UVals))
        array_bias_ih_l0 = np.array(bias_ih_l0)
        array_bias_hh_l0 = np.array(bias_hh_l0)
        BVals = (array_bias_ih_l0 + array_bias_hh_l0)
        lstm_weights.append(BVals) # BVals is (hidden_size*4, )
        lstm_layer = keras.layers.LSTM(hidden_size, activation=None, weights=lstm_weights, return_sequences=True, recurrent_activation=None, use_bias=bias_fl, unit_forget_bias=False)
        model.add(lstm_layer)
    elif unit_type == "GRU":
        gru_weights = []
        gru_weights.append(np.transpose(WVals))
        gru_weights.append(np.transpose(UVals))
        array_bias_ih_l0 = np.array(bias_ih_l0)
        array_bias_hh_l0 = np.array(bias_hh_l0)
        tmp = np.zeros((2, hidden_size*3))
        tmp[0] = np.transpose(array_bias_hh_l0)
        tmp[1] = np.transpose(array_bias_ih_l0)
        BVals = tmp
        gru_weights.append(BVals) # BVals is (2, hidden_size*3)
        gru_layer = keras.layers.GRU(hidden_size, activation=None, weights=gru_weights, return_sequences=True, recurrent_activation=None, use_bias=bias_fl)
        model.add(gru_layer)
    else:
        print("Cannot parse unit_type = %s" % unit_type)
        exit(1)

    dense_weights = []
    dense_weights.append(lin_weight.reshape(hidden_size, 1)) # lin_weight is (1, hidden_size)
    dense_weights.append(lin_bias) # lin_bias is (1,)
    dense_layer = keras.layers.Dense(1, weights=dense_weights, kernel_initializer="orthogonal", bias_initializer='random_normal')
    model.add(dense_layer)

    # Using save_model method from model_utils module from RTNeural project
    print("Exporting model...")
    save_model(model, out_path, keras.layers.InputLayer, skip=skip, verbose=False)