import argparse
import CoreAudioML.miscfuncs as miscfuncs

import numpy as np
from tensorflow import keras
from model_utils import save_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", default="RNN3")
    parser.add_argument("device", default="aidadsp-1")
    args = parser.parse_args()

    save_path = "Results/" + args.device + "-" + args.name + "-" + args.device

    model_data = miscfuncs.json_load('model_best', save_path)

    try:
        unit_type = model_data['model_data']['unit_type']
        input_size = model_data['model_data']['input_size']
        hidden_size = model_data['model_data']['hidden_size']
        output_size = model_data['model_data']['output_size']
        WVals = np.array(model_data['state_dict']['rec.weight_ih_l0'])
        UVals = np.array(model_data['state_dict']['rec.weight_hh_l0'])
        bias_ih_l0 =  model_data['state_dict']['rec.bias_ih_l0']
        bias_hh_l0 = model_data['state_dict']['rec.bias_hh_l0']
        array_bias_ih_l0 = np.array(bias_ih_l0)
        array_bias_hh_l0 = np.array(bias_hh_l0)
        BVals = (array_bias_ih_l0 + array_bias_hh_l0)
        lin_weight = np.array(model_data['state_dict']['lin.weight'])
        lin_bias = np.array(model_data['state_dict']['lin.bias'])
    except KeyError:
        print("Model file %s is corrupted" % (save_path + "/model.json"))

    # construct TensorFlow model
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(None, input_size)))

    if unit_type == "LSTM":
        lstm_weights = []
        lstm_weights.append(WVals.reshape(input_size, hidden_size*4)) # WVals is (hidden_size*4, input_size)
        lstm_weights.append(UVals.reshape(hidden_size, hidden_size*4)) # UVals is (hidden_size*4, hidden_size)
        lstm_weights.append(BVals) # BVals is (hidden_size*4, )
        lstm_layer = keras.layers.LSTM(hidden_size, activation=None, weights=lstm_weights, return_sequences=True, recurrent_activation=None, use_bias=True, kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal", bias_initializer="random_normal", unit_forget_bias=False)
        model.add(lstm_layer)

    elif unit_type == "GRU":
        print("Still need to implement GRU")
        exit(1)
    else:
        print("Cannot parse unit_type = %s" % unit_type)
        exit(1)

    dense_weights = []
    dense_weights.append(lin_weight.reshape(hidden_size, 1)) # lin_weight is (1, hidden_size)
    dense_weights.append(lin_bias) # lin_bias is (1,)
    dense_layer = keras.layers.Dense(1, weights=dense_weights, kernel_initializer="orthogonal", bias_initializer='random_normal')
    model.add(dense_layer)

    save_model(model, save_path + "/model_best_keras.json")