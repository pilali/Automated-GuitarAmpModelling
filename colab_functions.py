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


def steps_check(step):
    return "WIP"


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


def train_routine(load_config="RNN-aidadsp-1", config_location="Configs", segment_length=24000, epochs=2000,
                  validation_f=2, validation_p=25, batch_size=50, iter_num=None, learn_rate=.005, init_len=200,
                  up_fr=1000, cuda=1, loss_fcns={'ESRPre': 0.75, 'DC': 0.25}, pre_filt='high_pass',
                  val_chunk=100000, test_chunk=100000, model='SimpleRNN', input_size=1, output_size=1, num_blocks=1,
                  hidden_size=16, unit_type='LSTM', skip_con=1,
                  device="ht1", data_location='./Data', file_name="ht1", save_location="Results", load_model=0, seed=None,):

    args = {"load_config": load_config, "config_location": config_location, "segment_length": segment_length,
            "epochs": epochs, "validation_f": validation_f, "validation_p": validation_p,
            "batch_size": batch_size, "iter_num": iter_num, "learn_rate": learn_rate, "init_len": init_len,
            "up_fr": up_fr, "cuda": cuda, "loss_fcns": loss_fcns, "pre_filt": pre_filt,
            "val_chunk": val_chunk, "test_chunk": test_chunk, "model": model, "input_size": input_size,
            "output_size": output_size, "num_blocks": num_blocks, "hidden_size": hidden_size,
            "unit_type": unit_type, "skip_con": skip_con, "device": device, "data_location": data_location,
            "file_name": file_name, "save_location": save_location, "load_model": load_model, "seed": seed
            }

    args = argparse.Namespace(**args)

    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    start_time = time.time()

    # If a load_config argument was provided, construct the file path to the config file
    if args.load_config:
        # Load the configs and write them onto the args dictionary, this will add new args and/or overwrite old ones
        configs = miscfuncs.json_load(args.load_config, args.config_location)
        for parameters in configs:
            args.__setattr__(parameters, configs[parameters])

    if args.model == 'SimpleRNN':
        model_name = args.model + '_' + args.device + '_' + args.unit_type + '_hs' + str(
            args.hidden_size) + '_pre_' + args.pre_filt

    # Fix parameter in case input as argument
    if type(args.loss_fcns) is str:
        args.loss_fcns = eval(args.loss_fcns)

    # It's a good moment to print parameters
    print("")
    print("args.device = %s" % args.device)
    print("args.file_name = %s" % args.file_name)
    print("args.hidden_size = %d" % args.hidden_size)
    print("args.unit_type = %s" % args.unit_type)
    print("args.loss_fcns = %s" % str(args.loss_fcns))
    print("args.skip_con = %d" % args.skip_con)
    print("args.pre_filt = %s" % args.pre_filt)

    if args.pre_filt == 'A-Weighting':
        with open('Configs/' + 'b_Awght.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            args.pre_filt = list(reader)
            args.pre_filt = args.pre_filt[0]
            for item in range(len(args.pre_filt)):
                args.pre_filt[item] = float(args.pre_filt[item])
    elif args.pre_filt == 'high_pass':
        args.pre_filt = [-0.85, 1]
    elif args.pre_filt == 'None':
        args.pre_filt = None

    # Generate name of directory where results will be saved
    save_path = os.path.join(args.save_location, args.device + '-' + 'RNN' + '-' + args.file_name)

    # Check if an existing saved model exists, and load it, otherwise creates a new model
    network = init_model(save_path, args.load_model, args.unit_type, args.input_size, args.hidden_size, args.output_size, args.skip_con)

    # Check if a cuda device is available
    if not torch.cuda.is_available() or args.cuda == 0:
        print('cuda device not available/not selected')
        cuda = 0
    else:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.set_device(0)
        print('cuda device available')
        network = network.cuda()
        cuda = 1

    # Set up training optimiser + scheduler + loss fcns and training info tracker
    optimiser = torch.optim.Adam(network.parameters(), lr=args.learn_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', factor=0.5, patience=5, verbose=True)
    loss_functions = training.LossWrapper(args.loss_fcns, args.pre_filt)
    train_track = training.TrainTrack()
    writer = SummaryWriter(os.path.join('TensorboardData', model_name))

    # Load dataset
    dataset = CAMLdataset.DataSet(data_dir=args.data_location)

    # The train dataset is divided into frames of 0.5 seconds according to the paper. To achieve this
    # 22050 is used as segment_length since sample rate is 44100Hz.
    dataset.create_subset('train', frame_len=args.segment_length)
    dataset.load_file(os.path.join('train', args.file_name), 'train')

    dataset.create_subset('val')
    dataset.load_file(os.path.join('val', args.file_name), 'val')

    # If training is restarting, this will ensure the previously elapsed training time is added to the total
    init_time = time.time() - start_time + train_track['total_time'] * 3600
    # Set network save_state flag to true, so when the save_model method is called the network weights are saved
    network.save_state = True
    patience_counter = 0

    # This is where training happens
    # the network records the last epoch number, so if training is restarted it will start at the correct epoch number
    for epoch in range(train_track['current_epoch'] + 1, args.epochs + 1):
        ep_st_time = time.time()

        # Run 1 epoch of training,
        epoch_loss = network.train_epoch(dataset.subsets['train'].data['input'][0],
                                         dataset.subsets['train'].data['target'][0],
                                         loss_functions, optimiser, args.batch_size, args.init_len, args.up_fr)

        writer.add_scalar('Time/EpochTrainingTime', time.time() - ep_st_time, epoch)

        # Run validation
        if epoch % args.validation_f == 0:
            val_ep_st_time = time.time()
            val_output, val_loss = network.process_data(dataset.subsets['val'].data['input'][0],
                                                        dataset.subsets['val'].data['target'][0], loss_functions,
                                                        args.val_chunk)
            scheduler.step(val_loss)
            if val_loss < train_track['best_val_loss']:
                # print("new best val loss: %f" % val_loss.item())
                patience_counter = 0
                network.save_model('model_best', save_path)
                write(os.path.join(save_path, "best_val_out.wav"),
                      dataset.subsets['val'].fs, val_output.cpu().numpy()[:, 0, 0])
            else:
                patience_counter += 1
            train_track.val_epoch_update(val_loss.item(), val_ep_st_time, time.time())
            writer.add_scalar('TrainingAndValidation/ValidationLoss', train_track['validation_losses'][-1], epoch)

        # print('current learning rate: ' + str(optimiser.param_groups[0]['lr']))
        train_track.train_epoch_update(epoch_loss.item(), ep_st_time, time.time(), init_time, epoch)
        # write loss to the tensorboard (just for recording purposes)
        writer.add_scalar('TrainingAndValidation/TrainingLoss', train_track['training_losses'][-1], epoch)
        writer.add_scalar('TrainingAndValidation/LearningRate', optimiser.param_groups[0]['lr'], epoch)
        network.save_model('model', save_path)
        miscfuncs.json_save(train_track, 'training_stats', save_path)

        if args.validation_p and patience_counter > args.validation_p:
            print('validation patience limit reached at epoch ' + str(epoch))
            break

    # Remove dataset from memory
    del dataset
    # Empty the CUDA Cache
    # torch.cuda.empty_cache()

    # Create a new data set
    dataset = CAMLdataset.DataSet(data_dir=args.data_location)
    # Then load the Test data set
    dataset.create_subset('test')
    dataset.load_file(os.path.join('test', args.file_name), 'test')

    print("done training")
    lossESR = training.ESRLoss()
    lossDC = training.DCLoss()

    print("testing the final model")
    # Test the model the training ended with
    test_output, test_loss = network.process_data(dataset.subsets['test'].data['input'][0],
                                                  dataset.subsets['test'].data['target'][0], loss_functions,
                                                  args.test_chunk)
    test_loss_ESR = lossESR(test_output, dataset.subsets['test'].data['target'][0])
    test_loss_DC = lossDC(test_output, dataset.subsets['test'].data['target'][0])
    write(os.path.join(save_path, "test_out_final.wav"), dataset.subsets['test'].fs, test_output.cpu().numpy()[:, 0, 0])
    writer.add_scalar('Testing/FinalTestLoss', test_loss.item())
    writer.add_scalar('Testing/FinalTestESR', test_loss_ESR.item())
    writer.add_scalar('Testing/FinalTestDC', test_loss_DC.item())

    train_track['test_loss_final'] = test_loss.item()
    train_track['test_lossESR_final'] = test_loss_ESR.item()
    train_track['test_lossDC_final'] = test_loss_DC.item()

    print("testing the best model")
    # Test the best model
    best_val_net = miscfuncs.json_load('model_best', save_path)
    network = networks.load_model(best_val_net)
    test_output, test_loss = network.process_data(dataset.subsets['test'].data['input'][0],
                                                  dataset.subsets['test'].data['target'][0], loss_functions,
                                                  args.test_chunk)
    test_loss_ESR = lossESR(test_output, dataset.subsets['test'].data['target'][0])
    test_loss_DC = lossDC(test_output, dataset.subsets['test'].data['target'][0])
    write(os.path.join(save_path, "test_out_best.wav"),
          dataset.subsets['test'].fs, test_output.cpu().numpy()[:, 0, 0])
    writer.add_scalar('Testing/BestTestLoss', test_loss.item())
    writer.add_scalar('Testing/BestTestESR', test_loss_ESR.item())
    writer.add_scalar('Testing/BestTestDC', test_loss_DC.item())

    train_track['test_loss_best'] = test_loss.item()
    train_track['test_lossESR_best'] = test_loss_ESR.item()
    train_track['test_lossDC_best'] = test_loss_DC.item()

    print("finished training: " + model_name)

    miscfuncs.json_save(train_track, 'training_stats', save_path)
    if cuda:
        with open(os.path.join(save_path, 'maxmemusage.txt'), 'w') as f:
            f.write(str(torch.cuda.max_memory_allocated()))

    return network


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--files', '-f', nargs='+', help='provide input target files in pairs e.g. guitar_in.wav guitar_tg.wav bass_in.wav bass_tg.wav')
    # parser.add_argument('--load_config', '-l',
    #               help="File path, to a JSON config file, arguments listed in the config file will replace the defaults", default='RNN-aidadsp-1')
    # parser.add_argument('--csv_file', '-csv', action=argparse.BooleanOptionalAction, default=False, help='Use csv file for split bounds')
    # parser.add_argument('--config_location', '-cl', default='Configs', help='Location of the "Configs" directory')
    prep_audio(["Data/NeuralCoryWong/input.wav", "Data/NeuralCoryWong/target.wav"])
    train_routine(load_config="RNN-aidadsp-1", segment_length=24000, seed=39, )
