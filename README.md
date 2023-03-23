# Automated-Guitar Amplifier Modelling

This repository contains neural network training scripts and trained models of guitar amplifiers and distortion pedals. The 'Results' directory contains some example recurrent neural network models trained to emulate the ht-1 amplifier and Big Muff Pi fuzz pedal, these models are described in this [conference paper](https://www.dafx.de/paper-archive/2019/DAFx2019_paper_43.pdf)

## Aida DSP contributions

### What we implemented

- a way to export models generated here in a format compatible with [RTNeural](https://github.com/jatinchowdhury18/RTNeural)
- a way to customize the dataset with split bounds that are expressed with a csv file (see prep_wav.py)
- A-Weighting FIR filter coefficients to be used in the loss function pre-emphasis filter see PERCEPTUAL LOSS FUNCTION FOR NEURAL MODELLING OF AUDIO SYSTEMS
- a Docker container with CUDA support to perform training on local machines and running Jupyter Notebook
- a review of the Jupyter script .ipynb
- a [lv2 plugin](https://github.com/AidaDSP/aidadsp-lv2) to run the models generated here on the Mod Audio platform and derivatives

### Prerequisites to run the docker container

#### NVIDIA drivers

```
dpkg -l | grep nvidia-driver
ii  nvidia-driver-510                          510.47.03-0ubuntu0.20.04.1            amd64        NVIDIA driver metapackage

dpkg -S /usr/lib/i386-linux-gnu/libcuda.so
libnvidia-compute-510:i386: /usr/lib/i386-linux-gnu/libcuda.so
```

#### NVIDIA Container Toolkit

```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && distribution="ubuntu20.04" \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

now you can run containers with gpu support

### Example container usage

Build:

```
docker build -f Dockerfile_pytorch --build-arg host_uid=1000 --build-arg host_gid=1000 . -t pytorch
```

Run:

```
docker run --gpus all -v $PWD:/workdir:rw -w /workdir -p 8888:8888 -it pytorch:latest
```

#### Dataset

Since I was not satisfied with dataset proposed by original authors I've put together one:

- [Thomann Stompenberg Dataset](https://github.com/MaxPayne86/ThomannStompenbergDataset)

#### NAM Dataset

Since I've received a bunch of request from the NAM community, I leave some infos here. Since the
NAM models at the moment are not compatible with the inference engine used by rt-neural-generic (RTNeural), you can't
use them with our plugin directly. But you can still use our training script and the NAM Dataset, so that you will be able
to use the amplifiers that you are using on NAM with our plugin. In the end, training is 10mins on a Laptop with CUDA.

To do so, I'll leave a reference to NAM Dataset [v1_1_1.wav](https://drive.google.com/file/d/1v2xFXeQ9W2Ks05XrqsMCs2viQcKPAwBk/view?usp=share_link)

## Using this repository
It is possible to use this repository to train your own models. To model a different distortion pedal or amplifier, a dataset recorded from your target device is required, example datasets recorded from the ht1 and Big Muff Pi are contained in the 'Data' directory. 

### Cloning this repository

To create a working local copy of this repository, use the following command:

git clone --recurse-submodules https://github.com/Alec-Wright/NeuralGuitarAmpModelling

### Python Environment

Using this repository requires a python environment with the 'pytorch', 'scipy', 'tensorboard' and 'numpy' packages installed. 
Additionally this repository uses the 'CoreAudioML' package, which is included as a submodule. Cloining the repo as described in 'Cloning this repository' ensures the CoreAudioML package is also downloaded.

### Processing Audio

The 'proc_audio.py' script loads a neural network model and uses it to process some audio, then saving the processed audio. This is also a good way to check if your python environment is setup correctly. Running the script with no arguments:

python proc_audio.py

will use the default arguments, the script will load the 'model_best.json' file from the directory 'Results/ht1-ht11/' and use it to process the audio file 'Data/test/ht1-input.wav', then save the output audio as 'output.wav'
Different arguments can be used as follows

python proc_audio.py 'path/to/input_audio.wav' 'output_filename.wav' 'Results/path/to/model_best.json'

### Training Script

the 'dist_model_recnet.py' script was used to train the example models in the 'Results' directory. At the top of the file the 'argparser' contains a description of all the training script arguments, as well as their default values. To train a model using the default arguments, simply run the model from the command line as follows:

python dist_model_recnet.py

note that you must run this command from a python environment that has the libraries described in 'Python Environment' installed. To use different arguments in the training script you can change the default arguments directly in 'dist_model_recnet.py', or you can direct the 'dist_model_recnet.py' script to look for a config file that contains different arguments, for example by running the script using the following command:

python dist_model_recnet.py -l "ht11.json"

Where in this case the script will look for the file ht11.json in the the 'Configs' directory. To create custom config files, the ht11.json file provided can be edited in any text editor.

During training, the script will save some data to a folder in the Results directory. These are, the lowest loss achieved on the validation set so far in 'bestvloss.txt', as well as a copy of that model 'model_best.json', and the audio created by that model 'best_val_out.wav'. The neural network at the end of the most recent training epoch is also saved, as 'model.json'. When training is complete the test dataset is processed, and the audio produced and the test loss is also saved to the same directory.

A trained model contained in one of the 'model.json' or 'model_best.json' files can be loaded, see the 'proc_audio.py' script for an example of how this is done.

### Determinism

If determinism is desired, `dist_model_recnet.py` provides an option to seed all of the random number generators used at once. However, if NVIDIA CUDA is used, you must also handle the non-deterministic behavior of CUDA for RNN calculations as is described in the [Rev8 Release Notes](https://docs.nvidia.com/deeplearning/cudnn/release-notes/rel_8.html). The user can eliminate the non-deterministic behavior of cuDNN RNN and multi-head attention APIs, by setting a single buffer size in the CUBLAS_WORKSPACE_CONFIG environmental variable, for example, :16:8 or :4096:2
```
CUBLAS_WORKSPACE_CONFIG=:4096:2
```
or
```
CUBLAS_WORKSPACE_CONFIG=:16:8
```
Note: if you're in google colab, the following goes into a cell
```
!export CUBLAS_WORKSPACE_CONFIG=:4096:2
```

### Tensorboard
The `./dist_model_recnet.py` has been implemented with PyTorch's Tensorboard hooks. To see the data, run:
```
tensorboard --logdir ./TensorboardData
```

### Feedback

This repository is still a work in progress, and I welcome your feedback! Either by raising an Issue or in the 'Discussions' tab 
