# End-to-End Training Process for Conditioned Models (e.g., with "gain" and "tone")

This guide details how to train a neural network model that is conditioned on two external parameters, such as "gain" and "tone". The process involves two main Python scripts: `prep_wav.py` for data preparation and `dist_model.py` for model training. Both scripts are configured primarily through JSON files.

The goal is to create a model that can change its audio processing behavior based on the conditioning parameters provided alongside the input audio.

**Assumptions:**

*   You have a collection of raw audio data consisting of:
    *   **Input files**: The "dry" or unprocessed audio signals (e.g., direct instrument recordings from a guitar).
    *   **Target files**: The corresponding "wet" or processed audio signals (e.g., recordings of the same guitar performance after passing through an amplifier or effects unit where "gain" and "tone" settings were varied).
*   For each input/target audio pair, you have recorded the specific "gain" and "tone" values (or any other parameters you wish to condition on). These values should ideally be normalized (e.g., to a 0.0-1.0 range).
*   You have a CSV file (e.g., exported from a DAW like Reaper) that defines how your raw audio files are segmented into training, validation, and test sets. This CSV might also contain markers for blips (for auto-alignment) or noise regions (if denoising is used).

## Step 1: Configure `prep_wav.py`

The first step is to prepare your data using `prep_wav.py`. This script will take your raw audio files, resample them, align them (if necessary), and most importantly, embed the conditioning parameters ("gain" and "tone") into the input audio files as additional channels.

Create a JSON configuration file for `prep_wav.py`. Let's name it `config_prep_conditioned.json` and place it in the `Configs` directory.

**Key settings in `config_prep_conditioned.json`:**

*   `file_name` (string): A base name that will be used for the output processed files. For example, `"my_amp_conditioned"`.
*   `samplerate` (integer): The target sample rate to which all audio will be resampled (e.g., `44100` or `48000`).
*   `params` (object):
    *   `csv` (string): Path to your CSV file that defines data splits and markers. Example: `"Configs/Csv/my_audio_regions.csv"`.
    *   `n` (integer): **Set this to `2`**. This is crucial as it tells `prep_wav.py` that there are two conditioning parameters.
    *   `datasets` (list of objects): This is an array where each object defines one pair of raw input and target audio files, along with their associated conditioning parameter values.
        *   `input` (string): Path to the raw input WAV file.
        *   `target` (string): Path to the corresponding raw target WAV file.
        *   `params` (list of floats): **A list containing the two float values for your parameters: `[gain_value, tone_value]`**. The order of these parameters must be consistent across all entries in the `datasets` list (e.g., "gain" is always the first value, "tone" is always the second).

**Example `config_prep_conditioned.json`:**
```json
{
  "file_name": "my_amp_conditioned",
  "samplerate": 44100,
  "params": {
    "csv": "Configs/Csv/my_amp_splits.csv",
    "n": 2, // Signifies two conditioning parameters
    "datasets": [
      {
        "input": "RawAudio/DI_Track1_gain0.2_tone0.3.wav",
        "target": "RawAudio/Amp_Track1_gain0.2_tone0.3.wav",
        "params": [0.2, 0.3] // [gain_value, tone_value]
      },
      {
        "input": "RawAudio/DI_Track2_gain0.8_tone0.7.wav",
        "target": "RawAudio/Amp_Track2_gain0.8_tone0.7.wav",
        "params": [0.8, 0.7] // [gain_value, tone_value]
      }
      // ... more entries for all your raw audio file pairs
      // with their specific gain and tone values
    ]
  }
  // You can also include other top-level arguments for prep_wav.py here,
  // such as "norm: true" or "denoise: true" if needed.
}
```

## Step 2: Run `prep_wav.py`

Once the JSON configuration file is ready, execute `prep_wav.py` from your terminal. If your config file is in `Configs/config_prep_conditioned.json`, the command would be:

```bash
python prep_wav.py --load_config config_prep_conditioned.json
```
(Assuming `prep_wav.py` is in your current directory or Python path, and `Configs` is also accessible).

**Output of this step:**

*   `prep_wav.py` will read your raw audio files, process them according to the CSV splits, and apply any specified operations like normalization or denoising.
*   New WAV files will be generated and saved into `Data/train/`, `Data/val/`, and `Data/test/` subdirectories.
*   Crucially, the **input WAV files** (e.g., `Data/train/my_amp_conditioned-input.wav`) will now be **3-channel** audio files:
    *   **Channel 1**: The original audio signal.
    *   **Channel 2**: The first conditioning parameter value (e.g., "gain"), repeated as a constant signal for the length of the audio segment.
    *   **Channel 3**: The second conditioning parameter value (e.g., "tone"), also repeated as a constant signal.
*   The **target WAV files** (e.g., `Data/train/my_amp_conditioned-target.wav`) will remain single-channel audio (or however many channels your original target had, typically mono).

## Step 3: Configure `dist_model.py`

After preparing the data, the next step is to configure the model training script, `dist_model.py`. Create another JSON configuration file for this script, for example, `config_train_conditioned.json` in the `Configs` directory.

**Key settings in `config_train_conditioned.json`:**

*   `device` (string): A unique name for your model experiment (e.g., `"my_conditioned_amp_v1"`). This name will be used to create a directory in `Results` for saving the trained model and outputs.
*   `file_name` (string): **Must exactly match the `file_name` used in `prep_wav.py`'s configuration** (e.g., `"my_amp_conditioned"`). This ensures `dist_model.py` loads the correct multi-channel input files.
*   `samplerate` (integer): **Must match the `samplerate` used in `prep_wav.py`'s configuration** (e.g., `44100`).
*   `params` (object):
    *   `n` (integer): **Set this to `2` again**. This is critical. `dist_model.py` uses this value to automatically calculate and set the model's `input_size` argument to `n + 1` (i.e., `2 + 1 = 3`). This makes the neural network expect 3 input channels (audio + param1 + param2).
*   `model` (string): Choose the neural network architecture you want to train (e.g., `"SimpleRNN"`, `"GatedConvNet"`, `"RecNet"`).
*   `input_size` (integer): You typically **do not need to set this explicitly** in the JSON file if you have correctly set `params['n']`. The script will derive `input_size = params['n'] + 1`. If you were to set it, it should be `3` for this example.
*   Other training and model parameters: Configure `hidden_size`, `unit_type` (for RNNs), `num_layers`, `learn_rate`, `epochs`, `batch_size`, `loss_fcns`, `skip_con`, etc., according to your specific model architecture and training requirements.

**Example `config_train_conditioned.json`:**
```json
{
  "device": "my_conditioned_amp_v1",
  "file_name": "my_amp_conditioned", // Must match file_name from prep_wav.py config
  "samplerate": 44100,             // Must match samplerate from prep_wav.py config
  "params": {
    "n": 2 // Critical: informs dist_model.py about the number of conditioning params
           // This will result in model.input_size being set to 3
  },
  "model": "SimpleRNN", // Or "GatedConvNet", "RecNet", etc.
  "unit_type": "LSTM",  // If using an RNN-based model
  "hidden_size": 24,
  "num_layers": 1,
  "learn_rate": 0.0015,
  "epochs": 2000,
  "batch_size": 32,
  "validation_p": 30, // Early stopping patience
  "loss_fcns": {
    "ESRPre": 0.8,
    "DC": 0.2
  },
  "skip_con": 1
  // ... other model-specific and training settings as required
  // "input_size" is typically NOT set here, as it's derived from params.n
}
```

## Step 4: Run `dist_model.py`

With the training configuration file prepared, execute `dist_model.py` from your terminal:

```bash
python dist_model.py --load_config config_train_conditioned.json
```
(Again, assuming `dist_model.py` is in your current directory or Python path, and `Configs` is accessible).

**Output of this step:**

*   `dist_model.py` will load the 3-channel input data (audio + gain parameter + tone parameter) and the corresponding single-channel target data from the directories populated by `prep_wav.py`.
*   It will initialize the specified neural network model with an input layer compatible with 3 channels.
*   The script will then train the model using the defined training parameters.
*   Trained model files (e.g., `model.json`, `model_best.pth`), training statistics (like loss curves), and example output audio files will be saved in the `Results/your_device_name/` directory (e.g., `Results/my_conditioned_amp_v1/`).

## Summary of Parameter Flow for Conditioning

1.  **You define** the specific "gain" and "tone" values (or any other chosen parameters) for each of your raw audio file pairs within `prep_wav.py`'s JSON configuration, specifically in the `datasets[i].params` list (e.g., `[0.2, 0.3]`).
2.  **You inform `prep_wav.py`** that there are two such conditioning parameters by setting `params.n = 2` in its JSON configuration.
3.  `prep_wav.py` processes this information and creates new multi-channel `*-input.wav` files. These files contain the original audio on the first channel and the conditioning parameter values as constant signals on subsequent channels (channel 2 for "gain", channel 3 for "tone").
4.  **You inform `dist_model.py`** that it should expect data with two conditioning parameters by setting `params.n = 2` in its JSON configuration.
5.  `dist_model.py` uses this `params.n` value to automatically set its internal `input_size` variable to `n + 1` (which is `2 + 1 = 3` in this example).
6.  The neural network architecture defined in `dist_model.py` is then constructed with an input layer that accepts 3 channels.
7.  During training, the network learns to map the input audio *and* its associated "gain" and "tone" channel values to the desired target audio.

This end-to-end process enables the training of a single neural network model that can dynamically alter its audio processing behavior based on the provided "gain" and "tone" (or other) conditioning parameters. This is powerful for modeling devices with multiple controls or for creating adaptable audio effects.
