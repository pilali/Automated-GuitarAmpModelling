# Example JSON Configuration Snippets for Conditioned Training (2 Parameters: "gain", "tone")

This document provides focused example snippets for the JSON configuration files required by the `prep_wav.py` and `dist_model.py` scripts. These examples are tailored for training a neural network model that is conditioned on two external parameters, illustratively named "gain" and "tone".

These snippets highlight the most critical sections of the JSON files for setting up a conditioned training workflow.

## 1. `prep_wav.py` JSON Configuration Snippet

This configuration is used by `prep_wav.py` to process your raw audio dataset. It defines how to handle the audio files and, crucially, how to embed the "gain" and "tone" conditioning parameters into the input WAV files that will be fed to the model.

*   **Example File Location**: `Configs/prepare_my_device_conditioned.json`
*   **Purpose**: To take raw input and target audio files, along with their associated "gain" and "tone" values, and produce processed, multi-channel input WAV files (audio + gain + tone) and single-channel target WAV files, split into training, validation, and test sets.

```json
{
  "file_name": "my_device_conditioned_data",
  "samplerate": 48000, // Set to your project's target sample rate (e.g., 44100, 48000)
  "params": {
    "csv": "Configs/Csv/my_device_data_splits.csv", // Path to your CSV file defining train/validation/test splits and markers
    "n": 2, // CRITICAL: Number of conditioning parameters (e.g., gain, tone)
    "datasets": [
      {
        "input": "RawData/GuitarDI_Gain0.1_Tone0.2.wav", // Path to raw input audio
        "target": "RawData/AmpOut_Gain0.1_Tone0.2.wav", // Path to raw target audio
        "params": [0.1, 0.2] // [gain_value, tone_value] - Order must be consistent
      },
      {
        "input": "RawData/GuitarDI_Gain0.5_Tone0.5.wav",
        "target": "RawData/AmpOut_Gain0.5_Tone0.5.wav",
        "params": [0.5, 0.5] // [gain_value, tone_value]
      },
      {
        "input": "RawData/GuitarDI_Gain0.9_Tone0.8.wav",
        "target": "RawData/AmpOut_Gain0.9_Tone0.8.wav",
        "params": [0.9, 0.8] // [gain_value, tone_value]
      }
      // Add more entries for all your audio file pairs,
      // each with its specific gain and tone parameter values.
    ]
  }
  // Optional: You can include other top-level arguments for prep_wav.py here, e.g.:
  // "norm": true, // To normalize target tracks volume against input tracks
  // "denoise": false // To disable/enable denoising based on 'noise' markers in CSV
}
```

**Key points for `prep_wav.py` JSON:**
*   `params.n`: This **must** be set to `2` if you have two conditioning parameters. It tells the script to prepare for `n` additional channels for these parameters.
*   `params.datasets[i].params`: Each entry in the `datasets` array needs a `params` list. This list must contain exactly `n` (so, 2 in this case) floating-point values representing the conditioning parameters for that specific input/target audio pair. The order of parameters within this list (e.g., gain first, then tone) must be consistent across all entries.

## 2. `dist_model.py` JSON Configuration Snippet

This configuration is used by `dist_model.py` to train your neural network. It specifies the model architecture, training hyperparameters, and, importantly, tells the script to expect input data that has been augmented with conditioning parameters.

*   **Example File Location**: `Configs/train_my_device_conditioned.json`
*   **Purpose**: To define all aspects of the model training process, ensuring the model's input layer is correctly sized to accept the audio channel plus the two conditioning parameter channels.

```json
{
  "device": "MyConditionedDeviceV1", // A unique name for your model; results saved under this name
  "file_name": "my_device_conditioned_data", // MUST MATCH "file_name" from prep_wav.py JSON
  "samplerate": 48000, // MUST MATCH "samplerate" from prep_wav.py JSON
  "params": {
    "n": 2 // CRITICAL: Number of conditioning parameters.
           // This tells the script to set model input_size = n + 1 (i.e., 2 + 1 = 3)
  },
  "model": "SimpleRNN", // Specify the model architecture (e.g., "SimpleRNN", "GatedConvNet", "RecNet")
  "unit_type": "LSTM",  // For RNN-based models (e.g., "LSTM", "GRU", "RNN")
  "hidden_size": 20,    // Number of hidden units/channels in the model
  "num_layers": 1,      // Number of layers in the model architecture
  "skip_con": 1,        // Use skip connection (1 for true, 0 for false)
  "learn_rate": 0.0015,
  "epochs": 2000,
  "batch_size": 50,
  "validation_f": 5,    // Validate every 5 epochs
  "validation_p": 20,   // Early stopping patience (number of checks)
  "loss_fcns": {        // Dictionary of loss functions and their weights
    "ESRPre": 0.8,
    "DC": 0.2
  },
  "pre_filt": "high_pass" // Pre-emphasis filter for ESRPre loss
  // "input_size": 3, // This is generally NOT NEEDED.
                      // The script automatically calculates input_size = params.n + 1.
                      // If params.n is 2, input_size becomes 3.
  // Add any other model-specific or training-related parameters as needed.
}
```

**Key points for `dist_model.py` JSON:**
*   `file_name`: **Crucial**: This must be identical to the `file_name` specified in the `prep_wav.py` configuration. This ensures `dist_model.py` loads the correct multi-channel `.wav` files.
*   `samplerate`: Also **Crucial**: Must match the `samplerate` from the `prep_wav.py` configuration.
*   `params.n`: This **must** be set to `2` (for two conditioning parameters). The `dist_model.py` script uses this value to dynamically set `args.input_size = params.n + 1`. So, for `n=2`, `input_size` will be automatically configured to `3`.
*   `input_size`: You should generally **omit** this from your JSON configuration when `params.n` is set, as the script will correctly derive it. Explicitly setting it might be redundant or could lead to conflicts if `params.n` is the intended driver.

These snippets provide a clear template for setting up the JSON configuration files needed for training audio models conditioned on multiple parameters. Remember to adjust file paths, parameter values, and model hyperparameters to suit your specific dataset and experimental goals.
