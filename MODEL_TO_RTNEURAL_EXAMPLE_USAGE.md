# Example Usage Scenario for `modelToRTNeural.py`

This document illustrates a practical example of how to use the `modelToRTNeural.py` script. The primary goal in this scenario is to convert a model that was trained using the project's `dist_model.py` script into the RTNeural JSON format, automatically selecting the best performing version of the trained model.

## Scenario Context

Let's establish the context for this example:

*   **Training Configuration File**: You have previously configured and run a training session using a JSON file named `MyAmpModel_LSTM_Config.json`. This file is located in your `Configs/` directory.
*   **Device Name**: Inside the `MyAmpModel_LSTM_Config.json` file, the `"device"` parameter was set to `"MyAmpDevice"`. This is a common practice to identify and organize different model experiments.
*   **Training Results**: The `dist_model.py` script, using the above configuration, has completed training. The results, which include:
    *   `model.json` (the model from the final training epoch)
    *   `model_best.json` (the model that achieved the best score on the validation set)
    *   `training_stats.json` (containing loss metrics, metadata, and sample I/O batches)
    have been saved in the `Results/MyAmpDevice/` directory, as per the standard output structure of `dist_model.py`.
*   **Goal**: You now want to convert the *best performing* version of this trained model (`MyAmpDevice`) into an RTNeural-compatible JSON file.

## Command-Line Example

To convert the model based on the scenario above, you would execute `modelToRTNeural.py` from your terminal like this:

```bash
python modelToRTNeural.py --load_config MyAmpModel_LSTM_Config --config_location Configs --results_path Results/MyAmpDevice --verbose
```

## Breakdown of the Command

Let's dissect each part of this command:

*   `python modelToRTNeural.py`: This is the standard way to execute the Python script.
*   `--load_config MyAmpModel_LSTM_Config`:
    *   This argument tells the script to operate in its "config-based loading" mode.
    *   It specifies that the base name of the configuration file is `MyAmpModel_LSTM_Config` (the `.json` extension is often implied or handled by the script).
    *   The script will look for this configuration file in the directory provided by the `--config_location` argument.
*   `--config_location Configs`:
    *   This argument informs the script that the configuration file (`MyAmpModel_LSTM_Config.json`) is located within the `Configs/` directory relative to where the script is being run or relative to a predefined base path if the script handles that.
*   `--results_path Results/MyAmpDevice`:
    *   This argument explicitly tells the script where to find the training outputs associated with the `MyAmpModel_LSTM_Config`. Specifically, it will look for `training_stats.json`, `model.json`, and `model_best.json` inside the `Results/MyAmpDevice/` directory.
    *   While the script can often infer this path if `device` is correctly set in the config and the `Results` directory is standard, providing it explicitly ensures clarity and correctness.
*   `--verbose` (Optional):
    *   This flag enables detailed logging to the console during the script's execution. You'll see messages about which files are being loaded, what parameters are being parsed, which model is selected as "best," and the progress of the conversion. It's highly recommended for verifying the process and for troubleshooting if issues arise.

## What the Script Does (Based on the Command)

1.  **Reads Configuration**: The script starts by loading and parsing `Configs/MyAmpModel_LSTM_Config.json`. From this, it extracts model architecture details (like `unit_type`, `hidden_size`, `num_layers`, `skip_con`) and other metadata.
2.  **Accesses Training Statistics**: It navigates to `Results/MyAmpDevice/` and reads `training_stats.json`.
3.  **Determines Best Model**: By comparing the `test_lossESR_final` (associated with `model.json`) and `test_lossESR_best` (associated with `model_best.json`) values found in `training_stats.json`, the script decides which of the two model files represents the better performing model.
4.  **Loads Chosen Model**: It loads the selected model file (e.g., `Results/MyAmpDevice/model_best.json`). This file contains the model's architecture (`model_data`) and learned weights (`state_dict`).
5.  **Gathers Additional Data**: The script also extracts `input_batch`, `output_batch` (sample data), and enriches the `metadata` (e.g., by adding the chosen model's ESR) from `training_stats.json`.
6.  **Converts Model**: The core conversion logic is then applied. The script transforms the model's layers and weights into the structure required by RTNeural (as detailed in the functionality documentation).
7.  **Saves Output**: Finally, the converted model is saved to a new file.

## Expected Output

*   **Output File Name**: By default, the output file containing the RTNeural-compatible model will be named `model_rtneural.json`.
*   **Output File Location**: This file will be saved in the directory that was specified by the `--results_path` argument (or inferred from the configuration). In this specific example, the output file will be located at:
    `Results/MyAmpDevice/model_rtneural.json`
*   **Alternative Output (with `--aidax` flag)**: If you had included the `--aidax` (or `-ax`) flag in your command:
    ```bash
    python modelToRTNeural.py --load_config MyAmpModel_LSTM_Config --config_location Configs --results_path Results/MyAmpDevice --verbose --aidax
    ```
    Then the output file would instead be named `model_rtneural.aidax` and located at `Results/MyAmpDevice/model_rtneural.aidax`.

This example covers a typical use case where you want to convert the best model from a completed training run. For converting a specific, standalone model file (like a particular `.json` or a `.nam` file), you would use the `--load_model <path_to_your_model_file>` argument instead. In that case, arguments like `--load_config` and `--results_path` would be ignored, and the output RTNeural file would be saved in the same directory as the input model specified by `--load_model`.
