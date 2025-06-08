# Recommendations for Training Models with Highly Saturated Sounds

## Introduction

This document compiles general advice and recommendations for training neural network models specifically targeting highly saturated audio sounds. Modeling such sounds presents unique challenges due to their complex non-linear dynamics, rich harmonic content, and potentially extreme transient behavior. These recommendations cover data preparation, model architecture selection, and crucial training parameters, primarily focusing on the usage of the `prep_wav.py` and `dist_model.py` scripts.

## 1. Data Preparation (`prep_wav.py` considerations)

High-quality and thoughtfully prepared data is foundational for successfully modeling any audio effect, especially complex ones like heavy saturation.

*   **High-Quality and Varied Recordings are Key:**
    *   **Signal Chain Integrity:** Ensure your recording chain (instrument, preamps, converters) is as clean as possible to avoid introducing unwanted noise or distortion *before* the saturation stage you intend to model. The model will learn any characteristic in the target sounds.
    *   **Variety in Saturation:** Capture a wide range of saturation levels in your training data. This includes:
        *   Sounds just on the edge of breakup.
        *   Mild to moderate saturation.
        *   Heavy/intense saturation, and even "blocked" or "fuzz-like" sounds if that's part of what you want to model.
        *   Different input signal levels driving the saturator, as many devices react very differently to input strength.
    *   **Variety in Source Material:** If applicable, use diverse input material (e.g., different guitars, pickups, playing styles for guitar amps; different types of signals for general saturators). This helps the model generalize better.
    *   **Blip Alignment:** The blip system mentioned in `prep_wav.py` (using a CSV for alignment) is crucial. For heavily saturated sounds where transients might be squashed or altered, ensure your blips are clear and consistently placed in both input and target recordings for accurate delay compensation.

*   **Normalization Strategies:**
    *   **`prep_wav.py`'s `--norm` flag:** The `--norm` flag in `prep_wav.py` normalizes the target track to match the peak level of the input track.
        *   **For saturated sounds, this can be beneficial** as it ensures that the overall loudness of the target (saturated) signal is scaled relative to the input. This can prevent the model from having to learn drastic volume changes and instead focus on the timbral characteristics of saturation.
        *   However, be mindful: if the *perceived* loudness change is a key characteristic of the saturation you want to model, you might experiment with and without this flag, or implement a custom normalization strategy before using `prep_wav.py`.
    *   **Consider RMS Normalization (External Preprocessing):** For some applications, normalizing based on RMS (Root Mean Square) level *before* feeding data to `prep_wav.py` might be useful, especially if you want to maintain more consistent perceived loudness across different recordings with varying peak-to-RMS ratios (common in saturated signals). `prep_wav.py` itself doesn't offer RMS normalization directly.
    *   **Peak vs. RMS for Targets:** Heavy saturation often significantly reduces the peak-to-RMS ratio (crest factor). If you normalize everything to peak 0dBFS, your saturated targets might have a much higher average level than cleaner inputs. This is often desirable as it's part of the saturation effect.

*   **Denoising (`--denoise` flag in `prep_wav.py`):**
    *   **High-Gain Noise:** Saturated sounds, especially from high-gain amplifiers or distortion pedals, can also amplify noise from the input source or the device itself.
    *   If your target recordings have noticeable hiss, hum, or other steady background noise that you *don't* want the model to learn, the `--denoise` flag in `prep_wav.py` could be helpful.
    *   **Requires Noise Profile in CSV:** Remember, this feature relies on you defining 'noise' regions in your CSV file (as used by `params.csv` in the JSON config). The script uses these regions to learn a noise profile for subtraction.
    *   **Caution:** Be careful with denoising. Aggressive denoising can sometimes introduce artifacts or alter the character of the saturation itself, especially the delicate decay tails or quieter parts. Test its effect.

## 2. Model Architectures and Parameters (`dist_model.py`)

Choosing an appropriate model architecture and tuning its parameters are critical steps for capturing the complex characteristics of highly saturated audio.

*   **Model Architectures (`--model`):**
    Highly saturated audio often involves very complex non-linear dynamics, sharp clipping, and rich harmonic content. Some architectures might be better suited to capture these characteristics:
    *   **`GatedConvNet`:**
        *   **Potential Advantage:** Convolutional networks, especially with dilations (as `GatedConvNet` implies with `dilation_growth`), can have a large receptive field. This means they can model longer-term dependencies in the signal, which can be important for how saturation affects the envelope and character of a sound over time. Gated mechanisms can also help in controlling the information flow, potentially aiding in modeling complex behaviors.
        *   **Parameters to Tune:**
            *   `--hidden_size` (`-hs`): Number of channels in the convolutional layers. More channels can increase capacity.
            *   `--num_blocks` (`-nb`): Number of stacked gated convolutional blocks. More blocks deepen the network.
            *   `--num_layers` (`-nl`): Layers within each block.
            *   `--kernel_size` (`-ks`): Size of the convolutional kernels.
            *   `--dilation_growth` (`-dg`): Controls how rapidly the dilation factor increases, affecting the receptive field. For very complex saturation, ensuring a sufficiently large receptive field without an explosion in parameters can be key.

    *   **`SimpleRNN` (with appropriate `unit_type` like LSTM or GRU):**
        *   **Potential Advantage:** RNNs are naturally suited for sequential data. LSTMs and GRUs, with their gating mechanisms, are designed to capture temporal dependencies.
        *   **For Saturated Sounds:** You might need a "deeper" or "wider" RNN than for simpler effects:
            *   `--hidden_size` (`-hs`): Increasing this makes the recurrent layer "wider," giving it more capacity to store information about the signal's state. This is often a primary parameter to experiment with for complex sounds.
            *   `--num_layers` (`-nl`): Stacking multiple RNN layers (making it "deeper") allows the network to learn hierarchical features. For example, the first layer might capture raw waveform characteristics, while subsequent layers might learn more abstract features of the saturation.
        *   **`--unit_type` (`-ut`):** `LSTM` is often a good default. `GRU` is slightly simpler and can sometimes train faster or perform better on certain tasks. It's worth experimenting if you have the resources.

    *   **Other Models (`RecNet`, `ConvSimpleRNN`, `AsymmetricAdvancedClipSimpleRNN`):**
        *   These are more specialized. `AsymmetricAdvancedClipSimpleRNN` might be interesting if your saturation has a strong, known asymmetric clipping characteristic. `ConvSimpleRNN` attempts to combine convolutional front-ends with RNNs. Their suitability would depend on the specifics of your saturated sound.

*   **Skip Connection (`--skip_con`):**
    *   **Role:** The skip connection (`--skip_con 1` to enable) adds the input of the model (or a block) to its output.
    *   **For Saturated Sounds:**
        *   **Potential Benefit:** In scenarios where the saturation is an *addition* to or *modification* of the original signal (rather than a complete transformation), a skip connection can make it easier for the model to learn the "delta" or change introduced by the saturation effect. This is often the case; the output retains some characteristics of the input. For very extreme transformations, its utility might vary.
        *   It can also help with gradient flow during training, potentially leading to faster convergence or better performance, especially in deeper networks.
        *   Most of the models default to `--skip_con 1`, which is generally a good starting point.

*   **Experimentation is Key:**
    The optimal architecture and its parameters will heavily depend on the precise nature of the saturation you're trying to model. It's common to:
    *   Start with a simpler configuration (e.g., `SimpleRNN` with `LSTM`, moderate `hidden_size`) to establish a baseline.
    *   Gradually increase complexity (e.g., more `hidden_size`, `num_layers`, or try `GatedConvNet`) and observe the impact on validation loss and, importantly, the *sound quality* of the output.
    *   Be mindful that larger models require more data, more computational resources, and longer training times.

## 3. Loss Functions, Pre-filtering, and Training Hyperparameters (`dist_model.py`)

These settings directly influence how the model learns and what aspects of the sound it prioritizes.

### Loss Functions and Pre-filtering

*   **Loss Functions (`--loss_fcns`):**
    The `--loss_fcns` argument takes a dictionary specifying which loss functions to use and their relative weights. The default is `{'ESRPre': 0.75, 'DC': 0.25}`.
    *   **`ESRPre` (Error-to-Signal Ratio with Pre-emphasis):**
        *   **Importance:** This is often a very effective loss function for audio tasks. ESR measures the error relative to the energy of the target signal. The "Pre" (pre-emphasis) part means it's typically calculated on a version of the signal where higher frequencies have been boosted.
        *   **For Saturated Sounds:** Saturation adds significant harmonic content, often drastically changing the high-frequency spectrum. `ESRPre` can help the model focus on accurately reproducing these important high-frequency details. It's generally a good choice as a primary component of your loss.
    *   **`DC` (DC Loss):**
        *   **Importance:** This loss penalizes differences in the DC offset between the output and the target.
        *   **For Saturated Sounds:** Some saturation processes can introduce or alter DC offset. If accurately modeling this is important, keeping the DC loss is beneficial.
    *   **Tuning Loss Weights:** The default weights are a starting point. If spectral accuracy of harmonics is paramount, consider increasing the weight of `ESRPre` (e.g., `{'ESRPre': 0.9, 'DC': 0.1}`).

*   **Pre-emphasis Filter (`--pre_filt`):**
    This argument determines the filter applied before the `ESRPre` calculation. The default is `high_pass`.
    *   **`high_pass`:** Boosts high frequencies, ensuring the loss function is sensitive to errors in the upper harmonics generated by saturation. This is generally desirable.
    *   **Other `pre_filt` Options:**
        *   `A-weighting`: Mimics human hearing sensitivity. Might be useful if perceptual accuracy is the primary goal, as it emphasizes mid-range frequencies.
        *   `folded_differentiator`: Also tends to emphasize higher frequencies.
        *   **Custom CSV filter:** Offers maximum flexibility if you want to target specific frequency bands.
    *   **Experimentation:** For most cases, `high_pass` is a solid default. If struggling with high-frequency accuracy, `A-weighting` or a custom filter could be valid experiments.

### Training Hyperparameters

*   **Learning Rate (`--learn_rate` / `-lr`):**
    *   **Impact:** Controls weight adjustments during updates.
    *   **For Saturated Sounds:** Due to sharp non-linearities, a high learning rate can lead to instability.
        *   **Suggestion:** Start with a **smaller learning rate** (e.g., `0.001`, `0.0005`, or lower if the default `0.005` is unstable). The script's `ReduceLROnPlateau` scheduler will reduce it further if validation loss stagnates.

*   **Batch Size (`--batch_size` / `-bs`) and Segment Length (`--segment_length` / `-slen`):**
    *   **`--segment_length`:** Duration of audio chunks for training (default 500ms).
        *   **For Saturated Sounds:** Saturation can have temporal dependencies. If segments are too short, the model may miss this context. Consider if 500ms is sufficient; longer segments (e.g., 1000ms+) might capture effects like "sag" better but increase memory load.
    *   **`--batch_size`:** Number of segments per iteration.
        *   **Trade-offs:** Larger batches offer stable gradients but need more memory. Smaller batches are noisier but can escape local minima.
        *   **For Saturated Sounds:** Aim for a stable gradient with a moderate batch size (e.g., 32-64). If increasing `segment_length`, you may need to decrease `batch_size`. `--iter_num` is an alternative to control epoch length.

*   **Epochs (`--epochs` / `-eps`) and Early Stopping (`--validation_p` / `-vp`):**
    *   **`--epochs`:** Max number of passes through the training dataset.
        *   **For Saturated Sounds:** Complex non-linearities might require more training. The default of `2000` is substantial, but be prepared to increase it if validation loss is still improving.
    *   **`--validation_p` (Early Stopping Patience):**
        *   **Importance:** Prevents overfitting. Training stops if validation loss doesn't improve for `validation_p` checks.
        *   **For Saturated Sounds:** Progress can be slow. Ensure `validation_p` (default 25) isn't too small, to avoid stopping prematurely. Consider increasing it if small but consistent improvements are still occurring. The script always saves the best model found (`model_best.json`).

*   **General Training Considerations:**
    *   **Monitor GPU Memory:** Larger models, segments, or batches need more memory.
    *   **Listen to Validation Output:** Use your ears to judge `best_val_out.wav`, as loss values don't tell the full story.
    *   **Iterative Approach:** Experiment systematically. Start with sensible defaults, change one or two hyperparameters at a time, and log your results.

Finding the optimal combination of these settings requires careful experimentation and iteration, tailored to the specific characteristics of the saturated sound you aim to model.
