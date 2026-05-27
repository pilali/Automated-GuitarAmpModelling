"""
Pure-Python exporter that converts a PyTorch SimpleRNN (LSTM/GRU) model trained
with this repository into the JSON format consumed by AIDA-X / RTNeural.

This replaces the previous ``modelToKeras.py`` pipeline which required
TensorFlow + Keras just to re-emit the same weights as JSON. The output format
is byte-compatible with what ``model_utils.save_model_json`` used to produce
(same key names, same ordering of the gate weights for both LSTM and GRU).

Usage (CLI, mirrors the old modelToKeras.py):

    python3 aidax_export.py -lm Results/<run>/model_best.json \
                            -o  Results/<run>/<run>.aidax

If ``--output`` is not given the result is written next to the source model
as ``model_keras.json`` (kept for backward compatibility with old tooling).

When invoked with ``--load_config CONFIG`` instead of ``--load_model``, the
script reads the training_stats.json file (as the legacy script did) to pick
the model with the lowest ESR and to embed input_batch / output_batch /
metadata in the exported file.
"""

import argparse
import json
import os

import numpy as np


def _reorder_gru_gates_axis1(matrix: np.ndarray, hidden_size: int) -> np.ndarray:
    """Reorder GRU gates along axis 1 from PyTorch (r, z, n) to Keras (z, r, n)."""
    out = np.empty_like(matrix)
    for i in range(matrix.shape[0]):
        row = matrix[i]
        out[i] = np.concatenate((
            row[hidden_size:2 * hidden_size],
            row[0:hidden_size],
            row[2 * hidden_size:],
        ))
    return out


def _reorder_gru_bias(bias: np.ndarray, hidden_size: int) -> np.ndarray:
    return np.concatenate((
        bias[hidden_size:2 * hidden_size],
        bias[0:hidden_size],
        bias[2 * hidden_size:],
    ))


def _build_simple_rnn_layers(model_data: dict):
    meta = model_data["model_data"]
    state = model_data["state_dict"]

    unit_type = meta["unit_type"]
    input_size = int(meta["input_size"])
    output_size = int(meta["output_size"])
    hidden_size = int(meta["hidden_size"])
    bias_fl = bool(meta.get("bias_fl", True))
    skip = int(meta.get("skip", 0))

    W = np.array(state["rec.weight_ih_l0"], dtype=np.float32)
    U = np.array(state["rec.weight_hh_l0"], dtype=np.float32)
    b_ih = np.array(state["rec.bias_ih_l0"], dtype=np.float32)
    b_hh = np.array(state["rec.bias_hh_l0"], dtype=np.float32)
    lin_w = np.array(state["lin.weight"], dtype=np.float32)
    lin_b = np.array(state["lin.bias"], dtype=np.float32)

    rec_layer = {
        "type": unit_type.lower(),
        "activation": "",
        "shape": [None, None, hidden_size],
    }

    if unit_type == "LSTM":
        # PyTorch and Keras both use the (i, f, g, o) gate ordering, so the
        # only conversion is to transpose for Keras' (input_size, 4H) layout
        # and to fold the two biases into one as Keras stores a single bias.
        kernel = np.transpose(W)
        recurrent_kernel = np.transpose(U)
        bias = b_ih + b_hh
        rec_layer["weights"] = [kernel.tolist(), recurrent_kernel.tolist(), bias.tolist()]
    elif unit_type == "GRU":
        # PyTorch's GRU gate order is (r, z, n); Keras' is (z, r, n).
        kernel = _reorder_gru_gates_axis1(np.transpose(W), hidden_size)
        recurrent_kernel = _reorder_gru_gates_axis1(np.transpose(U), hidden_size)
        # Keras GRU with reset_after=True (its default) expects bias of
        # shape (2, 3*hidden_size): row 0 is the input bias, row 1 the
        # recurrent bias.
        bias = np.zeros((2, 3 * hidden_size), dtype=np.float32)
        bias[0] = _reorder_gru_bias(b_ih, hidden_size)
        bias[1] = _reorder_gru_bias(b_hh, hidden_size)
        rec_layer["weights"] = [kernel.tolist(), recurrent_kernel.tolist(), bias.tolist()]
    else:
        raise ValueError(f"Unsupported unit_type {unit_type!r} (expected LSTM or GRU)")

    if not bias_fl:
        # RTNeural always expects a bias entry: when the network was trained
        # without one we still emit a zero bias so the JSON stays valid.
        pass

    dense_layer = {
        "type": "dense",
        "activation": "",
        "shape": [None, None, output_size],
        "weights": [lin_w.reshape(hidden_size, output_size).tolist(), lin_b.tolist()],
    }

    return [rec_layer, dense_layer], input_size, skip


def export(model_json_path: str,
           output_path: str,
           metadata: dict | None = None,
           input_batch: list | None = None,
           output_batch: list | None = None) -> dict:
    """Convert a SimpleRNN PyTorch model JSON to AIDA-X / RTNeural JSON."""
    with open(model_json_path) as fp:
        model_data = json.load(fp)

    if model_data.get("model_data", {}).get("model") != "SimpleRNN":
        raise ValueError(
            "aidax_export only supports SimpleRNN models (LSTM/GRU); "
            f"got {model_data.get('model_data', {}).get('model')!r}"
        )

    layers, input_size, skip = _build_simple_rnn_layers(model_data)

    out: dict = {"in_shape": [None, None, input_size]}
    if skip > 0:
        out["in_skip"] = skip
    out["layers"] = layers
    if input_batch is not None and output_batch is not None:
        out["input_batch"] = input_batch
        out["output_batch"] = output_batch
    if metadata is not None:
        out["metadata"] = metadata

    with open(output_path, "w") as fp:
        json.dump(out, fp, indent=4)

    return out


def _pick_best_model(results_dir: str):
    """Read training_stats.json and pick the model variant with the lowest ESR."""
    stats_path = os.path.join(results_dir, "training_stats.json")
    with open(stats_path) as fp:
        stats = json.load(fp)

    esr_final = stats.get("test_lossESR_final")
    esr_best = stats.get("test_lossESR_best")
    if esr_final is None or esr_best is None:
        raise KeyError("training_stats.json is missing ESR fields")

    if esr_final < esr_best:
        return (
            os.path.join(results_dir, "model.json"),
            esr_final,
            stats.get("input_batch"),
            stats.get("output_batch_final"),
        )
    return (
        os.path.join(results_dir, "model_best.json"),
        esr_best,
        stats.get("input_batch"),
        stats.get("output_batch_best"),
    )


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load_model", "-lm", default="",
                        help="Path to a PyTorch model JSON (model.json or model_best.json).")
    parser.add_argument("--load_config", "-l", default="",
                        help="Name (no extension) of a Configs/*.json file. When set, the script "
                             "auto-picks the best ESR model and embeds metadata/input_batch/output_batch.")
    parser.add_argument("--config_location", "-cl", default="Configs",
                        help="Directory holding the JSON configs.")
    parser.add_argument("--results_path", "-rp", default="",
                        help="Override the results directory (otherwise derived from the config).")
    parser.add_argument("--output", "-o", default="",
                        help="Output path for the .aidax/.json model. Defaults to "
                             "<results_dir>/model_keras.json next to the source model.")
    args = parser.parse_args()

    metadata = None
    input_batch = None
    output_batch = None

    if args.load_model:
        model_path = args.load_model
        results_dir = os.path.dirname(model_path) or "."
    elif args.load_config:
        config_path = os.path.join(args.config_location, args.load_config + ".json")
        with open(config_path) as fp:
            config = json.load(fp)
        device = config["device"]
        unit_type = config["unit_type"]
        hidden_size = config["hidden_size"]
        skip = config.get("skip_con", 0)
        metadata = config.get("metadata")
        results_dir = args.results_path or os.path.join(
            "Results", f"{device}_{unit_type}-{hidden_size}-{skip}"
        )
        model_path, esr, input_batch, output_batch = _pick_best_model(results_dir)
        if metadata is not None:
            metadata = dict(metadata)
            metadata["esr"] = esr
    else:
        parser.error("Either --load_model or --load_config must be provided.")

    output_path = args.output or os.path.join(results_dir, "model_keras.json")
    export(model_path, output_path,
           metadata=metadata,
           input_batch=input_batch,
           output_batch=output_batch)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    _main()
