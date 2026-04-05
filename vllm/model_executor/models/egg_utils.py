# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for loading nano-egg checkpoint files into vLLM EGG models.

The nano-egg reference implementation stores weights as JAX arrays in
pickle format. This module provides conversion from the JAX parameter
tree to vLLM's PyTorch weight format.

Usage:
    python -m vllm.model_executor.models.egg_utils \\
        --input checkpoint.model \\
        --output_dir ./egg_model_hf

This produces a HuggingFace-compatible model directory with:
  - config.json (EGGConfig)
  - model.safetensors (converted weights)
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


def _jax_to_numpy(jax_array) -> np.ndarray:
    """Convert a JAX array to numpy, handling various array types."""
    if isinstance(jax_array, np.ndarray):
        return jax_array
    return np.asarray(jax_array)


def convert_nano_egg_checkpoint(
    checkpoint_path: str,
    output_dir: str,
    vocab_size: int = 256,
    hidden_size: int = 256,
    num_hidden_layers: int = 6,
    intermediate_size: int = 1024,
):
    """Convert a nano-egg .model checkpoint to HuggingFace format.

    Args:
        checkpoint_path: Path to the .model pickle file
        output_dir: Directory to save the converted model
        vocab_size: Vocabulary size (default 256 for byte-level)
        hidden_size: Hidden dimension size
        num_hidden_layers: Number of decoder layers
        intermediate_size: MLP intermediate dimension (hidden_size * 4)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load the pickle checkpoint
    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)

    # The nano-egg checkpoint structure is a nested dict:
    # params = {
    #     "emb": embedding weights,
    #     "blocks": {
    #         "ln1": {"weight": ...},
    #         "att": {"Wf": ..., "Uf": ..., "bf": ...,
    #                 "Wh": ..., "Uh": ..., "bh": ...},
    #         "ln2": {"weight": ...},
    #         "mlp": {"0": {"weight": ..., "bias": ...},
    #                 "1": {"weight": ..., "bias": ...}},
    #     },
    #     "ln_out": {"weight": ...},
    #     "head": head weights,
    # }
    params = checkpoint if isinstance(checkpoint, dict) else checkpoint[1]

    state_dict = {}

    # Convert embedding
    emb_weight = _jax_to_numpy(params["emb"])
    state_dict["backbone.embeddings.weight"] = torch.from_numpy(emb_weight).to(
        torch.int8
    )

    # Convert decoder layers
    blocks = params["blocks"]
    for layer_idx in range(num_hidden_layers):
        prefix = f"backbone.layers.{layer_idx}"

        # Extract per-layer params from stacked arrays
        # In nano-egg, blocks are stacked along axis 0 via scan_init
        def get_layer_param(param_tree, key, layer_idx=layer_idx):
            val = param_tree[key]
            if isinstance(val, dict):
                return {k: _jax_to_numpy(v)[layer_idx] for k, v in val.items()}
            arr = _jax_to_numpy(val)
            if arr.ndim > 0 and arr.shape[0] == num_hidden_layers:
                return arr[layer_idx]
            return arr

        # LayerNorm 1
        ln1_weight = get_layer_param(blocks["ln1"], "weight")
        if isinstance(ln1_weight, dict):
            ln1_weight = _jax_to_numpy(list(ln1_weight.values())[0])
        state_dict[f"{prefix}.ln1.weight"] = torch.from_numpy(
            np.asarray(ln1_weight)
        ).to(torch.int8)

        # GRU mixer
        att = blocks["att"]
        for gru_key in ["Wf", "Uf", "Wh", "Uh"]:
            weight = get_layer_param(att[gru_key], gru_key)
            if isinstance(weight, dict):
                # Linear layer has a nested structure
                weight = _jax_to_numpy(list(weight.values())[0])
            state_dict[f"{prefix}.mixer.{gru_key}.weight"] = torch.from_numpy(
                np.asarray(weight)
            ).to(torch.int8)

        for bias_key in ["bf", "bh"]:
            bias = get_layer_param(att, bias_key)
            if isinstance(bias, dict):
                bias = _jax_to_numpy(list(bias.values())[0])
            state_dict[f"{prefix}.mixer.{bias_key}"] = torch.from_numpy(
                np.asarray(bias)
            ).to(torch.int8)

        # LayerNorm 2
        ln2_weight = get_layer_param(blocks["ln2"], "weight")
        if isinstance(ln2_weight, dict):
            ln2_weight = _jax_to_numpy(list(ln2_weight.values())[0])
        state_dict[f"{prefix}.ln2.weight"] = torch.from_numpy(
            np.asarray(ln2_weight)
        ).to(torch.int8)

        # MLP
        mlp = blocks["mlp"]
        for mlp_idx, vllm_name in enumerate(["fc1", "fc2"]):
            mlp_layer = get_layer_param(mlp, str(mlp_idx))
            if isinstance(mlp_layer, dict):
                for k, v in mlp_layer.items():
                    arr = _jax_to_numpy(v)
                    if "weight" in k or arr.ndim == 2:
                        state_dict[f"{prefix}.mlp.{vllm_name}.weight"] = (
                            torch.from_numpy(arr).to(torch.int8)
                        )
                    elif "bias" in k or arr.ndim == 1:
                        state_dict[f"{prefix}.mlp.{vllm_name}.bias"] = torch.from_numpy(
                            arr
                        ).to(torch.int8)
            else:
                state_dict[f"{prefix}.mlp.{vllm_name}.weight"] = torch.from_numpy(
                    np.asarray(mlp_layer)
                ).to(torch.int8)

    # Final LayerNorm
    ln_out_weight = _jax_to_numpy(params["ln_out"]["weight"])
    state_dict["backbone.norm_f.weight"] = torch.from_numpy(ln_out_weight).to(
        torch.int8
    )

    # LM head
    head_weight = _jax_to_numpy(params["head"])
    state_dict["lm_head.weight"] = torch.from_numpy(head_weight).to(torch.int8)

    # Save weights
    save_file(state_dict, str(output_path / "model.safetensors"))

    # Save config
    config = {
        "model_type": "egg",
        "architectures": ["EGGForCausalLM"],
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "intermediate_size": intermediate_size,
        "fixed_point_bits": 4,
        "use_int8_weights": True,
        "tie_word_embeddings": False,
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Converted model saved to {output_path}")
    print(f"Total parameters: {sum(p.numel() for p in state_dict.values())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert nano-egg checkpoint to HuggingFace format"
    )
    parser.add_argument("--input", required=True, help="Path to .model checkpoint file")
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for HuggingFace model",
    )
    parser.add_argument("--vocab_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_hidden_layers", type=int, default=6)
    parser.add_argument("--intermediate_size", type=int, default=1024)
    args = parser.parse_args()

    convert_nano_egg_checkpoint(
        args.input,
        args.output_dir,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        intermediate_size=args.intermediate_size,
    )
