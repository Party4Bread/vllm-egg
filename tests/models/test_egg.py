# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the EGG (EGGROLL) model implementation."""

import json
import os
import tempfile

import torch
from safetensors.torch import save_file
from torch import nn

from vllm.model_executor.layers.mamba.egg_mixer import (
    FIXED_POINT,
    INT8_MAX,
    EGGFixedPointLayerNorm,
    EGGFixedPointLinear,
    clipped_add,
)
from vllm.model_executor.models.egg import (
    EGGMLP,
    EGGFixedPointEmbedding,
)
from vllm.transformers_utils.configs.egg import EGGConfig


class TestEGGConfig:
    def test_default_config(self):
        config = EGGConfig()
        assert config.vocab_size == 256
        assert config.hidden_size == 256
        assert config.num_hidden_layers == 6
        assert config.intermediate_size == 1024
        assert config.fixed_point_bits == 4
        assert config.use_int8_weights is True
        assert config.model_type == "egg"

    def test_custom_config(self):
        config = EGGConfig(
            vocab_size=512,
            hidden_size=128,
            num_hidden_layers=4,
            intermediate_size=512,
        )
        assert config.vocab_size == 512
        assert config.hidden_size == 128
        assert config.num_hidden_layers == 4
        assert config.intermediate_size == 512


class TestClippedAdd:
    def test_basic_add(self):
        a = torch.tensor([10, 20, 30], dtype=torch.int8)
        b = torch.tensor([5, 10, 15], dtype=torch.int8)
        result = clipped_add(a, b)
        expected = torch.tensor([15, 30, 45], dtype=torch.int8)
        assert torch.equal(result, expected)

    def test_saturation(self):
        a = torch.tensor([100, -100], dtype=torch.int8)
        b = torch.tensor([100, -100], dtype=torch.int8)
        result = clipped_add(a, b)
        assert result[0].item() == INT8_MAX
        assert result[1].item() == -INT8_MAX

    def test_three_operands(self):
        a = torch.tensor([10], dtype=torch.int8)
        b = torch.tensor([20], dtype=torch.int8)
        c = torch.tensor([30], dtype=torch.int8)
        result = clipped_add(a, b, c)
        assert result[0].item() == 60


class TestEGGFixedPointLinear:
    def test_shape(self):
        linear = EGGFixedPointLinear(16, 32)
        x = torch.randint(-127, 127, (4, 16), dtype=torch.int8)
        result = linear(x)
        assert result.shape == (4, 32)
        assert result.dtype == torch.int8

    def test_output_range(self):
        linear = EGGFixedPointLinear(16, 32)
        nn.init.constant_(linear.weight, 1)
        x = torch.randint(-10, 10, (4, 16), dtype=torch.int8)
        result = linear(x)
        assert (result >= -INT8_MAX).all()
        assert (result <= INT8_MAX).all()


class TestEGGFixedPointEmbedding:
    def test_lookup(self):
        emb = EGGFixedPointEmbedding(256, 16)
        nn.init.constant_(emb.weight, 42)
        ids = torch.tensor([0, 1, 255])
        result = emb(ids)
        assert result.shape == (3, 16)
        assert result.dtype == torch.int8


class TestEGGMLP:
    def test_shape(self):
        mlp = EGGMLP(16, 64)
        x = torch.randint(-50, 50, (4, 16), dtype=torch.int8)
        result = mlp(x)
        assert result.shape == (4, 16)
        assert result.dtype == torch.int8


class TestEGGFixedPointLayerNorm:
    def test_shape(self):
        ln = EGGFixedPointLayerNorm(16)
        x = torch.randint(-50, 50, (4, 16), dtype=torch.int8)
        result = ln(x)
        assert result.shape == (4, 16)
        assert result.dtype == torch.int8


class TestModelCheckpointCreation:
    """Test that a dummy EGG model config can be saved and loaded."""

    def test_save_load_config(self):
        config = EGGConfig(
            vocab_size=256,
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=256,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            config_dict = {
                "model_type": "egg",
                "architectures": ["EGGForCausalLM"],
                "vocab_size": config.vocab_size,
                "hidden_size": config.hidden_size,
                "num_hidden_layers": config.num_hidden_layers,
                "intermediate_size": config.intermediate_size,
                "fixed_point_bits": config.fixed_point_bits,
                "use_int8_weights": config.use_int8_weights,
            }
            with open(config_path, "w") as f:
                json.dump(config_dict, f)

            with open(config_path) as f:
                loaded = json.load(f)
            assert loaded["model_type"] == "egg"
            assert loaded["hidden_size"] == 64

    def test_create_dummy_weights(self):
        """Test that dummy weights can be created and saved."""
        hidden_size = 64
        vocab_size = 256
        intermediate_size = 256
        num_layers = 2

        state_dict = {}
        state_dict["backbone.embeddings.weight"] = torch.randint(
            -127, 127, (vocab_size, hidden_size), dtype=torch.int8
        )
        for i in range(num_layers):
            prefix = f"backbone.layers.{i}"
            state_dict[f"{prefix}.ln1.weight"] = torch.ones(
                hidden_size, dtype=torch.int8
            ) * (2**FIXED_POINT)
            state_dict[f"{prefix}.mixer.Wf.weight"] = torch.randint(
                -10, 10, (hidden_size, hidden_size), dtype=torch.int8
            )
            state_dict[f"{prefix}.mixer.Uf.weight"] = torch.randint(
                -10, 10, (hidden_size, hidden_size), dtype=torch.int8
            )
            state_dict[f"{prefix}.mixer.bf"] = torch.zeros(
                hidden_size, dtype=torch.int8
            )
            state_dict[f"{prefix}.mixer.Wh.weight"] = torch.randint(
                -10, 10, (hidden_size, hidden_size), dtype=torch.int8
            )
            state_dict[f"{prefix}.mixer.Uh.weight"] = torch.randint(
                -10, 10, (hidden_size, hidden_size), dtype=torch.int8
            )
            state_dict[f"{prefix}.mixer.bh"] = torch.zeros(
                hidden_size, dtype=torch.int8
            )
            state_dict[f"{prefix}.ln2.weight"] = torch.ones(
                hidden_size, dtype=torch.int8
            ) * (2**FIXED_POINT)
            state_dict[f"{prefix}.mlp.fc1.weight"] = torch.randint(
                -10, 10, (intermediate_size, hidden_size), dtype=torch.int8
            )
            state_dict[f"{prefix}.mlp.fc2.weight"] = torch.randint(
                -10, 10, (hidden_size, intermediate_size), dtype=torch.int8
            )
        state_dict["backbone.norm_f.weight"] = torch.ones(
            hidden_size, dtype=torch.int8
        ) * (2**FIXED_POINT)
        state_dict["lm_head.weight"] = torch.randint(
            -10, 10, (vocab_size, hidden_size), dtype=torch.int8
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_file(state_dict, os.path.join(tmpdir, "model.safetensors"))
            assert os.path.exists(os.path.join(tmpdir, "model.safetensors"))
