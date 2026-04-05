# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HuggingFace-compatible configuration for EGG (EGGROLL) models.

EGG is a minGRU-based language model trained using the EGGROLL evolutionary
strategy with int8 fixed-point arithmetic.

Reference: https://github.com/ESHyperscale/nano-egg
"""

from transformers import PretrainedConfig


class EGGConfig(PretrainedConfig):
    model_type = "egg"

    def __init__(
        self,
        vocab_size: int = 256,
        hidden_size: int = 256,
        num_hidden_layers: int = 6,
        intermediate_size: int = 1024,
        fixed_point_bits: int = 4,
        use_int8_weights: bool = True,
        tie_word_embeddings: bool = False,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.fixed_point_bits = fixed_point_bits
        self.use_int8_weights = use_int8_weights

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
