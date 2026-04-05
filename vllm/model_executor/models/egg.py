# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PyTorch EGG (EGGROLL) model for vLLM inference.

EGG is a minGRU-based language model trained using the EGGROLL evolutionary
strategy with int8 fixed-point arithmetic. This module provides inference
support for EGG models within vLLM's serving framework.

Architecture:
  - Embedding layer (int8 fixed-point)
  - N x EGGDecoderLayer:
      - LayerNorm (fixed-point division)
      - GRU mixer (minGRU with forget gate)
      - LayerNorm
      - MLP (with ReLU activation)
      - Residual connections
  - Final LayerNorm
  - LM head (int8 matmul)

Reference: https://github.com/ESHyperscale/nano-egg
"""

from collections.abc import Iterable
from itertools import islice
from typing import ClassVar

import torch
from torch import nn
from torch.nn.parameter import Parameter

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.egg_mixer import (
    EGGFixedPointLayerNorm,
    EGGFixedPointLinear,
    EGGGRUMixer,
    clipped_add,
)
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFunc
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsAttentionFree,
    SupportsMambaPrefixCaching,
    SupportsPP,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.egg import EGGConfig

from .utils import (
    AutoWeightsLoader,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)


class EGGFixedPointEmbedding(nn.Module):
    """Int8 fixed-point embedding layer for EGG models."""

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.weight = Parameter(torch.zeros(vocab_size, hidden_size, dtype=torch.int8))
        set_weight_attrs(self.weight, {"weight_loader": self._weight_loader})

    @staticmethod
    def _weight_loader(param: Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[input_ids]


class EGGMLP(nn.Module):
    """MLP block for EGG models with ReLU activation.

    Uses two fixed-point linear layers with ReLU in between:
      x -> Linear1 (hidden -> intermediate) -> ReLU -> Linear2 (intermediate -> hidden)
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = EGGFixedPointLinear(hidden_size, intermediate_size)
        self.fc2 = EGGFixedPointLinear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        # ReLU in int8: clamp negative values to 0
        x = torch.clamp(x.to(torch.int32), min=0).to(torch.int8)
        x = self.fc2(x)
        return x


class EGGDecoderLayer(nn.Module):
    """Single EGG decoder layer.

    Architecture: LN1 -> GRU -> residual -> LN2 -> MLP -> residual
    """

    def __init__(
        self,
        config: EGGConfig,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.ln1 = EGGFixedPointLayerNorm(config.hidden_size)
        self.mixer = EGGGRUMixer(
            hidden_size=config.hidden_size,
            model_config=model_config,
            cache_config=cache_config,
            prefix=f"{prefix}.mixer",
        )
        self.ln2 = EGGFixedPointLayerNorm(config.hidden_size)
        self.mlp = EGGMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.ln1(hidden_states)
        else:
            normed = self.ln1(hidden_states)
            residual = hidden_states
            hidden_states = normed

        # GRU mixing
        output = torch.empty_like(hidden_states)
        self.mixer(hidden_states, output)
        hidden_states = clipped_add(output, residual)

        # MLP
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = clipped_add(hidden_states, residual)

        return hidden_states, None


@support_torch_compile
class EGGModel(nn.Module):
    """EGG backbone model."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        self.config = config
        self.vocab_size = config.vocab_size

        self.embeddings = EGGFixedPointEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: EGGDecoderLayer(
                config,
                model_config=model_config,
                cache_config=cache_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        self.norm_f = EGGFixedPointLayerNorm(config.hidden_size)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states = self.norm_f(hidden_states)

        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if is_pp_missing_parameter(name, self):
                continue
            if name not in params_dict:
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class EGGForCausalLM(
    nn.Module,
    HasInnerState,
    IsAttentionFree,
    SupportsPP,
    SupportsMambaPrefixCaching,
):
    """EGG model for causal language modeling.

    Top-level model that wraps EGGModel backbone with an LM head
    for token prediction. Integrates with vLLM's state management
    for recurrent inference.
    """

    # Required by SupportsMambaPrefixCaching
    supports_mamba_prefix_caching: ClassVar[bool] = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config

        self.scheduler_config = vllm_config.scheduler_config

        super().__init__()
        self.config = config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.backbone = EGGModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "backbone"),
        )

        # LM head using fixed-point linear
        self.lm_head = EGGFixedPointLinear(config.hidden_size, config.vocab_size)

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.backbone.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.backbone.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        hidden_states = self.backbone(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, ...]:
        # GRU state stored as float32
        return (torch.float32,)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, ...], ...]:
        hf_config = vllm_config.model_config.hf_config
        return ((hf_config.hidden_size,),)

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, ...]:
        # Simple copy for GRU state - just memcpy the state tensor
        def gru_state_copy(
            state: torch.Tensor,
            block_ids: list[int],
            cur_block_idx: int,
            num_accepted_tokens: int,
        ):
            from vllm.model_executor.layers.mamba.mamba_utils import (
                MambaCopySpec,
            )

            src_start = block_ids[cur_block_idx] * state.shape[-1]
            return MambaCopySpec(
                start_addr=src_start,
                num_elements=state.shape[-1],
            )

        return (gru_state_copy,)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Convert int8 hidden states to float for logits processing
        float_hidden = hidden_states.to(torch.float32)
        # Use the LM head in float mode for logits
        weight = self.lm_head.weight.to(torch.float32)
        logits = torch.matmul(float_hidden, weight.t())
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
