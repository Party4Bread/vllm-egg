# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EGG GRU mixer layer for EGGROLL models.

Implements the minGRU-based recurrent mixing layer used in EGG models
trained with the EGGROLL evolutionary strategy. The layer uses int8
fixed-point arithmetic for weights and activations.

Reference: https://github.com/ESHyperscale/nano-egg
"""

import torch
from torch import nn
from torch.nn.parameter import Parameter

from vllm.config import CacheConfig, ModelConfig, get_current_vllm_config
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.egg_attn import EGGAttentionMetadata

# Fixed-point constants matching nano-egg reference implementation
FIXED_POINT = 4
INT8_MAX = 127
LOGMAX = 7


class EGGFixedPointLayerNorm(nn.Module):
    """Layer normalization using fixed-point int8 arithmetic.

    Implements the EGG_LN from nano-egg, which normalizes by dividing
    each element by the mean absolute value of the input vector.
    Uses a precomputed division lookup table for integer division.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = Parameter(
            torch.ones(hidden_size, dtype=torch.int8) * (2**FIXED_POINT),
        )
        set_weight_attrs(self.weight, {"weight_loader": self._weight_loader})

        # Precompute division table: result[divisor][numerator]
        # Maps (abs_mean, weighted_value) -> normalized_value
        numerators = torch.arange(2**16, dtype=torch.int16).view(torch.uint16)
        divisors = torch.arange(2**8, dtype=torch.uint8)
        # Clip division result to int8 range
        self.register_buffer(
            "division_table",
            torch.clamp(
                numerators[None, :].to(torch.int16)
                // divisors[:, None].clamp(min=1).to(torch.int16),
                -INT8_MAX,
                INT8_MAX,
            ).to(torch.int8),
            persistent=False,
        )

    @staticmethod
    def _weight_loader(param: Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(torch.int32)
        # Compute mean absolute value as divisor
        abs_sum = torch.clamp(
            torch.sum(torch.abs(x).to(torch.int32), dim=-1, keepdim=True)
            // x.shape[-1],
            min=1,
        )
        # Multiply by weight, cast to int16, then use division table
        numerator = (x.to(torch.int32) * weight).to(torch.int16)
        numerator_u16 = numerator.view(torch.uint16)
        abs_sum_idx = abs_sum.squeeze(-1).clamp(0, 255).to(torch.long)
        # Perform lookup: division_table[abs_sum][numerator]
        # We need to do this element-wise
        batch_shape = x.shape[:-1]
        flat_x = numerator_u16.reshape(-1, self.hidden_size)
        flat_abs = abs_sum_idx.reshape(-1)
        results = []
        for i in range(flat_x.shape[0]):
            div_row = self.division_table[flat_abs[i]]
            results.append(div_row[flat_x[i].to(torch.long)])
        return torch.stack(results).reshape(*batch_shape, self.hidden_size)


class EGGFixedPointLinear(nn.Module):
    """Fixed-point int8 matrix multiplication.

    Implements the MM class from nano-egg. Performs int8 matmul
    with fixed-point scaling.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.zeros(out_features, in_features, dtype=torch.int8)
        )
        set_weight_attrs(self.weight, {"weight_loader": self._weight_loader})
        self._scale = 1.0 / ((2**FIXED_POINT) * int(in_features**0.5))

    @staticmethod
    def _weight_loader(param: Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Int8 matmul: x @ weight^T, scaled by fixed-point factor
        result = torch.matmul(x.to(torch.int32), self.weight.to(torch.int32).t())
        result = torch.clamp(
            (result * self._scale).to(torch.int64) // 1,
            -INT8_MAX,
            INT8_MAX,
        ).to(torch.int8)
        return result


def clipped_add(*tensors: torch.Tensor) -> torch.Tensor:
    """Add tensors with int8 saturation."""
    result = sum(t.to(torch.int32) for t in tensors)
    return torch.clamp(result, -INT8_MAX, INT8_MAX).to(torch.int8)


@PluggableLayer.register("egg_gru_mixer")
class EGGGRUMixer(MambaBase, PluggableLayer):
    """GRU-based recurrent mixer for EGG models.

    Implements the EGG_GRU from nano-egg using int8 fixed-point arithmetic.
    The GRU uses the minGRU formulation with forget gate and candidate state.

    State: single hidden state tensor of shape (hidden_size,) per sequence.
    """

    def __init__(
        self,
        hidden_size: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.model_config = model_config
        self.cache_config = cache_config
        self.prefix = prefix

        # Forget gate: ft = sigmoid(Wf @ x + Uf @ state + bf)
        self.Wf = EGGFixedPointLinear(hidden_size, hidden_size)
        self.Uf = EGGFixedPointLinear(hidden_size, hidden_size)
        self.bf = Parameter(torch.zeros(hidden_size, dtype=torch.int8))
        set_weight_attrs(self.bf, {"weight_loader": self._bias_loader})

        # Candidate state: ht = tanh(Wh @ x + Uh @ gated_past + bh)
        self.Wh = EGGFixedPointLinear(hidden_size, hidden_size)
        self.Uh = EGGFixedPointLinear(hidden_size, hidden_size)
        self.bh = Parameter(torch.zeros(hidden_size, dtype=torch.int8))
        set_weight_attrs(self.bh, {"weight_loader": self._bias_loader})

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        # Single GRU state tensor
        self.kv_cache = (torch.tensor([]),)

    @staticmethod
    def _bias_loader(param: Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def _gru_step(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Compute one GRU step using int8 fixed-point arithmetic.

        Args:
            x: Input tensor of shape (..., hidden_size), dtype int8
            state: Previous hidden state (..., hidden_size), dtype int8

        Returns:
            New hidden state (..., hidden_size), dtype int8
        """
        # Forget gate: ft = Wf(x) + Uf(state) + bf
        # In nano-egg, sigmoid is identity for int8
        ft = clipped_add(
            self.Wf(x),
            self.Uf(state),
            self.bf,
        )

        # Gated past: gated_past = ((ft + MAX) * state) >> (LOGMAX + 1)
        ft_i32 = ft.to(torch.int32)
        state_i32 = state.to(torch.int32)
        gated_past = ((ft_i32 + INT8_MAX) * state_i32 >> (LOGMAX + 1)).to(x.dtype)

        # Candidate: ht_candidate = Wh(x) + Uh(gated_past) + bh
        # In nano-egg, tanh is identity for int8
        ht_candidate = clipped_add(
            self.Wh(x),
            self.Uh(gated_past),
            self.bh,
        )

        # New state: state + ((ft + MAX) * (candidate - state)) >> (LOGMAX+1)
        ht_cand_i32 = ht_candidate.to(torch.int32)
        new_state = state_i32 + (
            (ft_i32 + INT8_MAX) * (ht_cand_i32 - state_i32) >> (LOGMAX + 1)
        )
        return torch.clamp(new_state, -INT8_MAX, INT8_MAX).to(x.dtype)

    def forward(self, hidden_states: torch.Tensor, output: torch.Tensor):
        torch.ops.vllm.egg_gru_mixer(
            hidden_states,
            output,
            self.prefix,
        )

    def forward_impl(self, hidden_states: torch.Tensor, output: torch.Tensor):
        """Run the EGG GRU mixer.

        Handles both prefill (sequential scan over tokens) and decode
        (single-step state update) cases.
        """
        forward_context: ForwardContext = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if attn_metadata is None:
            # Profile run - return input unchanged
            output[: hidden_states.shape[0]] = hidden_states
            return

        assert isinstance(attn_metadata, dict)
        attn_metadata = attn_metadata[self.prefix]
        assert isinstance(attn_metadata, EGGAttentionMetadata)

        gru_state = self.kv_cache[0]
        state_indices_tensor_p = attn_metadata.state_indices_tensor_p
        state_indices_tensor_d = attn_metadata.state_indices_tensor_d
        has_initial_states_p = attn_metadata.has_initial_states_p

        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_actual_tokens = num_prefill_tokens + num_decode_tokens

        results = []

        # Process decode tokens (come first in v1)
        if num_decode_tokens > 0:
            decode_hidden = hidden_states[:num_decode_tokens]
            assert state_indices_tensor_d is not None

            state_indices = state_indices_tensor_d.to(torch.long)
            current_state = gru_state[state_indices]
            new_state = self._gru_step(decode_hidden, current_state)
            gru_state[state_indices] = new_state
            results.append(new_state)

        # Process prefill tokens (come after decode in v1)
        if num_prefill_tokens > 0:
            prefill_hidden = hidden_states[num_decode_tokens:num_actual_tokens]
            assert state_indices_tensor_p is not None
            assert attn_metadata.query_start_loc_p is not None

            query_start_loc = attn_metadata.query_start_loc_p
            num_prefills = attn_metadata.num_prefills

            prefill_outputs = []
            for i in range(num_prefills):
                start = query_start_loc[i].item()
                end = query_start_loc[i + 1].item()
                seq_tokens = prefill_hidden[start:end]
                state_idx = state_indices_tensor_p[i].to(torch.long)

                # Initialize state
                if has_initial_states_p is not None and has_initial_states_p[i]:
                    state = gru_state[state_idx].clone()
                else:
                    state = torch.zeros(
                        self.hidden_size,
                        dtype=seq_tokens.dtype,
                        device=seq_tokens.device,
                    )

                # Sequential scan over tokens
                seq_outputs = []
                for t in range(seq_tokens.shape[0]):
                    state = self._gru_step(
                        seq_tokens[t : t + 1], state.unsqueeze(0)
                    ).squeeze(0)
                    seq_outputs.append(state)

                prefill_outputs.append(torch.stack(seq_outputs))
                gru_state[state_idx] = state

            results.append(torch.cat(prefill_outputs, dim=0))

        combined = results[0] if len(results) == 1 else torch.cat(results, dim=0)
        output[:num_actual_tokens] = combined

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        # GRU state is stored as float32 for numerical stability
        # even though computations use int8
        return (torch.float32,)

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        return ((self.hidden_size,),)

    @property
    def mamba_type(self) -> str:
        return "egg_gru"


def egg_gru_mixer(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self.forward_impl(hidden_states=hidden_states, output=output)


def egg_gru_mixer_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="egg_gru_mixer",
    op_func=egg_gru_mixer,
    mutates_args=["output"],
    fake_impl=egg_gru_mixer_fake,
)
