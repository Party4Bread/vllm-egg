# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention backend for EGG GRU state management."""

from dataclasses import dataclass

from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder,
)


class EGGAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "EGG_GRU_ATTN"

    @staticmethod
    def get_builder_cls() -> type["EGGAttentionMetadataBuilder"]:
        return EGGAttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True


@dataclass
class EGGAttentionMetadata(BaseMambaAttentionMetadata):
    pass


class EGGAttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[EGGAttentionMetadata]
):
    metadata_cls = EGGAttentionMetadata
