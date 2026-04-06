# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EGGROLL: Evolutionary Gradient-based Generation with LoRA over vLLM.

This package implements the EGGROLL evolutionary strategy for training/
fine-tuning language models using vLLM as the inference backend.

Reference: https://github.com/ESHyperscale/HyperscaleES
"""

from vllm.eggroll.noiser import EggRoll, OpenES
from vllm.eggroll.trainer import EggRollTrainer, MultiGPUEggRollTrainer

__all__ = ["EggRoll", "OpenES", "EggRollTrainer", "MultiGPUEggRollTrainer"]
