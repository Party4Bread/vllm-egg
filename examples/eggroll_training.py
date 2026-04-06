# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Example: EGGROLL evolutionary training with vLLM.

This example demonstrates using the EGGROLL evolutionary strategy
to fine-tune a language model using vLLM for efficient inference.

The EGGROLL algorithm from HyperscaleES replaces backpropagation with:
1. Population-based perturbation of model weights (LoRA-style)
2. Fitness evaluation via text generation
3. Evolutionary gradient estimation
4. Optimizer-based parameter updates

Usage:
    python examples/eggroll_training.py \\
        --model meta-llama/Llama-3.1-8B \\
        --sigma 1e-3 \\
        --lr 1e-4 \\
        --population-size 64 \\
        --num-epochs 100

Reference: https://github.com/ESHyperscale/HyperscaleES
"""

import argparse
import logging

import torch

from vllm.eggroll import EggRollTrainer
from vllm.eggroll.trainer import EggRollConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def simple_reward_fn(prompts: list[str], generations: list[str]) -> torch.Tensor:
    """Example fitness function: reward longer, more diverse generations.

    In practice, replace this with your task-specific reward function
    (e.g., RLHF reward model, code execution, math verification).
    """
    scores = []
    for gen in generations:
        # Simple heuristic: reward length and vocabulary diversity
        length_score = min(len(gen) / 200.0, 1.0)
        unique_words = len(set(gen.split()))
        diversity_score = min(unique_words / 50.0, 1.0)
        scores.append(length_score * 0.5 + diversity_score * 0.5)
    return torch.tensor(scores, dtype=torch.float32)


def get_prompts(epoch: int, num_prompts: int) -> list[str]:
    """Generate training prompts for each epoch."""
    base_prompts = [
        "Explain the concept of evolutionary strategies in machine learning.",
        "Write a Python function that sorts a list of numbers.",
        "What are the benefits of using LoRA for model fine-tuning?",
        "Describe how antithetical sampling reduces variance.",
        "Explain the difference between gradient descent and evolution.",
        "Write a haiku about artificial intelligence.",
        "What is the EGGROLL algorithm and how does it work?",
        "Explain why vLLM is efficient for batched text generation.",
    ]
    # Cycle through prompts
    prompts = []
    for i in range(num_prompts):
        prompts.append(base_prompts[i % len(base_prompts)])
    return prompts


def main():
    parser = argparse.ArgumentParser(
        description="EGGROLL evolutionary training with vLLM"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-125m",
        help="HuggingFace model name or path",
    )
    parser.add_argument("--sigma", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--population-size", type=int, default=16)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--noiser", type=str, default="eggroll", choices=["eggroll", "open_es"]
    )
    parser.add_argument(
        "--freeze-nonlora",
        action="store_true",
        default=True,
        help="Freeze non-matrix parameters",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-lora",
        action="store_true",
        default=True,
        help="Use LoRA-batched generation (recommended)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs for multi-GPU population sharding",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to save/load checkpoints",
    )
    args = parser.parse_args()

    config = EggRollConfig(
        sigma=args.sigma,
        lr=args.lr,
        population_size=args.population_size,
        rank=args.rank,
        noise_reuse=0,
        freeze_nonlora=args.freeze_nonlora,
        noiser_type=args.noiser,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        num_epochs=args.num_epochs,
        seed=args.seed,
        use_lora=args.use_lora,
        optimizer_cls="Adam",
        optimizer_kwargs={"betas": (0.9, 0.999)},
    )

    if args.num_gpus > 1:
        from vllm.eggroll import MultiGPUEggRollTrainer

        trainer = MultiGPUEggRollTrainer(
            model_name=args.model,
            config=config,
            fitness_fn=simple_reward_fn,
            num_workers=args.num_gpus,
            prompt_fn=get_prompts,
        )
    else:
        trainer = EggRollTrainer(
            model_name=args.model,
            config=config,
            fitness_fn=simple_reward_fn,
            prompt_fn=get_prompts,
        )

    logger.info("Starting EGGROLL training with config:")
    logger.info("  Model: %s", args.model)
    logger.info("  Population: %d", config.population_size)
    logger.info("  Sigma: %e", config.sigma)
    logger.info("  LR: %e", config.lr)
    logger.info("  Rank: %d", config.rank)
    logger.info("  Noiser: %s", config.noiser_type)
    logger.info("  Mode: %s", "lora-batched" if args.use_lora else "sequential")
    logger.info("  GPUs: %d", args.num_gpus)

    stats = trainer.train()

    logger.info("Training complete!")
    logger.info("Final epoch stats:")
    final = stats[-1]
    logger.info("  Mean fitness: %.4f", final.mean_fitness)
    logger.info("  Max fitness: %.4f", final.max_fitness)
    logger.info("  Param change: %.6f", final.param_change_norm)

    if args.checkpoint_path:
        trainer.save_checkpoint(args.checkpoint_path)


if __name__ == "__main__":
    main()
