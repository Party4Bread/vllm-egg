# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EGGROLL trainer using vLLM for efficient batched generation.

This module provides the training loop that integrates the EGGROLL
evolutionary strategy with vLLM's high-throughput inference engine.

The key optimization: vLLM can efficiently generate text for an entire
population of perturbed models using batched inference, which is the
bottleneck in evolutionary strategy training for LLMs.

Workflow:
  1. For each population member, apply LoRA perturbation to weights
  2. Use vLLM to generate text from all perturbed models
  3. Evaluate fitness of each generation
  4. Estimate gradients from fitness scores
  5. Update model weights via optimizer

Reference: https://github.com/ESHyperscale/HyperscaleES
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from vllm.eggroll.noiser import (
    EggRoll,
    FrozenNoiserParams,
    NoisedParams,
    OpenES,
)

logger = logging.getLogger(__name__)


@dataclass
class EggRollConfig:
    """Configuration for EGGROLL training."""

    # Evolutionary strategy parameters
    sigma: float = 1e-3
    lr: float = 1e-4
    population_size: int = 64
    rank: int = 8
    noise_reuse: int = 0
    freeze_nonlora: bool = True
    group_size: int = 0
    noiser_type: str = "eggroll"  # "eggroll" or "open_es"

    # Optimizer
    optimizer_cls: str = "Adam"
    optimizer_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"betas": (0.9, 0.999)}
    )

    # Generation parameters
    max_tokens: int = 100
    temperature: float = 0.0
    top_p: float = 1.0

    # Training loop
    num_epochs: int = 100
    validate_every: int = 10
    log_every: int = 1

    # vLLM parameters
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9

    # Random seed
    seed: int = 42


@dataclass
class EpochStats:
    """Statistics for a single training epoch."""

    epoch: int
    mean_fitness: float
    std_fitness: float
    max_fitness: float
    min_fitness: float
    median_fitness: float
    generation_time: float
    fitness_time: float
    update_time: float
    param_change_norm: float


class EggRollTrainer:
    """EGGROLL evolutionary strategy trainer using vLLM.

    This trainer wraps a language model and uses vLLM for efficient
    batched text generation during evolutionary optimization.

    Example usage::

        from vllm import LLM
        from vllm.eggroll import EggRollTrainer, EggRollConfig

        config = EggRollConfig(
            sigma=1e-3,
            lr=1e-4,
            population_size=64,
            rank=8,
            num_epochs=100,
        )


        def fitness_fn(prompts, generations):
            # Return fitness score for each generation
            scores = []
            for prompt, gen in zip(prompts, generations):
                scores.append(evaluate(prompt, gen))
            return torch.tensor(scores)


        trainer = EggRollTrainer(
            model_name="meta-llama/Llama-3.1-8B",
            config=config,
            fitness_fn=fitness_fn,
        )
        trainer.train(prompts=my_prompts)
    """

    def __init__(
        self,
        model_name: str,
        config: EggRollConfig,
        fitness_fn: Callable[[list[str], list[str]], torch.Tensor],
        prompt_fn: Callable[[int, int], list[str]] | None = None,
        validation_fn: Callable[[Any, int], float] | None = None,
    ):
        """Initialize the EGGROLL trainer.

        Args:
            model_name: HuggingFace model name or path.
            config: Training configuration.
            fitness_fn: Function mapping (prompts, generations) -> fitness
                scores tensor of shape (batch_size,).
            prompt_fn: Optional function that generates prompts for each epoch.
                Signature: (epoch, num_prompts) -> list[str].
            validation_fn: Optional validation function.
                Signature: (llm, epoch) -> validation_score.
        """
        self.model_name = model_name
        self.config = config
        self.fitness_fn = fitness_fn
        self.prompt_fn = prompt_fn
        self.validation_fn = validation_fn

        # These are initialized in setup()
        self._llm = None
        self._model = None
        self._noiser_cls = None
        self._frozen: FrozenNoiserParams | None = None
        self._noised: NoisedParams | None = None
        self._param_seeds: dict[str, int] | None = None
        self._original_params: dict[str, torch.Tensor] | None = None

    def setup(self) -> None:
        """Initialize vLLM engine and evolutionary strategy components."""
        from vllm import LLM

        logger.info("Initializing vLLM with model: %s", self.model_name)
        self._llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            seed=self.config.seed,
            enforce_eager=True,  # Required for weight modification
        )

        # Get the underlying model for weight perturbation
        self._model = self._get_model()

        # Select noiser
        if self.config.noiser_type == "eggroll":
            self._noiser_cls = EggRoll
        elif self.config.noiser_type == "open_es":
            self._noiser_cls = OpenES
        else:
            raise ValueError(f"Unknown noiser type: {self.config.noiser_type}")

        # Get optimizer class
        optim_cls = getattr(torch.optim, self.config.optimizer_cls)

        # Initialize noiser
        self._frozen, self._noised = self._noiser_cls.init(
            model=self._model,
            sigma=self.config.sigma,
            lr=self.config.lr,
            optimizer_cls=optim_cls,
            optimizer_kwargs=self.config.optimizer_kwargs,
            group_size=self.config.group_size,
            freeze_nonlora=self.config.freeze_nonlora,
            noise_reuse=self.config.noise_reuse,
            rank=self.config.rank,
        )

        # Generate per-parameter seeds
        self._param_seeds = self._noiser_cls.get_param_seeds(
            self._model, self.config.seed
        )

        # Snapshot original parameters
        self._original_params = {
            name: param.data.clone() for name, param in self._model.named_parameters()
        }

        logger.info(
            "EGGROLL initialized: pop_size=%d, sigma=%e, lr=%e, rank=%d",
            self.config.population_size,
            self.config.sigma,
            self.config.lr,
            self.config.rank,
        )

    def _get_model(self) -> nn.Module:
        """Extract the underlying nn.Module from the vLLM engine."""
        # Access the model runner's model
        workers = self._llm.llm_engine.model_executor.drivers
        if hasattr(workers, "__iter__"):
            worker = list(workers)[0]
        else:
            worker = self._llm.llm_engine.model_executor.driver_worker
        return worker.model_runner.model

    def _restore_params(self) -> None:
        """Restore model to original (unperturbed) parameters."""
        with torch.no_grad():
            for name, param in self._model.named_parameters():
                if name in self._original_params:
                    param.copy_(self._original_params[name])

    def _snapshot_params(self) -> None:
        """Save current model parameters as the new originals."""
        with torch.no_grad():
            for name, param in self._model.named_parameters():
                self._original_params[name] = param.data.clone()

    def _generate_batch(
        self,
        prompts: list[str],
        epoch: int,
    ) -> tuple[list[list[str]], torch.Tensor]:
        """Generate text for the full population.

        For each population member:
        1. Apply perturbation to model weights
        2. Generate text using vLLM
        3. Restore original weights

        Args:
            prompts: Input prompts.
            epoch: Current epoch number.

        Returns:
            (all_generations, fitness_scores) where all_generations[i]
            is a list of generated texts for population member i.
        """
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        pop_size = self.config.population_size
        all_generations = []
        all_prompts_flat = []
        member_indices = []

        for member_id in range(pop_size):
            # Apply perturbation for this population member
            self._noiser_cls.perturb_model(
                self._frozen,
                self._noised.sigma,
                self._model,
                epoch,
                member_id,
                self._param_seeds,
                self._original_params,
            )

            # Generate with perturbed model
            outputs = self._llm.generate(prompts, sampling_params)
            generations = [out.outputs[0].text for out in outputs]
            all_generations.append(generations)
            all_prompts_flat.extend(prompts)
            member_indices.extend([member_id] * len(prompts))

            # Restore original weights
            self._restore_params()

        return all_generations

    def _evaluate_fitness(
        self,
        prompts: list[str],
        all_generations: list[list[str]],
    ) -> torch.Tensor:
        """Evaluate fitness for all population members.

        Args:
            prompts: Input prompts (shared across population).
            all_generations: List of generation lists, one per member.

        Returns:
            Fitness scores tensor of shape (population_size,).
        """
        pop_size = len(all_generations)
        scores = torch.zeros(pop_size)

        for member_id in range(pop_size):
            member_score = self.fitness_fn(prompts, all_generations[member_id])
            if isinstance(member_score, torch.Tensor):
                scores[member_id] = member_score.mean()
            else:
                scores[member_id] = float(member_score)

        return scores

    def train_epoch(
        self,
        prompts: list[str],
        epoch: int,
    ) -> EpochStats:
        """Run a single epoch of EGGROLL training.

        Args:
            prompts: Training prompts for this epoch.
            epoch: Current epoch number.

        Returns:
            Statistics for this epoch.
        """
        pop_size = self.config.population_size

        # 1. Generate text for all population members
        t0 = time.time()
        all_generations = self._generate_batch(prompts, epoch)
        gen_time = time.time() - t0

        # 2. Evaluate fitness
        t0 = time.time()
        raw_scores = self._evaluate_fitness(prompts, all_generations)
        fitness_time = time.time() - t0

        # 3. Normalize fitness
        fitnesses = self._noiser_cls.convert_fitnesses(self._frozen, raw_scores)

        # 4. Compute gradient estimates
        t0 = time.time()
        epochs_tensor = torch.full((pop_size,), epoch, dtype=torch.int32)
        thread_ids = torch.arange(pop_size, dtype=torch.int32)

        gradients = self._noiser_cls.compute_gradients(
            self._frozen,
            self._noised.sigma,
            self._model,
            self._param_seeds,
            fitnesses,
            epochs_tensor,
            thread_ids,
        )

        # 5. Apply updates
        # Compute param change norm before update
        old_params = {n: p.data.clone() for n, p in self._model.named_parameters()}
        self._noiser_cls.update_params(self._noised, self._model, gradients)

        param_change = 0.0
        n_params = 0
        for name, param in self._model.named_parameters():
            if name in old_params:
                diff = (param.data - old_params[name]).float()
                param_change += (diff**2).sum().item()
                n_params += diff.numel()
        param_change_norm = (param_change / max(n_params, 1)) ** 0.5

        update_time = time.time() - t0

        # 6. Update original params snapshot
        self._snapshot_params()

        stats = EpochStats(
            epoch=epoch,
            mean_fitness=raw_scores.mean().item(),
            std_fitness=raw_scores.std().item(),
            max_fitness=raw_scores.max().item(),
            min_fitness=raw_scores.min().item(),
            median_fitness=raw_scores.median().item(),
            generation_time=gen_time,
            fitness_time=fitness_time,
            update_time=update_time,
            param_change_norm=param_change_norm,
        )

        return stats

    def train(
        self,
        prompts: list[str] | None = None,
    ) -> list[EpochStats]:
        """Run the full EGGROLL training loop.

        Args:
            prompts: Fixed set of training prompts. If None, prompt_fn
                must be provided at init time.

        Returns:
            List of per-epoch statistics.
        """
        if self._llm is None:
            self.setup()

        all_stats = []

        for epoch in range(self.config.num_epochs):
            # Get prompts for this epoch
            if self.prompt_fn is not None:
                epoch_prompts = self.prompt_fn(epoch, self.config.population_size)
            elif prompts is not None:
                epoch_prompts = prompts
            else:
                raise ValueError("Either prompts or prompt_fn must be provided")

            # Validation
            if (
                self.validation_fn is not None
                and epoch % self.config.validate_every == 0
            ):
                val_score = self.validation_fn(self._llm, epoch)
                logger.info("Epoch %d validation: %.4f", epoch, val_score)

            # Train
            stats = self.train_epoch(epoch_prompts, epoch)
            all_stats.append(stats)

            if epoch % self.config.log_every == 0:
                logger.info(
                    "Epoch %d: fitness=%.4f +/- %.4f "
                    "[%.4f, %.4f] gen=%.1fs update=%.1fs",
                    epoch,
                    stats.mean_fitness,
                    stats.std_fitness,
                    stats.min_fitness,
                    stats.max_fitness,
                    stats.generation_time,
                    stats.update_time,
                )

        return all_stats

    def save_checkpoint(self, path: str) -> None:
        """Save current model weights and optimizer state."""
        checkpoint = {
            "model_state_dict": {
                n: p.data.clone() for n, p in self._model.named_parameters()
            },
            "optimizer_state_dict": self._noised.optimizer.state_dict(),
            "step": self._noised.step,
            "config": self.config,
        }
        torch.save(checkpoint, path)
        logger.info("Checkpoint saved to %s", path)

    def load_checkpoint(self, path: str) -> None:
        """Load model weights and optimizer state from checkpoint."""
        checkpoint = torch.load(path, weights_only=False)
        with torch.no_grad():
            for name, param in self._model.named_parameters():
                if name in checkpoint["model_state_dict"]:
                    param.copy_(checkpoint["model_state_dict"][name])
        self._noised.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._noised.step = checkpoint["step"]
        self._snapshot_params()
        logger.info("Checkpoint loaded from %s", path)
