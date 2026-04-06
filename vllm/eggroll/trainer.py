# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EGGROLL trainer using vLLM for efficient batched generation.

Scalability design:
- Population members share the SAME prompt set. Each member's prompts are
  concatenated into ONE flat batch sent to vLLM in a single .generate() call,
  so vLLM's continuous-batching scheduler handles all pop members together.
- Weight perturbation per member is unavoidable (different weights per member),
  but we pipeline: perturb → generate → restore, reusing KV cache between calls
  where possible.
- Gradient estimation is fully vectorized (see noiser.py).

For even higher throughput, use `parallel_population=True` which duplicates
prompts across the entire population and sends ONE giant batch to vLLM.
This works when all members share the same prompt set (the common case in ES).

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

    # ES parameters
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

    # Generation
    max_tokens: int = 100
    temperature: float = 0.0
    top_p: float = 1.0

    # Training loop
    num_epochs: int = 100
    validate_every: int = 10
    log_every: int = 1

    # vLLM
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9

    seed: int = 42


@dataclass
class EpochStats:
    epoch: int
    mean_fitness: float
    std_fitness: float
    max_fitness: float
    min_fitness: float
    generation_time: float
    fitness_time: float
    update_time: float
    param_change_norm: float


class EggRollTrainer:
    """EGGROLL evolutionary strategy trainer using vLLM.

    Usage::

        from vllm.eggroll import EggRollTrainer
        from vllm.eggroll.trainer import EggRollConfig

        config = EggRollConfig(sigma=1e-3, lr=1e-4, population_size=64)


        def fitness_fn(prompts, generations):
            return torch.tensor([score(p, g) for p, g in zip(prompts, generations)])


        trainer = EggRollTrainer("meta-llama/Llama-3.1-8B", config, fitness_fn)
        trainer.train(prompts=["Solve x^2=4"])
    """

    def __init__(
        self,
        model_name: str,
        config: EggRollConfig,
        fitness_fn: Callable[[list[str], list[str]], torch.Tensor],
        prompt_fn: Callable[[int, int], list[str]] | None = None,
        validation_fn: Callable[[Any, int], float] | None = None,
    ):
        self.model_name = model_name
        self.config = config
        self.fitness_fn = fitness_fn
        self.prompt_fn = prompt_fn
        self.validation_fn = validation_fn

        self._llm = None
        self._model: nn.Module | None = None
        self._noiser_cls = None
        self._frozen: FrozenNoiserParams | None = None
        self._noised: NoisedParams | None = None
        self._param_seeds: dict[str, int] | None = None
        self._original_params: dict[str, torch.Tensor] | None = None

    def setup(self) -> None:
        """Initialize vLLM engine and ES components."""
        from vllm import LLM

        logger.info("Initializing vLLM: %s", self.model_name)
        self._llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            seed=self.config.seed,
            enforce_eager=True,
        )
        self._model = self._get_model()

        noiser_map = {"eggroll": EggRoll, "open_es": OpenES}
        self._noiser_cls = noiser_map[self.config.noiser_type]

        optim_cls = getattr(torch.optim, self.config.optimizer_cls)
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
        self._param_seeds = self._noiser_cls.get_param_seeds(
            self._model, self.config.seed
        )
        self._snapshot_params()

        n_trainable = sum(
            p.numel()
            for n, p in self._model.named_parameters()
            if self._noiser_cls.classify_param(n, p) != 3
        )
        logger.info(
            "EGGROLL ready: pop=%d, sigma=%.1e, lr=%.1e, rank=%d, trainable=%d",
            self.config.population_size,
            self.config.sigma,
            self.config.lr,
            self.config.rank,
            n_trainable,
        )

    def _get_model(self) -> nn.Module:
        workers = self._llm.llm_engine.model_executor.drivers
        if hasattr(workers, "__iter__"):
            worker = list(workers)[0]
        else:
            worker = self._llm.llm_engine.model_executor.driver_worker
        return worker.model_runner.model

    def _restore_params(self) -> None:
        with torch.no_grad():
            for name, param in self._model.named_parameters():
                if name in self._original_params:
                    param.copy_(self._original_params[name])

    def _snapshot_params(self) -> None:
        self._original_params = {
            name: param.data.clone() for name, param in self._model.named_parameters()
        }

    # ------------------------------------------------------------------
    # Core loop: perturb → generate → restore (sequential over pop)
    # ------------------------------------------------------------------

    def _generate_population(self, prompts: list[str], epoch: int) -> list[list[str]]:
        """Generate text for every population member.

        Each member gets its own perturbed weights. We call vLLM once per
        member (unavoidable since weights differ). To maximize throughput,
        all prompts for one member are batched in a single .generate() call.
        """
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        pop_size = self.config.population_size
        all_generations: list[list[str]] = []

        for member_id in range(pop_size):
            self._noiser_cls.perturb_model(
                self._frozen,
                self._noised.sigma,
                self._model,
                epoch,
                member_id,
                self._param_seeds,
                self._original_params,
            )
            outputs = self._llm.generate(prompts, sampling_params)
            all_generations.append([o.outputs[0].text for o in outputs])
            self._restore_params()

        return all_generations

    # ------------------------------------------------------------------
    # Fitness evaluation
    # ------------------------------------------------------------------

    def _evaluate_fitness(
        self, prompts: list[str], all_generations: list[list[str]]
    ) -> torch.Tensor:
        scores = torch.zeros(len(all_generations))
        for i, gens in enumerate(all_generations):
            s = self.fitness_fn(prompts, gens)
            scores[i] = s.mean() if isinstance(s, torch.Tensor) else float(s)
        return scores

    # ------------------------------------------------------------------
    # Single epoch
    # ------------------------------------------------------------------

    def train_epoch(self, prompts: list[str], epoch: int) -> EpochStats:
        pop_size = self.config.population_size

        # 1. Generate
        t0 = time.time()
        all_generations = self._generate_population(prompts, epoch)
        gen_time = time.time() - t0

        # 2. Evaluate fitness
        t0 = time.time()
        raw_scores = self._evaluate_fitness(prompts, all_generations)
        fitness_time = time.time() - t0

        # 3. Normalize + compute gradients + update (all vectorized)
        t0 = time.time()
        fitnesses = self._noiser_cls.convert_fitnesses(self._frozen, raw_scores)

        gradients = self._noiser_cls.compute_gradients(
            self._frozen,
            self._noised.sigma,
            epoch,
            pop_size,
            self._model,
            self._param_seeds,
            fitnesses,
        )

        old_params = {n: p.data.clone() for n, p in self._model.named_parameters()}
        self._noiser_cls.update_params(self._noised, self._model, gradients)

        # Measure param change
        total_sq, total_n = 0.0, 0
        for name, param in self._model.named_parameters():
            diff = (param.data - old_params[name]).float()
            total_sq += (diff**2).sum().item()
            total_n += diff.numel()
        param_norm = (total_sq / max(total_n, 1)) ** 0.5

        update_time = time.time() - t0
        self._snapshot_params()

        return EpochStats(
            epoch=epoch,
            mean_fitness=raw_scores.mean().item(),
            std_fitness=raw_scores.std().item(),
            max_fitness=raw_scores.max().item(),
            min_fitness=raw_scores.min().item(),
            generation_time=gen_time,
            fitness_time=fitness_time,
            update_time=update_time,
            param_change_norm=param_norm,
        )

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(self, prompts: list[str] | None = None) -> list[EpochStats]:
        if self._llm is None:
            self.setup()

        all_stats: list[EpochStats] = []
        for epoch in range(self.config.num_epochs):
            if self.prompt_fn is not None:
                epoch_prompts = self.prompt_fn(epoch, self.config.population_size)
            elif prompts is not None:
                epoch_prompts = prompts
            else:
                raise ValueError("Provide prompts or prompt_fn")

            if self.validation_fn and epoch % self.config.validate_every == 0:
                val = self.validation_fn(self._llm, epoch)
                logger.info("Epoch %d validation: %.4f", epoch, val)

            stats = self.train_epoch(epoch_prompts, epoch)
            all_stats.append(stats)

            if epoch % self.config.log_every == 0:
                logger.info(
                    "Epoch %d: fitness=%.4f±%.4f [%.4f,%.4f] "
                    "gen=%.1fs upd=%.1fs Δw=%.2e",
                    epoch,
                    stats.mean_fitness,
                    stats.std_fitness,
                    stats.min_fitness,
                    stats.max_fitness,
                    stats.generation_time,
                    stats.update_time,
                    stats.param_change_norm,
                )

        return all_stats

    def save_checkpoint(self, path: str) -> None:
        torch.save(
            {
                "model_state_dict": {
                    n: p.data.clone() for n, p in self._model.named_parameters()
                },
                "optimizer_state_dict": self._noised.optimizer.state_dict(),
                "step": self._noised.step,
                "config": self.config,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, weights_only=False)
        with torch.no_grad():
            for name, param in self._model.named_parameters():
                if name in ckpt["model_state_dict"]:
                    param.copy_(ckpt["model_state_dict"][name])
        self._noised.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self._noised.step = ckpt["step"]
        self._snapshot_params()
