# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EGGROLL trainer using vLLM for efficient batched generation.

Two generation strategies:

1. **LoRA-batched** (default, `use_lora=True`):
   Each population member's perturbation is exported as a PEFT LoRA adapter.
   All members' prompts are submitted to vLLM in ONE generate() call with
   per-request LoRARequest objects. vLLM's continuous-batching scheduler
   handles all members in parallel — true population-level batching.

2. **Sequential** (`use_lora=False`):
   Each member's weights are applied in-place, generated, then restored.
   One vLLM.generate() call per member. Simpler but slower.

Multi-GPU support:
- `tensor_parallel_size`: splits the model across GPUs (model parallelism).
  All population members still share the same TP group.
- `gpu_parallel_popsize`: splits the population across independent vLLM
  instances on different GPUs. Each instance handles a subset of the
  population. Requires multiple GPUs not used by TP.

Reference: https://github.com/ESHyperscale/HyperscaleES
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
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

    # LoRA-batched mode (the scalable path)
    use_lora: bool = True
    max_loras_per_batch: int = 16  # vLLM max_loras (GPU LoRA slots)

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
    """EGGROLL trainer with LoRA-batched population evaluation.

    Usage::

        from vllm.eggroll import EggRollTrainer
        from vllm.eggroll.trainer import EggRollConfig

        config = EggRollConfig(
            sigma=1e-3,
            lr=1e-4,
            population_size=64,
            rank=8,
            use_lora=True,  # Enable LoRA-batched generation
        )


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
        self._adapter_tmpdir: str | None = None

    def setup(self) -> None:
        """Initialize vLLM engine and ES components."""
        from vllm import LLM

        logger.info("Initializing vLLM: %s", self.model_name)

        llm_kwargs = {
            "model": self.model_name,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "seed": self.config.seed,
        }

        if self.config.use_lora:
            llm_kwargs.update(
                {
                    "enable_lora": True,
                    "max_loras": min(
                        self.config.max_loras_per_batch,
                        self.config.population_size,
                    ),
                    "max_lora_rank": self.config.rank,
                    "max_cpu_loras": self.config.population_size,
                }
            )
        else:
            llm_kwargs["enforce_eager"] = True

        self._llm = LLM(**llm_kwargs)
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

        # Temp directory for LoRA adapter files
        if self.config.use_lora:
            self._adapter_tmpdir = tempfile.mkdtemp(prefix="eggroll_lora_")

        n_lora = sum(
            1
            for n, p in self._model.named_parameters()
            if self._noiser_cls.classify_param(n, p) == 1  # MM_PARAM
        )
        logger.info(
            "EGGROLL ready: pop=%d, sigma=%.1e, lr=%.1e, rank=%d, "
            "lora_modules=%d, mode=%s",
            self.config.population_size,
            self.config.sigma,
            self.config.lr,
            self.config.rank,
            n_lora,
            "lora-batched" if self.config.use_lora else "sequential",
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
    # LoRA-batched generation (scalable)
    # ------------------------------------------------------------------

    def _generate_lora_batched(
        self,
        prompts: list[str],
        epoch: int,
    ) -> list[list[str]]:
        """Generate text for ALL population members in one vLLM batch.

        Each member's perturbation is a LoRA adapter. All prompts × members
        are submitted as separate requests with per-request LoRARequest.
        vLLM's scheduler batches them, serving multiple LoRAs concurrently.
        """
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        pop_size = self.config.population_size
        sampling_params = SamplingParams(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        # Export all population members as LoRA adapters
        adapter_dirs = self._noiser_cls.export_lora_adapters(
            self._frozen,
            self._noised.sigma,
            epoch,
            pop_size,
            self._model,
            self._param_seeds,
            self._adapter_tmpdir,
        )

        # Build flat list of (prompt, lora_request) for all members
        all_prompts = []
        all_lora_requests = []
        for member_id in range(pop_size):
            lora_req = LoRARequest(
                lora_name=f"member_{member_id}",
                lora_int_id=member_id + 1,  # Must be > 0
                lora_path=adapter_dirs[member_id],
            )
            for prompt in prompts:
                all_prompts.append(prompt)
                all_lora_requests.append(lora_req)

        # Single batched generate call — vLLM handles scheduling
        outputs = self._llm.generate(
            all_prompts,
            sampling_params,
            lora_request=all_lora_requests,
        )

        # Reshape outputs: [pop_size][num_prompts]
        n_prompts = len(prompts)
        all_generations: list[list[str]] = []
        for member_id in range(pop_size):
            start = member_id * n_prompts
            end = start + n_prompts
            member_gens = [o.outputs[0].text for o in outputs[start:end]]
            all_generations.append(member_gens)

        return all_generations

    # ------------------------------------------------------------------
    # Sequential generation (fallback)
    # ------------------------------------------------------------------

    def _generate_sequential(
        self,
        prompts: list[str],
        epoch: int,
    ) -> list[list[str]]:
        """Generate text member-by-member with weight perturbation."""
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

        # 1. Generate (LoRA-batched or sequential)
        t0 = time.time()
        if self.config.use_lora:
            all_generations = self._generate_lora_batched(prompts, epoch)
        else:
            all_generations = self._generate_sequential(prompts, epoch)
        gen_time = time.time() - t0

        # 2. Evaluate fitness
        t0 = time.time()
        raw_scores = self._evaluate_fitness(prompts, all_generations)
        fitness_time = time.time() - t0

        # 3. Normalize + gradient + update (vectorized)
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

    def cleanup(self) -> None:
        """Remove temporary LoRA adapter files."""
        if self._adapter_tmpdir and os.path.exists(self._adapter_tmpdir):
            shutil.rmtree(self._adapter_tmpdir)
            self._adapter_tmpdir = None

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

    def __del__(self):
        self.cleanup()


class MultiGPUEggRollTrainer:
    """Multi-GPU EGGROLL trainer that shards population across GPU workers.

    Splits the population across `num_workers` independent vLLM instances,
    each running on separate GPUs. Each worker evaluates a subset of the
    population, then gradients are aggregated.

    This is orthogonal to tensor parallelism (which splits the model).

    Usage::

        trainer = MultiGPUEggRollTrainer(
            model_name="meta-llama/Llama-3.1-8B",
            config=EggRollConfig(population_size=256, rank=8),
            fitness_fn=my_fitness_fn,
            num_workers=4,  # 4 GPUs, 64 members each
        )
        trainer.train(prompts=["Solve x^2=4"])
    """

    def __init__(
        self,
        model_name: str,
        config: EggRollConfig,
        fitness_fn: Callable[[list[str], list[str]], torch.Tensor],
        num_workers: int = 1,
        prompt_fn: Callable[[int, int], list[str]] | None = None,
        validation_fn: Callable[[Any, int], float] | None = None,
        gpu_ids: list[int] | None = None,
    ):
        self.model_name = model_name
        self.config = config
        self.fitness_fn = fitness_fn
        self.num_workers = num_workers
        self.prompt_fn = prompt_fn
        self.validation_fn = validation_fn
        self.gpu_ids = gpu_ids or list(range(num_workers))

        assert config.population_size % num_workers == 0, (
            f"population_size ({config.population_size}) must be divisible "
            f"by num_workers ({num_workers})"
        )
        self._pop_per_worker = config.population_size // num_workers

        # These are shared across workers
        self._noiser_cls = None
        self._frozen: FrozenNoiserParams | None = None
        self._noised: NoisedParams | None = None
        self._model: nn.Module | None = None
        self._param_seeds: dict[str, int] | None = None
        self._original_params: dict[str, torch.Tensor] | None = None
        self._workers: list[Any] = []

    def setup(self) -> None:
        """Initialize workers and ES state."""
        noiser_map = {"eggroll": EggRoll, "open_es": OpenES}
        self._noiser_cls = noiser_map[self.config.noiser_type]

        # Initialize first worker to get model reference
        logger.info(
            "Initializing %d GPU workers for population of %d",
            self.num_workers,
            self.config.population_size,
        )

        for worker_idx in range(self.num_workers):
            gpu_id = self.gpu_ids[worker_idx]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            worker_config = EggRollConfig(
                **{
                    k: v
                    for k, v in self.config.__dict__.items()
                    if k != "population_size"
                }
            )
            worker_config.population_size = self._pop_per_worker

            worker = EggRollTrainer(
                model_name=self.model_name,
                config=worker_config,
                fitness_fn=self.fitness_fn,
                prompt_fn=self.prompt_fn,
            )
            worker.setup()
            self._workers.append(worker)

            if worker_idx == 0:
                self._model = worker._model
                self._param_seeds = worker._param_seeds
                self._original_params = worker._original_params
                self._frozen = worker._frozen
                self._noised = worker._noised

        logger.info(
            "Multi-GPU ready: %d workers × %d members = %d total",
            self.num_workers,
            self._pop_per_worker,
            self.config.population_size,
        )

    def train_epoch(self, prompts: list[str], epoch: int) -> EpochStats:
        """Run one epoch across all GPU workers."""
        pop_size = self.config.population_size

        t0 = time.time()
        all_generations: list[list[str]] = []
        for worker in self._workers:
            if self.config.use_lora:
                gens = worker._generate_lora_batched(prompts, epoch)
            else:
                gens = worker._generate_sequential(prompts, epoch)
            all_generations.extend(gens)
        gen_time = time.time() - t0

        # Evaluate fitness
        t0 = time.time()
        raw_scores = torch.zeros(pop_size)
        for i, gens in enumerate(all_generations):
            s = self.fitness_fn(prompts, gens)
            raw_scores[i] = s.mean() if isinstance(s, torch.Tensor) else float(s)
        fitness_time = time.time() - t0

        # Gradient + update (centralized on first worker's model)
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

        # Sync updated params to all workers
        for worker in self._workers[1:]:
            with torch.no_grad():
                for name, param in self._model.named_parameters():
                    w_param = dict(worker._model.named_parameters())[name]
                    w_param.copy_(param.data)
            worker._snapshot_params()

        total_sq, total_n = 0.0, 0
        for name, param in self._model.named_parameters():
            diff = (param.data - old_params[name]).float()
            total_sq += (diff**2).sum().item()
            total_n += diff.numel()
        param_norm = (total_sq / max(total_n, 1)) ** 0.5

        update_time = time.time() - t0
        self._workers[0]._snapshot_params()

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

    def train(self, prompts: list[str] | None = None) -> list[EpochStats]:
        if not self._workers:
            self.setup()

        all_stats: list[EpochStats] = []
        for epoch in range(self.config.num_epochs):
            if self.prompt_fn is not None:
                epoch_prompts = self.prompt_fn(epoch, self.config.population_size)
            elif prompts is not None:
                epoch_prompts = prompts
            else:
                raise ValueError("Provide prompts or prompt_fn")

            stats = self.train_epoch(epoch_prompts, epoch)
            all_stats.append(stats)

            if epoch % self.config.log_every == 0:
                logger.info(
                    "Epoch %d [%d GPUs]: fitness=%.4f±%.4f gen=%.1fs upd=%.1fs",
                    epoch,
                    self.num_workers,
                    stats.mean_fitness,
                    stats.std_fitness,
                    stats.generation_time,
                    stats.update_time,
                )

        return all_stats

    def cleanup(self) -> None:
        for worker in self._workers:
            worker.cleanup()
