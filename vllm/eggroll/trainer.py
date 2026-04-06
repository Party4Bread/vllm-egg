# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EGGROLL trainer using vLLM for efficient batched generation.

Generation strategies:
  LoRA-batched (default): Each member → PEFT LoRA adapter → single vLLM batch.
  Sequential (fallback):  perturb → generate → restore per member.

Multi-node:
  DistributedEggRollTrainer uses torch.distributed to shard the population
  across nodes. Each node runs its own vLLM instance, evaluates a pop slice,
  then all-gathers fitness scores so every node can compute identical
  gradients deterministically — no gradient transfer required.

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

_NOISER_MAP = {"eggroll": EggRoll, "open_es": OpenES}


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
    noiser_type: str = "eggroll"

    # Optimizer
    optimizer_cls: str = "Adam"
    optimizer_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"betas": (0.9, 0.999)}
    )

    # Generation
    max_tokens: int = 100
    temperature: float = 0.0
    top_p: float = 1.0

    # LoRA-batched mode
    use_lora: bool = True
    max_loras_per_batch: int = 16

    # Training
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
    """Single-node EGGROLL trainer.

    Caches param_plan at init to avoid repeated classify_param() in hot paths.
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
        self._param_plan = None  # Cached (name, classification, seed) list
        self._original_params: dict[str, torch.Tensor] | None = None
        self._adapter_tmpdir: str | None = None

    def setup(self) -> None:
        from vllm import LLM

        llm_kwargs: dict[str, Any] = {
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

        self._noiser_cls = _NOISER_MAP[self.config.noiser_type]
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
        self._param_plan = self._noiser_cls.get_param_plan(
            self._model, self.config.seed
        )
        self._snapshot_params()

        if self.config.use_lora:
            self._adapter_tmpdir = tempfile.mkdtemp(prefix="eggroll_lora_")

        logger.info(
            "EGGROLL ready: pop=%d sigma=%.1e rank=%d mode=%s",
            self.config.population_size,
            self.config.sigma,
            self.config.rank,
            "lora" if self.config.use_lora else "seq",
        )

    def _get_model(self) -> nn.Module:
        executor = self._llm.llm_engine.model_executor
        workers = getattr(executor, "drivers", None)
        if workers and hasattr(workers, "__iter__"):
            worker = list(workers)[0]
        else:
            worker = executor.driver_worker
        return worker.model_runner.model

    def _restore_params(self) -> None:
        with torch.no_grad():
            for name, param in self._model.named_parameters():
                if name in self._original_params:
                    param.copy_(self._original_params[name])

    def _snapshot_params(self) -> None:
        self._original_params = {
            n: p.data.clone() for n, p in self._model.named_parameters()
        }

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate_lora_batched(
        self,
        prompts: list[str],
        epoch: int,
        member_offset: int = 0,
    ) -> list[list[str]]:
        """All members in one vLLM batch via LoRA adapters."""
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        pop = self.config.population_size
        sp = SamplingParams(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        dirs = self._noiser_cls.export_lora_adapters(
            self._frozen,
            self._noised.sigma,
            epoch,
            pop,
            self._model,
            self._param_seeds,
            self._adapter_tmpdir,
            param_plan=self._param_plan,
        )

        all_prompts: list[str] = []
        all_lora: list[LoRARequest] = []
        for mid in range(pop):
            lr = LoRARequest(
                lora_name=f"m{member_offset + mid}_e{epoch}",
                lora_int_id=member_offset + mid + 1,
                lora_path=dirs[mid],
            )
            for p in prompts:
                all_prompts.append(p)
                all_lora.append(lr)

        outputs = self._llm.generate(all_prompts, sp, lora_request=all_lora)

        n = len(prompts)
        return [
            [o.outputs[0].text for o in outputs[i * n : (i + 1) * n]]
            for i in range(pop)
        ]

    def _generate_sequential(
        self,
        prompts: list[str],
        epoch: int,
        member_offset: int = 0,
    ) -> list[list[str]]:
        from vllm import SamplingParams

        sp = SamplingParams(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        result: list[list[str]] = []
        for mid in range(self.config.population_size):
            self._noiser_cls.perturb_model(
                self._frozen,
                self._noised.sigma,
                self._model,
                epoch,
                member_offset + mid,
                self._param_seeds,
                self._original_params,
            )
            outs = self._llm.generate(prompts, sp)
            result.append([o.outputs[0].text for o in outs])
            self._restore_params()
        return result

    # ------------------------------------------------------------------
    # Fitness
    # ------------------------------------------------------------------

    def _evaluate_fitness(
        self, prompts: list[str], all_gens: list[list[str]]
    ) -> torch.Tensor:
        scores = torch.zeros(len(all_gens))
        for i, gens in enumerate(all_gens):
            s = self.fitness_fn(prompts, gens)
            scores[i] = s.mean() if isinstance(s, torch.Tensor) else float(s)
        return scores

    # ------------------------------------------------------------------
    # Epoch
    # ------------------------------------------------------------------

    def train_epoch(self, prompts: list[str], epoch: int) -> EpochStats:
        pop = self.config.population_size

        t0 = time.time()
        if self.config.use_lora:
            gens = self._generate_lora_batched(prompts, epoch)
        else:
            gens = self._generate_sequential(prompts, epoch)
        gen_time = time.time() - t0

        t0 = time.time()
        raw = self._evaluate_fitness(prompts, gens)
        fit_time = time.time() - t0

        t0 = time.time()
        fitnesses = self._noiser_cls.convert_fitnesses(self._frozen, raw)
        grads = self._noiser_cls.compute_gradients(
            self._frozen,
            self._noised.sigma,
            epoch,
            pop,
            self._model,
            self._param_seeds,
            fitnesses,
            param_plan=self._param_plan,
        )
        old = {n: p.data.clone() for n, p in self._model.named_parameters()}
        self._noiser_cls.update_params(self._noised, self._model, grads)

        sq, n_el = 0.0, 0
        for name, p in self._model.named_parameters():
            d = (p.data - old[name]).float()
            sq += (d**2).sum().item()
            n_el += d.numel()
        upd_time = time.time() - t0
        self._snapshot_params()

        return EpochStats(
            epoch=epoch,
            mean_fitness=raw.mean().item(),
            std_fitness=raw.std().item(),
            max_fitness=raw.max().item(),
            min_fitness=raw.min().item(),
            generation_time=gen_time,
            fitness_time=fit_time,
            update_time=upd_time,
            param_change_norm=(sq / max(n_el, 1)) ** 0.5,
        )

    def train(self, prompts: list[str] | None = None) -> list[EpochStats]:
        if self._llm is None:
            self.setup()

        stats: list[EpochStats] = []
        for epoch in range(self.config.num_epochs):
            ep = (
                self.prompt_fn(epoch, self.config.population_size)
                if self.prompt_fn
                else prompts
            )
            if ep is None:
                raise ValueError("Provide prompts or prompt_fn")

            if self.validation_fn and epoch % self.config.validate_every == 0:
                logger.info(
                    "Epoch %d val: %.4f",
                    epoch,
                    self.validation_fn(self._llm, epoch),
                )

            s = self.train_epoch(ep, epoch)
            stats.append(s)
            if epoch % self.config.log_every == 0:
                logger.info(
                    "Epoch %d: fit=%.4f±%.4f gen=%.1fs upd=%.1fs",
                    epoch,
                    s.mean_fitness,
                    s.std_fitness,
                    s.generation_time,
                    s.update_time,
                )
        return stats

    def cleanup(self) -> None:
        if self._adapter_tmpdir and os.path.exists(self._adapter_tmpdir):
            shutil.rmtree(self._adapter_tmpdir)
            self._adapter_tmpdir = None

    def save_checkpoint(self, path: str) -> None:
        torch.save(
            {
                "model": {n: p.data.cpu() for n, p in self._model.named_parameters()},
                "optim": self._noised.optimizer.state_dict(),
                "step": self._noised.step,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, weights_only=False)
        with torch.no_grad():
            for n, p in self._model.named_parameters():
                if n in ckpt["model"]:
                    p.copy_(ckpt["model"][n])
        self._noised.optimizer.load_state_dict(ckpt["optim"])
        self._noised.step = ckpt["step"]
        self._snapshot_params()

    def __del__(self):
        self.cleanup()


# ======================================================================
# Multi-node distributed trainer
# ======================================================================


class DistributedEggRollTrainer:
    """Multi-node EGGROLL trainer using torch.distributed.

    Key insight: ES gradients are deterministic given the fitness scores.
    So we only need to all-gather the fitness vector (pop_size floats),
    not the gradients (millions of floats). Each node computes identical
    gradients locally from the shared fitness vector.

    Architecture::

        Node 0: vLLM instance → evaluates members [0, pop/N)
        Node 1: vLLM instance → evaluates members [pop/N, 2*pop/N)
        ...
        Node N-1: → evaluates members [(N-1)*pop/N, pop)

        All-gather fitness scores (tiny: pop_size floats)
        Each node computes same gradients locally → same update

    Launch with torchrun::

        torchrun --nproc_per_node=1 --nnodes=4 \\
            --rdzv_backend=c10d --rdzv_endpoint=head:29500 \\
            my_script.py

    Or with SLURM + srun.
    """

    def __init__(
        self,
        model_name: str,
        config: EggRollConfig,
        fitness_fn: Callable[[list[str], list[str]], torch.Tensor],
        prompt_fn: Callable[[int, int], list[str]] | None = None,
        validation_fn: Callable[[Any, int], float] | None = None,
        backend: str = "nccl",
    ):
        self.model_name = model_name
        self.config = config
        self.fitness_fn = fitness_fn
        self.prompt_fn = prompt_fn
        self.validation_fn = validation_fn
        self.backend = backend

        self._trainer: EggRollTrainer | None = None
        self._rank = 0
        self._world_size = 1
        self._local_pop = 0
        self._member_offset = 0

    def setup(self) -> None:
        """Initialize distributed process group and local vLLM trainer."""
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(backend=self.backend)

        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()

        pop = self.config.population_size
        assert pop % self._world_size == 0, (
            f"pop_size ({pop}) must be divisible by world_size ({self._world_size})"
        )
        self._local_pop = pop // self._world_size
        self._member_offset = self._rank * self._local_pop

        # Each node creates its own vLLM instance with a pop slice
        local_config = EggRollConfig(**{k: v for k, v in self.config.__dict__.items()})
        local_config.population_size = self._local_pop

        self._trainer = EggRollTrainer(
            model_name=self.model_name,
            config=local_config,
            fitness_fn=self.fitness_fn,
            prompt_fn=self.prompt_fn,
            validation_fn=self.validation_fn,
        )
        self._trainer.setup()

        logger.info(
            "Distributed rank %d/%d: members [%d, %d)",
            self._rank,
            self._world_size,
            self._member_offset,
            self._member_offset + self._local_pop,
        )

    def train_epoch(self, prompts: list[str], epoch: int) -> EpochStats:
        """Run one epoch with distributed fitness gathering."""
        import torch.distributed as dist

        pop = self.config.population_size
        local_pop = self._local_pop
        trainer = self._trainer

        # 1. Generate locally for our population slice
        t0 = time.time()
        if self.config.use_lora:
            local_gens = trainer._generate_lora_batched(
                prompts, epoch, member_offset=self._member_offset
            )
        else:
            local_gens = trainer._generate_sequential(
                prompts, epoch, member_offset=self._member_offset
            )
        gen_time = time.time() - t0

        # 2. Evaluate local fitness
        t0 = time.time()
        local_scores = trainer._evaluate_fitness(prompts, local_gens)
        fit_time = time.time() - t0

        # 3. All-gather fitness scores across all nodes
        # This is the ONLY communication: pop_size floats (~256 bytes for pop=64)
        t0 = time.time()
        device = next(trainer._model.parameters()).device
        local_scores_gpu = local_scores.to(device)
        gathered = [
            torch.zeros(local_pop, device=device) for _ in range(self._world_size)
        ]
        dist.all_gather(gathered, local_scores_gpu)
        all_scores = torch.cat(gathered).cpu()

        # 4. Each node computes identical gradients from full fitness
        fitnesses = trainer._noiser_cls.convert_fitnesses(trainer._frozen, all_scores)
        grads = trainer._noiser_cls.compute_gradients(
            trainer._frozen,
            trainer._noised.sigma,
            epoch,
            pop,
            trainer._model,
            trainer._param_seeds,
            fitnesses,
            param_plan=trainer._param_plan,
        )

        old = {n: p.data.clone() for n, p in trainer._model.named_parameters()}
        trainer._noiser_cls.update_params(trainer._noised, trainer._model, grads)

        sq, n_el = 0.0, 0
        for name, p in trainer._model.named_parameters():
            d = (p.data - old[name]).float()
            sq += (d**2).sum().item()
            n_el += d.numel()

        upd_time = time.time() - t0
        trainer._snapshot_params()

        return EpochStats(
            epoch=epoch,
            mean_fitness=all_scores.mean().item(),
            std_fitness=all_scores.std().item(),
            max_fitness=all_scores.max().item(),
            min_fitness=all_scores.min().item(),
            generation_time=gen_time,
            fitness_time=fit_time,
            update_time=upd_time,
            param_change_norm=(sq / max(n_el, 1)) ** 0.5,
        )

    def train(self, prompts: list[str] | None = None) -> list[EpochStats]:
        if self._trainer is None:
            self.setup()

        stats: list[EpochStats] = []
        for epoch in range(self.config.num_epochs):
            ep = (
                self.prompt_fn(epoch, self.config.population_size)
                if self.prompt_fn
                else prompts
            )
            if ep is None:
                raise ValueError("Provide prompts or prompt_fn")

            if (
                self.validation_fn
                and epoch % self.config.validate_every == 0
                and self._rank == 0
            ):
                logger.info(
                    "Epoch %d val: %.4f",
                    epoch,
                    self.validation_fn(self._trainer._llm, epoch),
                )

            s = self.train_epoch(ep, epoch)
            stats.append(s)

            if epoch % self.config.log_every == 0 and self._rank == 0:
                logger.info(
                    "Epoch %d [%d nodes]: fit=%.4f±%.4f gen=%.1fs upd=%.1fs",
                    epoch,
                    self._world_size,
                    s.mean_fitness,
                    s.std_fitness,
                    s.generation_time,
                    s.update_time,
                )

        return stats

    def cleanup(self) -> None:
        if self._trainer:
            self._trainer.cleanup()
