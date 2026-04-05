# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EGGROLL and OpenES noiser implementations in PyTorch.

Port of HyperscaleES evolutionary strategy noisers from JAX to PyTorch
for integration with vLLM's inference engine.

The noiser handles:
- Parameter perturbation via LoRA-style low-rank noise (for weight matrices)
  or full noise (for biases/scalars)
- Antithetical sampling (paired +/- perturbations)
- Fitness normalization
- Gradient estimation from fitness scores
- Parameter updates via standard optimizers (Adam, SGD)

Reference: https://github.com/ESHyperscale/HyperscaleES
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

# Parameter classification constants (matching HyperscaleES)
PARAM = 0  # Standard parameter (full perturbation)
MM_PARAM = 1  # Matrix multiply parameter (LoRA perturbation)
EMB_PARAM = 2  # Embedding parameter (LoRA perturbation on embeddings)
EXCLUDED = 3  # Excluded from evolution


@dataclass
class NoisedParams:
    """Container for parameter state during evolution."""

    sigma: float
    optimizer: torch.optim.Optimizer
    step: int = 0


@dataclass
class FrozenNoiserParams:
    """Immutable noiser configuration."""

    group_size: int = 0
    freeze_nonlora: bool = False
    noise_reuse: int = 0
    rank: int = 1
    use_batched_update: bool = False


def _deterministic_key(base_seed: int, epoch: int, thread_idx: int) -> torch.Generator:
    """Create a deterministic RNG from (base_seed, epoch, thread_idx).

    This mirrors JAX's fold_in(fold_in(key, epoch), thread_idx) pattern
    for reproducible antithetical noise generation.
    """
    gen = torch.Generator()
    # Combine seeds deterministically
    combined = base_seed ^ (epoch * 2654435761) ^ (thread_idx * 40503)
    gen.manual_seed(combined & 0xFFFFFFFF)
    return gen


def get_lora_perturbation(
    frozen: FrozenNoiserParams,
    sigma: float,
    epoch: int,
    thread_id: int,
    param: torch.Tensor,
    param_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate LoRA-style low-rank perturbation for a matrix parameter.

    Uses antithetical sampling: even thread_ids get +sigma, odd get -sigma.

    Args:
        frozen: Frozen noiser configuration.
        sigma: Base perturbation magnitude.
        epoch: Current epoch (for noise seed).
        thread_id: Thread/member index in population.
        param: The parameter tensor (out_dim, in_dim).
        param_seed: Base random seed for this parameter.

    Returns:
        (A, B) where A is (out_dim, rank) and B is (in_dim, rank),
        such that the perturbation is A @ B^T.
    """
    true_epoch = 0 if frozen.noise_reuse == 0 else epoch // frozen.noise_reuse
    true_thread = thread_id // 2
    sign = 1.0 if thread_id % 2 == 0 else -1.0

    a, b = param.shape
    r = frozen.rank
    gen = _deterministic_key(param_seed, true_epoch, true_thread)

    # Generate (a+b) x r random values
    lora_params = torch.randn(
        a + b, r, dtype=param.dtype, device=param.device, generator=gen
    )
    B = lora_params[:b]  # (in_dim, rank)
    A = lora_params[b:]  # (out_dim, rank)

    effective_sigma = sign * sigma / math.sqrt(r)
    return A * effective_sigma, B


def get_full_perturbation(
    frozen: FrozenNoiserParams,
    sigma: float,
    epoch: int,
    thread_id: int,
    param: torch.Tensor,
    param_seed: int,
) -> torch.Tensor:
    """Generate full-rank noise perturbation for a standard parameter.

    Uses antithetical sampling: even thread_ids get +sigma, odd get -sigma.
    """
    true_epoch = 0 if frozen.noise_reuse == 0 else epoch // frozen.noise_reuse
    true_thread = thread_id // 2
    sign = 1.0 if thread_id % 2 == 0 else -1.0

    gen = _deterministic_key(param_seed, true_epoch, true_thread)
    noise = torch.randn(
        *param.shape, dtype=param.dtype, device=param.device, generator=gen
    )
    return noise * sign * sigma


def apply_lora_perturbation(
    frozen: FrozenNoiserParams,
    sigma: float,
    epoch: int,
    thread_id: int,
    weight: torch.Tensor,
    x: torch.Tensor,
    param_seed: int,
    transpose: bool = False,
) -> torch.Tensor:
    """Apply LoRA-perturbed matrix multiplication.

    Computes x @ (W + A @ B^T)^T  (or non-transposed variant).
    The perturbation is applied implicitly: x @ W^T + (x @ B) @ A^T.
    """
    base = x @ weight if transpose else x @ weight.T

    A, B = get_lora_perturbation(frozen, sigma, epoch, thread_id, weight, param_seed)
    perturbation = (x @ A) @ B.T if transpose else (x @ B) @ A.T

    return base + perturbation


def compute_lora_gradient(
    frozen: FrozenNoiserParams,
    sigma: float,
    param: torch.Tensor,
    param_seed: int,
    fitnesses: torch.Tensor,
    epochs: torch.Tensor,
    thread_ids: torch.Tensor,
) -> torch.Tensor:
    """Estimate gradient for a LoRA-perturbed matrix parameter.

    Computes: mean_i[ fitness_i * (A_i @ B_i^T) ]

    This is the evolutionary strategy gradient estimate using
    antithetical sampling and LoRA-style perturbations.
    """
    num_envs = fitnesses.shape[0]
    a, b = param.shape
    r = frozen.rank

    grad_accum = torch.zeros_like(param)
    for i in range(num_envs):
        A, B = get_lora_perturbation(
            frozen,
            sigma / math.sqrt(r),
            epochs[i].item(),
            thread_ids[i].item(),
            param,
            param_seed,
        )
        # Outer product: A @ B^T weighted by fitness
        grad_accum += fitnesses[i] * (A @ B.T)

    return grad_accum / num_envs


def compute_full_gradient(
    frozen: FrozenNoiserParams,
    sigma: float,
    param: torch.Tensor,
    param_seed: int,
    fitnesses: torch.Tensor,
    epochs: torch.Tensor,
    thread_ids: torch.Tensor,
) -> torch.Tensor:
    """Estimate gradient for a fully-perturbed parameter."""
    if frozen.freeze_nonlora:
        return torch.zeros_like(param)

    num_envs = fitnesses.shape[0]
    grad_accum = torch.zeros_like(param)
    for i in range(num_envs):
        noise = get_full_perturbation(
            frozen,
            sigma,
            epochs[i].item(),
            thread_ids[i].item(),
            param,
            param_seed,
        )
        grad_accum += fitnesses[i] * noise

    return grad_accum / num_envs


def normalize_fitnesses(raw_scores: torch.Tensor, group_size: int = 0) -> torch.Tensor:
    """Normalize fitness scores using z-score normalization.

    If group_size > 0, normalizes within groups (for shared-prompt settings).
    Global variance is always used for the denominator.

    Args:
        raw_scores: Raw fitness scores, shape (population_size,).
        group_size: If > 0, subtract group mean instead of global mean.

    Returns:
        Normalized fitness scores.
    """
    global_var = torch.var(raw_scores) + 1e-5
    global_std = torch.sqrt(global_var)

    if group_size == 0:
        return (raw_scores - raw_scores.mean()) / global_std
    else:
        grouped = raw_scores.reshape(-1, group_size)
        group_means = grouped.mean(dim=-1, keepdim=True)
        normalized = (grouped - group_means) / global_std
        return normalized.reshape(-1)


class BaseNoiser:
    """Base noiser with no perturbation (evaluation mode)."""

    @classmethod
    def init(
        cls,
        named_params: dict[str, torch.Tensor],
        sigma: float,
        lr: float,
        **kwargs,
    ) -> tuple[FrozenNoiserParams, NoisedParams]:
        return FrozenNoiserParams(), NoisedParams(sigma=sigma, optimizer=None)

    @classmethod
    def perturb_forward(
        cls,
        frozen: FrozenNoiserParams,
        noised: NoisedParams,
        model: nn.Module,
        epoch: int,
        thread_id: int,
    ) -> None:
        """No-op: base noiser does not perturb."""
        pass

    @classmethod
    def restore(cls, model: nn.Module, original_params: dict) -> None:
        """Restore model parameters to unperturbed values."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in original_params:
                    param.copy_(original_params[name])


class EggRoll:
    """EGGROLL evolutionary strategy noiser.

    Uses LoRA-style low-rank perturbations for weight matrices and
    full perturbations for biases/scalars. Integrates with standard
    PyTorch optimizers for parameter updates.

    Key features:
    - Antithetical sampling (paired +/- perturbations)
    - LoRA perturbation for memory-efficient matrix noise
    - Compatible with Adam, SGD, or any PyTorch optimizer
    - Group-based fitness normalization
    """

    @classmethod
    def init(
        cls,
        model: nn.Module,
        sigma: float,
        lr: float,
        optimizer_cls: type = None,
        optimizer_kwargs: dict[str, Any] | None = None,
        group_size: int = 0,
        freeze_nonlora: bool = True,
        noise_reuse: int = 0,
        rank: int = 1,
        use_batched_update: bool = False,
    ) -> tuple[FrozenNoiserParams, NoisedParams]:
        """Initialize the EGGROLL noiser.

        Args:
            model: The model to train.
            sigma: Perturbation magnitude.
            lr: Learning rate.
            optimizer_cls: PyTorch optimizer class (default: Adam).
            optimizer_kwargs: Additional optimizer kwargs.
            group_size: Group size for fitness normalization.
            freeze_nonlora: If True, don't perturb non-matrix params.
            noise_reuse: Reuse noise across this many epochs.
            rank: Rank of LoRA perturbations.
            use_batched_update: Whether to batch gradient computation.

        Returns:
            (frozen_params, noised_params) tuple.
        """
        if optimizer_cls is None:
            optimizer_cls = torch.optim.Adam
        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        optimizer = optimizer_cls(model.parameters(), lr=lr, **optimizer_kwargs)

        frozen = FrozenNoiserParams(
            group_size=group_size,
            freeze_nonlora=freeze_nonlora,
            noise_reuse=noise_reuse,
            rank=rank,
            use_batched_update=use_batched_update,
        )
        noised = NoisedParams(sigma=sigma, optimizer=optimizer)
        return frozen, noised

    @classmethod
    def classify_param(cls, name: str, param: torch.Tensor) -> int:
        """Classify a parameter for ES update strategy.

        Returns:
            PARAM (0) for biases/scalars (full perturbation),
            MM_PARAM (1) for weight matrices (LoRA perturbation),
            EMB_PARAM (2) for embeddings (LoRA perturbation),
            EXCLUDED (3) for frozen parameters.
        """
        if "embed" in name.lower() or "lm_head" in name.lower():
            return EXCLUDED
        elif param.ndim >= 2 and "weight" in name.lower():
            return MM_PARAM
        elif param.ndim <= 1:
            return PARAM
        else:
            return MM_PARAM

    @classmethod
    def get_param_seeds(cls, model: nn.Module, base_seed: int = 42) -> dict:
        """Generate deterministic per-parameter seeds."""
        seeds = {}
        for i, (name, _) in enumerate(model.named_parameters()):
            seeds[name] = base_seed + i * 7919  # Prime-spaced seeds
        return seeds

    @classmethod
    def perturb_model(
        cls,
        frozen: FrozenNoiserParams,
        sigma: float,
        model: nn.Module,
        epoch: int,
        thread_id: int,
        param_seeds: dict[str, int],
        original_params: dict[str, torch.Tensor],
    ) -> None:
        """Apply perturbations to model parameters in-place.

        For LoRA params (matrices): W' = W + A @ B^T
        For full params (biases): p' = p + noise
        For excluded params: no change
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                classification = cls.classify_param(name, param)
                if classification == EXCLUDED:
                    continue

                seed = param_seeds[name]
                original = original_params[name]

                if classification == MM_PARAM:
                    A, B = get_lora_perturbation(
                        frozen, sigma, epoch, thread_id, original, seed
                    )
                    param.copy_(original + A @ B.T)
                elif classification == PARAM:
                    if frozen.freeze_nonlora:
                        continue
                    noise = get_full_perturbation(
                        frozen, sigma, epoch, thread_id, original, seed
                    )
                    param.copy_(original + noise)

    @classmethod
    def compute_gradients(
        cls,
        frozen: FrozenNoiserParams,
        sigma: float,
        model: nn.Module,
        param_seeds: dict[str, int],
        fitnesses: torch.Tensor,
        epochs: torch.Tensor,
        thread_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute ES gradient estimates from population fitnesses.

        Args:
            frozen: Frozen noiser configuration.
            sigma: Perturbation magnitude.
            model: The model (with original unperturbed weights).
            param_seeds: Per-parameter random seeds.
            fitnesses: Normalized fitness scores, shape (pop_size,).
            epochs: Epoch indices for each population member.
            thread_ids: Thread indices for each population member.

        Returns:
            Dictionary mapping parameter names to gradient tensors.
        """
        gradients = {}
        pop_size = fitnesses.shape[0]

        for name, param in model.named_parameters():
            classification = cls.classify_param(name, param)
            seed = param_seeds[name]

            if classification == EXCLUDED:
                gradients[name] = torch.zeros_like(param)
            elif classification == MM_PARAM:
                grad = compute_lora_gradient(
                    frozen, sigma, param, seed, fitnesses, epochs, thread_ids
                )
                # Scale by sqrt(pop_size) and negate (for maximization)
                gradients[name] = -(grad * math.sqrt(pop_size))
            elif classification == PARAM:
                grad = compute_full_gradient(
                    frozen, sigma, param, seed, fitnesses, epochs, thread_ids
                )
                gradients[name] = -(grad * math.sqrt(pop_size))

        return gradients

    @classmethod
    def update_params(
        cls,
        noised: NoisedParams,
        model: nn.Module,
        gradients: dict[str, torch.Tensor],
    ) -> None:
        """Apply gradient estimates to update model parameters.

        Uses the configured optimizer (Adam/SGD) to apply the
        evolutionary gradient estimates.
        """
        noised.optimizer.zero_grad()

        # Set .grad on each parameter from our ES gradient estimates
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in gradients:
                    param.grad = gradients[name].to(param.dtype)

        noised.optimizer.step()
        noised.step += 1

    @classmethod
    def convert_fitnesses(
        cls, frozen: FrozenNoiserParams, raw_scores: torch.Tensor
    ) -> torch.Tensor:
        """Normalize raw fitness scores."""
        return normalize_fitnesses(raw_scores, frozen.group_size)


class OpenES:
    """OpenAI-style Evolutionary Strategy.

    Uses full-rank perturbations for all parameters (no LoRA).
    Simpler but more memory-intensive than EggRoll.
    """

    @classmethod
    def init(
        cls,
        model: nn.Module,
        sigma: float,
        lr: float,
        optimizer_cls: type = None,
        optimizer_kwargs: dict[str, Any] | None = None,
        group_size: int = 0,
        freeze_nonlora: bool = False,
        noise_reuse: int = 0,
    ) -> tuple[FrozenNoiserParams, NoisedParams]:
        if optimizer_cls is None:
            optimizer_cls = torch.optim.SGD
        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        optimizer = optimizer_cls(model.parameters(), lr=lr, **optimizer_kwargs)
        frozen = FrozenNoiserParams(
            group_size=group_size,
            freeze_nonlora=freeze_nonlora,
            noise_reuse=noise_reuse,
            rank=1,
        )
        noised = NoisedParams(sigma=sigma, optimizer=optimizer)
        return frozen, noised

    @classmethod
    def classify_param(cls, name: str, param: torch.Tensor) -> int:
        if "embed" in name.lower() or "lm_head" in name.lower():
            return EXCLUDED
        return PARAM  # OpenES uses full perturbation for everything

    @classmethod
    def get_param_seeds(cls, model: nn.Module, base_seed: int = 42) -> dict:
        return EggRoll.get_param_seeds(model, base_seed)

    @classmethod
    def perturb_model(
        cls,
        frozen: FrozenNoiserParams,
        sigma: float,
        model: nn.Module,
        epoch: int,
        thread_id: int,
        param_seeds: dict[str, int],
        original_params: dict[str, torch.Tensor],
    ) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                classification = cls.classify_param(name, param)
                if classification == EXCLUDED:
                    continue
                if frozen.freeze_nonlora and classification == PARAM:
                    continue

                seed = param_seeds[name]
                original = original_params[name]
                noise = get_full_perturbation(
                    frozen, sigma, epoch, thread_id, original, seed
                )
                param.copy_(original + noise)

    @classmethod
    def compute_gradients(
        cls,
        frozen: FrozenNoiserParams,
        sigma: float,
        model: nn.Module,
        param_seeds: dict[str, int],
        fitnesses: torch.Tensor,
        epochs: torch.Tensor,
        thread_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        gradients = {}
        pop_size = fitnesses.shape[0]
        for name, param in model.named_parameters():
            classification = cls.classify_param(name, param)
            seed = param_seeds[name]
            if classification == EXCLUDED:
                gradients[name] = torch.zeros_like(param)
            else:
                grad = compute_full_gradient(
                    frozen, sigma, param, seed, fitnesses, epochs, thread_ids
                )
                gradients[name] = -(grad * math.sqrt(pop_size))
        return gradients

    @classmethod
    def update_params(
        cls,
        noised: NoisedParams,
        model: nn.Module,
        gradients: dict[str, torch.Tensor],
    ) -> None:
        EggRoll.update_params(noised, model, gradients)

    @classmethod
    def convert_fitnesses(
        cls, frozen: FrozenNoiserParams, raw_scores: torch.Tensor
    ) -> torch.Tensor:
        return normalize_fitnesses(raw_scores, frozen.group_size)
