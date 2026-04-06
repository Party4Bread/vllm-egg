# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EGGROLL and OpenES noiser implementations in PyTorch.

Port of HyperscaleES evolutionary strategy noisers from JAX to PyTorch
for integration with vLLM's inference engine.

Scalability design:
- Noise generation is batched: all population members' noise is generated
  in a single GPU kernel call per parameter (no Python loops over pop).
- Gradient estimation uses vectorized einsum (LoRA) or broadcast-multiply
  (full), producing gradients in O(1) kernel launches per parameter.
- Antithetical sampling is handled via sign vectors, not branching.

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
EMB_PARAM = 2  # Embedding parameter (excluded in current impl)
EXCLUDED = 3  # Excluded from evolution


@dataclass
class NoisedParams:
    """Mutable state for the noiser during evolution."""

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


# ---------------------------------------------------------------------------
# Vectorized noise generation (all pop members at once)
# ---------------------------------------------------------------------------


def _batch_generators(
    param_seed: int,
    noise_reuse: int,
    epoch: int,
    pop_size: int,
) -> list[torch.Generator]:
    """Create pop_size//2 deterministic generators (one per antithetical pair).

    Returns pop_size generators where pairs share the same seed.
    """
    true_epoch = 0 if noise_reuse == 0 else epoch // noise_reuse
    gens = []
    for tid in range(pop_size):
        true_thread = tid // 2
        gen = torch.Generator()
        combined = param_seed ^ (true_epoch * 2654435761) ^ (true_thread * 40503)
        gen.manual_seed(combined & 0xFFFFFFFF)
        gens.append(gen)
    return gens


def batch_lora_noise(
    frozen: FrozenNoiserParams,
    sigma: float,
    epoch: int,
    pop_size: int,
    param: torch.Tensor,
    param_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate LoRA noise for ALL population members at once.

    Returns:
        A: (pop_size, out_dim, rank) - already scaled by ±sigma/sqrt(rank)
        B: (pop_size, in_dim, rank) - unscaled random directions
    """
    a, b = param.shape
    r = frozen.rank
    device = param.device
    dtype = param.dtype

    # Sign vector for antithetical sampling: +1 for even, -1 for odd
    signs = torch.ones(pop_size, 1, 1, device=device, dtype=dtype)
    signs[1::2] = -1.0
    effective_sigma = sigma / math.sqrt(r)

    # Generate noise for each unique pair on CPU then move to GPU
    # Only need pop_size//2 unique noise vectors (antithetical pairs share)
    n_unique = (pop_size + 1) // 2
    gens = _batch_generators(param_seed, frozen.noise_reuse, epoch, pop_size)

    # Pre-allocate on device
    all_A = torch.empty(pop_size, a, r, device=device, dtype=dtype)
    all_B = torch.empty(pop_size, b, r, device=device, dtype=dtype)

    for pair_idx in range(n_unique):
        gen = gens[pair_idx * 2]
        lora_params = torch.randn(a + b, r, generator=gen, dtype=dtype, device=device)
        B_i = lora_params[:b]
        A_i = lora_params[b:]

        all_A[pair_idx * 2] = A_i
        all_B[pair_idx * 2] = B_i
        if pair_idx * 2 + 1 < pop_size:
            all_A[pair_idx * 2 + 1] = A_i  # Same noise, sign applied later
            all_B[pair_idx * 2 + 1] = B_i

    # Apply antithetical signs and sigma to A only (matching HyperscaleES)
    all_A = all_A * signs * effective_sigma
    return all_A, all_B


def batch_full_noise(
    frozen: FrozenNoiserParams,
    sigma: float,
    epoch: int,
    pop_size: int,
    param: torch.Tensor,
    param_seed: int,
) -> torch.Tensor:
    """Generate full-rank noise for ALL population members at once.

    Returns:
        noise: (pop_size, *param.shape) - scaled by ±sigma
    """
    device = param.device
    dtype = param.dtype
    shape = param.shape

    signs = torch.ones(pop_size, *([1] * len(shape)), device=device, dtype=dtype)
    signs[1::2] = -1.0

    n_unique = (pop_size + 1) // 2
    gens = _batch_generators(param_seed, frozen.noise_reuse, epoch, pop_size)

    all_noise = torch.empty(pop_size, *shape, device=device, dtype=dtype)
    for pair_idx in range(n_unique):
        gen = gens[pair_idx * 2]
        noise_i = torch.randn(*shape, generator=gen, dtype=dtype, device=device)
        all_noise[pair_idx * 2] = noise_i
        if pair_idx * 2 + 1 < pop_size:
            all_noise[pair_idx * 2 + 1] = noise_i

    return all_noise * signs * sigma


# ---------------------------------------------------------------------------
# Vectorized gradient estimation (single kernel per parameter)
# ---------------------------------------------------------------------------


def compute_lora_gradient_batched(
    frozen: FrozenNoiserParams,
    sigma: float,
    epoch: int,
    pop_size: int,
    param: torch.Tensor,
    param_seed: int,
    fitnesses: torch.Tensor,
) -> torch.Tensor:
    """Estimate gradient for a LoRA parameter using vectorized ops.

    grad = mean_i[ fitness_i * A_i @ B_i^T ]
         = einsum('nir,njr->ij', fitness*A, B) / pop_size

    This runs in O(1) kernel launches regardless of pop_size.
    """
    # (pop_size, out, rank), (pop_size, in, rank)
    all_A, all_B = batch_lora_noise(frozen, sigma, epoch, pop_size, param, param_seed)

    # Broadcast fitness: (pop_size, 1, 1)
    f = fitnesses.reshape(pop_size, 1, 1)
    weighted_A = f * all_A  # (pop_size, out, rank)

    # Batched outer product sum via einsum
    grad = torch.einsum("nir,njr->ij", weighted_A, all_B) / pop_size
    return grad


def compute_full_gradient_batched(
    frozen: FrozenNoiserParams,
    sigma: float,
    epoch: int,
    pop_size: int,
    param: torch.Tensor,
    param_seed: int,
    fitnesses: torch.Tensor,
) -> torch.Tensor:
    """Estimate gradient for a full-noise parameter using vectorized ops.

    grad = mean_i[ fitness_i * noise_i ]

    Single broadcast-multiply + mean, O(1) kernel launches.
    """
    if frozen.freeze_nonlora:
        return torch.zeros_like(param)

    all_noise = batch_full_noise(frozen, sigma, epoch, pop_size, param, param_seed)
    f = fitnesses.reshape(pop_size, *([1] * param.ndim))
    return (f * all_noise).mean(dim=0)


# ---------------------------------------------------------------------------
# Fitness normalization
# ---------------------------------------------------------------------------


def normalize_fitnesses(raw_scores: torch.Tensor, group_size: int = 0) -> torch.Tensor:
    """Z-score normalize fitness. Group-aware if group_size > 0."""
    global_std = torch.sqrt(torch.var(raw_scores) + 1e-5)
    if group_size == 0:
        return (raw_scores - raw_scores.mean()) / global_std
    grouped = raw_scores.reshape(-1, group_size)
    group_means = grouped.mean(dim=-1, keepdim=True)
    return ((grouped - group_means) / global_std).reshape(-1)


# ---------------------------------------------------------------------------
# Noiser classes
# ---------------------------------------------------------------------------


class EggRoll:
    """EGGROLL evolutionary strategy noiser.

    Uses LoRA-style low-rank perturbations for weight matrices (MM_PARAM)
    and full perturbations for biases/scalars (PARAM). Embeddings and
    LM heads are excluded from perturbation.

    All noise generation and gradient estimation is vectorized over the
    population dimension — no Python loops over pop_size.
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
        **kwargs,
    ) -> tuple[FrozenNoiserParams, NoisedParams]:
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
        )
        return frozen, NoisedParams(sigma=sigma, optimizer=optimizer)

    @classmethod
    def classify_param(cls, name: str, param: torch.Tensor) -> int:
        if "embed" in name.lower() or "lm_head" in name.lower():
            return EXCLUDED
        if param.ndim >= 2 and "weight" in name.lower():
            return MM_PARAM
        if param.ndim <= 1:
            return PARAM
        return MM_PARAM

    @classmethod
    def get_param_seeds(cls, model: nn.Module, base_seed: int = 42) -> dict:
        return {
            name: base_seed + i * 7919
            for i, (name, _) in enumerate(model.named_parameters())
        }

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
        """Apply perturbation for a single population member in-place."""
        true_epoch = 0 if frozen.noise_reuse == 0 else epoch // frozen.noise_reuse
        true_thread = thread_id // 2
        sign = 1.0 if thread_id % 2 == 0 else -1.0
        r = frozen.rank

        with torch.no_grad():
            for name, param in model.named_parameters():
                classification = cls.classify_param(name, param)
                if classification == EXCLUDED:
                    continue
                if classification == PARAM and frozen.freeze_nonlora:
                    continue

                seed = param_seeds[name]
                original = original_params[name]
                gen = torch.Generator()
                combined = seed ^ (true_epoch * 2654435761) ^ (true_thread * 40503)
                gen.manual_seed(combined & 0xFFFFFFFF)

                if classification == MM_PARAM:
                    a, b = original.shape
                    lora = torch.randn(
                        a + b,
                        r,
                        generator=gen,
                        dtype=original.dtype,
                        device=original.device,
                    )
                    B_i, A_i = lora[:b], lora[b:]
                    eff_sigma = sign * sigma / math.sqrt(r)
                    param.copy_(original + (A_i * eff_sigma) @ B_i.T)
                else:
                    noise = torch.randn(
                        *original.shape,
                        generator=gen,
                        dtype=original.dtype,
                        device=original.device,
                    )
                    param.copy_(original + noise * sign * sigma)

    @classmethod
    def compute_gradients(
        cls,
        frozen: FrozenNoiserParams,
        sigma: float,
        epoch: int,
        pop_size: int,
        model: nn.Module,
        param_seeds: dict[str, int],
        fitnesses: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Vectorized gradient estimation for all parameters.

        All population members' noise is generated in batch and combined
        via a single einsum/broadcast per parameter — no Python loop
        over population.
        """
        gradients = {}

        for name, param in model.named_parameters():
            classification = cls.classify_param(name, param)
            seed = param_seeds[name]

            if classification == EXCLUDED:
                gradients[name] = torch.zeros_like(param)
            elif classification == MM_PARAM:
                grad = compute_lora_gradient_batched(
                    frozen, sigma, epoch, pop_size, param, seed, fitnesses
                )
                gradients[name] = -(grad * math.sqrt(pop_size)).to(param.dtype)
            elif classification == PARAM:
                grad = compute_full_gradient_batched(
                    frozen, sigma, epoch, pop_size, param, seed, fitnesses
                )
                gradients[name] = -(grad * math.sqrt(pop_size)).to(param.dtype)

        return gradients

    @classmethod
    def update_params(
        cls,
        noised: NoisedParams,
        model: nn.Module,
        gradients: dict[str, torch.Tensor],
    ) -> None:
        """Apply ES gradient estimates via the configured optimizer."""
        noised.optimizer.zero_grad()
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
        return normalize_fitnesses(raw_scores, frozen.group_size)

    @classmethod
    def export_lora_adapters(
        cls,
        frozen: FrozenNoiserParams,
        sigma: float,
        epoch: int,
        pop_size: int,
        model: nn.Module,
        param_seeds: dict[str, int],
        output_dir: str,
        target_modules: list[str] | None = None,
    ) -> list[str]:
        """Export each population member's perturbation as a PEFT LoRA adapter.

        This allows vLLM to serve all population members in a single batch
        by loading each member as a separate LoRA adapter.

        Args:
            frozen: Frozen noiser config.
            sigma: Perturbation magnitude.
            epoch: Current epoch.
            pop_size: Population size.
            model: The base model.
            param_seeds: Per-parameter seeds.
            output_dir: Base directory for adapter files.
            target_modules: List of module name suffixes to include.
                If None, auto-detected from model.

        Returns:
            List of adapter directory paths (one per pop member).
        """
        import json
        import os

        from safetensors.torch import save_file

        r = frozen.rank
        adapter_dirs = []

        # Detect target modules from the model
        if target_modules is None:
            target_modules = []
            for name, param in model.named_parameters():
                if cls.classify_param(name, param) == MM_PARAM:
                    # Extract the module suffix (e.g. "q_proj", "k_proj")
                    parts = name.split(".")
                    # Remove ".weight" suffix
                    if parts[-1] == "weight":
                        parts = parts[:-1]
                    target_modules.append(parts[-1])
            target_modules = sorted(set(target_modules))

        for member_id in range(pop_size):
            adapter_dir = os.path.join(output_dir, f"member_{member_id}")
            os.makedirs(adapter_dir, exist_ok=True)

            # Build LoRA tensors for this member
            tensors = {}
            true_epoch = 0 if frozen.noise_reuse == 0 else epoch // frozen.noise_reuse
            true_thread = member_id // 2
            sign = 1.0 if member_id % 2 == 0 else -1.0

            for name, param in model.named_parameters():
                classification = cls.classify_param(name, param)
                if classification != MM_PARAM:
                    continue

                seed = param_seeds[name]
                a, b = param.shape
                gen = torch.Generator()
                combined = seed ^ (true_epoch * 2654435761) ^ (true_thread * 40503)
                gen.manual_seed(combined & 0xFFFFFFFF)

                lora = torch.randn(
                    a + b, r, generator=gen, dtype=param.dtype, device="cpu"
                )
                B_i = lora[:b]  # (in_dim, rank) -> lora_A
                A_i = lora[b:]  # (out_dim, rank) -> lora_B

                eff_sigma = sign * sigma / math.sqrt(r)

                # Strip "weight" suffix for module path
                module_name = name
                if module_name.endswith(".weight"):
                    module_name = module_name[: -len(".weight")]

                # PEFT naming: base_model.model.<module>.lora_A.weight
                peft_prefix = f"base_model.model.{module_name}"
                # lora_A is (rank, in_dim), lora_B is (out_dim, rank)
                tensors[f"{peft_prefix}.lora_A.weight"] = B_i.T.contiguous()
                tensors[f"{peft_prefix}.lora_B.weight"] = (A_i * eff_sigma).contiguous()

            # Save adapter weights
            save_file(tensors, os.path.join(adapter_dir, "adapter_model.safetensors"))

            # Save adapter config
            config = {
                "r": r,
                "lora_alpha": r,  # alpha=r means scaling=1.0
                "target_modules": target_modules,
                "bias": "none",
                "task_type": "CAUSAL_LM",
                "peft_type": "LORA",
            }
            with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
                json.dump(config, f)

            adapter_dirs.append(adapter_dir)

        return adapter_dirs


class OpenES:
    """OpenAI-style ES. Full-rank perturbations for all parameters."""

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
        **kwargs,
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
        return frozen, NoisedParams(sigma=sigma, optimizer=optimizer)

    @classmethod
    def classify_param(cls, name: str, param: torch.Tensor) -> int:
        if "embed" in name.lower() or "lm_head" in name.lower():
            return EXCLUDED
        return PARAM

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
        EggRoll.perturb_model(
            frozen, sigma, model, epoch, thread_id, param_seeds, original_params
        )

    @classmethod
    def compute_gradients(
        cls,
        frozen: FrozenNoiserParams,
        sigma: float,
        epoch: int,
        pop_size: int,
        model: nn.Module,
        param_seeds: dict[str, int],
        fitnesses: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        gradients = {}
        for name, param in model.named_parameters():
            classification = cls.classify_param(name, param)
            seed = param_seeds[name]
            if classification == EXCLUDED:
                gradients[name] = torch.zeros_like(param)
            else:
                grad = compute_full_gradient_batched(
                    frozen, sigma, epoch, pop_size, param, seed, fitnesses
                )
                gradients[name] = -(grad * math.sqrt(pop_size)).to(param.dtype)
        return gradients

    @classmethod
    def update_params(
        cls, noised: NoisedParams, model: nn.Module, gradients: dict[str, torch.Tensor]
    ) -> None:
        EggRoll.update_params(noised, model, gradients)

    @classmethod
    def convert_fitnesses(
        cls, frozen: FrozenNoiserParams, raw_scores: torch.Tensor
    ) -> torch.Tensor:
        return normalize_fitnesses(raw_scores, frozen.group_size)
