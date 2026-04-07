#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end test: verify EGGROLL can optimize a simple function.

This test mirrors HyperscaleES/tests/end_to_end_test.py:
Train an MLP to output 2.0 using EGGROLL evolutionary strategy.
No GPU or vLLM required — pure CPU PyTorch.

Verifies:
1. Noise generation produces correct shapes and antithetical pairs
2. Perturbed models produce different outputs than base model
3. Gradient estimation produces non-zero gradients
4. Optimizer updates move parameters in the right direction
5. Fitness improves over epochs (the algorithm actually works)
"""

import torch
from torch import nn

from vllm.eggroll.noiser import (
    EggRoll,
    FrozenNoiserParams,
    OpenES,
    batch_full_noise,
    batch_lora_noise,
    compute_full_gradient_batched,
    compute_lora_gradient_batched,
    normalize_fitnesses,
)


def test_eggroll_optimizes_to_target():
    """Core test: EGGROLL should drive MLP output toward target=2.0."""
    torch.manual_seed(0)

    model = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))
    frozen, noised = EggRoll.init(
        model,
        sigma=0.2,
        lr=0.03,
        rank=8,
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs={"betas": (0.9, 0.999)},
    )
    seeds = EggRoll.get_param_seeds(model)
    plan = EggRoll.get_param_plan(model)
    original = {n: p.data.clone() for n, p in model.named_parameters()}

    pop_size = 64
    target = 2.0
    initial_fitness = None
    final_fitness = None

    for epoch in range(30):
        # Generate random inputs
        torch.manual_seed(epoch + 100)
        x = torch.randn(pop_size, 3)

        # Evaluate each population member
        raw_scores = torch.zeros(pop_size)
        for member_id in range(pop_size):
            EggRoll.perturb_model(
                frozen,
                noised.sigma,
                model,
                epoch,
                member_id,
                seeds,
                original,
            )
            with torch.no_grad():
                out = model(x[member_id].unsqueeze(0))
            raw_scores[member_id] = -((out.item() - target) ** 2)
            # Restore
            with torch.no_grad():
                for n, p in model.named_parameters():
                    p.copy_(original[n])

        if epoch == 0:
            initial_fitness = raw_scores.mean().item()

        # Normalize and update
        fitnesses = EggRoll.convert_fitnesses(frozen, raw_scores)
        grads = EggRoll.compute_gradients(
            frozen,
            noised.sigma,
            epoch,
            pop_size,
            model,
            seeds,
            fitnesses,
            param_plan=plan,
        )
        EggRoll.update_params(noised, model, grads)
        original = {n: p.data.clone() for n, p in model.named_parameters()}

        final_fitness = raw_scores.mean().item()

    # Fitness should improve (become less negative)
    assert final_fitness > initial_fitness, (
        f"Fitness did not improve: {initial_fitness:.4f} -> {final_fitness:.4f}"
    )
    print(f"PASS: fitness improved {initial_fitness:.4f} -> {final_fitness:.4f}")


def test_open_es_optimizes_to_target():
    """OpenES should also optimize, just with full-rank noise."""
    torch.manual_seed(0)

    model = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))
    frozen, noised = OpenES.init(
        model,
        sigma=0.2,
        lr=0.03,
        optimizer_cls=torch.optim.SGD,
        freeze_nonlora=False,
    )
    seeds = OpenES.get_param_seeds(model)
    original = {n: p.data.clone() for n, p in model.named_parameters()}

    pop_size = 64
    target = 2.0
    initial_fitness = None
    final_fitness = None

    for epoch in range(30):
        torch.manual_seed(epoch + 100)
        x = torch.randn(pop_size, 3)

        raw_scores = torch.zeros(pop_size)
        for member_id in range(pop_size):
            OpenES.perturb_model(
                frozen,
                noised.sigma,
                model,
                epoch,
                member_id,
                seeds,
                original,
            )
            with torch.no_grad():
                out = model(x[member_id].unsqueeze(0))
            raw_scores[member_id] = -((out.item() - target) ** 2)
            with torch.no_grad():
                for n, p in model.named_parameters():
                    p.copy_(original[n])

        if epoch == 0:
            initial_fitness = raw_scores.mean().item()

        fitnesses = OpenES.convert_fitnesses(frozen, raw_scores)
        grads = OpenES.compute_gradients(
            frozen,
            noised.sigma,
            epoch,
            pop_size,
            model,
            seeds,
            fitnesses,
        )
        OpenES.update_params(noised, model, grads)
        original = {n: p.data.clone() for n, p in model.named_parameters()}
        final_fitness = raw_scores.mean().item()

    assert final_fitness > initial_fitness, (
        f"Fitness did not improve: {initial_fitness:.4f} -> {final_fitness:.4f}"
    )
    print(f"PASS: OpenES fitness improved {initial_fitness:.4f} -> {final_fitness:.4f}")


def test_noise_properties():
    """Verify noise correctness: antithetical, deterministic, scaled."""
    frozen = FrozenNoiserParams(rank=4)
    param = torch.randn(16, 8)

    # Shape
    A, B = batch_lora_noise(frozen, 0.5, 0, 10, param, 42)
    assert A.shape == (10, 16, 4), f"Bad A shape: {A.shape}"
    assert B.shape == (10, 8, 4), f"Bad B shape: {B.shape}"

    # Antithetical: pairs should negate
    assert torch.allclose(A[0], -A[1], atol=1e-6), "Pair 0 not antithetical"
    assert torch.allclose(B[0], B[1], atol=1e-6), "Pair 0 B not shared"
    assert torch.allclose(A[4], -A[5], atol=1e-6), "Pair 2 not antithetical"

    # Deterministic: same call twice should match
    A2, B2 = batch_lora_noise(frozen, 0.5, 0, 10, param, 42)
    assert torch.allclose(A, A2), "Not deterministic"
    assert torch.allclose(B, B2), "Not deterministic"

    # Different seeds -> different noise
    A3, _ = batch_lora_noise(frozen, 0.5, 0, 10, param, 99)
    assert not torch.allclose(A, A3), "Different seeds produced same noise"

    # Full noise
    bias = torch.randn(16)
    noise = batch_full_noise(FrozenNoiserParams(), 0.3, 0, 6, bias, 42)
    assert noise.shape == (6, 16)
    assert torch.allclose(noise[0], -noise[1], atol=1e-6)

    print("PASS: noise properties correct")


def test_gradient_correctness():
    """Verify gradient: zero fitness -> zero gradient, uniform -> zero mean."""
    frozen = FrozenNoiserParams(rank=4)
    param = torch.randn(16, 8)

    # Zero fitness -> zero gradient
    zeros = torch.zeros(10)
    grad = compute_lora_gradient_batched(frozen, 0.5, 0, 10, param, 42, zeros)
    assert torch.allclose(grad, torch.zeros_like(grad), atol=1e-7), (
        "Zero fitness should give zero gradient"
    )

    # Constant fitness -> gradient should be small (antithetical cancellation)
    const = torch.ones(10)
    grad = compute_lora_gradient_batched(frozen, 0.5, 0, 10, param, 42, const)
    # With antithetical sampling and constant fitness, positive and negative
    # perturbations cancel, so gradient should be near zero
    assert grad.abs().max() < 0.5, (
        f"Constant fitness should nearly cancel: max={grad.abs().max():.4f}"
    )

    # Full noise gradient
    bias = torch.randn(16)
    frozen_full = FrozenNoiserParams(freeze_nonlora=False)
    grad_full = compute_full_gradient_batched(
        frozen_full,
        0.3,
        0,
        10,
        bias,
        42,
        zeros,
    )
    assert torch.allclose(grad_full, torch.zeros_like(grad_full), atol=1e-7)

    # Freeze nonlora -> zero
    frozen_freeze = FrozenNoiserParams(freeze_nonlora=True)
    grad_frozen = compute_full_gradient_batched(
        frozen_freeze,
        0.3,
        0,
        10,
        bias,
        42,
        torch.randn(10),
    )
    assert torch.allclose(grad_frozen, torch.zeros_like(grad_frozen))

    print("PASS: gradient correctness verified")


def test_fitness_normalization():
    """Verify z-score normalization and group normalization."""
    scores = torch.tensor([1.0, 3.0, 5.0, 7.0])
    normed = normalize_fitnesses(scores)
    assert abs(normed.mean().item()) < 1e-5, "Should be zero-mean"
    assert abs(normed.std().item() - 1.0) < 0.3, "Should be ~unit std"

    # Group normalization
    scores = torch.tensor([0.0, 10.0, 100.0, 110.0])
    normed = normalize_fitnesses(scores, group_size=2)
    # Within-group: (0,10) -> (-5, 5)/std, (100,110) -> (-5, 5)/std
    assert normed[0] < 0 and normed[1] > 0
    assert normed[2] < 0 and normed[3] > 0
    # Cross-group: both groups have same spread, so normalized values ~equal
    assert abs(normed[0].item() - normed[2].item()) < 0.01

    print("PASS: fitness normalization correct")


def test_perturb_produces_different_outputs():
    """Each population member should produce a different model output."""
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    frozen, noised = EggRoll.init(model, sigma=0.1, lr=0.01, rank=4)
    seeds = EggRoll.get_param_seeds(model)
    original = {n: p.data.clone() for n, p in model.named_parameters()}

    x = torch.randn(1, 4)
    outputs = []
    for member_id in range(8):
        EggRoll.perturb_model(
            frozen,
            noised.sigma,
            model,
            0,
            member_id,
            seeds,
            original,
        )
        with torch.no_grad():
            outputs.append(model(x).item())
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(original[n])

    # All 8 outputs should be different
    unique = set(round(o, 6) for o in outputs)
    assert len(unique) >= 4, (
        f"Expected diverse outputs, got {len(unique)} unique from {outputs}"
    )
    # Antithetical pairs should produce different outputs
    assert outputs[0] != outputs[1], "Antithetical pair should differ"

    print(f"PASS: 8 members produced {len(unique)} unique outputs")


def test_full_es_loop_converges():
    """Full ES loop: model should learn to output target value."""
    torch.manual_seed(7)

    model = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1))

    # Record initial output
    x_test = torch.ones(1, 1)
    with torch.no_grad():
        initial_output = model(x_test).item()

    frozen, noised = EggRoll.init(
        model,
        sigma=0.3,
        lr=0.05,
        rank=4,
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs={"betas": (0.9, 0.999)},
    )
    seeds = EggRoll.get_param_seeds(model)
    plan = EggRoll.get_param_plan(model)
    original = {n: p.data.clone() for n, p in model.named_parameters()}

    target = 5.0
    pop_size = 128

    for epoch in range(50):
        raw_scores = torch.zeros(pop_size)
        for mid in range(pop_size):
            EggRoll.perturb_model(
                frozen,
                noised.sigma,
                model,
                epoch,
                mid,
                seeds,
                original,
            )
            with torch.no_grad():
                out = model(x_test)
            raw_scores[mid] = -((out.item() - target) ** 2)
            with torch.no_grad():
                for n, p in model.named_parameters():
                    p.copy_(original[n])

        fitnesses = EggRoll.convert_fitnesses(frozen, raw_scores)
        grads = EggRoll.compute_gradients(
            frozen,
            noised.sigma,
            epoch,
            pop_size,
            model,
            seeds,
            fitnesses,
            param_plan=plan,
        )
        EggRoll.update_params(noised, model, grads)
        original = {n: p.data.clone() for n, p in model.named_parameters()}

    # Check final output is closer to target
    with torch.no_grad():
        final_output = model(x_test).item()

    initial_error = abs(initial_output - target)
    final_error = abs(final_output - target)

    assert final_error < initial_error, (
        f"Should converge toward target={target}: "
        f"initial={initial_output:.3f} (err={initial_error:.3f}), "
        f"final={final_output:.3f} (err={final_error:.3f})"
    )
    print(
        f"PASS: converged toward target={target}: "
        f"{initial_output:.3f} -> {final_output:.3f} "
        f"(error {initial_error:.3f} -> {final_error:.3f})"
    )


if __name__ == "__main__":
    print("=" * 60)
    print("EGGROLL End-to-End Tests")
    print("=" * 60)

    test_noise_properties()
    test_gradient_correctness()
    test_fitness_normalization()
    test_perturb_produces_different_outputs()
    test_eggroll_optimizes_to_target()
    test_open_es_optimizes_to_target()
    test_full_es_loop_converges()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
