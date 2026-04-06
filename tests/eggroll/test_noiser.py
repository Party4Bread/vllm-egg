# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the EGGROLL noiser module."""

import torch
from torch import nn

from vllm.eggroll.noiser import (
    EXCLUDED,
    MM_PARAM,
    PARAM,
    EggRoll,
    FrozenNoiserParams,
    OpenES,
    batch_full_noise,
    batch_lora_noise,
    compute_full_gradient_batched,
    compute_lora_gradient_batched,
    normalize_fitnesses,
)


class TestBatchNoiseGeneration:
    def test_lora_shapes(self):
        frozen = FrozenNoiserParams(rank=4)
        param = torch.randn(32, 16)
        A, B = batch_lora_noise(frozen, 0.1, 0, 8, param, 42)
        assert A.shape == (8, 32, 4)
        assert B.shape == (8, 16, 4)

    def test_full_noise_shapes(self):
        frozen = FrozenNoiserParams()
        param = torch.randn(16)
        noise = batch_full_noise(frozen, 0.1, 0, 8, param, 42)
        assert noise.shape == (8, 16)

    def test_antithetical_lora(self):
        """Even/odd pairs: A has opposite sign, B is same."""
        frozen = FrozenNoiserParams(rank=2)
        param = torch.randn(8, 4)
        A, B = batch_lora_noise(frozen, 1.0, 0, 4, param, 42)
        # Pair 0: indices 0,1
        assert torch.allclose(A[0], -A[1], atol=1e-6)
        assert torch.allclose(B[0], B[1], atol=1e-6)
        # Pair 1: indices 2,3
        assert torch.allclose(A[2], -A[3], atol=1e-6)
        assert torch.allclose(B[2], B[3], atol=1e-6)

    def test_antithetical_full(self):
        frozen = FrozenNoiserParams()
        param = torch.randn(8)
        noise = batch_full_noise(frozen, 1.0, 0, 4, param, 42)
        assert torch.allclose(noise[0], -noise[1], atol=1e-6)
        assert torch.allclose(noise[2], -noise[3], atol=1e-6)

    def test_different_epochs_different_noise(self):
        frozen = FrozenNoiserParams(rank=2)
        param = torch.randn(8, 4)
        A1, _ = batch_lora_noise(frozen, 1.0, 0, 2, param, 42)
        A2, _ = batch_lora_noise(frozen, 1.0, 1, 2, param, 42)
        assert not torch.allclose(A1, A2)

    def test_noise_reuse(self):
        frozen = FrozenNoiserParams(noise_reuse=2, rank=2)
        param = torch.randn(8, 4)
        A0, B0 = batch_lora_noise(frozen, 1.0, 0, 2, param, 42)
        A1, B1 = batch_lora_noise(frozen, 1.0, 1, 2, param, 42)
        assert torch.allclose(A0, A1)
        assert torch.allclose(B0, B1)

    def test_different_pairs_different_noise(self):
        frozen = FrozenNoiserParams(rank=2)
        param = torch.randn(8, 4)
        _, B = batch_lora_noise(frozen, 1.0, 0, 4, param, 42)
        # Pair 0 and pair 1 should have different B directions
        assert not torch.allclose(B[0], B[2])


class TestBatchGradients:
    def test_lora_gradient_shape(self):
        frozen = FrozenNoiserParams(rank=4)
        param = torch.randn(32, 16)
        fitnesses = torch.randn(8)
        grad = compute_lora_gradient_batched(frozen, 0.1, 0, 8, param, 42, fitnesses)
        assert grad.shape == (32, 16)

    def test_full_gradient_shape(self):
        frozen = FrozenNoiserParams()
        param = torch.randn(16)
        fitnesses = torch.randn(8)
        grad = compute_full_gradient_batched(frozen, 0.1, 0, 8, param, 42, fitnesses)
        assert grad.shape == (16,)

    def test_zero_fitness_zero_gradient(self):
        frozen = FrozenNoiserParams(rank=2)
        param = torch.randn(8, 4)
        fitnesses = torch.zeros(8)
        grad = compute_lora_gradient_batched(frozen, 0.1, 0, 8, param, 42, fitnesses)
        assert torch.allclose(grad, torch.zeros_like(grad), atol=1e-7)

    def test_frozen_nonlora_zero_gradient(self):
        frozen = FrozenNoiserParams(freeze_nonlora=True)
        param = torch.randn(16)
        fitnesses = torch.randn(8)
        grad = compute_full_gradient_batched(frozen, 0.1, 0, 8, param, 42, fitnesses)
        assert torch.allclose(grad, torch.zeros_like(grad))


class TestFitnessNormalization:
    def test_zero_mean(self):
        scores = torch.tensor([1.0, 2.0, 3.0, 4.0])
        normalized = normalize_fitnesses(scores)
        assert abs(normalized.mean().item()) < 1e-5

    def test_group_normalization(self):
        scores = torch.tensor([1.0, 2.0, 10.0, 11.0])
        normalized = normalize_fitnesses(scores, group_size=2)
        assert normalized[0].item() < 0
        assert normalized[1].item() > 0
        assert normalized[2].item() < 0
        assert normalized[3].item() > 0


class TestEggRoll:
    def _make_model(self):
        return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    def test_init(self):
        model = self._make_model()
        frozen, noised = EggRoll.init(model, sigma=0.1, lr=0.01, rank=2)
        assert frozen.rank == 2
        assert noised.sigma == 0.1

    def test_classify_param(self):
        assert EggRoll.classify_param("layer.weight", torch.randn(4, 4)) == MM_PARAM
        assert EggRoll.classify_param("layer.bias", torch.randn(4)) == PARAM
        name = "embed_tokens.weight"
        assert EggRoll.classify_param(name, torch.randn(100, 4)) == EXCLUDED

    def test_perturb_changes_weights(self):
        model = self._make_model()
        frozen, noised = EggRoll.init(model, sigma=0.1, lr=0.01, rank=2)
        seeds = EggRoll.get_param_seeds(model)
        original = {n: p.data.clone() for n, p in model.named_parameters()}

        EggRoll.perturb_model(frozen, noised.sigma, model, 0, 0, seeds, original)

        changed = any(
            not torch.allclose(p.data, original[n]) for n, p in model.named_parameters()
        )
        assert changed

    def test_perturb_restore_roundtrip(self):
        model = self._make_model()
        frozen, noised = EggRoll.init(model, sigma=0.1, lr=0.01, rank=2)
        seeds = EggRoll.get_param_seeds(model)
        original = {n: p.data.clone() for n, p in model.named_parameters()}

        EggRoll.perturb_model(frozen, noised.sigma, model, 0, 0, seeds, original)
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(original[n])

        for n, p in model.named_parameters():
            assert torch.allclose(p.data, original[n])

    def test_gradient_shapes(self):
        model = self._make_model()
        frozen, noised = EggRoll.init(model, sigma=0.1, lr=0.01, rank=2)
        seeds = EggRoll.get_param_seeds(model)

        fitnesses = torch.randn(8)
        grads = EggRoll.compute_gradients(
            frozen, noised.sigma, 0, 8, model, seeds, fitnesses
        )

        for name, param in model.named_parameters():
            assert grads[name].shape == param.shape

    def test_full_update_cycle(self):
        model = self._make_model()
        frozen, noised = EggRoll.init(model, sigma=0.1, lr=0.01, rank=2)
        seeds = EggRoll.get_param_seeds(model)
        original = {n: p.data.clone() for n, p in model.named_parameters()}

        fitnesses = torch.randn(8)
        normalized = EggRoll.convert_fitnesses(frozen, fitnesses)
        grads = EggRoll.compute_gradients(
            frozen, noised.sigma, 0, 8, model, seeds, normalized
        )
        EggRoll.update_params(noised, model, grads)

        changed = any(
            not torch.allclose(p.data, original[n], atol=1e-8)
            for n, p in model.named_parameters()
        )
        assert changed


class TestOpenES:
    def test_init(self):
        model = nn.Linear(4, 2)
        frozen, noised = OpenES.init(model, sigma=0.1, lr=0.01)
        assert frozen.rank == 1
        assert noised.sigma == 0.1

    def test_classify_all_full(self):
        assert OpenES.classify_param("layer.weight", torch.randn(4, 4)) == PARAM
        assert OpenES.classify_param("layer.bias", torch.randn(4)) == PARAM
        assert OpenES.classify_param("embed.weight", torch.randn(100, 4)) == EXCLUDED
