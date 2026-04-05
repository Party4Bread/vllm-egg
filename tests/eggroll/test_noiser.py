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
    get_full_perturbation,
    get_lora_perturbation,
    normalize_fitnesses,
)


class TestPerturbations:
    def test_lora_shape(self):
        frozen = FrozenNoiserParams(rank=4)
        param = torch.randn(32, 16)
        A, B = get_lora_perturbation(frozen, 0.1, 0, 0, param, 42)
        assert A.shape == (32, 4)
        assert B.shape == (16, 4)

    def test_full_perturbation_shape(self):
        frozen = FrozenNoiserParams()
        param = torch.randn(16)
        noise = get_full_perturbation(frozen, 0.1, 0, 0, param, 42)
        assert noise.shape == (16,)

    def test_antithetical_sampling(self):
        """Even and odd thread IDs should produce opposite perturbations."""
        frozen = FrozenNoiserParams(rank=2)
        param = torch.randn(8, 4)

        A_even, B_even = get_lora_perturbation(frozen, 1.0, 0, 0, param, 42)
        A_odd, B_odd = get_lora_perturbation(frozen, 1.0, 0, 1, param, 42)

        # A should be negated, B should be the same
        assert torch.allclose(A_even, -A_odd, atol=1e-6)
        assert torch.allclose(B_even, B_odd, atol=1e-6)

    def test_antithetical_full(self):
        frozen = FrozenNoiserParams()
        param = torch.randn(8)
        noise_even = get_full_perturbation(frozen, 1.0, 0, 0, param, 42)
        noise_odd = get_full_perturbation(frozen, 1.0, 0, 1, param, 42)
        assert torch.allclose(noise_even, -noise_odd, atol=1e-6)

    def test_different_epochs_different_noise(self):
        frozen = FrozenNoiserParams(rank=2)
        param = torch.randn(8, 4)
        A1, _ = get_lora_perturbation(frozen, 1.0, 0, 0, param, 42)
        A2, _ = get_lora_perturbation(frozen, 1.0, 1, 0, param, 42)
        assert not torch.allclose(A1, A2)

    def test_noise_reuse(self):
        frozen = FrozenNoiserParams(noise_reuse=2, rank=2)
        param = torch.randn(8, 4)
        A0, B0 = get_lora_perturbation(frozen, 1.0, 0, 0, param, 42)
        A1, B1 = get_lora_perturbation(frozen, 1.0, 1, 0, param, 42)
        # Epochs 0 and 1 should use same noise (reuse=2)
        assert torch.allclose(A0, A1)
        assert torch.allclose(B0, B1)


class TestFitnessNormalization:
    def test_zero_mean(self):
        scores = torch.tensor([1.0, 2.0, 3.0, 4.0])
        normalized = normalize_fitnesses(scores)
        assert abs(normalized.mean().item()) < 1e-5

    def test_unit_variance(self):
        scores = torch.randn(100)
        normalized = normalize_fitnesses(scores)
        assert abs(normalized.std().item() - 1.0) < 0.2

    def test_group_normalization(self):
        scores = torch.tensor([1.0, 2.0, 10.0, 11.0])
        normalized = normalize_fitnesses(scores, group_size=2)
        # Within groups: (1,2) and (10,11) should have mean subtracted
        # Group 1: 1-1.5=-0.5, 2-1.5=0.5; Group 2: 10-10.5=-0.5, 11-10.5=0.5
        # All divided by global std
        assert normalized[0].item() < 0
        assert normalized[1].item() > 0
        assert normalized[2].item() < 0
        assert normalized[3].item() > 0


class TestEggRoll:
    def _make_model(self):
        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
        return model

    def test_init(self):
        model = self._make_model()
        frozen, noised = EggRoll.init(model, sigma=0.1, lr=0.01, rank=2)
        assert frozen.rank == 2
        assert noised.sigma == 0.1
        assert noised.optimizer is not None

    def test_classify_param(self):
        assert EggRoll.classify_param("layer.weight", torch.randn(4, 4)) == MM_PARAM
        assert EggRoll.classify_param("layer.bias", torch.randn(4)) == PARAM
        result = EggRoll.classify_param("embed_tokens.weight", torch.randn(100, 4))
        assert result == EXCLUDED

    def test_param_seeds(self):
        model = self._make_model()
        seeds = EggRoll.get_param_seeds(model)
        assert len(seeds) == len(list(model.named_parameters()))
        # Seeds should be unique
        assert len(set(seeds.values())) == len(seeds)

    def test_perturb_model(self):
        model = self._make_model()
        frozen, noised = EggRoll.init(model, sigma=0.1, lr=0.01, rank=2)
        seeds = EggRoll.get_param_seeds(model)
        original = {n: p.data.clone() for n, p in model.named_parameters()}

        EggRoll.perturb_model(frozen, noised.sigma, model, 0, 0, seeds, original)

        # Some parameters should have changed
        changed = False
        for name, param in model.named_parameters():
            if not torch.allclose(param.data, original[name]):
                changed = True
                break
        assert changed

    def test_perturb_restore_cycle(self):
        model = self._make_model()
        frozen, noised = EggRoll.init(model, sigma=0.1, lr=0.01, rank=2)
        seeds = EggRoll.get_param_seeds(model)
        original = {n: p.data.clone() for n, p in model.named_parameters()}

        # Perturb
        EggRoll.perturb_model(frozen, noised.sigma, model, 0, 0, seeds, original)
        # Restore
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(original[name])

        # Should match original
        for name, param in model.named_parameters():
            assert torch.allclose(param.data, original[name])

    def test_gradient_computation(self):
        model = self._make_model()
        frozen, noised = EggRoll.init(model, sigma=0.1, lr=0.01, rank=2)
        seeds = EggRoll.get_param_seeds(model)

        pop_size = 8
        fitnesses = torch.randn(pop_size)
        epochs = torch.zeros(pop_size, dtype=torch.int32)
        thread_ids = torch.arange(pop_size, dtype=torch.int32)

        grads = EggRoll.compute_gradients(
            frozen,
            noised.sigma,
            model,
            seeds,
            fitnesses,
            epochs,
            thread_ids,
        )

        assert len(grads) == len(list(model.named_parameters()))
        for name, grad in grads.items():
            param = dict(model.named_parameters())[name]
            assert grad.shape == param.shape

    def test_full_update_cycle(self):
        model = self._make_model()
        frozen, noised = EggRoll.init(model, sigma=0.1, lr=0.01, rank=2)
        seeds = EggRoll.get_param_seeds(model)
        original = {n: p.data.clone() for n, p in model.named_parameters()}

        pop_size = 8
        fitnesses = torch.randn(pop_size)
        normalized = EggRoll.convert_fitnesses(frozen, fitnesses)
        epochs = torch.zeros(pop_size, dtype=torch.int32)
        thread_ids = torch.arange(pop_size, dtype=torch.int32)

        grads = EggRoll.compute_gradients(
            frozen,
            noised.sigma,
            model,
            seeds,
            normalized,
            epochs,
            thread_ids,
        )
        EggRoll.update_params(noised, model, grads)

        # Parameters should have changed
        changed = False
        for name, param in model.named_parameters():
            if not torch.allclose(param.data, original[name], atol=1e-8):
                changed = True
                break
        assert changed


class TestOpenES:
    def test_init(self):
        model = nn.Linear(4, 2)
        frozen, noised = OpenES.init(model, sigma=0.1, lr=0.01)
        assert frozen.rank == 1
        assert noised.sigma == 0.1

    def test_classify_all_full(self):
        # OpenES uses full perturbation for non-excluded params
        assert OpenES.classify_param("layer.weight", torch.randn(4, 4)) == PARAM
        assert OpenES.classify_param("layer.bias", torch.randn(4)) == PARAM
        assert OpenES.classify_param("embed.weight", torch.randn(100, 4)) == EXCLUDED
