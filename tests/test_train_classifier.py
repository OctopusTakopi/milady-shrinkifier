from __future__ import annotations

import pytest
import torch
from torch import nn

import milady.train_classifier as train_classifier
from milady.train_classifier import RegularizedBatch, compute_regularized_loss, create_cutmix_batch, create_mixup_batch


def test_create_mixup_batch_is_noop_when_alpha_is_disabled() -> None:
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    labels = torch.tensor([0, 1])
    weights = torch.tensor([1.0, 0.5])

    batch = create_mixup_batch(inputs, labels, weights, mixup_alpha=0.0)

    assert batch.active is False
    assert batch.method == "off"
    assert batch.lambda_value == pytest.approx(1.0)
    assert batch.effective_primary_ratio == pytest.approx(1.0)
    assert torch.equal(batch.inputs, inputs)
    assert torch.equal(batch.primary_labels, labels)
    assert torch.equal(batch.secondary_labels, labels)
    assert torch.equal(batch.primary_contributions, weights)
    assert torch.equal(batch.secondary_contributions, torch.zeros_like(weights))


def test_create_mixup_batch_weights_inputs_by_sample_contribution(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = torch.tensor(
        [
            [[1.0, 10.0]],
            [[2.0, 20.0]],
            [[3.0, 30.0]],
            [[4.0, 40.0]],
        ]
    )
    labels = torch.tensor([0, 1, 1, 0])
    weights = torch.tensor([1.0, 0.0, 0.5, 1.0])
    permutation = torch.tensor([1, 0, 3, 2])
    expected_lambda = 0.25

    monkeypatch.setattr(train_classifier.np.random, "beta", lambda *_args: expected_lambda)
    monkeypatch.setattr(train_classifier.torch, "randperm", lambda n, device=None: permutation.to(device=device))

    batch = create_mixup_batch(inputs, labels, weights, mixup_alpha=0.2)

    assert batch.active is True
    assert batch.method == "mixup"
    assert batch.lambda_value == pytest.approx(expected_lambda)
    assert torch.equal(batch.secondary_labels, labels[permutation])

    paired_weights = weights[permutation]
    expected_primary_contributions = expected_lambda * weights
    expected_secondary_contributions = (1.0 - expected_lambda) * paired_weights
    expected_total = expected_primary_contributions + expected_secondary_contributions
    expected_primary_ratio = torch.where(
        expected_total > 0,
        expected_primary_contributions / expected_total,
        torch.ones_like(expected_total),
    )
    expected_secondary_ratio = torch.where(
        expected_total > 0,
        expected_secondary_contributions / expected_total,
        torch.zeros_like(expected_total),
    )
    expected_inputs = (
        expected_primary_ratio.view(-1, 1, 1) * inputs
        + expected_secondary_ratio.view(-1, 1, 1) * inputs[permutation]
    )

    assert torch.allclose(batch.primary_contributions, expected_primary_contributions)
    assert torch.allclose(batch.secondary_contributions, expected_secondary_contributions)
    assert torch.allclose(batch.inputs, expected_inputs)
    assert torch.equal(batch.inputs[paired_weights == 0], inputs[paired_weights == 0])


def test_create_cutmix_batch_does_not_let_zero_weight_partner_change_pixels(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = torch.tensor(
        [
            [[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]],
            [[[9.0, 9.0, 9.0, 9.0], [9.0, 9.0, 9.0, 9.0], [9.0, 9.0, 9.0, 9.0], [9.0, 9.0, 9.0, 9.0]]],
        ]
    )
    labels = torch.tensor([0, 1])
    weights = torch.tensor([1.0, 0.0])

    monkeypatch.setattr(train_classifier.np.random, "beta", lambda *_args: 0.25)
    monkeypatch.setattr(train_classifier.torch, "randperm", lambda n, device=None: torch.tensor([0, 1], device=device))

    batch = create_cutmix_batch(inputs, labels, weights, cutmix_alpha=1.0)

    assert batch.method == "cutmix"
    assert torch.equal(batch.inputs, inputs)
    assert torch.equal(batch.primary_contributions, weights)
    assert torch.equal(batch.secondary_contributions, torch.zeros_like(weights))


def test_create_cutmix_batch_preserves_pairwise_sample_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = torch.tensor(
        [
            [[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]],
            [[[9.0, 9.0, 9.0, 9.0], [9.0, 9.0, 9.0, 9.0], [9.0, 9.0, 9.0, 9.0], [9.0, 9.0, 9.0, 9.0]]],
        ]
    )
    labels = torch.tensor([0, 1])
    weights = torch.tensor([0.25, 1.0])

    monkeypatch.setattr(train_classifier.np.random, "beta", lambda *_args: 0.25)
    monkeypatch.setattr(train_classifier.torch, "randperm", lambda n, device=None: torch.tensor([0, 1], device=device))
    monkeypatch.setattr(train_classifier.random, "randrange", lambda _limit: 2)

    batch = create_cutmix_batch(inputs, labels, weights, cutmix_alpha=1.0)

    expected_secondary_ratio = 9.0 / 16.0

    assert batch.method == "cutmix"
    assert torch.equal(batch.secondary_labels, labels[torch.tensor([1, 0])])
    assert batch.primary_contributions[0].item() == pytest.approx((1.0 - expected_secondary_ratio) * weights[0].item())
    assert batch.secondary_contributions[0].item() == pytest.approx(expected_secondary_ratio * weights[1].item())
    assert batch.primary_contributions[1].item() == pytest.approx((1.0 - expected_secondary_ratio) * weights[1].item())
    assert batch.secondary_contributions[1].item() == pytest.approx(expected_secondary_ratio * weights[0].item())
    assert batch.primary_contributions[0].item() + batch.secondary_contributions[1].item() == pytest.approx(weights[0].item())
    assert batch.primary_contributions[1].item() + batch.secondary_contributions[0].item() == pytest.approx(weights[1].item())


def test_create_cutmix_batch_covers_full_image_when_ratio_reaches_one(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = torch.tensor(
        [
            [[[1.0, 1.0], [1.0, 1.0]]],
            [[[9.0, 9.0], [9.0, 9.0]]],
        ]
    )
    labels = torch.tensor([0, 1])
    weights = torch.tensor([1.0, 1.0])

    monkeypatch.setattr(train_classifier.np.random, "beta", lambda *_args: 0.0)
    monkeypatch.setattr(train_classifier.torch, "randperm", lambda n, device=None: torch.tensor([0, 1], device=device))
    monkeypatch.setattr(train_classifier.random, "randrange", lambda _limit: 0)

    batch = create_cutmix_batch(inputs, labels, weights, cutmix_alpha=1.0)

    assert batch.method == "cutmix"
    assert batch.inputs[0].equal(inputs[1])
    assert batch.inputs[1].equal(inputs[0])
    assert batch.primary_contributions[0].item() == pytest.approx(0.0)
    assert batch.primary_contributions[1].item() == pytest.approx(0.0)
    assert batch.secondary_contributions[0].item() == pytest.approx(weights[1].item())
    assert batch.secondary_contributions[1].item() == pytest.approx(weights[0].item())


def test_compute_regularized_loss_matches_weighted_contributions() -> None:
    criterion = nn.CrossEntropyLoss(reduction="none")
    logits = torch.tensor(
        [
            [2.0, 0.5],
            [0.1, 1.7],
        ],
        dtype=torch.float32,
    )
    batch = RegularizedBatch(
        inputs=torch.empty(0),
        primary_labels=torch.tensor([0, 1]),
        secondary_labels=torch.tensor([1, 0]),
        primary_contributions=torch.tensor([0.3, 0.25]),
        secondary_contributions=torch.tensor([0.7, 2.0]),
        lambda_value=0.3,
        effective_primary_ratio=0.275,
        method="mixup",
        active=True,
    )

    primary_loss = criterion(logits, batch.primary_labels)
    secondary_loss = criterion(logits, batch.secondary_labels)
    expected = (
        (primary_loss * batch.primary_contributions)
        + (secondary_loss * batch.secondary_contributions)
    ).sum() / (
        batch.primary_contributions + batch.secondary_contributions
    ).sum()

    actual = compute_regularized_loss(criterion, logits, batch)

    assert actual.item() == pytest.approx(float(expected.item()))
