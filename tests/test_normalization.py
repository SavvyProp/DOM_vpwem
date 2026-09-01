import torch

from dom_vpwem.normalization import NormalizationStats


def test_minmax_proprio_and_identity_action_round_trip() -> None:
    stats = NormalizationStats(
        proprio_min=(0.0,) * 7,
        proprio_max=(2.0,) * 7,
        action_min=(-0.5,) * 7,
        action_max=(0.5,) * 7,
        action_mode="identity",
    )
    proprio = torch.tensor([[0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0]])
    assert torch.allclose(
        stats.normalize_proprio(proprio),
        torch.tensor([[-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 0.0]]),
    )
    action = torch.linspace(-0.5, 0.5, 7)
    assert torch.equal(stats.unnormalize_action(stats.normalize_action(action)), action)


def test_constant_proprio_dimension_normalizes_to_zero() -> None:
    stats = NormalizationStats(
        proprio_min=(1.0,) * 7,
        proprio_max=(1.0,) * 7,
        action_min=(-1.0,) * 7,
        action_max=(1.0,) * 7,
    )
    assert torch.equal(stats.normalize_proprio(torch.ones(2, 7)), torch.zeros(2, 7))
