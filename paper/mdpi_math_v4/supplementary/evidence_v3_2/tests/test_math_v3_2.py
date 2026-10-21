"""Boundary regressions and analytic-envelope consequences for v3.2."""
import pytest
import torch

from sokkanaem.certified_scalar_cache import adaptive_nodes, interpolate, interval_envelope, rows
from sokkanaem.range_cache_filter import bilateral_two_read


def test_constant_image_with_legacy_denominator_floor_failure():
    value = torch.ones(1, 1, 8, 8, dtype=torch.float64)
    guide = torch.full_like(value, 0.5)
    actual = bilateral_two_read(value, guide, 2, 0.05, 2)
    torch.testing.assert_close(actual, value, atol=1e-14, rtol=0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_local_underflow_preserves_constants_and_offsets(dtype):
    guide = torch.zeros(1, 1, 8, 16, dtype=dtype)
    guide[..., 8:] = 1
    values = torch.ones_like(guide)
    result = bilateral_two_read(values, guide, 1, 0.01, 2)
    torch.testing.assert_close(result, values, atol=2e-6 if dtype == torch.float32 else 1e-14, rtol=0)
    shifted = bilateral_two_read(values + 7, guide, 1, 0.01, 2)
    torch.testing.assert_close(shifted, result + 7, atol=2e-6 if dtype == torch.float32 else 1e-14, rtol=0)


def test_filter_matches_independent_logsumexp_snapshot_oracle():
    torch.manual_seed(32)
    guide = torch.rand(1, 1, 7, 9, dtype=torch.float64)
    values = torch.randn_like(guide)
    actual = bilateral_two_read(values, guide, 0.8, 0.04, 4)
    # One interior pixel: build the spatially weighted normalized snapshots
    # directly, without the implementation's blur or underflow detection.
    yy, xx = 3, 4
    offset = torch.arange(-2, 3, dtype=torch.float64)
    spatial = -(offset[:, None].square() + offset[None, :].square()) / (2 * 0.8 ** 2)
    keys = guide[0, 0, yy-2:yy+3, xx-2:xx+3]
    vals = values[0, 0, yy-2:yy+3, xx-2:xx+3]
    levels = torch.linspace(0, 1, 4, dtype=torch.float64)
    logits = spatial[None] - (levels[:, None, None] - keys).square() / (2 * 0.04 ** 2)
    reference = (logits.flatten(1).softmax(-1) * vals.flatten()).sum(-1)
    wanted = interpolate(guide[0, 0, yy, xx].reshape(1), levels, reference)[0]
    torch.testing.assert_close(actual[0, 0, yy, xx], wanted, atol=1e-12, rtol=1e-12)


def test_adaptive_certificate_bounds_all_values_at_dense_queries():
    keys = torch.tensor([0, 0.2, 0.5, 1.0], dtype=torch.float64) * 40
    biases = torch.tensor([1, -3, 2, -15], dtype=torch.float64)
    nodes, bounds = adaptive_nodes(keys, biases, 0, 1, 1e-3)
    query = torch.linspace(0, 1, 4097, dtype=torch.float64)
    error = (rows(query, keys, biases) - interpolate(query, nodes, rows(nodes, keys, biases))).abs().sum(-1)
    cell = (torch.searchsorted(nodes, query, right=True) - 1).clamp(0, len(nodes)-2)
    assert bool((error <= bounds[cell] + 1e-13).all())
    assert float(bounds.max()) <= 1e-3


def test_constant_keys_need_only_two_nodes_and_have_zero_error():
    keys = torch.ones(5, dtype=torch.float64)
    biases = torch.arange(5, dtype=torch.float64)
    nodes, bounds = adaptive_nodes(keys, biases, -2, 3, 1e-8)
    assert len(nodes) == 2 and float(bounds.max()) == 0


def test_log_domain_envelope_keeps_tiny_midpoint_masses():
    keys = torch.tensor([0, 2000.0], dtype=torch.float64)
    biases = torch.tensor([0, -1800.0], dtype=torch.float64)
    left, right = torch.tensor([0.0]), torch.tensor([1.0])
    bound, variance = interval_envelope(keys, biases, left, right)
    # The midpoint probability of the second key underflows in a naive
    # variance calculation, but the interval contains its mass transition.
    assert float(variance) > 0 and float(bound) > 0


def test_invalid_duplicate_levels_are_rejected():
    value = torch.ones(1, 1, 4, 4)
    with pytest.raises(ValueError, match='strictly increasing'):
        bilateral_two_read(value, value / 2, 1, 0.2, 3, [0, 0, 1])
