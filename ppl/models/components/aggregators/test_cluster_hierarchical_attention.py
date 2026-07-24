import pytest
import torch

from ppl.models.components.aggregators.cluster_hierarchical_attention import (
    ClusterHierarchicalAttentionAggregator,
)

DIM, BIG, SMALL = 32, 100, 2


def _agg(**overrides):
    kwargs = dict(
        num_heads=4,
        num_query_tokens=1,
        dropout=0.0,
        attn_dropout=0.0,
        cluster_refinement_mode="soft",
        refine_mix=1.0,
        temperature_init=0.15,
    )
    kwargs.update(overrides)
    return ClusterHierarchicalAttentionAggregator(DIM, **kwargs).eval()


def _two_clusters():
    ids = torch.cat([torch.zeros(BIG), torch.ones(SMALL)]).long().unsqueeze(0)
    return ids, torch.ones(1, BIG + SMALL, dtype=torch.bool)


def test_uniform_term_ranks_by_population_share_not_attention():
    """The beta_c/|c| term inverts cluster ordering -- the reason refine_mix is pinned to 1."""
    agg = _agg()
    ids, valid = _two_clusters()
    counts = torch.tensor([[float(BIG), float(SMALL)]])

    # cluster 0 is far MORE attended, but has 50x the members
    alpha = agg._cluster_uniform_alpha(
        torch.tensor([[0.9, 0.1]]), counts, ids, valid
    )[0]
    assert alpha[-1] > alpha[0], "expected the size-diluted inversion"

    # ordering flips exactly at attention share == population share
    share = SMALL / (BIG + SMALL)
    for eps, expect_small_wins in ((+1e-3, True), (-1e-3, False)):
        b = share + eps
        a = agg._cluster_uniform_alpha(
            torch.tensor([[1.0 - b, b]]), counts, ids, valid
        )[0]
        assert bool(a[-1] > a[0]) is expect_small_wins


def test_identical_instances_get_uniform_attention_regardless_of_cluster_size():
    """Option 1 invariant: with p_i and beta_c symmetric, cluster size must not tilt alpha."""
    ids, _ = _two_clusters()
    n = BIG + SMALL
    h = torch.randn(1, 1, DIM).expand(1, n, DIM).contiguous()

    with torch.no_grad():
        _, extras = _agg(refine_mix=1.0)(h, cluster_ids=ids)
    alpha = extras["alpha"][0]

    assert alpha.sum() == pytest.approx(1.0, abs=1e-5)
    assert torch.allclose(alpha, torch.full_like(alpha, 1.0 / n), atol=1e-5)

    # the mixture regime fails this: the small cluster's members are inflated ~10x
    with torch.no_grad():
        _, mixed = _agg(refine_mix=0.7)(h, cluster_ids=ids)
    m = mixed["alpha"][0]
    assert m[-1] / m[0] > 5.0


def test_refine_mix_one_uses_only_the_globally_normalized_alpha():
    ids, _ = _two_clusters()
    h = torch.randn(1, BIG + SMALL, DIM)
    with torch.no_grad():
        _, extras = _agg(refine_mix=1.0)(h, cluster_ids=ids)
    assert torch.allclose(extras["alpha"], extras["alpha_instance_refined"], atol=1e-6)


def test_cluster_bias_power_zero_drops_the_cluster_prior():
    ids, _ = _two_clusters()
    h = torch.randn(1, BIG + SMALL, DIM)
    with torch.no_grad():
        _, extras = _agg(refine_mix=1.0, cluster_bias_power=0.0)(h, cluster_ids=ids)
    assert torch.allclose(extras["alpha"], extras["alpha_instance_raw"], atol=1e-6)


def test_cluster_bias_power_rejects_negative():
    with pytest.raises(ValueError, match="cluster_bias_power"):
        _agg(cluster_bias_power=-0.5)
