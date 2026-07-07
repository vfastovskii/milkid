"""Aggregator components for the Multi-Instance Learning Kit (MILK)."""

# Expose key modules at the package level
from ppl.models.components.aggregators.base_aggregator import AggregatorBase
from ppl.models.components.aggregators.multihead_attention_aggregator_v4 import MultiHeadAttentionAggregatorV4
from ppl.models.components.aggregators.vit_aggregator import VITAggregator
from ppl.models.components.aggregators.mha_v5 import MultiHeadAttentionAggregatorV5
from ppl.models.components.aggregators.cluster_hierarchical_attention import ClusterHierarchicalAttentionAggregator
