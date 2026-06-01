"""Typed output heads (pooled embedding -> ModelOutput)."""
from model.heads.typed_matrix_head import TypedMatrixHead
from model.heads.node_head import NodeHead
from model.heads.pairwise_edge_head import PairwiseEdgeHead

__all__ = ["TypedMatrixHead", "NodeHead", "PairwiseEdgeHead"]
