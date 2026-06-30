from services.retrieval.hybrid_search import reciprocal_rank_fusion


def test_rrf_prefers_docs_ranked_high_by_both():
    dense = [(0, 0.95), (1, 0.80), (2, 0.60)]
    sparse = [(2, 8.5), (0, 7.1), (3, 5.0)]
    fused = reciprocal_rank_fusion(dense, sparse)
    assert fused[0][0] == 0  # top in dense, 2nd in sparse -> should win
    assert fused[1][0] == 2  # top in sparse, 3rd in dense -> should be second


def test_rrf_handles_disjoint_sets():
    dense = [(0, 0.9), (1, 0.8)]
    sparse = [(5, 10.0), (6, 9.0)]
    fused = reciprocal_rank_fusion(dense, sparse)
    assert set(idx for idx, _ in fused) == {0, 1, 5, 6}
