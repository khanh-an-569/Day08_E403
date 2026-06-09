from concurrent.futures import ThreadPoolExecutor

try:
    from .task5_semantic_search import semantic_search
    from .task6_lexical_search import lexical_search
    from .task7_reranking import rerank, rerank_rrf
    from .task8_pageindex_vectorless import pageindex_search
except ImportError:
    from task5_semantic_search import semantic_search
    from task6_lexical_search import lexical_search
    from task7_reranking import rerank, rerank_rrf
    from task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Hybrid Retrieval Pipeline:
        1. Dense Retrieval
        2. BM25 Retrieval
        3. RRF Fusion
        4. Rerank
        5. PageIndex Fallback
    """

    try:
        # ==========================================================
        # STEP 1 - RUN DENSE + SPARSE IN PARALLEL
        # ==========================================================

        dense_results = []
        sparse_results = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(
                semantic_search,
                query,
                top_k * 2
            )

            sparse_future = executor.submit(
                lexical_search,
                query,
                top_k * 2
            )

            try:
                dense_results = dense_future.result()
            except Exception as e:
                print(f"Semantic search failed: {e}")

            try:
                sparse_results = sparse_future.result()
            except Exception as e:
                print(f"Lexical search failed: {e}")

        print(
            f"Dense results: {len(dense_results)} | "
            f"Sparse results: {len(sparse_results)}"
        )

        # ==========================================================
        # STEP 2 - RRF FUSION
        # ==========================================================

        merged_results = rerank_rrf(
            [dense_results, sparse_results],
            top_k=top_k * 2
        )

        for item in merged_results:
            item["source"] = "hybrid"

        print(
            f"Hybrid merged results: {len(merged_results)}"
        )

        # ==========================================================
        # STEP 3 - RERANK
        # ==========================================================

        if use_reranking and merged_results:

            try:
                final_results = rerank(
                    query=query,
                    candidates=merged_results,
                    top_k=top_k,
                    method=RERANK_METHOD
                )

            except Exception as e:

                print(
                    f"Rerank failed: {e}"
                )

                final_results = merged_results[:top_k]

        else:
            final_results = merged_results[:top_k]

        # ==========================================================
        # STEP 4 - THRESHOLD CHECK
        # ==========================================================

        best_score = 0.0

        if final_results:
            # Map content to its original dense/semantic score to perform threshold check
            # because final_results contain fused RRF scores (which are very small, < 0.05).
            dense_scores = {item["content"]: item["score"] for item in dense_results}
            # Find the maximum semantic score among the top returned results
            semantic_scores_in_final = [
                dense_scores.get(item["content"], 0.0) for item in final_results
            ]
            best_semantic_score = max(semantic_scores_in_final) if semantic_scores_in_final else 0.0
            
            # Use the actual score of the top candidate if it's not an RRF score,
            # or fallback to the best semantic score.
            top_candidate_score = float(final_results[0].get("score", 0.0))
            best_score = max(top_candidate_score, best_semantic_score)

        if (
            not final_results
            or best_score < score_threshold
        ):

            print(
                f"Hybrid score "
                f"({best_score:.3f}) "
                f"< threshold "
                f"({score_threshold:.3f})"
            )

            print(
                "Fallback → PageIndex"
            )

            fallback_results = pageindex_search(
                query=query,
                top_k=top_k
            )

            if fallback_results:

                for item in fallback_results:
                    item["source"] = "pageindex"

                return fallback_results

        # ==========================================================
        # STEP 5 - RETURN HYBRID RESULTS
        # ==========================================================

        return final_results[:top_k]

    except Exception as e:

        print(
            f"Retrieval pipeline error: {e}"
        )

        try:

            fallback_results = pageindex_search(
                query=query,
                top_k=top_k
            )

            for item in fallback_results:
                item["source"] = "pageindex"

            return fallback_results

        except Exception as fallback_error:

            print(
                f"Fallback failed: {fallback_error}"
            )

            return []