import os
import cohere
from typing import List
from config import get_settings
from models.domain import Chunk
import structlog

logger = structlog.get_logger(__name__)
settings = get_settings()


class CrossEncoderReranker:
    """
    CrossEncoderReranker handles scoring search candidates against the original query
    using Cohere's Rerank API.
    Falls back to RRF-only ordering if the Cohere API key is missing or call fails.
    """

    def rerank(self, query: str, chunks: List[Chunk], top_n: int = 8) -> List[Chunk]:
        """
        Scores input chunks against the query using Cohere's Rerank API.
        Keeps top_n scored chunks.
        """
        if not chunks:
            return []

        api_key = settings.cohere_api_key or os.environ.get("COHERE_API_KEY", "")
        if not api_key:
            logger.warning("reranker_model_not_loaded_falling_back_to_rrf")
            for chunk in chunks:
                chunk.rerank_score = None
            return chunks[:top_n]

        try:
            co = cohere.ClientV2(api_key=api_key)
            documents = [chunk.text for chunk in chunks]

            response = co.rerank(
                model="rerank-v3.5",
                query=query,
                documents=documents,
                top_n=top_n
            )

            score_map = {res.index: res.relevance_score for res in response.results}

            for idx, chunk in enumerate(chunks):
                if idx in score_map:
                    chunk.rerank_score = float(score_map[idx])
                else:
                    chunk.rerank_score = -999999.0

            reranked = sorted(
                chunks,
                key=lambda x: x.rerank_score if x.rerank_score is not None else -999999.0,
                reverse=True,
            )

            logger.info(
                "reranking_success",
                input_count=len(chunks),
                output_count=min(len(reranked), top_n),
            )
            return reranked[:top_n]

        except Exception as e:
            logger.exception("reranking_inference_failed_falling_back_to_rrf", error=str(e))
            for chunk in chunks:
                chunk.rerank_score = None
            return chunks[:top_n]

