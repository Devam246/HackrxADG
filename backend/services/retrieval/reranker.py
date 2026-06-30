from functools import lru_cache
from typing import List, Optional
import structlog

from config import get_settings
from models.domain import Chunk

logger = structlog.get_logger(__name__)
settings = get_settings()


@lru_cache(maxsize=1)
def get_reranker_model(model_name: str):
    """
    Lazy load the BAAI/bge-reranker-v2-m3 cross-encoder model.
    Loads using sentence-transformers or FlagEmbedding.
    Attempts to use CUDA and FP16/float16 if GPU is available.
    """
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("loading_reranker_model", model_name=model_name, device=device)

        # 1. Attempt using sentence-transformers
        try:
            from sentence_transformers import CrossEncoder

            automodel_args = {}
            if device == "cuda":
                automodel_args["torch_dtype"] = torch.float16

            model = CrossEncoder(
                model_name,
                device=device,
                automodel_args=automodel_args,
            )
            logger.info("reranker_model_loaded_sentence_transformers", device=device)
            return model
        except ImportError:
            # 2. Fall back to FlagEmbedding's FlagReranker
            from FlagEmbedding import FlagReranker

            use_fp16 = torch.cuda.is_available()
            model = FlagReranker(model_name, use_fp16=use_fp16)
            logger.info("reranker_model_loaded_flagembedding", use_fp16=use_fp16)
            return model

    except Exception as e:
        logger.exception("reranker_model_load_failed", model_name=model_name, error=str(e))
        return None


class CrossEncoderReranker:
    """
    CrossEncoderReranker handles scoring search candidates against the original query.
    Falls back to original ordering if model loading or scoring fails.
    """

    def rerank(self, query: str, chunks: List[Chunk], top_n: int = 8) -> List[Chunk]:
        """
        Scores input chunks against the query using BAAI/bge-reranker-v2-m3.
        Keeps top_n scored chunks.
        """
        if not chunks:
            return []

        model = get_reranker_model(settings.reranker_model)
        if model is None:
            logger.warning("reranker_model_not_loaded_falling_back_to_rrf")
            for chunk in chunks:
                chunk.rerank_score = None
            return chunks[:top_n]

        try:
            pairs = [(query, chunk.text) for chunk in chunks]

            # Run predictions depending on loaded model type
            if hasattr(model, "predict"):
                scores = model.predict(pairs)
            else:
                scores = model.compute_score(pairs)

            # Assign float scores to chunk copies
            for chunk, score in zip(chunks, scores):
                chunk.rerank_score = float(score)

            # Sort descending by score. If a score is missing/None (should not happen), treat as very low.
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
