from collections import Counter
import re


class DocumentClassifier:
    """Automatically detect document type and adjust processing accordingly"""

    DOCUMENT_TYPES = {
        "legal": ["constitution", "law", "legal", "article", "section", "chapter", "act", "rule", "regulation"],
        "insurance": ["policy", "premium", "coverage", "claim", "benefit", "maternity", "waiting", "grace"],
        "technical": ["specification", "manual", "part", "component", "system", "procedure", "operation"],
        "academic": ["chapter", "theorem", "principle", "equation", "formula", "hypothesis", "research"],
        "medical": ["diagnosis", "treatment", "patient", "disease", "symptom", "medication", "therapy"],
        "financial": ["investment", "portfolio", "return", "risk", "asset", "liability", "balance"],
        "general": [],  # fallback
    }

    @classmethod
    def classify_document(cls, text_sample: str) -> str:
        """Classify document type based on content analysis"""
        text_lower = text_sample.lower()
        word_counts = Counter(re.findall(r"\b\w+\b", text_lower))

        type_scores = {}
        for doc_type, keywords in cls.DOCUMENT_TYPES.items():
            if doc_type == "general":
                continue
            score = sum(word_counts.get(keyword, 0) for keyword in keywords)
            type_scores[doc_type] = score

        if not type_scores or max(type_scores.values()) < 3:
            return "general"

        return max(type_scores, key=type_scores.get)
