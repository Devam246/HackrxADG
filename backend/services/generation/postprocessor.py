import json
import re
from typing import Dict, List
import structlog

logger = structlog.get_logger(__name__)


def parse_json_response(content: str) -> List[str]:
    """
    Robust JSON parsing with multiple fallback strategies
    """
    try:
        return json.loads(content)["answers"]
    except Exception:
        pass

    try:
        cleaned = re.sub(r"^```json\s*", "", content, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
        cleaned = cleaned.strip()
        return json.loads(cleaned)["answers"]
    except Exception:
        pass

    try:
        json_match = re.search(r'\{.*?"answers"\s*:\s*\[.*?\].*?\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)["answers"]
    except Exception:
        pass

    try:
        answers_match = re.search(r'"answers"\s*:\s*\[(.*?)\]', content, re.DOTALL)
        if answers_match:
            answers_str = f'{{"answers": [{answers_match.group(1)}]}}'
            return json.loads(answers_str)["answers"]
    except Exception:
        pass

    return None


def extract_answers_fallback(content: str) -> List[str]:
    """
    Manual extraction when JSON parsing completely fails
    """
    answers = []

    quoted_text = re.findall(r'"([^"]{20,500})"', content)
    for text in quoted_text:
        if any(
            keyword in text.lower()
            for keyword in [
                "period",
                "coverage",
                "policy",
                "days",
                "months",
                "years",
                "benefit",
                "claim",
            ]
        ):
            answers.append(text)

    if not answers:
        lines = content.split("\n")
        for line in lines:
            if re.match(r"^\d+[.)]\s*", line.strip()):
                answer = re.sub(r"^\d+[.)]\s*", "", line.strip())
                if len(answer) > 10:
                    answers.append(answer)

    if not answers:
        sentences = re.split(r"[.!?]+", content)
        for sentence in sentences:
            if len(sentence.strip()) > 20 and any(
                keyword in sentence.lower()
                for keyword in [
                    "period",
                    "coverage",
                    "policy",
                    "waiting",
                    "claim",
                    "benefit",
                ]
            ):
                answers.append(sentence.strip())

    answers = answers[:10]
    answers = [ans[:500] for ans in answers]

    return answers if answers else ["Unable to extract answer from response"] * 5


def calculate_universal_confidence(query: str, retrieved_chunks: List[Dict], doc_type: str) -> float:
    """Calculate normalized confidence score for any document type"""
    confidence_signals = []

    query_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
    all_chunk_words = set()
    for chunk in retrieved_chunks:
        chunk_words = set(re.findall(r"\b\w{3,}\b", chunk["text"].lower()))
        all_chunk_words.update(chunk_words)

    if query_words:
        overlap_score = len(query_words.intersection(all_chunk_words)) / len(query_words)
        confidence_signals.append(min(overlap_score, 1.0))
    else:
        confidence_signals.append(0.0)

    specificity_score = 0.0
    for chunk in retrieved_chunks:
        text = chunk["text"]
        specifics = len(re.findall(r"\b\d+\b|\b[A-Z]{2,}\d+\b|\b\d{4}-\d{2}-\d{2}\b", text))
        specificity_score += min(specifics / 100.0, 0.3)

    confidence_signals.append(min(specificity_score, 1.0))

    title_relevance = 0.0
    query_words_title = set(query.lower().split())
    for chunk in retrieved_chunks:
        title_words = set(chunk.get("section_title", "").lower().split())
        if query_words_title:
            overlap = len(title_words.intersection(query_words_title)) / len(query_words_title)
            title_relevance = max(title_relevance, overlap)

    confidence_signals.append(min(title_relevance, 1.0))

    type_confidence = calculate_type_specific_confidence(query, retrieved_chunks, doc_type)
    confidence_signals.append(min(type_confidence, 1.0))

    consistency_score = calculate_chunk_consistency(retrieved_chunks)
    confidence_signals.append(min(consistency_score, 1.0))

    avg_confidence = sum(confidence_signals) / len(confidence_signals)
    return min(max(avg_confidence, 0.0), 1.0)


def calculate_type_specific_confidence(query: str, chunks: List[Dict], doc_type: str) -> float:
    type_indicators = {
        "legal": ["article", "section", "constitution", "law", "right", "duty"],
        "insurance": ["policy", "premium", "coverage", "claim", "benefit"],
        "technical": ["part", "component", "procedure", "specification"],
        "academic": ["theorem", "principle", "equation", "research", "study"],
    }

    if doc_type not in type_indicators:
        return 0.5

    indicators = type_indicators[doc_type]
    query_indicators = sum(1 for ind in indicators if ind in query.lower())

    chunk_indicators = 0
    for chunk in chunks:
        chunk_indicators += sum(1 for ind in indicators if ind in chunk["text"].lower())

    if query_indicators > 0:
        return min(chunk_indicators / (query_indicators * len(chunks)), 1.0)

    return 0.5


def calculate_chunk_consistency(chunks: List[Dict]) -> float:
    if len(chunks) < 2:
        return 1.0

    sections = [chunk.get("section_id", "") for chunk in chunks]
    section_prefixes = [s.split(".")[0] if "." in s else s for s in sections]

    unique_prefixes = len(set(section_prefixes))
    consistency = 1.0 - (unique_prefixes - 1) / len(chunks)

    return max(consistency, 0.2)
