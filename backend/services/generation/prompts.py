from typing import Dict, List, Tuple


def build_batch_prompt(queries: List[str], top_chunks: List[List[Dict]], snippet_len: int = 200) -> Tuple[str, str]:
    """
    Returns (system_message, user_prompt) where:
      - system_message: instructions + JSON schema
      - user_prompt: numbered list of queries + their top chunk snippets
    """
    system = (
        "You are a helpful insurance assistant. You will be asked multiple questions "
        "related to an insurance policy. Each question will be accompanied by relevant clauses "
        "from the policy document. For each question:\n\n"
        "- Read the question carefully.\n"
        "- Use the supporting clauses provided to generate a clear, correct, and concise answer.\n"
        "- Do not make up facts not present in the clauses.\n"
        "- Your final output must ONLY be a JSON object with an 'answers' key, "
        "which contains a list of plain English answers (one for each question) in order.\n\n"
        "Example output:\n"
        "{\n"
        '  "answers": [\n'
        '    "Yes, the policy covers cataract surgery after a waiting period of 2 years.",\n'
        '    "A grace period of 30 days is provided for premium payment.",\n'
        '    "AYUSH treatment is covered up to the sum insured when taken in AYUSH hospitals."\n'
        "  ]\n"
        "}\n"
        "Do not wrap your output in markdown, triple backticks, or any code block. Return only raw JSON.\n"
    )

    lines = []
    for i, q in enumerate(queries, start=1):
        lines.append(f"{i}) Question: {q}\nClauses:")
        for c in top_chunks[i - 1]:
            snippet = c["text"].replace("\n", " ")[:snippet_len]
            # Uses clause_id or defaults to chunk_index / section_id
            clause_id = c.get("clause_id", c.get("chunk_index", "N/A"))
            lines.append(f"- [Clause {clause_id}] {snippet}")
        lines.append("---")
    user = "\n".join(lines) + "\nRespond with JSON:"

    return system, user


def build_universal_prompt(
    queries: List[str],
    top_chunks: List[List[Dict]],
    confidence_scores: List[float],
    doc_type: str,
    snippet_len: int = 600,
) -> Tuple[str, str]:
    """Build prompts that work optimally for any document type"""
    type_instructions = {
        "legal": (
            "You are analyzing legal documents. Provide precise answers with exact article/section references. "
            "Include specific legal provisions, rights, duties, and procedures. "
            "When citing provisions, use exact numbering (e.g., 'Article 19(1)(a)')."
        ),
        "insurance": (
            "You are analyzing insurance policy documents. Provide detailed answers with specific terms, "
            "conditions, waiting periods, coverage limits, and exclusions. "
            "Include exact timeframes (days/months/years) and monetary amounts where applicable."
        ),
        "technical": (
            "You are analyzing technical documentation. Provide precise answers with specific part numbers, "
            "procedures, specifications, and safety requirements. "
            "Include step-by-step processes and exact technical parameters."
        ),
        "academic": (
            "You are analyzing academic/scientific content. Provide comprehensive answers with definitions, "
            "principles, formulas, and supporting evidence. "
            "Include mathematical expressions and cite specific theorems or principles."
        ),
        "general": (
            "You are analyzing document content. Provide accurate, detailed answers based on the provided text. "
            "Include specific facts, numbers, and relevant details from the source material."
        ),
    }

    type_instruction = type_instructions.get(doc_type, type_instructions["general"])

    system = f"""You are an expert document analyst specializing in {doc_type} documents.

{type_instruction}

CRITICAL INSTRUCTIONS:
- You MUST respond with ONLY a valid JSON object
- NO additional text, explanations, or markdown formatting
- NO backticks, code blocks, or other formatting
- The JSON must have exactly this structure: {{"answers": ["answer1", "answer2", ...]}}
- Provide COMPLETE, DETAILED answers with SPECIFIC information
- Include exact numbers, dates, references, and conditions
- Use information ONLY from the provided document sections
- Along with Specific Information, if clause contain some necessary sounding details along.. add them as well.
- Also State Yes and No wherever recquired.
- For Questions with low confidence :- Provide Best Possible Answer

Example of correct response format:
{{"answers": ["The grace period is 30 days from the due date.", "Pre-existing diseases have a waiting period of 36 months."]}}"""

    lines = []
    for i, (query, chunks, conf) in enumerate(zip(queries, top_chunks, confidence_scores), 1):
        confidence_level = "HIGH" if conf > 0.7 else "MEDIUM" if conf > 0.5 else "LOW"

        lines.append(f"\n{i}) QUERY: {query}")
        lines.append(f"CONFIDENCE: {confidence_level}")
        lines.append("RELEVANT SECTIONS:")

        for j, chunk in enumerate(chunks, 1):
            section_info = f"Section {chunk.get('section_id', 'N/A')}: {chunk.get('section_title', 'N/A')}"
            snippet = chunk["text"][:snippet_len].replace("\n", " ")

            lines.append(f"\n[{j}] {section_info}")
            lines.append(f"Content: {snippet}")

        lines.append("\n" + "=" * 50)

    user = "\n".join(lines) + "\n\nRespond with valid JSON only:"

    return system, user
