"""LLM-as-a-judge evaluation for the rag01 baseline.

Run from the project root:
    uv run python -m rag01.evaluation

Evaluates:
- Faithfulness: answer claims supported by retrieved context
- Answer correctness: answer matches the expected answer
- Citation correctness: citations are present, refer to retrieved pages,
  and support the claims attributed to them

Writes detailed results to evaluation_results.json.
"""

import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

from langchain_openai import ChatOpenAI

from rag01.generation import generate_answer
from rag01.retrieval import retrieve_documents


# Keep this configuration aligned with generation.py's local LM Studio server.
judge_llm = ChatOpenAI(
    model="local-model",
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    temperature=0,
    max_tokens=700,
    timeout=180,
)

TEST_CASES = [
    {
        "question": "Who are Mr. and Mrs. Dursley?",
        "expected_answer": (
            "They live at Number Four, Privet Drive, "
            "and are proud to be perfectly normal."
        ),
    },
    {
        "question": "What does Grunnings manufacture?",
        "expected_answer": "Grunnings manufactures drills.",
    },
    {
        "question": "Who is Dudley?",
        "expected_answer": "Dudley is the Dursleys' son.",
    },
    {
        "question": "Where do the Dursleys live?",
        "expected_answer": "Number Four, Privet Drive.",
    },
]


def build_context(documents) -> str:
    """Format retrieved chunks and preserve the metadata page convention."""
    parts = []
    for rank, doc in enumerate(documents, start=1):
        page = doc.metadata.get("page", "unknown")
        source = doc.metadata.get("source", "unknown")
        parts.append(
            f"[Retrieved chunk {rank} | Source: {source} | Page: {page}]\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(parts)


def parse_judge_json(raw_text: str) -> dict[str, Any]:
    """Parse JSON even if the local model wraps it in Markdown fences."""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Recover the first JSON object if the model adds text around it.
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Judge did not return valid JSON: {raw_text}")
        parsed = json.loads(match.group(0))

    required = {
        "faithfulness_score",
        "faithfulness_reason",
        "answer_correctness_score",
        "answer_correctness_reason",
        "citation_correctness_score",
        "citation_correctness_reason",
    }
    missing = required - parsed.keys()
    if missing:
        raise ValueError(f"Judge JSON is missing fields: {sorted(missing)}")

    # Clamp scores so malformed outputs don't break aggregation.
    score_fields = (
        "faithfulness_score",
        "answer_correctness_score",
        "citation_correctness_score",
    )
    for field in score_fields:
        parsed[field] = max(1, min(5, int(parsed[field])))

    return parsed


def judge_answer(
    question: str,
    expected_answer: str,
    answer: str,
    context: str,
) -> dict[str, Any]:
    """Ask the LLM judge to score the answer against context and reference."""
    prompt = f"""
You are a strict evaluator for a retrieval-augmented generation (RAG) system.
Judge the answer using only the supplied question, reference answer, retrieved
context, and generated answer. Do not reward confident wording by itself.

Return ONLY one valid JSON object with these exact keys:
{{
  "faithfulness_score": 1,
  "faithfulness_reason": "brief explanation",
  "answer_correctness_score": 1,
  "answer_correctness_reason": "brief explanation",
  "citation_correctness_score": 1,
  "citation_correctness_reason": "brief explanation"
}}

Use integer scores from 1 to 5:
1 = very poor / unsupported
2 = mostly poor
3 = partially correct or partially supported
4 = good, with minor issues
5 = fully correct / fully supported

Scoring rules:
- Faithfulness: score whether factual claims in the generated answer are
  entailed by the retrieved context. Penalize unsupported claims. Do not use
  outside knowledge to fill gaps.
- Answer correctness: compare the generated answer with the reference answer
  and question. Accept semantically equivalent answers; do not require exact
  wording. Penalize missing key information or answering a different question.
- Citation correctness: check whether citations such as [Page 0] are included
  when factual claims need support, whether each cited page appears in the
  retrieved context below, and whether that page's text supports the relevant
  claims. Penalize missing, invalid, or mismatched citations. If the answer has
  no factual claims, score based on the available evidence and citation need.
- The page identifiers in the context are the application's stored page
  metadata. They are currently zero-based; evaluate citations against these
  identifiers as written, not against a PDF viewer's one-based page numbering.
- Be strict and evidence-based. Do not assume that a claim is supported just
  because it is plausible.

Question:
{question}

Reference answer:
{expected_answer}

Retrieved context:
{context}

Generated answer:
{answer}
"""
    response = judge_llm.invoke(prompt)
    return parse_judge_json(str(response.content))


def evaluate_case(case: dict[str, str], k: int = 3) -> dict[str, Any]:
    question = case["question"]
    expected_answer = case["expected_answer"]

    # Retrieve context for evaluation evidence and diagnostics.
    documents = retrieve_documents(question, k=k)
    context = build_context(documents)

    # Existing generation function retrieves the same question again.
    answer = generate_answer(question)

    try:
        scores = judge_answer(
            question=question,
            expected_answer=expected_answer,
            answer=answer,
            context=context,
        )
        judge_error = None
    except Exception as exc:
        # Keep the run going if the judge fails for one case.
        scores = {
            "faithfulness_score": None,
            "faithfulness_reason": "Judge failed; review this case manually.",
            "answer_correctness_score": None,
            "answer_correctness_reason": "Judge failed; review this case manually.",
            "citation_correctness_score": None,
            "citation_correctness_reason": "Judge failed; review this case manually.",
        }
        judge_error = f"{type(exc).__name__}: {exc}"

    return {
        "question": question,
        "expected_answer": expected_answer,
        "generated_answer": answer,
        "retrieved_pages": [
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "unknown"),
                "content_preview": doc.page_content[:500],
            }
            for doc in documents
        ],
        "scores": scores,
        "judge_error": judge_error,
    }


def average_score(results: list[dict[str, Any]], field: str) -> float | None:
    values = [
        result["scores"].get(field)
        for result in results
        if isinstance(result["scores"].get(field), (int, float))
    ]
    return round(mean(values), 2) if values else None


def main() -> None:
    results: list[dict[str, Any]] = []

    print("=" * 72)
    print("LLM-AS-A-JUDGE: RAG GENERATION EVALUATION")
    print("=" * 72)

    for index, case in enumerate(TEST_CASES, start=1):
        print(f"\n[{index}/{len(TEST_CASES)}] {case['question']}")
        try:
            result = evaluate_case(case, k=3)
        except Exception as exc:
            result = {
                "question": case["question"],
                "expected_answer": case["expected_answer"],
                "generated_answer": None,
                "retrieved_pages": [],
                "scores": {},
                "judge_error": f"{type(exc).__name__}: {exc}",
            }

        results.append(result)
        print(f"Expected: {result['expected_answer']}")
        print(f"Generated: {result['generated_answer']}")

        scores = result["scores"]
        for label, field in (
            ("Faithfulness", "faithfulness_score"),
            ("Answer correctness", "answer_correctness_score"),
            ("Citation correctness", "citation_correctness_score"),
        ):
            value = scores.get(field)
            reason_key = field.replace("_score", "_reason")
            if value is None:
                print(f"{label}: FAILED")
            else:
                print(f"{label}: {value}/5 — {scores.get(reason_key, '')}")

        if result.get("judge_error"):
            print(f"Evaluation error: {result['judge_error']}")

    print("\n" + "=" * 72)
    print("AGGREGATE SCORES (mean score out of 5)")
    print("=" * 72)
    metrics = (
        ("Faithfulness", "faithfulness_score"),
        ("Answer correctness", "answer_correctness_score"),
        ("Citation correctness", "citation_correctness_score"),
    )
    for label, field in metrics:
        score = average_score(results, field)
        print(f"{label}: {score}/5" if score is not None else f"{label}: unavailable")

    output_path = Path("evaluation_results.json")
    output_path.write_text(
        json.dumps(
            {
                "evaluator": "LM Studio local LLM-as-a-judge",
                "k": 3,
                "metrics_scale": "1-5; higher is better",
                "results": results,
                "aggregate": {
                    field: average_score(results, field)
                    for _, field in metrics
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nDetailed results saved to: {output_path.resolve()}")
    print(
        "\nNote: LLM-judge scores are estimates, not ground truth. "
        "Periodically review a small sample manually."
    )


if __name__ == "__main__":
    main()
