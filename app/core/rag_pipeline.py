"""
Shared RAG Pipeline

Used by both Part 1 (assessment) and Part 2 (chat).
Provides retrieve(query, top_k, patient_data) function.

When patient_data is provided (Part 1), applies deterministic score boosts
based on age, symptoms, smoking history, and gender match.
"""

import json
import re
from typing import Any

from app.core import vector_store


def retrieve(
    query: str,
    top_k: int = 5,
    patient_data: dict | None = None,
) -> list[dict[str, Any]]:
    """Retrieve the most relevant NG12 guideline chunks for a query.

    When patient_data is provided (Part 1 assessment), applies deterministic
    weighting to re-rank results based on patient characteristics:
      - age match:    +0.15 if patient age >= chunk age_min
      - age match:    +0.15 if patient age < chunk age_max (under condition)
      - symptoms:     +0.10 per overlapping symptom keyword
      - smoking:      +0.10 if patient smokes and chunk mentions smoking
      - gender match: +0.05 if gender matches chunk gender_specific
      - gender clash: -0.30 if gender mismatches (e.g. male + breast cancer)

    When patient_data is None (Part 2 chat), applies lightweight query-aware
    reranking (urgency, age, duration, exact-wording boosts) on a 3x
    candidate pool before returning the top results.

    Args:
        query: The search query string.
        top_k: Number of top results to return.
        patient_data: Optional patient dict with keys:
            age, symptoms, smoking_history, gender.

    Returns:
        List of result dicts with keys: chunk_id, text, metadata, score.
    """
    # Fetch 3x candidates for both modes so re-ranking has room to work
    fetch_k = top_k * 3
    results = vector_store.query(query, top_k=fetch_k)

    if not patient_data:
        results = _chat_rerank(query, results)
        top_results = results[:top_k]
        _attach_canonicals(top_results)
        return top_results

    # Apply deterministic score adjustments
    patient_age = patient_data.get("age", 0)
    patient_symptoms = [s.lower() for s in patient_data.get("symptoms", [])]
    patient_smoking = patient_data.get("smoking_history", "Never Smoked")
    patient_gender = patient_data.get("gender", "")

    for result in results:
        meta = result["metadata"]
        boost = 0.0

        # Age match: patient meets age_min threshold
        age_min = meta.get("age_min")
        if age_min is not None and patient_age >= age_min:
            boost += 0.15

        # Age match: patient meets "under X" threshold
        age_max = meta.get("age_max")
        if age_max is not None and patient_age < age_max:
            boost += 0.15

        # Symptom overlap
        chunk_symptoms = meta.get("symptom_keywords", [])
        if isinstance(chunk_symptoms, str):
            try:
                chunk_symptoms = json.loads(chunk_symptoms)
            except (json.JSONDecodeError, TypeError):
                chunk_symptoms = []

        overlap_count = 0
        for ps in patient_symptoms:
            for cs in chunk_symptoms:
                if cs in ps or ps in cs:
                    overlap_count += 1
                    break
        boost += 0.1 * overlap_count

        # Smoking risk factor
        if (
            patient_smoking != "Never Smoked"
            and meta.get("risk_factor_smoking")
        ):
            boost += 0.1

        # Gender match / clash
        gender_specific = meta.get("gender_specific")
        if gender_specific:
            if (
                (gender_specific == "Female" and patient_gender == "Female")
                or (gender_specific == "Male" and patient_gender == "Male")
            ):
                boost += 0.05
            elif (
                (gender_specific == "Female" and patient_gender == "Male")
                or (gender_specific == "Male" and patient_gender == "Female")
            ):
                boost -= 0.3

        result["score"] += boost

    # Re-sort by adjusted score and take top_k
    results.sort(key=lambda r: r["score"], reverse=True)
    results = results[:top_k]

    # Enrich results with canonical text from the canonical collection
    _attach_canonicals(results)

    return results


# Precompiled patterns for chat-mode query-aware reranking
_URGENCY_RE = re.compile(r"urgent|red\s*flag|emergency|immediate", re.IGNORECASE)
_AGE_RE = re.compile(r"age|under\s+\d|over\s+\d|years?\s*old|\byo\b|\byrs?\b", re.IGNORECASE)
_DURATION_RE = re.compile(r"weeks?|months?|persistent|duration|lasting", re.IGNORECASE)
_EXACT_RE = re.compile(r"quote|exact|wording|verbatim", re.IGNORECASE)

# Urgency values that warrant a boost
_URGENCY_VALUES = {"immediate", "very_urgent", "urgent"}


def _chat_rerank(
    query: str,
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply lightweight query-aware boosts for chat-mode retrieval.

    Adjusts scores based on query intent signals (urgency, age, duration,
    exact-wording) and re-sorts.  Does not mutate the original score
    permanently - boosts are additive for ranking purposes only.
    """
    q_urgency = bool(_URGENCY_RE.search(query))
    q_age = bool(_AGE_RE.search(query))
    q_duration = bool(_DURATION_RE.search(query))
    q_exact = bool(_EXACT_RE.search(query))

    for result in results:
        meta = result.get("metadata", {})
        boost = 0.0

        # Urgency boost
        if q_urgency:
            urgency = meta.get("urgency", "").lower()
            if urgency in _URGENCY_VALUES:
                boost += 0.1

        # Age boost - chunk has age thresholds
        if q_age:
            if meta.get("age_min") is not None or meta.get("age_max") is not None:
                boost += 0.1

        # Duration boost - chunk mentions duration-related terms
        if q_duration:
            chunk_text = result.get("text", "").lower()
            if _DURATION_RE.search(chunk_text):
                boost += 0.1

        # Exact wording boost - canonical text is available
        if q_exact and meta.get("doc_type") == "rule_search":
            boost += 0.15

        result["score"] += boost

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def _attach_canonicals(results: list[dict[str, Any]]) -> None:
    """Attach canonical original text to retrieval results.

    For rule_search docs: attaches the single canonical entry as
    ``canonical_text`` and ``canonical_metadata``.

    For symptom_index docs: collects all referenced canonical entries
    into ``referenced_canonicals``.

    Silently skips any rule_id that cannot be found in the canonical
    collection.
    """
    for result in results:
        meta = result.get("metadata", {})
        doc_type = meta.get("doc_type")

        if doc_type == "rule_search":
            rule_id = meta.get("rule_id")
            if not rule_id:
                continue
            canonical = vector_store.get_canonical(rule_id)
            if canonical:
                result["canonical_text"] = canonical["text"]
                result["canonical_metadata"] = canonical["metadata"]

        elif doc_type == "symptom_index":
            refs_json = meta.get("references_json", "[]")
            if isinstance(refs_json, str):
                try:
                    refs = json.loads(refs_json)
                except (json.JSONDecodeError, TypeError):
                    refs = []
            else:
                refs = refs_json

            referenced = []
            for ref in refs:
                # Strip brackets: "[1.5.2]" -> "1.5.2"
                rule_id = ref.strip("[]")
                if not rule_id:
                    continue
                canonical = vector_store.get_canonical(rule_id)
                if canonical:
                    referenced.append({
                        "rule_id": rule_id,
                        "text": canonical["text"],
                        "metadata": canonical["metadata"],
                    })
            if referenced:
                result["referenced_canonicals"] = referenced
