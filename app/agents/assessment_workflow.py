"""
Part 1: Clinical Decision Support Workflow

LangGraph workflow: fetch_patient -> retrieve_guidelines -> assess_risk
Orchestrates patient lookup, RAG retrieval, and Gemini-based risk assessment.

Uses Gemini function calling for patient data retrieval when credentials
are available, with a direct fallback otherwise.
"""

import json
import logging
import re
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from app.core import patient_db, rag_pipeline
from app.core.gemini_client import gemini_client
from app.prompts.assessment import (
    ASSESSMENT_SYSTEM_PROMPT,
    format_assessment_prompt,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class AssessmentState(TypedDict):
    patient_id: str
    patient: Optional[dict]
    chunks: Optional[list]
    assessment: Optional[dict]
    citations: Optional[list]
    error: Optional[str]


# ---------------------------------------------------------------------------
# Node 1: fetch_patient
# ---------------------------------------------------------------------------

# Gemini function calling tool schema for patient lookup
_PATIENT_TOOL_SCHEMA = {
    "function_declarations": [
        {
            "name": "get_patient_data",
            "description": (
                "Retrieve patient record from the clinical database "
                "by patient ID"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "string",
                        "description": "The patient identifier, e.g. PT-101",
                    }
                },
                "required": ["patient_id"],
            },
        }
    ]
}


async def fetch_patient(state: AssessmentState) -> dict:
    """Look up the patient record.

    Attempts Gemini function calling first.  If Gemini is unavailable
    (e.g. missing credentials), falls back to a direct patient_db call.
    """
    patient_id = state["patient_id"]
    patient = None
    method = "direct"

    # --- Try Gemini function calling ---
    if gemini_client.is_available:
        try:
            from vertexai.generative_models import Tool

            tool = Tool.from_dict(_PATIENT_TOOL_SCHEMA)
            response = await gemini_client.generate_with_tools(
                prompt=f"Retrieve the patient data for patient {patient_id}",
                tools=[tool],
            )

            # Check if the model returned a function call
            candidate = response.candidates[0]
            part = candidate.content.parts[0]
            if hasattr(part, "function_call") and part.function_call:
                fn_call = part.function_call
                called_id = fn_call.args.get("patient_id", patient_id)
                patient = patient_db.get_patient(called_id)
                method = "gemini_function_calling"
        except Exception as exc:
            logger.warning(
                "Gemini function calling failed, using fallback: %s", exc
            )

    # --- Fallback: direct lookup ---
    if patient is None:
        patient = patient_db.get_patient(patient_id)
        if method != "gemini_function_calling":
            method = "direct_fallback"

    logger.info("fetch_patient: method=%s, found=%s", method, patient is not None)

    if patient is None:
        return {"error": f"Patient {patient_id} not found"}
    return {"patient": patient}


# ---------------------------------------------------------------------------
# Node 2: retrieve_guidelines
# ---------------------------------------------------------------------------

async def retrieve_guidelines(state: AssessmentState) -> dict:
    """Retrieve relevant NG12 guideline chunks for the patient."""
    patient = state["patient"]
    query = (
        f"{' '.join(patient['symptoms'])} "
        f"age {patient['age']} "
        f"{patient['gender']} "
        f"{patient['smoking_history']}"
    )
    chunks = rag_pipeline.retrieve(query, top_k=8, patient_data=patient)

    if not chunks:
        return {"error": "No relevant NG12 guideline passages found"}

    logger.info("retrieve_guidelines: %d chunks retrieved", len(chunks))
    return {"chunks": chunks}


# ---------------------------------------------------------------------------
# Node 3: assess_risk
# ---------------------------------------------------------------------------

def _clean_json_text(text: str) -> str:
    """Strip markdown code fences and leading/trailing whitespace."""
    text = text.strip()
    # Remove ```json ... ``` wrapper
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


_DEMO_ASSESSMENT = {
    "risk_level": "Demo Mode - Gemini not configured",
    "cancer_type": "N/A",
    "recommended_action": (
        "Configure GOOGLE_CLOUD_PROJECT in .env to enable Gemini"
    ),
    "reasoning": (
        "This is a demo response. Configure Vertex AI credentials "
        "to get real assessments."
    ),
    "matched_recommendations": [],
}


async def assess_risk(state: AssessmentState) -> dict:
    """Call Gemini to assess the patient against retrieved guidelines."""
    patient = state["patient"]
    chunks = state["chunks"]

    # Build citations from chunk metadata regardless of Gemini availability
    citations = []
    for i, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        canonical_meta = chunk.get("canonical_metadata", {})
        section = meta.get("section") or canonical_meta.get("section", "Part B")
        page = meta.get("page") or canonical_meta.get("page", 0)
        citations.append({
            "source": "NG12 PDF",
            "section": section,
            "page": page,
            "chunk_id": chunk.get("chunk_id", f"chunk_{i}"),
            "excerpt": chunk["text"][:200],
        })

    # --- Try Gemini ---
    if not gemini_client.is_available:
        logger.info("assess_risk: Gemini unavailable, returning demo result")
        return {"assessment": _DEMO_ASSESSMENT, "citations": citations}

    try:
        user_prompt = format_assessment_prompt(patient, chunks)
        raw = await gemini_client.generate(
            system_prompt=ASSESSMENT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
            max_output_tokens=2048,
        )

        if raw is None:
            return {"assessment": _DEMO_ASSESSMENT, "citations": citations}

        cleaned = _clean_json_text(raw)
        assessment = json.loads(cleaned)
        logger.info("assess_risk: Gemini assessment completed")
        return {"assessment": assessment, "citations": citations}

    except json.JSONDecodeError as exc:
        logger.error("JSON parse error from Gemini response: %s", exc)
        return {
            "error": f"Failed to parse Gemini response as JSON: {exc}",
            "citations": citations,
        }
    except Exception as exc:
        logger.error("Gemini call failed: %s", exc)
        return {
            "assessment": _DEMO_ASSESSMENT,
            "citations": citations,
        }


# ---------------------------------------------------------------------------
# Node 4: handle_error
# ---------------------------------------------------------------------------

async def handle_error(state: AssessmentState) -> dict:
    """Terminal node reached when a previous step sets an error."""
    logger.error("Assessment workflow error: %s", state.get("error"))
    return {"error": state.get("error", "Unknown error")}


# ---------------------------------------------------------------------------
# Conditional routing helpers
# ---------------------------------------------------------------------------

def _has_error(state: AssessmentState) -> str:
    """Route to handle_error if an error is present, otherwise continue."""
    if state.get("error"):
        return "handle_error"
    return "continue"


# ---------------------------------------------------------------------------
# Build and compile the graph
# ---------------------------------------------------------------------------

def build_assessment_graph() -> StateGraph:
    """Construct the LangGraph StateGraph for clinical assessment.

    Returns:
        A compiled LangGraph graph.
    """
    graph = StateGraph(AssessmentState)

    # Add nodes
    graph.add_node("fetch_patient", fetch_patient)
    graph.add_node("retrieve_guidelines", retrieve_guidelines)
    graph.add_node("assess_risk", assess_risk)
    graph.add_node("handle_error", handle_error)

    # Entry point
    graph.set_entry_point("fetch_patient")

    # Edges
    graph.add_conditional_edges(
        "fetch_patient",
        _has_error,
        {"handle_error": "handle_error", "continue": "retrieve_guidelines"},
    )
    graph.add_conditional_edges(
        "retrieve_guidelines",
        _has_error,
        {"handle_error": "handle_error", "continue": "assess_risk"},
    )
    graph.add_edge("assess_risk", END)
    graph.add_edge("handle_error", END)

    return graph.compile()


# Compiled workflow singleton
workflow = build_assessment_graph()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_assessment(patient_id: str) -> dict:
    """Run the full assessment workflow and return the final result.

    Args:
        patient_id: The patient identifier (e.g. "PT-101").

    Returns:
        Dict with keys: patient, assessment, citations (on success),
        or: error (on failure).
    """
    initial_state: AssessmentState = {
        "patient_id": patient_id,
        "patient": None,
        "chunks": None,
        "assessment": None,
        "citations": None,
        "error": None,
    }

    result = await workflow.ainvoke(initial_state)

    if result.get("error"):
        return {"error": result["error"]}

    return {
        "patient": result["patient"],
        "assessment": result["assessment"],
        "citations": result["citations"],
    }
