"""
Assessment Router

POST /assess/{patient_id} - Run the clinical decision support workflow
for a given patient and return risk assessment with citations.

GET /assess/patients - Return a summary list of all patients for the UI.
"""

from fastapi import APIRouter, HTTPException

from app.agents.assessment_workflow import run_assessment
from app.core import patient_db
from app.models.schemas import (
    AssessResponse,
    AssessmentResult,
    Citation,
    MatchedRecommendation,
    PatientData,
)

router = APIRouter()


@router.get("/patients")
async def list_patients() -> list[dict]:
    """Return a summary of all patients for the UI quick-select buttons.

    Each entry contains patient_id, name, and a short symptom summary.
    """
    patients = patient_db.list_patients()
    return [
        {
            "patient_id": p["patient_id"],
            "name": p["name"],
            "symptoms_summary": ", ".join(p.get("symptoms", [])[:3]),
        }
        for p in patients
    ]


@router.post("/{patient_id}", response_model=AssessResponse)
async def assess_patient(patient_id: str) -> AssessResponse:
    """Assess cancer risk for the specified patient.

    Runs the full LangGraph assessment workflow:
    fetch_patient -> retrieve_guidelines -> assess_risk

    Returns structured assessment results with citations.
    """
    result = await run_assessment(patient_id)

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    # Build PatientData from the raw patient dict
    p = result["patient"]
    patient_data = PatientData(
        patient_id=p["patient_id"],
        name=p["name"],
        age=p["age"],
        gender=p["gender"],
        smoking_history=p["smoking_history"],
        symptoms=p["symptoms"],
        symptom_duration_days=p["symptom_duration_days"],
    )

    # Build AssessmentResult from the workflow output
    a = result["assessment"]
    matched = [
        MatchedRecommendation(
            section=m.get("section", "N/A"),
            action_type=m.get("action_type", "N/A"),
            criteria_met=m.get("criteria_met", ""),
            criteria_from_guideline=m.get("criteria_from_guideline", ""),
        )
        for m in a.get("matched_recommendations", [])
    ]
    assessment = AssessmentResult(
        risk_level=a.get("risk_level", "Unknown"),
        cancer_type=a.get("cancer_type", "Unknown"),
        recommended_action=a.get("recommended_action", "N/A"),
        reasoning=a.get("reasoning", ""),
        matched_recommendations=matched,
    )

    # Build Citation list
    citations = [
        Citation(
            source=c.get("source", "NG12 PDF"),
            section=str(c.get("section", "N/A")),
            page=int(c.get("page", 0)),
            chunk_id=c.get("chunk_id", ""),
            excerpt=c.get("excerpt", ""),
        )
        for c in result.get("citations", [])
    ]

    return AssessResponse(
        patient=patient_data,
        assessment=assessment,
        citations=citations,
    )
