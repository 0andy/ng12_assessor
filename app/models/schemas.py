"""
Pydantic Schemas

Defines request and response models for all API endpoints:
- PatientData, AssessmentResult, Citation for assessment
- ChatRequest / ChatResponse for conversational Q&A
- RefreshResponse for admin operations
"""

from pydantic import BaseModel


class PatientData(BaseModel):
    """Patient demographic and symptom information."""

    patient_id: str
    name: str
    age: int
    gender: str
    smoking_history: str
    symptoms: list[str]
    symptom_duration_days: int


class MatchedRecommendation(BaseModel):
    """A single matched NG12 guideline recommendation."""

    section: str
    action_type: str
    criteria_met: str
    criteria_from_guideline: str = ""


class AssessmentResult(BaseModel):
    """Clinical risk assessment output."""

    risk_level: str
    cancer_type: str
    recommended_action: str
    reasoning: str
    matched_recommendations: list[MatchedRecommendation] = []


class Citation(BaseModel):
    """A reference back to a specific NG12 guideline chunk."""

    source: str
    section: str
    page: int
    chunk_id: str
    excerpt: str


class AssessResponse(BaseModel):
    """Full response for the /assess endpoint."""

    patient: PatientData
    assessment: AssessmentResult
    citations: list[Citation]


class ChatRequest(BaseModel):
    """Incoming message for the /chat endpoint."""

    session_id: str
    message: str


class ChatResponse(BaseModel):
    """Response from the /chat endpoint."""

    session_id: str
    answer: str
    citations: list[Citation]


class RefreshResponse(BaseModel):
    """Response from the /admin/refresh endpoint."""

    status: str
    chunks_indexed: int
    canonical_stored: int = 0
    sessions_cleared: bool
