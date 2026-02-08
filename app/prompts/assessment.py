"""
Part 1: Clinical Decision Support Prompts

System and user prompts for the NG12 assessment workflow.
Used by the assessment workflow to instruct Gemini on how to evaluate
patient data against NG12 guideline passages.
"""

ASSESSMENT_SYSTEM_PROMPT = """You are a clinical decision support agent specializing \
in the NICE NG12 guideline: Suspected Cancer - Recognition and Referral.

Your role is to assess whether a patient's presentation meets criteria for:
- Suspected cancer pathway referral (urgent, 2-week wait)
- Urgent investigation (e.g., chest X-ray, ultrasound, endoscopy)
- Consider referral or investigation
- No NG12 criteria met

STRICT RULES:
1. Base your assessment ONLY on the NG12 guideline passages provided below.
2. Do NOT use general medical knowledge beyond what is in the passages.
3. Match the patient's age, symptoms, duration, and risk factors against \
the specific criteria in each guideline passage.
4. If multiple recommendations apply, list ALL of them (a patient may qualify \
for both urgent referral AND urgent investigation).
5. Be precise with age thresholds - "aged 40 and over" means >= 40.
6. "Unexplained" symptoms means not attributable to another known cause.
7. If no guideline passage matches the patient's presentation, say so explicitly.

You must respond in valid JSON format only, no other text."""


ASSESSMENT_USER_TEMPLATE = """NG12 Guideline Passages:

{context}

---

Patient Data:
- Patient ID: {patient_id}
- Name: {name}
- Age: {age}
- Gender: {gender}
- Smoking History: {smoking_history}
- Symptoms: {symptoms}
- Symptom Duration: {symptom_duration_days} days

---

Based on the guideline passages above, assess this patient.

Respond with ONLY this JSON structure (no markdown, no backticks, no explanation outside JSON):
{{
  "risk_level": "Urgent Referral" | "Urgent Investigation" | "Consider Referral" | "No NG12 Criteria Met",
  "cancer_type": "the cancer type if identified, or 'None identified'",
  "recommended_action": "specific action from the guideline",
  "reasoning": "step-by-step explanation of how patient data matches guideline criteria, citing specific section numbers",
  "matched_recommendations": [
    {{
      "section": "1.1.1",
      "action_type": "Urgent Referral",
      "criteria_met": "Patient aged 55 (>=40) with unexplained haemoptysis",
      "criteria_from_guideline": "exact text from the guideline passage"
    }}
  ]
}}

If multiple recommendations apply, include all in matched_recommendations.
The risk_level should reflect the HIGHEST priority recommendation matched.
Priority order: Urgent Referral > Urgent Investigation > Consider Referral > No NG12 Criteria Met."""


def format_context(chunks: list[dict]) -> str:
    """Format RAG chunks into numbered context text for the prompt.

    Args:
        chunks: List of dicts with keys 'text' and 'metadata'.

    Returns:
        Formatted string with source headers and chunk text.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        canonical_meta = chunk.get("canonical_metadata", {})
        section = meta.get("section") or canonical_meta.get("section", "Part B")
        page = meta.get("page") or canonical_meta.get("page", "N/A")
        header = (
            f"[Source {i} | Section {section} "
            f"| Page {page} "
            f"| {meta.get('cancer_type', 'N/A')}]"
        )
        context_parts.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(context_parts)


def format_assessment_prompt(patient: dict, chunks: list[dict]) -> str:
    """Assemble the complete user prompt from patient data and RAG chunks.

    Args:
        patient: Patient dict with keys: patient_id, name, age, gender,
                 smoking_history, symptoms, symptom_duration_days.
        chunks: List of RAG result dicts with keys 'text' and 'metadata'.

    Returns:
        Fully interpolated user prompt string.
    """
    context = format_context(chunks)
    symptoms_str = ", ".join(patient["symptoms"])
    return ASSESSMENT_USER_TEMPLATE.format(
        context=context,
        patient_id=patient["patient_id"],
        name=patient["name"],
        age=patient["age"],
        gender=patient["gender"],
        smoking_history=patient["smoking_history"],
        symptoms=symptoms_str,
        symptom_duration_days=patient["symptom_duration_days"],
    )
