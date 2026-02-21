# PROMPTS.md — Part 1: Clinical Decision Support Prompt Strategy

This document explains the prompt engineering design for the **NG12 Cancer Risk Assessor** (Part 1). All prompts live in [`app/prompts/assessment.py`](app/prompts/assessment.py).

---

## Overview

The assessment pipeline uses a **two-prompt architecture**:

1. `ASSESSMENT_SYSTEM_PROMPT` — defines the agent's role, strict grounding rules, and output contract
2. `ASSESSMENT_USER_TEMPLATE` — injects RAG-retrieved guideline passages and structured patient data at runtime

Both prompts are assembled by `format_assessment_prompt()` and passed to Gemini 1.5 Pro via Vertex AI.

---

## System Prompt

### Role Definition

```
You are a clinical decision support agent specialising in the NICE NG12 guideline:
Suspected Cancer — Recognition and Referral.
```

The system prompt opens by anchoring the model to a specific clinical role and a specific guideline document. This prevents the model from drawing on general medical knowledge outside of NG12.

### Output Taxonomy

The model is explicitly given four — and only four — valid risk levels:

| Risk Level | Clinical Meaning |
|---|---|
| `Urgent Referral` | Suspected cancer pathway, 2-week wait |
| `Urgent Investigation` | e.g. chest X-ray, CT, endoscopy within 2 weeks |
| `Consider Referral` | Lower certainty; clinician discretion |
| `No NG12 Criteria Met` | Patient presentation not covered by guideline |

Enumerating the exact output values eliminates free-form risk labelling and keeps downstream parsing deterministic.

### Grounding Rules

Seven numbered rules enforce strict grounding:

| Rule | Purpose |
|---|---|
| 1. Base assessment ONLY on provided passages | Prevents hallucination from parametric knowledge |
| 2. Do NOT use general medical knowledge | Reinforces rule 1 for clinical facts (drug names, diagnostic criteria) |
| 3. Match age, symptoms, duration, risk factors against each passage | Forces explicit criteria-matching, not pattern recognition |
| 4. If multiple recommendations apply, list ALL | Ensures completeness — a patient may qualify for both Urgent Referral AND Urgent Investigation |
| 5. Precise age thresholds — "aged 40 and over" means ≥40 | Prevents off-by-one errors at age boundaries |
| 6. "Unexplained" means not attributable to another known cause | Operationalises an ambiguous clinical qualifier |
| 7. If no passage matches, say so explicitly | Enforces honest failure rather than a forced answer |

### Output Contract

```
You must respond in valid JSON format only, no other text.
```

Requiring JSON-only output (no markdown, no backticks, no preamble) allows the application to parse the response directly without post-processing. The exact schema is defined in the user template.

---

## User Prompt Template

The user turn is constructed by `format_assessment_prompt()` at inference time and has three sections:

### Section 1 — RAG Context

```
NG12 Guideline Passages:

[Source 1 | Section 1.1.1 | Page 9 | Lung Cancer]
<retrieved chunk text>

---

[Source 2 | Section 1.4.1 | Page 23 | Colorectal Cancer]
<retrieved chunk text>
```

Each RAG chunk is rendered with a structured header containing:
- `Section` — NG12 section number (e.g. `1.1.1`) or `Part B` for the symptom index
- `Page` — PDF page number for citation traceability
- `Cancer type` — e.g. `Lung Cancer`, `Colorectal Cancer`

The numbered `[Source N]` format directly maps to the `matched_recommendations` section numbers in the JSON output.

### Section 2 — Patient Data

```
Patient Data:
- Patient ID: PT-101
- Name: John Doe
- Age: 55
- Gender: Male
- Smoking History: Current Smoker
- Symptoms: unexplained hemoptysis, fatigue
- Symptom Duration: 14 days
```

All structured patient fields are explicitly labeled. This prevents the model from inferring values (e.g. assuming a duration if not provided).

### Section 3 — Output Schema

The user prompt ends with the exact JSON schema the model must populate:

```json
{
  "risk_level": "Urgent Referral | Urgent Investigation | Consider Referral | No NG12 Criteria Met",
  "cancer_type": "the cancer type if identified, or 'None identified'",
  "recommended_action": "specific action from the guideline",
  "reasoning": "step-by-step explanation citing specific section numbers",
  "matched_recommendations": [
    {
      "section": "1.1.1",
      "action_type": "Urgent Referral",
      "criteria_met": "Patient aged 55 (>=40) with unexplained haemoptysis",
      "criteria_from_guideline": "exact text from the guideline passage"
    }
  ]
}
```

Key design decisions in the schema:

- **`criteria_met`** — forces the model to explicitly state which patient facts matched the guideline criterion (age, symptom, duration, risk factor)
- **`criteria_from_guideline`** — forces the model to quote the exact guideline text, making the citation verifiable against the source PDF
- **`matched_recommendations` as a list** — allows multiple qualifying criteria to be captured in a single response (e.g. a patient meeting both lung and colorectal referral thresholds)
- **`risk_level` reflects the HIGHEST priority** — if a patient qualifies for both `Urgent Referral` and `Consider Referral`, the top-level `risk_level` is always `Urgent Referral`

---

## Context Formatting

`format_context()` assembles the RAG chunks into the prompt. It applies a fallback chain for metadata:

```python
section = meta.get("section") or canonical_meta.get("section", "Part B")
page    = meta.get("page")    or canonical_meta.get("page", "N/A")
```

This handles two cases:
- **`rule_search` chunks** may have their section/page in `canonical_metadata` (attached at query time by `_attach_canonicals()`)
- **`symptom_index` chunks** have no section number — they fall back to `"Part B"` to reflect the PDF's symptom index section

---

## Design Rationale Summary

| Decision | Rationale |
|---|---|
| JSON-only output | Deterministic parsing; no LLM text wrapping the JSON |
| Numbered source headers in context | Enables the model to cite `[Source N]` without inventing citation formats |
| Explicit schema in user prompt | Reduces schema drift across model versions |
| `criteria_from_guideline` field | Verifiable citation — assessor can check the quoted text against the PDF |
| Priority ordering in prompt | Prevents ambiguous multi-match results from surfacing the wrong risk level |
| Fallback to `"No NG12 Criteria Met"` | Forces honest negative result rather than a speculative answer |
