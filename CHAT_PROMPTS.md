# CHAT_PROMPTS.md — Part 2: Conversational Chat Prompt Strategy

This document explains the prompt engineering design for the **NG12 Conversational Chat** (Part 2). All prompts and helper functions live in [`app/prompts/chat.py`](app/prompts/chat.py). The chat workflow is orchestrated by [`app/agents/chat_workflow.py`](app/agents/chat_workflow.py).

---

## Overview

The chat pipeline uses a **multi-stage prompt architecture** across a LangGraph workflow:

| Stage | Prompt / Component | Purpose |
|---|---|---|
| 1. Input Classification | `classify_input` (**no LLM** — pure regex/keyword) | Route the message before hitting RAG |
| 2. Query Building | `REWRITE_PROMPT` (LLM call, conditional) | Reformulate follow-ups into standalone search queries |
| 3. Answer Generation | `CHAT_SYSTEM_PROMPT` + `CHAT_USER_TEMPLATE` | Ground answer in retrieved passages with citations |
| 4. Post-processing | `clean_answer_sources()` | Replace `[Source N]` with human-readable citation refs |

---

## Stage 1 — Input Guardrail (classify_input)

Before any RAG retrieval occurs, every incoming message is classified into one of seven categories:

| Classification | Trigger | Response |
|---|---|---|
| `smalltalk` | Greetings, thanks, farewells — **no medical signal** | `SMALLTALK_RESPONSE` (welcome + example questions) |
| `meta` | Questions about the assistant itself ("what can you do?") | `META_RESPONSE` (capability description + disclaimer) |
| `chitchat_redirect` | Off-topic conversational messages | `CHAT_NON_MEDICAL_REDIRECT_RESPONSE` |
| `safety_urgent` | Emergency-like language ("I think I have cancer", "should I go to A&E") | `CHAT_SAFETY_RESPONSE` (emergency guidance + redirect) |
| `medical_out_of_scope` | Medical questions outside NG12 scope (treatment, prognosis, other guidelines) | `CHAT_OUT_OF_SCOPE_RESPONSE` |
| `needs_clarification` | Too vague to retrieve meaningfully ("tell me about cancer") | `CHAT_CLARIFY_RESPONSE` (asks for age, sex, symptoms, duration) |
| `proceed` | Clinical question answerable from NG12 | Enters the RAG pipeline |

### Key Design Decision — Medical Signal Guard

A helper function `_has_medical_signal()` scans the message for clinical terms (symptom names, cancer types, NG12 keywords). This prevents `smalltalk` and `chitchat_redirect` from capturing messages that happen to be short but contain medical content — for example, a single-word message like `"haematuria"` is routed to `proceed`, not `smalltalk`.

### Priority Order

The classifier evaluates in a strict priority cascade:

```text
smalltalk → meta → chitchat_redirect → safety_urgent → medical_out_of_scope → needs_clarification → proceed
```

`safety_urgent` is intentionally placed before `medical_out_of_scope` so that distressed messages are caught and redirected to emergency services before any other handling.

### Why Guardrail Before RAG

Calling RAG on every message — including greetings and off-topic messages — wastes vector search and LLM inference. The guardrail also prevents the model from generating clinical-sounding answers to vague or non-clinical inputs, which is a patient safety consideration.

---

## Stage 2 — Query Building (REWRITE_PROMPT)

Follow-up messages in a multi-turn conversation are often context-dependent and cannot be searched directly. The `QueryBuilder` class (in [`app/core/query_builder.py`](app/core/query_builder.py)) applies a three-tier strategy:

### Tier A — Direct Use

Short messages (≤3 words) or messages starting with known follow-up phrases (e.g. `"earlier"`, `"you mentioned"`, `"go back to"`) are enriched using the active session topic rather than rewritten via LLM. This avoids an extra LLM call for simple follow-ups.

### Tier B — LLM Rewrite (REWRITE_PROMPT)

Longer follow-ups that reference prior context implicitly (pronouns like `"it"`, `"they"`, `"that condition"`) are rewritten using the `REWRITE_PROMPT`:

```text
Rewrite this message into a standalone search query for NICE NG12 guidelines.

RULES:
1. Do NOT add facts (ages, durations, symptoms) not in the conversation
2. Keep the user's exact medical terms (e.g., "haemoptysis" not "coughing blood")
3. If information is missing, keep the query general — do not guess
4. Under 20 words
5. Do NOT answer — only rewrite for search

Conversation:
{history}

Message: {message}

Query:
```

Key constraints in the rewrite prompt:

- **Rule 1** — prevents hallucinating patient ages or durations into the search query
- **Rule 2** — preserves exact NG12 terminology so the vector search matches the right chunks
- **Rule 3** — a vague follow-up generates a general query, not a specific one with fabricated details
- **Rule 5** — explicitly tells the model its output will be used as a search query, not an answer

### Tier C — Direct Pass-through

Fresh, self-contained queries that pass no follow-up signal are used as-is.

---

## Stage 3 — Answer Generation

### System Prompt (CHAT_SYSTEM_PROMPT)

The system prompt has three groups of rules:

#### Grounding Rules (1–6)

```text
1. ONLY answer based on the NG12 guideline passages provided below.
   Do NOT use your general medical knowledge.
2. Every factual claim must cite a specific source using [Source N] format.
3. Be precise with age thresholds, action types, and clinical criteria.
   Do not paraphrase in a way that changes the clinical meaning.
4. If multiple sources are relevant, cite all of them.
5. Distinguish clearly between:
   - "Suspected cancer pathway referral" (urgent, 2-week wait)
   - "Urgent investigation" (e.g., chest X-ray within 2 weeks)
   - "Consider" recommendations (lower certainty)
6. When quoting criteria, include ALL conditions (age AND symptom AND duration etc.)
   Do not omit qualifying conditions.
```

Rule 3 is particularly important: NG12 thresholds are clinically precise (e.g. `"aged 40 and over with unexplained haemoptysis"`). A paraphrase like `"older patients with coughing blood"` would be clinically misleading. The model is explicitly instructed not to soften or generalise these criteria.

Rule 6 prevents partial citation — e.g. citing only the age threshold without the symptom requirement, which would make a criterion appear broader than it is.

#### Conversation Rules (7–10)

```text
7. Use the conversation history for context, but always ground answers in provided passages.
8. For follow-up questions, use context to understand the question, cite passages for the answer.
9. Keep answers focused and clinical. Use clear, professional language.
10. Structure longer answers with clear paragraphs, not bullet points.
```

Rules 7–8 implement the multi-turn grounding contract: the model is allowed to use history to understand *what* is being asked, but must always cite the retrieved passages — not the conversation history — when making clinical claims.

#### Missing Information Rules (11–13)

```text
11. If asked about criteria NOT in the provided passages:
    - State clearly: "The specific [criteria type] is not found in these passages."
    - Do NOT guess or infer numbers/thresholds.
    - Suggest rephrasing.
12. Never fabricate numbers. If a passage says "persistent" without defining duration,
    say "persistent (duration not specified)" rather than inventing "2 weeks".
13. If you cannot fully answer, acknowledge what you DO know, then state what is missing.
```

Rule 12 addresses a specific failure mode: the NG12 guideline uses the word `"persistent"` for some symptoms without defining a duration threshold. A model trained on general medical text might confidently fill this gap with `"2 weeks"` or `"3 weeks"` from other sources. These rules explicitly prohibit that.

### User Prompt Template (CHAT_USER_TEMPLATE)

```text
NG12 Guideline Passages:
{context}

---

Conversation History:
{history}

---

Current Question: {message}

---

Instructions:
- Answer using ONLY the guideline passages above
- Cite using [Source N] format for EVERY factual claim
- If the passages don't contain enough evidence, say so explicitly
- Be precise with clinical criteria (age, symptoms, action types)
```

The three-section structure (context → history → question) places the most constrained information (retrieved passages) first, so the model's attention is anchored to the guideline text before it reads the user's question.

### Context Formatting

`format_chat_context()` renders each RAG chunk with a structured header:

```text
[Source 1 | Section 1.1.1 | Page 9 | Lung Cancer | Urgent Referral]
<chunk text>
```

The header includes `action_type` (e.g. `Urgent Referral`) for chat chunks — this allows the model to cite the action type without inferring it from the passage text.

### Conversation History Formatting

`format_history()` applies two constraints:

- **Last 6 messages only** — keeps the prompt within token limits for long sessions
- **Assistant messages truncated to 200 characters** — assistant responses often contain quoted guideline text; truncating prevents the context window from being dominated by prior answers rather than retrieved passages

---

## The [Source N] Citation Contract

The citation system works as a **pre-agreed convention** between the code and the LLM. Three pieces must work together:

1. **`format_chat_context()` assigns numbers** — each retrieved chunk is labelled `[Source 1]`, `[Source 2]`, etc. in the prompt. The LLM sees these labels alongside the chunk text.

2. **System prompt Rule 2 instructs the LLM to use them** — `"Every factual claim must cite a specific source using [Source N] format."` The LLM is explicitly told that its output will be held to this standard.

3. **Post-processing parses and replaces** — after the LLM responds, the code scans for `[Source N]` markers in the answer text. These markers are the signal used to (a) identify which chunks were cited and (b) replace the raw index with a readable reference.

Neither piece works without the other: without the numbered labels in the prompt, the LLM has nothing to reference; without the system prompt instruction, the LLM may answer without citing. The code **trusts but does not verify** — if the LLM writes `[Source 3]`, the code assumes chunk 3 was used. There is no semantic check. If the LLM produces no `[Source N]` markers at all, `build_citations_from_chunks()` returns an empty list and a transparency note is appended to the answer.

---

## Stage 4 — Citation Post-Processing

After the LLM generates its answer, two functions process the citation references:

### build_citations_from_chunks()

Scans the answer text for `[Source N]` and `[Source 1, 2, 3]` patterns and maps them back to the original RAG chunks. Only chunks that are actually cited in the answer are included in the `citations` array — no fake citations are added if the model fails to cite.

Each citation includes:

```json
{
  "source": "NG12 PDF",
  "section": "1.1.1",
  "page": 9,
  "chunk_id": "ng12_0045_02",
  "excerpt": "first 200 characters of the chunk text"
}
```

### clean_answer_sources()

Replaces `[Source N]` markers in the answer text with human-readable references before returning to the client:

| Chunk Type | Input | Output |
|---|---|---|
| `rule_search` / `rule_canonical` | `[Source 1]` | `[NG12 §1.1.1, p.9]` |
| `symptom_index` | `[Source 2]` | `[NG12 Part B, p.43]` |

Multi-source citations are also handled: `[Source 1, 2, 3]` → `[NG12 §1.1.1, p.9; NG12 §1.1.4, p.11; NG12 Part B, p.43]`.

---

## Canned Responses

Seven pre-written responses handle non-RAG paths. Each is designed for a specific classification:

| Constant | When Used | Key Content |
|---|---|---|
| `SMALLTALK_RESPONSE` | Greetings / farewells | Welcome message + 3 example questions to prompt engagement |
| `META_RESPONSE` | "What can you do?" | Capability description + safety disclaimer (not a doctor) |
| `CHAT_NON_MEDICAL_REDIRECT_RESPONSE` | Off-topic chat | Brief redirect to NG12 scope |
| `CHAT_OUT_OF_SCOPE_RESPONSE` | Medical but outside NG12 | Lists 4 things the assistant CAN help with |
| `CHAT_SAFETY_RESPONSE` | Emergency/self-diagnosis signals | Emergency services redirect (999/911) + offer to help with guideline questions |
| `CHAT_CLARIFY_RESPONSE` | Too vague | 6-field checklist (age, sex, symptoms, duration, smoking, red flags) |
| `CHAT_REFUSE_RESPONSE` | RAG returns irrelevant passages | Acknowledges evidence gap + 3 rephrasing suggestions |

The `CHAT_CLARIFY_RESPONSE` is particularly important for clinical safety: NG12 referral criteria are highly specific to age, sex, and symptom combination. Returning a vague answer to an underspecified question risks the user drawing incorrect clinical conclusions.

---

## Grounding Strategy Summary

The overall grounding strategy operates at three levels:

| Level | Mechanism | Protection Against |
|---|---|---|
| **Pre-retrieval** | `classify_input` guardrail | Hallucination on non-clinical or vague inputs |
| **Prompt-level** | System prompt rules 1–6, 11–13 | Parametric knowledge leak, threshold fabrication |
| **Post-generation** | `build_citations_from_chunks()` + `CHAT_REFUSE_RESPONSE` | Fake citations, uncited clinical claims |

No answer that cites clinical thresholds or referral criteria reaches the client without being traceable to a specific retrieved chunk from the NG12 PDF.
