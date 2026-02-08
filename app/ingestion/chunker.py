"""
NG12 Structured Chunker

Splits the NG12 PDF into chunks by clinical recommendation.
Each chunk is one complete recommendation with rich metadata.

State-machine approach:
  PART_A — Clinical recommendation sections (1.x / 1.x.y)
    1. Identify major sections (1.x) to track cancer_type
    2. Identify subsections (1.x.y) to delineate recommendations
    3. Within subsections, split by recommendation verbs (with protection rules)
  PART_B — "Recommendations organised by symptom" (collected, processed later)
  STOP   — Appendix material (discarded entirely)
"""

import json
import re
from collections import Counter
from typing import Any

import fitz  # pymupdf


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

RE_SUBSECTION = re.compile(r"^\s*(1\.\d{1,2}\.\d{1,2})\b")
RE_MAJOR_SECTION = re.compile(r"^\s*(1\.\d{1,2})\s+(.+?)$")
RE_REC_VERB = re.compile(
    r"^\s*(Refer|Offer|Consider|Perform|Test|Arrange|Do not)\b", re.IGNORECASE
)
RE_BULLET = re.compile(r"^\s*[\u2022\u2013\u2212\u2023\u25cf\u00b7\u2015\u2010•\-\xb7\u25cb]\s")
RE_NUMBERED_BULLET = re.compile(r"^\s*\d+\.\s")

RE_AGE_AND_OVER = re.compile(r"aged?\s+(\d+)\s+and\s+over", re.IGNORECASE)
RE_AGE_OR_OVER = re.compile(r"aged?\s+(\d+)\s+or\s+over", re.IGNORECASE)
RE_AGE_UNDER = re.compile(r"(?:aged?\s+)?under\s+(\d+)", re.IGNORECASE)

SYMPTOM_KEYWORDS = [
    "haemoptysis", "hemoptysis", "dysphagia", "haematuria", "hematuria",
    "breast lump", "rectal bleeding", "hoarseness", "weight loss",
    "jaundice", "dyspepsia", "cough", "shortness of breath", "fatigue",
    "chest pain", "abdominal pain", "iron-deficiency anaemia",
    "iron deficiency anaemia", "lymphadenopathy", "night sweats",
    "abdominal mass", "post-menopausal bleeding", "unexplained lump",
    "change in bowel habit", "rectal mass", "anal mass",
    "appetite loss", "nausea", "vomiting", "bloating",
]

REC_CONTEXT_WORDS = [
    "people", "patient", "suspected cancer", "urgent", "pathway",
    "adults", "children", "women", "men", "referral",
]

# Synonym expansions for rule_search text (kept small and curated)
SYNONYM_MAP: dict[str, list[str]] = {
    "haemoptysis": ["hemoptysis", "coughing blood"],
    "hoarseness": ["voice change", "persistent voice change"],
    "chest x-ray": ["CXR"],
    "suspected cancer pathway referral": ["urgent referral", "2-week wait", "2WW"],
    "haematuria": ["hematuria", "blood in urine"],
    "dysphagia": ["difficulty swallowing"],
    "dyspepsia": ["indigestion"],
    "iron-deficiency anaemia": ["iron deficiency anemia"],
}

# Part B table header markers (noise lines to skip, lowercase)
TABLE_HEADER_MARKERS = [
    "symptom and specific features",
    "possible cancer",
    "recommendation",
    "investigation findings and specific features",
    "examination findings and specific features",
    "symptoms and signs",
    "symptoms and specific features",
]

# Known cancer type keywords for detecting the "Possible cancer" column
KNOWN_CANCER_TYPES = [
    "lung", "mesothelioma", "oesophageal", "stomach", "pancreatic",
    "colorectal", "anal", "breast", "ovarian", "endometrial",
    "cervical", "vulval", "vaginal", "prostate", "bladder",
    "renal", "testicular", "penile", "melanoma", "squamous cell",
    "basal cell", "laryngeal", "oral", "thyroid", "brain",
    "leukaemia", "leukemia", "myeloma", "non-hodgkin", "hodgkin",
    "sarcoma", "neuroblastoma", "retinoblastoma", "wilms",
    "gall bladder", "liver",
]

# Part B cross-reference pattern, e.g. [1.5.2]
RE_PART_B_REF = re.compile(r"\[1\.\d{1,2}\.\d{1,2}\]")

# Part B top-level system titles (these group multiple sub-tables)
SYSTEM_TITLES = {
    "abdominal symptoms",
    "bleeding",
    "gynaecological symptoms",
    "lumps or masses",
    "neurological symptoms in adults",
    "pain",
    "respiratory symptoms",
    "skeletal symptoms",
    "skin or surface symptoms",
    "urological symptoms",
    "non-specific features of cancer",
    "primary care investigations",
    "symptoms in children and young people",
}

# Regex for residual page headers/footers inside Part B lines
RE_PAGE_FOOTER = re.compile(r"Page \d+ of\s*\d+")

# Part B recommendation action verb prefixes (lowercase).
# Lines starting with these are additional actions for the current record,
# NOT the beginning of a new symptom row.
RECOMMENDATION_VERBS = [
    "refer", "offer", "consider", "carry out", "measure",
    "arrange", "if serum", "see the section", "advise",
]


# ---------------------------------------------------------------------------
# A) Parse PDF to lines
# ---------------------------------------------------------------------------

def parse_pdf_to_lines(pdf_path: str) -> list[dict]:
    """Parse the NG12 PDF and return cleaned lines with page numbers.

    Cleaning steps:
      1. Merge hyphenated line breaks (e.g. "haemop-\\ntysis" -> "haemoptysis")
      2. Collapse consecutive blank lines
      3. Merge short fragment lines (< 10 chars) into previous line,
         unless the line is a structural marker (section number, verb, bullet)

    Returns:
        List of dicts: [{"text": "line text", "page": 9}, ...]
    """
    doc = fitz.open(pdf_path)
    raw_lines: list[dict] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        for line in text.split("\n"):
            raw_lines.append({"text": line, "page": page_num + 1})

    doc.close()

    # Step 1: merge hyphenated line breaks
    merged: list[dict] = []
    i = 0
    while i < len(raw_lines):
        current = raw_lines[i]
        txt = current["text"]
        if (
            txt.rstrip().endswith("-")
            and i + 1 < len(raw_lines)
            and raw_lines[i + 1]["text"]
            and raw_lines[i + 1]["text"][0].islower()
        ):
            joined_text = txt.rstrip()[:-1] + raw_lines[i + 1]["text"]
            merged.append({"text": joined_text, "page": current["page"]})
            i += 2
        else:
            merged.append(current)
            i += 1

    # Step 2: collapse consecutive blank lines
    deduped: list[dict] = []
    prev_blank = False
    for item in merged:
        is_blank = item["text"].strip() == ""
        if is_blank and prev_blank:
            continue
        deduped.append(item)
        prev_blank = is_blank

    # Step 2b: filter out page headers / footers
    RE_PAGE_NUM = re.compile(r"Page \d+ of\s*\d+")
    RE_NG12_TITLE = re.compile(r"Suspected cancer: recognition and referral \(NG12\)")
    filtered: list[dict] = []
    for item in deduped:
        line_text = item["text"]
        if "\u00a9 NICE" in line_text or "© NICE" in line_text:
            continue
        if RE_PAGE_NUM.search(line_text):
            continue
        if RE_NG12_TITLE.search(line_text):
            continue
        filtered.append(item)

    # Step 3: merge short fragment lines
    result: list[dict] = []
    for item in filtered:
        txt = item["text"].strip()
        if not result:
            result.append(item)
            continue

        if (
            len(txt) < 10
            and txt != ""
            and not RE_SUBSECTION.match(txt)
            and not RE_MAJOR_SECTION.match(txt)
            and not RE_REC_VERB.match(txt)
            and not RE_BULLET.match(txt)
            and not RE_NUMBERED_BULLET.match(txt)
        ):
            result[-1] = {
                "text": result[-1]["text"].rstrip() + " " + txt,
                "page": result[-1]["page"],
            }
        else:
            result.append(item)

    return result


# ---------------------------------------------------------------------------
# B) Identify major section titles -> cancer_type mapping
# ---------------------------------------------------------------------------

def _build_major_section_map(lines: list[dict]) -> dict[str, str]:
    """Scan lines and build section_prefix -> cancer_type title mapping.

    E.g. "1.1" -> "Lung and pleural cancers"

    Two-pass: first collect all subsection numbers, then identify major
    section headings that are NOT subsection numbers.
    """
    subsection_nums = set()
    for item in lines:
        m = RE_SUBSECTION.match(item["text"])
        if m:
            subsection_nums.add(m.group(1))

    section_map: dict[str, str] = {}
    for idx, item in enumerate(lines):
        m = RE_MAJOR_SECTION.match(item["text"])
        if not m:
            continue
        section_num = m.group(1)
        title = m.group(2).strip()

        if section_num in subsection_nums:
            continue
        if RE_SUBSECTION.match(item["text"]):
            continue

        next_text = _next_nonblank_text(lines, idx)
        if next_text and RE_REC_VERB.match(next_text):
            continue

        section_map[section_num] = title

    return section_map


def _next_nonblank_text(lines: list[dict], idx: int) -> str | None:
    """Return the text of the next non-blank line after idx."""
    for i in range(idx + 1, min(idx + 5, len(lines))):
        t = lines[i]["text"].strip()
        if t:
            return t
    return None


# ---------------------------------------------------------------------------
# C) Main chunking logic
# ---------------------------------------------------------------------------

def chunk_ng12(lines: list[dict]) -> list[dict]:
    """Split NG12 lines into structured chunks with rich metadata.

    Uses a state machine with three states:
      - PART_A: Clinical recommendations (section/subsection structure)
      - PART_B: Recommendations organised by symptom (collected for later)
      - STOP:   Appendix material (discarded)

    Returns:
        List of chunk dicts with keys: chunk_id, text, metadata.
    """
    section_map = _build_major_section_map(lines)

    chunks: list[dict] = []
    current_section: str | None = None
    current_lines: list[dict] = []
    current_cancer_type = "General"
    part_b_lines: list[dict] = []

    state = "PART_A"  # initial state

    # STOP markers (normalized, lowercase)
    STOP_MARKERS = [
        "terms used in this guideline",
        "rationale and impact",
        "recommendations for research",
        "finding more information",
        "update information",
    ]

    for item in lines:
        text = item["text"]
        normalized = re.sub(r"\s+", " ", text).strip().lower()

        # Skip TOC lines (contain runs of dots like ".....") — they are
        # not real section headings and would cause premature transitions.
        is_toc_line = "....." in normalized

        # --- State transition checks (before processing the line) ---
        if (
            not is_toc_line
            and normalized.startswith("recommendations organised by symptom")
        ):
            # Finalize any open PART_A section before switching
            if state == "PART_A" and current_section and current_lines:
                chunks.extend(
                    _finalize_section(current_section, current_lines, current_cancer_type)
                )
                current_section = None
                current_lines = []
            state = "PART_B"
            continue

        if not is_toc_line and any(
            normalized.startswith(marker) for marker in STOP_MARKERS
        ):
            # Finalize any open PART_A section before switching
            if state == "PART_A" and current_section and current_lines:
                chunks.extend(
                    _finalize_section(current_section, current_lines, current_cancer_type)
                )
                current_section = None
                current_lines = []
            state = "STOP"
            continue

        # "context" is too generic; only match short standalone lines
        if (
            not is_toc_line
            and (normalized == "context" or normalized.startswith("context"))
            and len(normalized) < 20
        ):
            if state == "PART_A" and current_section and current_lines:
                chunks.extend(
                    _finalize_section(current_section, current_lines, current_cancer_type)
                )
                current_section = None
                current_lines = []
            state = "STOP"
            continue

        # --- State behaviour ---
        if state == "STOP":
            continue

        if state == "PART_B":
            part_b_lines.append(item)
            continue

        # state == "PART_A" — original section/subsection logic
        # Check if this is a major section heading
        m_major = RE_MAJOR_SECTION.match(text)
        if m_major and m_major.group(1) in section_map:
            if current_section and current_lines:
                chunks.extend(
                    _finalize_section(current_section, current_lines, current_cancer_type)
                )
                current_section = None
                current_lines = []
            current_cancer_type = section_map[m_major.group(1)]
            continue

        # Check for subsection match
        m_sub = RE_SUBSECTION.match(text)
        if m_sub:
            if current_section and current_lines:
                chunks.extend(
                    _finalize_section(current_section, current_lines, current_cancer_type)
                )
            current_section = m_sub.group(1)
            current_lines = [item]
            continue

        # Inside a subsection: accumulate lines
        if current_section:
            current_lines.append(item)

    # Save last open section (if still in PART_A)
    if current_section and current_lines:
        chunks.extend(
            _finalize_section(current_section, current_lines, current_cancer_type)
        )

    # Deduplicate chunk IDs (PDF may contain repeated section numbers)
    chunks = _deduplicate_ids(chunks)

    # Generate a rule_search companion for every rule_canonical chunk
    search_chunks: list[dict] = []
    for c in chunks:
        if c["metadata"].get("doc_type") == "rule_canonical":
            search_chunks.append(_generate_rule_search(c))
    search_chunks = _deduplicate_ids(search_chunks)
    chunks.extend(search_chunks)

    print(f"Part B lines collected: {len(part_b_lines)}")

    # Parse Part B table lines into symptom_index chunks
    symptom_chunks = _parse_part_b(part_b_lines)
    symptom_chunks = _deduplicate_ids(symptom_chunks)
    chunks.extend(symptom_chunks)

    _print_stats(chunks)
    return chunks


def _deduplicate_ids(chunks: list[dict]) -> list[dict]:
    """Ensure all chunk IDs are unique by appending _dup2, _dup3, etc."""
    seen: dict[str, int] = {}
    for chunk in chunks:
        cid = chunk["chunk_id"]
        if cid in seen:
            seen[cid] += 1
            new_id = f"{cid}_dup{seen[cid]}"
            chunk["chunk_id"] = new_id
            chunk["metadata"]["chunk_id"] = new_id
        else:
            seen[cid] = 1
    return chunks


# ---------------------------------------------------------------------------
# D) Finalize a subsection: optionally split by recommendation verbs
# ---------------------------------------------------------------------------

def _finalize_section(
    section: str, lines: list[dict], cancer_type: str
) -> list[dict]:
    """Process a subsection's lines into one or more chunks.

    If the section contains multiple recommendation verbs, split into
    sub-chunks with protection rules to prevent over-splitting.
    """
    full_text = "\n".join(line["text"] for line in lines).strip()
    page_start = lines[0]["page"]
    page_end = lines[-1]["page"]

    # Strip the leading section number from text body
    text_body = RE_SUBSECTION.sub("", full_text, count=1).strip()

    verb_positions = _find_rec_verb_positions(text_body)

    if len(verb_positions) <= 1:
        metadata = _build_chunk_metadata(
            section=section,
            cancer_type=cancer_type,
            page_start=page_start,
            page_end=page_end,
            text=text_body,
        )
        chunk_id = "ng12_" + section.replace(".", "_")
        metadata["chunk_id"] = chunk_id
        return [{"chunk_id": chunk_id, "text": text_body, "metadata": metadata}]

    # Multiple recommendations -> split into sub-chunks
    sub_chunks = []
    text_lines = text_body.split("\n")
    suffix_ord = ord("a")

    for i, pos in enumerate(verb_positions):
        end_pos = verb_positions[i + 1] if i + 1 < len(verb_positions) else len(text_lines)
        sub_text = "\n".join(text_lines[pos:end_pos]).strip()
        if not sub_text:
            continue

        suffix = chr(suffix_ord)
        suffix_ord += 1
        chunk_id = "ng12_" + section.replace(".", "_") + "_" + suffix

        metadata = _build_chunk_metadata(
            section=section,
            cancer_type=cancer_type,
            page_start=page_start,
            page_end=page_end,
            text=sub_text,
        )
        metadata["chunk_id"] = chunk_id
        sub_chunks.append({"chunk_id": chunk_id, "text": sub_text, "metadata": metadata})

    return sub_chunks


def _find_rec_verb_positions(text: str) -> list[int]:
    """Find line indices where valid recommendation verbs start.

    Protection: verb line must contain context words (e.g. "people",
    "patient", "suspected cancer") to confirm it is a standalone
    recommendation and not a continuation.
    """
    text_lines = text.split("\n")
    positions: list[int] = []

    for i, line in enumerate(text_lines):
        if not RE_REC_VERB.match(line):
            continue
        line_lower = line.lower()
        if any(w in line_lower for w in REC_CONTEXT_WORDS):
            positions.append(i)

    if not positions:
        return [0] if text_lines else []

    if positions[0] != 0:
        positions.insert(0, 0)

    return positions


# ---------------------------------------------------------------------------
# E) Build chunk metadata with rule extraction
# ---------------------------------------------------------------------------

def _build_chunk_metadata(
    section: str,
    cancer_type: str,
    page_start: int,
    page_end: int,
    text: str,
) -> dict[str, Any]:
    """Build rich metadata for a recommendation chunk."""
    metadata: dict[str, Any] = {
        "source": "NG12",
        "doc_type": "rule_canonical",
        "section": section,
        "cancer_type": cancer_type,
        "page": page_start,
        "page_end": page_end,
    }
    rule_meta = extract_rule_metadata(text)
    metadata.update(rule_meta)
    return metadata


def extract_rule_metadata(text: str) -> dict[str, Any]:
    """Extract structured rule conditions from recommendation text.

    Returns:
        Dict with action_type, age_min/max/operator, symptom_keywords_json,
        risk_factor_smoking, gender_specific as applicable.
    """
    metadata: dict[str, Any] = {}
    text_lower = text.lower()

    # 1. action_type (priority-ordered)
    if (
        "suspected cancer pathway" in text_lower
        or "two week" in text_lower
        or ("refer" in text_lower and "suspected cancer" in text_lower)
    ):
        metadata["action_type"] = "Urgent Referral"
    elif "urgent" in text_lower and any(
        w in text_lower
        for w in [
            "x-ray", "ct", "ultrasound", "endoscopy",
            "investigation", "dermoscopy", "test", "imaging",
        ]
    ):
        metadata["action_type"] = "Urgent Investigation"
    elif text_lower.lstrip().startswith("do not"):
        metadata["action_type"] = "Do Not"
    elif any(w in text_lower for w in ["safety net", "advise", "information"]):
        metadata["action_type"] = "Safety Net"
    elif text_lower.lstrip().startswith("consider"):
        metadata["action_type"] = "Consider"
    else:
        metadata["action_type"] = "Other"

    # 2. age_threshold
    m = RE_AGE_AND_OVER.search(text)
    if m:
        metadata["age_min"] = int(m.group(1))
        metadata["age_operator"] = "and_over"
    else:
        m = RE_AGE_OR_OVER.search(text)
        if m:
            metadata["age_min"] = int(m.group(1))
            metadata["age_operator"] = "or_over"
        else:
            m = RE_AGE_UNDER.search(text)
            if m:
                metadata["age_max"] = int(m.group(1))
                metadata["age_operator"] = "under"

    # 3. symptom_keywords
    matched_symptoms = [kw for kw in SYMPTOM_KEYWORDS if kw in text_lower]
    if matched_symptoms:
        metadata["symptom_keywords_json"] = json.dumps(matched_symptoms)

    # 4. risk_factor_smoking (kept for backward compatibility)
    if "smoked" in text_lower or "smoker" in text_lower:
        metadata["risk_factor_smoking"] = True

    # 5. urgency (priority-ordered)
    if "immediate" in text_lower:
        metadata["urgency"] = "immediate"
    elif "within 48 hours" in text_lower or "very urgent" in text_lower:
        metadata["urgency"] = "very_urgent"
    elif "within 2 weeks" in text_lower or "suspected cancer pathway" in text_lower:
        metadata["urgency"] = "urgent"
    elif "routine referral" in text_lower or "non-urgent" in text_lower:
        metadata["urgency"] = "non_urgent"

    # 6. qualifiers
    qualifier_terms = ["persistent", "unexplained", "recurrent"]
    matched_quals = [q for q in qualifier_terms if q in text_lower]
    if matched_quals:
        metadata["qualifiers_json"] = json.dumps(matched_quals)

    # 7. risk_factors (list, superset of the boolean smoking field)
    risk_factors: list[str] = []
    if "smoked" in text_lower or "smoker" in text_lower:
        risk_factors.append("ever_smoked")
    if "asbestos" in text_lower:
        risk_factors.append("asbestos_exposure")
    if risk_factors:
        metadata["risk_factors_json"] = json.dumps(risk_factors)

    # 8. gender_specific
    female_terms = [
        "breast", "gynaecological", "ovarian", "cervical",
        "endometrial", "vulval", "vaginal", "post-menopausal",
    ]
    male_terms = ["prostate", "testicular", "penile"]
    if any(t in text_lower for t in female_terms):
        metadata["gender_specific"] = "Female"
    elif any(t in text_lower for t in male_terms):
        metadata["gender_specific"] = "Male"

    return metadata


# ---------------------------------------------------------------------------
# F) Generate rule_search companion chunk
# ---------------------------------------------------------------------------

def _generate_rule_search(canonical_chunk: dict) -> dict:
    """Build a rule_search chunk from a rule_canonical chunk.

    The search chunk contains a short, template-based text optimised for
    embedding similarity retrieval, plus synonym expansions.
    """
    meta = canonical_chunk["metadata"]
    section = meta["section"]
    cancer_type = meta.get("cancer_type", "")
    action_type = meta.get("action_type", "")

    parts: list[str] = [f"NG12 Rule {section}"]

    if cancer_type:
        parts.append(f"Cancer site: {cancer_type}")

    action_line = f"Action: {action_type}" if action_type else ""
    urgency = meta.get("urgency")
    if action_line and urgency:
        action_line += f" ({urgency})"
    if action_line:
        parts.append(action_line)

    if "age_min" in meta:
        parts.append(f"Criteria: age >= {meta['age_min']}")
    elif "age_max" in meta:
        parts.append(f"Criteria: age < {meta['age_max']}")

    if "qualifiers_json" in meta:
        qualifiers = json.loads(meta["qualifiers_json"])
        parts.append(f"Qualifiers: {', '.join(qualifiers)}")

    if "risk_factors_json" in meta:
        risk_factors = json.loads(meta["risk_factors_json"])
        parts.append(f"Risk factors: {', '.join(risk_factors)}")

    if "symptom_keywords_json" in meta:
        symptoms = json.loads(meta["symptom_keywords_json"])
        # Expand with synonyms
        expanded: list[str] = []
        for sym in symptoms:
            expanded.append(sym)
            sym_lower = sym.lower()
            if sym_lower in SYNONYM_MAP:
                expanded.extend(SYNONYM_MAP[sym_lower])
        parts.append(f"Symptoms: {', '.join(expanded)}")

    if "gender_specific" in meta:
        parts.append(f"Gender: {meta['gender_specific']}")

    search_text = "\n".join(parts)

    chunk_id = "ng12_search_" + section.replace(".", "_")
    search_meta: dict[str, Any] = {
        "source": "NG12",
        "doc_type": "rule_search",
        "rule_id": section,
        "section": section,
        "cancer_type": cancer_type,
        "page": meta.get("page", 0),
        "page_end": meta.get("page_end", 0),
        "chunk_id": chunk_id,
    }

    return {"chunk_id": chunk_id, "text": search_text, "metadata": search_meta}


# ---------------------------------------------------------------------------
# G) Part B: symptom_index parsing
# ---------------------------------------------------------------------------

def _is_part_b_section_title(
    line_text: str, next_line_text: str | None
) -> tuple[str, str] | None:
    """Detect whether a Part B line is a system or sub-section title.

    Returns:
        ("system", title) or ("sub", title) if the line is a title,
        None otherwise.
    """
    normalized = line_text.strip()
    if not normalized:
        return None

    norm_lower = normalized.lower()

    # Noise lines are not titles
    if any(m in norm_lower for m in TABLE_HEADER_MARKERS):
        return None
    # Lines containing cross-references are table body, not titles
    if "[" in normalized:
        return None
    # Lines matching known cancer types are table body
    if any(ct in norm_lower for ct in KNOWN_CANCER_TYPES):
        return None
    # "See also ..." is a cross-reference note, not a title
    if norm_lower.startswith("see also"):
        return None

    # Check against hardcoded system-level titles
    if norm_lower in SYSTEM_TITLES:
        return ("system", normalized)

    # Heuristic for sub-titles: short line starting with uppercase,
    # and the NEXT line is a table header row.  This is conservative
    # but avoids false positives on multi-line table cell fragments.
    if len(normalized) < 80 and normalized[0].isupper():
        if next_line_text is not None:
            next_lower = next_line_text.strip().lower()
            if any(m in next_lower for m in TABLE_HEADER_MARKERS):
                return ("sub", normalized)
    return None


def _parse_part_b(part_b_lines: list[dict]) -> list[dict]:
    """Parse Part B table lines into symptom_index chunks.

    Flush strategy: accumulate lines into a record until the *next*
    record begins.  A record is considered complete once it contains at
    least one [1.x.y] cross-reference.  A new record starts when a
    non-tail-note, non-ref line arrives after a complete record.

    Returns a list of chunk dicts with doc_type="symptom_index".
    """
    # --- Pre-filter: remove page footers and blank lines ---
    cleaned: list[dict] = []
    for item in part_b_lines:
        txt = item["text"].strip()
        if not txt:
            continue
        if RE_PAGE_FOOTER.search(txt):
            continue
        cleaned.append(item)

    chunks: list[dict] = []
    current_system_title = ""
    current_sub_title = ""
    current_record_lines: list[dict] = []
    current_record_page_start = 0
    row_index = 0

    # Lookahead helper
    def _next_text(idx: int) -> str | None:
        if idx + 1 < len(cleaned):
            return cleaned[idx + 1]["text"]
        return None

    # Tail-note prefixes that belong to the current record (lowercase)
    TAIL_PREFIXES = ("these ", "separate ", "see ", "also ", "for ", "if ")

    def _record_has_refs() -> bool:
        """Return True if the accumulated record already has a [1.x.y] ref."""
        return any(
            RE_PART_B_REF.search(ln["text"]) for ln in current_record_lines
        )

    def _flush_record() -> None:
        nonlocal current_record_lines, row_index
        if not current_record_lines:
            return

        raw_text = " ".join(
            line["text"].strip() for line in current_record_lines
        )
        # Collapse multiple spaces
        raw_text = re.sub(r"\s+", " ", raw_text).strip()

        if not raw_text:
            current_record_lines = []
            return

        page_start = current_record_lines[0]["page"]
        page_end = current_record_lines[-1]["page"]

        # Extract cross-references (deduplicated, order-preserved)
        refs = RE_PART_B_REF.findall(raw_text)
        seen_refs: set[str] = set()
        unique_refs: list[str] = []
        for r in refs:
            if r not in seen_refs:
                seen_refs.add(r)
                unique_refs.append(r)

        # Skip noise records that contain no [1.x.y] back-references
        # (intro paragraphs, orphaned notes, fragment lines)
        if not unique_refs:
            current_record_lines = []
            return

        # Extract possible_cancer: first matching KNOWN_CANCER_TYPES token
        raw_lower = raw_text.lower()
        possible_cancer = ""
        cancer_pos = len(raw_text)
        for ct in KNOWN_CANCER_TYPES:
            idx = raw_lower.find(ct)
            if idx != -1 and idx < cancer_pos:
                cancer_pos = idx
                possible_cancer = ct.title()

        # Symptom: text before the first cancer-type mention (trimmed)
        symptom = raw_text[:cancer_pos].strip() if cancer_pos < len(raw_text) else raw_text
        symptom = symptom[:200]

        chunk_id = f"ng12_symptom_{page_start}_{row_index}"
        text = (
            f"NG12 Part B \u2014 Symptom index\n"
            f"System: {current_system_title}\n"
            f"Subsection: {current_sub_title}\n"
            f"Row: {raw_text}"
        )
        metadata: dict[str, Any] = {
            "source": "NG12",
            "doc_type": "symptom_index",
            "system_title": current_system_title,
            "sub_title": current_sub_title,
            "symptom": symptom,
            "possible_cancer": possible_cancer,
            "references_json": json.dumps(unique_refs),
            "page": page_start,
            "page_end": page_end,
            "chunk_id": chunk_id,
        }
        chunks.append({"chunk_id": chunk_id, "text": text, "metadata": metadata})
        row_index += 1
        current_record_lines = []

    # --- Main loop ---
    in_tail_note = False  # sticky flag for multi-line tail notes
    in_orphan_note = False  # sticky flag for multi-line orphaned section notes

    i = 0
    while i < len(cleaned):
        item = cleaned[i]
        line_text = item["text"]
        line_stripped = line_text.strip()
        line_lower = line_stripped.lower()

        # 1. Section titles always trigger a flush
        title_result = _is_part_b_section_title(line_text, _next_text(i))
        if title_result is not None:
            _flush_record()
            in_tail_note = False
            in_orphan_note = False
            kind, title = title_result
            if kind == "system":
                current_system_title = title
                current_sub_title = ""
            else:
                current_sub_title = title
            i += 1
            continue

        # 2. Skip table header noise (also breaks tail-note streak).
        #    Guard: "these recommendations" contains "recommendation" as a
        #    substring, so exclude lines starting with a tail-note prefix.
        if (any(m in line_lower for m in TABLE_HEADER_MARKERS)
                and not line_lower.startswith(TAIL_PREFIXES)):
            in_tail_note = False
            in_orphan_note = False
            i += 1
            continue

        # 2b. Skip orphaned section-level notes (tail-note prefix lines
        #     that appear at the start of a sub-section, before any
        #     symptom record has accumulated refs).  E.g. "These
        #     recommendations apply to women aged 18 and over" can
        #     span multiple PDF lines — use sticky in_orphan_note.
        if line_lower.startswith(TAIL_PREFIXES) and (
            not current_record_lines or not _record_has_refs()
        ):
            in_orphan_note = True
            i += 1
            continue
        if in_orphan_note:
            has_cancer = any(ct in line_lower for ct in KNOWN_CANCER_TYPES)
            has_ref = bool(RE_PART_B_REF.search(line_stripped))
            if len(line_stripped) < 80 and not has_cancer and not has_ref:
                i += 1
                continue
            else:
                in_orphan_note = False

        # 3. Decide whether to flush the current record before adding
        #    this line.  A record is "complete" once it has >= 1 ref.
        #    Only flush when the incoming line is genuinely the start of
        #    a new symptom description.
        if current_record_lines and _record_has_refs():
            # --- Tail-note detection (sticky across continuation lines) ---
            is_tail = False
            if line_lower.startswith(TAIL_PREFIXES):
                is_tail = True
                in_tail_note = True
            elif in_tail_note:
                # Continuation of a multi-line tail note: short line,
                # no cancer type, no new [1.x.y] ref, starts lowercase.
                # Uppercase start signals new content, not a continuation.
                has_cancer = any(ct in line_lower for ct in KNOWN_CANCER_TYPES)
                has_ref = bool(RE_PART_B_REF.search(line_stripped))
                starts_upper = line_stripped[0].isupper() if line_stripped else False
                if (len(line_stripped) < 80 and not has_cancer
                        and not has_ref and not starts_upper):
                    is_tail = True
                else:
                    in_tail_note = False

            if not is_tail:
                in_tail_note = False  # reset for next cycle
                has_ref = bool(RE_PART_B_REF.search(line_stripped))
                is_action = any(
                    line_lower.startswith(v) for v in RECOMMENDATION_VERBS
                )

                if is_action or has_ref:
                    # Additional recommendation action or extra ref for
                    # the current row — keep accumulating.
                    #
                    # Exception: if the line contains a cancer-type keyword
                    # BEFORE its first [1.x.y] ref, it is a brand-new
                    # single-line row (symptom+cancer+ref in one line).
                    if has_ref and not is_action:
                        ref_match = RE_PART_B_REF.search(line_lower)
                        ref_pos = ref_match.start() if ref_match else len(line_lower)
                        cancer_before_ref = any(
                            line_lower.find(ct) != -1
                            and line_lower.find(ct) < ref_pos
                            for ct in KNOWN_CANCER_TYPES
                        )
                        if cancer_before_ref:
                            _flush_record()
                    # else: keep accumulating
                else:
                    # Line has no ref, no action verb, not a tail note
                    # → this is a new symptom description → flush.
                    _flush_record()

        # 4. Accumulate into current record
        if not current_record_lines:
            current_record_page_start = item["page"]
        current_record_lines.append(item)

        i += 1

    # Flush any remaining record
    _flush_record()

    return chunks


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _safe_print(text: str) -> None:
    """Print text with defensive encoding for non-Unicode consoles."""
    print(text.encode("ascii", "replace").decode("ascii"))


def _print_stats(chunks: list[dict]) -> None:
    """Print chunking statistics grouped by doc_type."""
    print(f"\n{'='*60}")
    print("NG12 Chunking Statistics")
    print(f"{'='*60}")
    print(f"Total chunks: {len(chunks)}")

    # Group by doc_type
    doc_type_counter = Counter(
        c["metadata"].get("doc_type", "unknown") for c in chunks
    )
    print("\nDoc type distribution:")
    for dtype, count in doc_type_counter.most_common():
        print(f"  {dtype}: {count}")

    # Action type distribution (rule_canonical only)
    canonical = [c for c in chunks if c["metadata"].get("doc_type") == "rule_canonical"]
    action_counter = Counter(
        c["metadata"].get("action_type", "Unknown") for c in canonical
    )
    print("\nAction type distribution (rule_canonical):")
    for action, count in action_counter.most_common():
        print(f"  {action}: {count}")

    # Urgency distribution
    urgency_counter = Counter(
        c["metadata"].get("urgency", "none") for c in canonical
    )
    print("\nUrgency distribution (rule_canonical):")
    for urg, count in urgency_counter.most_common():
        print(f"  {urg}: {count}")

    # Cancer type distribution
    cancer_counter = Counter(
        c["metadata"].get("cancer_type", "Unknown") for c in canonical
    )
    print("\nCancer type distribution (rule_canonical):")
    for cancer, count in cancer_counter.most_common():
        print(f"  {cancer}: {count}")

    has_age = sum(
        1 for c in canonical
        if "age_min" in c["metadata"] or "age_max" in c["metadata"]
    )
    has_symptoms = sum(
        1 for c in canonical if "symptom_keywords_json" in c["metadata"]
    )
    has_quals = sum(
        1 for c in canonical if "qualifiers_json" in c["metadata"]
    )
    has_risk = sum(
        1 for c in canonical if "risk_factors_json" in c["metadata"]
    )
    print(f"\nChunks with age thresholds: {has_age}")
    print(f"Chunks with symptom keywords: {has_symptoms}")
    print(f"Chunks with qualifiers: {has_quals}")
    print(f"Chunks with risk factors: {has_risk}")

    # Print first 2 examples per doc_type
    for dtype in doc_type_counter:
        group = [c for c in chunks if c["metadata"].get("doc_type") == dtype]
        print(f"\n{'='*60}")
        print(f"First 2 '{dtype}' chunks:")
        print(f"{'='*60}")
        for c in group[:2]:
            print(f"\n--- {c['chunk_id']} ---")
            print(f"Metadata: ", end="")
            # Show all metadata keys except chunk_id (already shown)
            display_meta = {k: v for k, v in c["metadata"].items() if k != "chunk_id"}
            _safe_print(json.dumps(display_meta, ensure_ascii=False))
            display_text = c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"]
            _safe_print(f"Text: {display_text}")
