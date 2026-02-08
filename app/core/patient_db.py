"""
Patient Database

Loads patient records from patients.json.
Provides get_patient() function used as Gemini function calling tool.
Can be run standalone: python -m app.core.patient_db
"""

import json
from pathlib import Path
from typing import Optional

from app.config import settings

_PATIENTS: dict[str, dict] = {}


def _load_patients() -> None:
    """Load patient records from the JSON file into memory."""
    global _PATIENTS
    path = Path(settings.PATIENTS_PATH)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        _PATIENTS = {p["patient_id"]: p for p in records}


def get_patient(patient_id: str) -> Optional[dict]:
    """Look up a patient record by ID.

    Args:
        patient_id: The patient identifier (e.g. "PT-101").

    Returns:
        The patient dict if found, otherwise None.
    """
    if not _PATIENTS:
        _load_patients()
    return _PATIENTS.get(patient_id)


def list_patients() -> list[dict]:
    """Return all patient records.

    Returns:
        A list of all patient dicts.
    """
    if not _PATIENTS:
        _load_patients()
    return list(_PATIENTS.values())


if __name__ == "__main__":
    patients = list_patients()
    print(f"Loaded {len(patients)} patients:")
    for p in patients:
        print(f"  {p['patient_id']}: {p['name']} (age {p['age']})")
