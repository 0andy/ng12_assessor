"""Regression tests for classify_input() in chat_workflow.

Run with:  python -m pytest tests/test_classify_input.py -v
"""

import pytest

from app.agents.chat_workflow import classify_input


# ── Smalltalk ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("msg", [
    "hi",
    "hello",
    "hey!",
    "Hi there",
    "Good morning",
    "thanks",
    "bye",
    "how are you?",
    "how are you doing?",
    "you there?",
    "are you there?",
    "sup",
    "what's up",
    "lol",
    "haha",
    "good thanks",
    "fine thanks",
])
def test_smalltalk(msg: str):
    assert classify_input(msg) == "smalltalk", f"Expected smalltalk for: {msg!r}"


# ── Meta ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("msg", [
    "who are you?",
    "what can you do?",
    "what are you?",
    "how does this work?",
    "tell me about yourself",
    "what do you know?",
    "help",
    "are you a doctor?",
    "can you diagnose cancer for me?",
])
def test_meta(msg: str):
    assert classify_input(msg) == "meta", f"Expected meta for: {msg!r}"


# ── Chitchat redirect ───────────────────────────────────────────────────
@pytest.mark.parametrize("msg", [
    "tell me a joke",
    "what's the weather today?",
    "what time is it now?",
    "do you remember my name?",
    "what kind of AI model are you using?",
    "can you explain how you were built?",
])
def test_chitchat_redirect(msg: str):
    assert classify_input(msg) == "chitchat_redirect", f"Expected chitchat_redirect for: {msg!r}"


# ── Proceed (must NOT be misclassified as smalltalk/chitchat) ────────────
@pytest.mark.parametrize("msg", [
    "hi, does haemoptysis require urgent referral?",
    "how are you — does persistent hoarseness need urgent referral?",
    "what are the referral criteria for lung cancer?",
    "does a breast lump require a 2-week-wait referral?",
    "what age threshold applies for haematuria investigation?",
    "sup, what does NG12 say about dysphagia referral?",
])
def test_proceed_not_misclassified(msg: str):
    result = classify_input(msg)
    assert result == "proceed", (
        f"Expected proceed for: {msg!r}, got {result!r}"
    )


# ── Medical out-of-scope ────────────────────────────────────────────────
@pytest.mark.parametrize("msg", [
    "what chemotherapy is used for lung cancer?",
    "what is the prognosis for stage 4 melanoma?",
    "what is the survival rate for pancreatic cancer?",
])
def test_medical_out_of_scope(msg: str):
    assert classify_input(msg) == "medical_out_of_scope", (
        f"Expected medical_out_of_scope for: {msg!r}"
    )


# ── Safety urgent ─────────────────────────────────────────────────────
@pytest.mark.parametrize("msg", [
    "Should I go to the emergency room right now?",
    "Based on NG12, do I definitely have cancer?",
    "Can you confirm this is cancer and not anxiety?",
    "Tell me whether I should skip seeing a doctor.",
    "Can I treat this myself at home?",
    "Should I call 911?",
    "Can I self-treat this?",
])
def test_safety_urgent(msg: str):
    assert classify_input(msg) == "safety_urgent", (
        f"Expected safety_urgent for: {msg!r}"
    )


# ── Needs clarification ──────────────────────────────────────────────
@pytest.mark.parametrize("msg", [
    "I feel unwell — what should I do?",
    "Something feels wrong in my body.",
    "I've been tired lately, is that cancer?",
    "I'm not feeling well.",
    "I've been feeling sick recently.",
])
def test_needs_clarification(msg: str):
    assert classify_input(msg) == "needs_clarification", (
        f"Expected needs_clarification for: {msg!r}"
    )


# ── Needs clarification should NOT trigger with specific symptoms ──
@pytest.mark.parametrize("msg", [
    "I feel unwell and have rectal bleeding",
    "I'm tired and noticed a lump in my neck",
    "Something feels wrong — I have haematuria",
])
def test_specific_symptoms_bypass_clarification(msg: str):
    result = classify_input(msg)
    assert result != "needs_clarification", (
        f"Should NOT be needs_clarification for: {msg!r}, got {result!r}"
    )
