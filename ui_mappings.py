"""
LinguaLabel — User Experience & Plain-Language Mappings (Part 10 & 10.1)

Provides:
- Reversible bidirectional mappings between beginner-friendly display labels
  and the canonical internal LinguaLabel schema.
- Plain-language confidence descriptions replacing technical percentages,
  properly distinguishing heuristic scores, uncalibrated model probabilities,
  and fallback predictions.
- Natural-language explanation generator grounded strictly in empirical model predictions,
  supporting English, Hindi, Hinglish, punctuation-only inputs, and explicit sarcasm caveats.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from ml.base import PredictionResult


# =============================================================================
# REVERSIBLE LABEL MAPPINGS
# =============================================================================

# Language options
DISPLAY_LANGUAGE_OPTIONS = ["English", "Hindi", "Hinglish", "Other", "Not sure"]

DISPLAY_TO_INTERNAL_LANGUAGE: Dict[str, str] = {
    "English": "English",
    "Hindi": "Hindi",
    "Hinglish": "Hinglish",
    "Other": "Other",
    "Not sure": "Uncertain",
    "— Select —": "— Select —",
    "": "",
}

INTERNAL_TO_DISPLAY_LANGUAGE: Dict[str, str] = {
    "English": "English",
    "Hindi": "Hindi",
    "Hinglish": "Hinglish",
    "Other": "Other",
    "Uncertain": "Not sure",
    "— Select —": "— Select —",
    "": "— Select —",
}

# Sentiment options
DISPLAY_SENTIMENT_OPTIONS = ["Positive", "Negative", "Neutral", "Not applicable", "Not sure"]

DISPLAY_TO_INTERNAL_SENTIMENT: Dict[str, str] = {
    "Positive": "Positive",
    "Negative": "Negative",
    "Neutral": "Neutral",
    "Not applicable": "Not applicable",
    "Not sure": "Uncertain",
    "— Select —": "— Select —",
    "": "",
}

INTERNAL_TO_DISPLAY_SENTIMENT: Dict[str, str] = {
    "Positive": "Positive",
    "Negative": "Negative",
    "Neutral": "Neutral",
    "Not applicable": "Not applicable",
    "Uncertain": "Not sure",
    "— Select —": "— Select —",
    "": "— Select —",
}

# Text-Type / Intent options
# 1-to-1 bijective mapping with LinguaLabel canonical schema to prevent information loss
DISPLAY_TEXT_TYPE_OPTIONS = [
    "Asking a question",
    "Sharing an opinion or statement",
    "Making a request or instruction",
    "Greeting",
    "Complaining or reporting an issue",
    "Other",
    "Not sure",
]

DISPLAY_TO_INTERNAL_TEXT_TYPE: Dict[str, str] = {
    "Asking a question": "Question",
    "Sharing an opinion or statement": "Statement",
    "Making a request or instruction": "Request / Instruction",
    "Greeting": "Greeting",
    "Complaining or reporting an issue": "Complaining",
    "Complaining": "Complaining",
    "Complaint": "Complaining",
    "Other": "Other",
    "Not sure": "Uncertain",
    "— Select —": "— Select —",
    "": "",
    # Aliases for backward compatibility
    "Sharing an opinion": "Statement",
    "Giving information": "Statement",
    "Making a request": "Request / Instruction",
    "Giving instructions": "Request / Instruction",
    "Question": "Question",
    "Statement": "Statement",
    "Request / Instruction": "Request / Instruction",
}

INTERNAL_TO_DISPLAY_TEXT_TYPE: Dict[str, str] = {
    "Question": "Asking a question",
    "Statement": "Sharing an opinion or statement",
    "Request / Instruction": "Making a request or instruction",
    "Greeting": "Greeting",
    "Complaining": "Complaining or reporting an issue",
    "Complaint": "Complaining or reporting an issue",
    "Other": "Other",
    "Uncertain": "Not sure",
    "— Select —": "— Select —",
    "": "— Select —",
}

# =============================================================================
# ABUSIVE LANGUAGE & PROFANITY MAPPINGS (Part 11.2)
# =============================================================================

DISPLAY_ABUSE_OPTIONS = [
    "Not abusive",
    "Profanity / Vulgarity",
    "Severe Personal Insult",
    "Sexualized Abuse",
    "Harassment",
    "Threat",
    "Not sure",
]

DISPLAY_TO_INTERNAL_ABUSE: Dict[str, str] = {
    "Not abusive": "Not abusive",
    "Profanity / Vulgarity": "Profanity / Vulgarity",
    "Severe Personal Insult": "Severe Personal Insult",
    "Sexualized Abuse": "Sexualized Abuse",
    "Harassment": "Harassment",
    "Threat": "Threat",
    "Not sure": "Uncertain",
    "— Select —": "— Select —",
    "": "",
}

INTERNAL_TO_DISPLAY_ABUSE: Dict[str, str] = {
    "Not abusive": "Not abusive",
    "Profanity / Vulgarity": "Profanity / Vulgarity",
    "Severe Personal Insult": "Severe Personal Insult",
    "Sexualized Abuse": "Sexualized Abuse",
    "Harassment": "Harassment",
    "Threat": "Threat",
    "Uncertain": "Not sure",
    "— Select —": "— Select —",
    "": "— Select —",
}

DISPLAY_SEVERITY_OPTIONS = [
    "None",
    "Mild",
    "Moderate",
    "Severe",
]

DISPLAY_TO_INTERNAL_SEVERITY: Dict[str, str] = {
    "None": "None",
    "Mild": "Mild",
    "Moderate": "Moderate",
    "Severe": "Severe",
    "— Select —": "— Select —",
    "": "",
}

INTERNAL_TO_DISPLAY_SEVERITY: Dict[str, str] = {
    "None": "None",
    "Mild": "Mild",
    "Moderate": "Moderate",
    "Severe": "Severe",
    "— Select —": "— Select —",
    "": "— Select —",
}

# Complaining Detection mappings (Part 11.4)
DISPLAY_COMPLAINT_OPTIONS = [
    "Complaining",
    "Not complaining",
    "Not sure",
]

DISPLAY_TO_INTERNAL_COMPLAINT: Dict[str, str] = {
    "Complaining": "Complaining",
    "Not complaining": "Not complaining",
    "Complaining or reporting an issue": "Complaining",
    "Complaint": "Complaining",
    "Non-complaint": "Not complaining",
    "Not sure": "Uncertain",
    "— Select —": "— Select —",
    "": "",
}

INTERNAL_TO_DISPLAY_COMPLAINT: Dict[str, str] = {
    "Complaining": "Complaining",
    "Not complaining": "Not complaining",
    "Complaint": "Complaining",
    "Uncertain": "Not sure",
    "— Select —": "— Select —",
    "": "— Select —",
}

# Review Status options
DISPLAY_REVIEW_STATUS_OPTIONS = [
    "Reviewed & approved",
    "Needs second review",
    "Unclear / Ambiguous",
    "Skip for now",
]

DISPLAY_TO_INTERNAL_REVIEW_STATUS: Dict[str, str] = {
    "Reviewed & approved": "Reviewed — looks correct",
    "Needs second review": "Needs second review",
    "Unclear / Ambiguous": "Ambiguous",
    "Skip for now": "Skip",
    "— Select —": "— Select —",
    "Pending review": "",
    "": "",
}

INTERNAL_TO_DISPLAY_REVIEW_STATUS: Dict[str, str] = {
    "Reviewed — looks correct": "Reviewed & approved",
    "Needs second review": "Needs second review",
    "Ambiguous": "Unclear / Ambiguous",
    "Skip": "Skip for now",
    "— Select —": "— Select —",
    "": "— Select —",
}


def to_internal_label(task: str, display_label: str) -> str:
    """Converts a user-friendly display label to the canonical internal schema label."""
    dl = str(display_label).strip()
    if task == "language":
        return DISPLAY_TO_INTERNAL_LANGUAGE.get(dl, dl)
    elif task == "sentiment":
        return DISPLAY_TO_INTERNAL_SENTIMENT.get(dl, dl)
    elif task == "text_type":
        return DISPLAY_TO_INTERNAL_TEXT_TYPE.get(dl, dl)
    elif task in ("complaint", "complaining"):
        return DISPLAY_TO_INTERNAL_COMPLAINT.get(dl, dl)
    elif task in ("abuse", "abuse_category"):
        return DISPLAY_TO_INTERNAL_ABUSE.get(dl, dl)
    elif task in ("abuse_severity", "severity"):
        return DISPLAY_TO_INTERNAL_SEVERITY.get(dl, dl)
    elif task == "review_status":
        return DISPLAY_TO_INTERNAL_REVIEW_STATUS.get(dl, dl)
    return dl


def to_display_label(task: str, internal_label: str) -> str:
    """Converts a canonical internal schema label to the user-friendly display label."""
    il = str(internal_label).strip()
    if task == "language":
        return INTERNAL_TO_DISPLAY_LANGUAGE.get(il, il if il else "— Select —")
    elif task == "sentiment":
        return INTERNAL_TO_DISPLAY_SENTIMENT.get(il, il if il else "— Select —")
    elif task == "text_type":
        return INTERNAL_TO_DISPLAY_TEXT_TYPE.get(il, il if il else "— Select —")
    elif task in ("complaint", "complaining"):
        return INTERNAL_TO_DISPLAY_COMPLAINT.get(il, il if il else "— Select —")
    elif task in ("abuse", "abuse_category"):
        return INTERNAL_TO_DISPLAY_ABUSE.get(il, il if il else "— Select —")
    elif task in ("abuse_severity", "severity"):
        return INTERNAL_TO_DISPLAY_SEVERITY.get(il, il if il else "— Select —")
    elif task == "review_status":
        return INTERNAL_TO_DISPLAY_REVIEW_STATUS.get(il, il if il else "— Select —")
    return il if il else "— Select —"


# =============================================================================
# SIMPLIFIED CONFIDENCE TRANSLATIONS & AUDIT
# =============================================================================

def simplify_confidence(pred: Optional[PredictionResult]) -> Tuple[str, str, str]:
    """
    Translates statistical confidence and backend metadata into beginner-friendly guidance:
    - High confidence (>= 0.80): "AI is fairly sure."
    - Medium confidence (0.60 to 0.79): "AI is somewhat unsure."
    - Low confidence (< 0.60 or is_uncertain): "This sentence needs your attention."
    - Fallback: Explicitly identifies fallback predictions from failed transformers.

    Distinguishes heuristic scores, uncalibrated model probabilities, and calibrated probabilities.
    Returns: (display_text, badge_color_hex, detail_tooltip)
    """
    if pred is None:
        return "This sentence needs your attention.", "#ef4444", "No prediction available; manual review required."

    conf = float(pred.confidence) if pred.confidence is not None else 0.0
    conf_type = getattr(pred, "confidence_type", "heuristic") or "heuristic"
    is_unc = bool(getattr(pred, "is_uncertain", False) or conf < 0.60)
    is_fallback = bool(
        getattr(pred, "actual_backend", "") == "fallback_baseline"
        or (pred.metadata and pred.metadata.get("is_fallback"))
    )

    # Ground tooltip in confidence type honesty
    if conf_type == "heuristic":
        score_desc = f"Heuristic rule score ({conf:.0%}); not a calibrated statistical probability."
    elif conf_type == "model_probability":
        score_desc = f"Uncalibrated model output probability ({conf:.0%}); not a calibrated statistical certainty."
    elif conf_type == "calibrated_probability":
        score_desc = f"Empirically calibrated probability ({conf:.0%})."
    else:
        score_desc = f"Advisory confidence score ({conf:.0%})."

    if is_fallback:
        return (
            "AI fallback (needs review)",
            "#f59e0b" if conf >= 0.70 else "#ef4444",
            f"Fallback baseline prediction ({score_desc}) used because transformer was unavailable or errored. Human review recommended."
        )

    if is_unc or conf < 0.60:
        return (
            "This sentence needs your attention.",
            "#ef4444",
            f"{score_desc} Confidence is below the 60% safety threshold; please verify."
        )
    elif conf >= 0.80:
        return (
            "AI is fairly sure.",
            "#22c55e",
            f"{score_desc} High model certainty based on clear linguistic markers."
        )
    else:
        return (
            "AI is somewhat unsure.",
            "#f59e0b",
            f"{score_desc} Moderate model certainty; quick human check suggested."
        )


# =============================================================================
# PLAIN-LANGUAGE EXPLANATION GENERATOR
# =============================================================================

def generate_sentence_explanation(
    text: str,
    preds: Dict[str, PredictionResult],
) -> str:
    """
    Constructs a plain-language explanation of what the sentence expresses,
    strictly grounded in the model's actual predictions.
    
    Handles:
    - Multilingual inputs (Hindi, English, Hinglish, Other).
    - Punctuation-only inputs (e.g. '???').
    - Negation structures.
    - Sarcasm/irony detection caveat (never claims reliable sarcasm classification).
    """
    if not preds or not text or not str(text).strip():
        return "Enter or select a sentence to see an AI explanation."

    clean_text = str(text).strip()

    # Check for punctuation-only inputs
    if not re.search(r"[\w\u0900-\u097F]", clean_text):
        return "This input consists only of punctuation symbols without text words; please review and clarify."

    p_lang = preds.get("language")
    p_sent = preds.get("sentiment")
    p_tt = preds.get("text_type")

    lang_lbl = p_lang.label if p_lang else "Unknown"
    sent_lbl = p_sent.label if p_sent else "Unknown"
    tt_lbl = p_tt.label if p_tt else "Unknown"

    # Language insight
    if lang_lbl == "Hindi":
        lang_str = "This sentence is in Hindi"
    elif lang_lbl == "Hinglish":
        lang_str = "This sentence is in Hinglish (code-mixed Hindi-English)"
    elif lang_lbl == "English":
        lang_str = "This sentence is in English"
    else:
        lang_str = "This sentence"

    # Intent insight
    if tt_lbl == "Question":
        intent_str = "asking a question"
    elif tt_lbl in ("Request / Instruction", "Request", "Instruction"):
        intent_str = "making a request or giving an instruction"
    elif tt_lbl == "Greeting":
        intent_str = "offering a greeting or conversational remark"
    elif tt_lbl in ("Complaining", "Complaint"):
        intent_str = "expressing a complaint or reporting an issue"
    elif tt_lbl == "Statement":
        intent_str = "stating information or sharing an opinion"
    else:
        intent_str = "expressing text"

    # Sentiment insight
    if sent_lbl == "Positive":
        sent_str = "with a positive, satisfied tone."
    elif sent_lbl == "Negative":
        sent_str = "expressing dissatisfaction, frustration, or a complaint."
    elif sent_lbl == "Neutral":
        sent_str = "without strong emotional tone (neutral statement)."
    elif sent_lbl == "Not applicable":
        sent_str = "where emotional sentiment does not apply."
    else:
        sent_str = "with uncertain emotional tone."

    explanation = f"{lang_str}, {intent_str} {sent_str}"

    # Sarcasm / Irony detection warning (rule: never claim reliable sarcasm detection)
    text_lower = clean_text.lower()
    has_positive_marker = any(w in text_lower for w in ["great", "love", "wonderful", "perfect", "awesome", "fantastic"])
    has_negative_marker = any(w in text_lower for w in ["delay", "terrible", "waste", "slow", "broken", "worst", "hate", "problem"])
    
    if has_positive_marker and has_negative_marker:
        explanation += " Note: Potential irony or contrasting cues detected; AI models cannot reliably detect sarcasm without human judgment."

    # Negation notice
    has_negation = bool(re.search(r"\b(don't|dont|not|never|didn't|didnt|nahi|nahin|nhi|bilkul nahi)\b", text_lower))
    if has_negation and sent_lbl == "Negative":
        explanation += " (Negation cues present)."

    # Abusive language notice
    p_abuse = preds.get("abuse")
    if p_abuse and p_abuse.label != "Not abusive":
        sev = p_abuse.metadata.get("severity", "Moderate") if p_abuse.metadata else "Moderate"
        explanation += f" ⚠️ Content warning: AI detected {p_abuse.label.lower()} ({sev.lower()} severity)."

    return explanation
