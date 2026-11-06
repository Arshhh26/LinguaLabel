"""
LinguaLabel — Multilingual Abusive Language & Profanity Detector (Part 11.2)

Classifies text across 7 standardized categories:
1. Not abusive
2. Profanity / Vulgarity
3. Severe Personal Insult
4. Sexualized Abuse
5. Harassment
6. Threat
7. Uncertain

Severity Levels:
- None
- Mild
- Moderate
- Severe

Key Capabilities:
- Explicit detection of mandatory vocabulary: 'madarchod', 'chutiya', 'chutiye', 'gaand', 'gand', 'lund', 'teri maa ki chut'.
- Robust transliteration, leetspeak unmasking, masking, and spacing collapse.
- Context-aware distinction:
  * Product criticism vs. personal insult ('your app is terrible' vs 'you are an idiot')
  * Quoted abuse ('someone called me an idiot')
  * Academic / meta-discussion ('the word madarchod is used in this example')
  * Exclamations vs. targeted attacks ('what the hell' vs 'you asshole')
- Multi-label category extraction with distinct primary category.
"""

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from ml.base import BasePredictor, PredictionResult
from ml.normalization import detect_context_flags, get_normalized_representations


LEXICONS_DIR = Path(__file__).parent / "lexicons"

# Standard Abuse Categories
ABUSE_CATEGORIES = [
    "Not abusive",
    "Profanity / Vulgarity",
    "Severe Personal Insult",
    "Sexualized Abuse",
    "Harassment",
    "Threat",
    "Uncertain",
]

# Standard Severity Levels
SEVERITY_LEVELS = ["None", "Mild", "Moderate", "Severe"]


class AbuseDetector(BasePredictor):
    """
    Multilingual Abusive Language & Profanity Detector for English, Hindi, and Hinglish.
    """

    def __init__(self, uncertainty_threshold: float = 0.60):
        self.uncertainty_threshold = uncertainty_threshold
        self.lexicons: Dict[str, Dict[str, List[str]]] = {}
        self._load_lexicons()

    def _load_lexicons(self):
        p = LEXICONS_DIR / "abuse_lexicon.json"
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    self.lexicons = json.load(f)
            except Exception:
                pass

    def is_ready(self) -> bool:
        return bool(self.lexicons)

    def _match_category_terms(self, text_views: Dict[str, str], cat_key: str) -> List[str]:
        """
        Matches terms of a category across clean, unmasked, reduced, and collapsed text representations.
        """
        cat_data = self.lexicons.get(cat_key, {})
        matched_terms: Set[str] = set()

        all_terms = (
            cat_data.get("english", []) +
            cat_data.get("hinglish", []) +
            cat_data.get("devanagari", [])
        )

        for term in all_terms:
            t_clean = term.strip().lower()
            if not t_clean:
                continue

            # Appropriate word boundary regex pattern (handles Indic script matras safely)
            is_dev = bool(re.search(r"[\u0900-\u097F]", t_clean))
            if is_dev:
                pattern = re.compile(r"(?<![\u0900-\u097F\w])" + re.escape(t_clean) + r"(?![\u0900-\u097F\w])", re.IGNORECASE)
            else:
                pattern = re.compile(r"\b" + re.escape(t_clean) + r"\b", re.IGNORECASE)

            for view_name in ("clean_lower", "unmasked", "reduced", "hinglish_norm"):
                txt = text_views.get(view_name, "")
                if pattern.search(txt) or (" " in t_clean and t_clean in txt):
                    matched_terms.add(term)
                    break

            # Also check collapsed string only for spaced multi-word phrases (e.g. 'madar chod' -> 'madarchod')
            if " " in t_clean:
                term_collapsed = re.sub(r"[^a-z0-9]", "", t_clean)
                if len(term_collapsed) >= 6 and term_collapsed in text_views.get("collapsed", ""):
                    matched_terms.add(term)

        return sorted(list(matched_terms))

    def predict(self, text: str) -> PredictionResult:
        raw_text = (text or "").strip()
        if not raw_text:
            return PredictionResult(
                label="Uncertain",
                confidence=0.0,
                method="Fallback",
                is_uncertain=True,
                confidence_type="heuristic",
                metadata={
                    "actual_backend": "baseline",
                    "is_abusive": False,
                    "primary_category": "Uncertain",
                    "all_categories": [],
                    "severity": "None",
                    "explanation": "Empty text provided.",
                },
            )

        views = get_normalized_representations(raw_text)
        flags = detect_context_flags(raw_text)

        # 1. Scan for terms in each category
        detected_by_cat: Dict[str, List[str]] = {}
        detected_by_cat["threat"] = self._match_category_terms(views, "threat")
        detected_by_cat["sexualized_abuse"] = self._match_category_terms(views, "sexualized_abuse")
        detected_by_cat["severe_personal_insult"] = self._match_category_terms(views, "severe_personal_insult")
        detected_by_cat["harassment"] = self._match_category_terms(views, "harassment")
        detected_by_cat["profanity_vulgarity"] = self._match_category_terms(views, "profanity_vulgarity")

        # Include regex threat flags
        if flags["is_threat"] and ("threat" not in detected_by_cat or not detected_by_cat["threat"]):
            detected_by_cat["threat"].append("explicit_threat_expression")

        # Map internal category keys to official UI categories
        CAT_NAME_MAP = {
            "threat": "Threat",
            "sexualized_abuse": "Sexualized Abuse",
            "severe_personal_insult": "Severe Personal Insult",
            "harassment": "Harassment",
            "profanity_vulgarity": "Profanity / Vulgarity",
        }

        active_categories: List[str] = [
            CAT_NAME_MAP[k] for k, terms in detected_by_cat.items() if terms
        ]
        all_detected_terms: List[str] = [
            term for terms in detected_by_cat.values() for term in terms
        ]

        # ---------------------------------------------------------------------
        # 2. CONTEXTUAL OVERRIDES & REFINEMENTS
        # ---------------------------------------------------------------------

        # A. Meta / Educational / Reporting discussion of profanity
        if flags["is_discussion"]:
            return PredictionResult(
                label="Not abusive",
                confidence=0.92,
                method="Contextual Discourse Analyzer",
                is_uncertain=False,
                confidence_type="heuristic",
                metadata={
                    "actual_backend": "baseline",
                    "is_abusive": False,
                    "primary_category": "Not abusive",
                    "all_categories": [],
                    "severity": "None",
                    "is_discussion": True,
                    "detected_terms": all_detected_terms,
                    "explanation": "Meta-discussion, linguistic analysis, or reporting of abusive terms without targeting individuals.",
                },
            )

        # B. Quoted abuse (the author is reporting what someone else said)
        if flags["is_quoted"]:
            return PredictionResult(
                label="Not abusive",
                confidence=0.88,
                method="Contextual Attribution Analyzer",
                is_uncertain=False,
                confidence_type="heuristic",
                metadata={
                    "actual_backend": "baseline",
                    "is_abusive": False,
                    "primary_category": "Not abusive",
                    "all_categories": active_categories,
                    "severity": "None",
                    "is_quoted": True,
                    "detected_terms": all_detected_terms,
                    "explanation": "Quoted expression reported by author; not an active attack by the speaker.",
                },
            )

        # C. Product criticism vs targeted personal abuse
        # 'Your application is terrible' or 'Ye bakwas app hai' -> Not targeted abuse
        if flags["is_product_criticism"] and not flags["is_threat"] and "Sexualized Abuse" not in active_categories:
            # If mild profanity like 'bakwaas' was used to describe the app
            if "Severe Personal Insult" in active_categories:
                active_categories.remove("Severe Personal Insult")
            if not active_categories or active_categories == ["Profanity / Vulgarity"]:
                return PredictionResult(
                    label="Not abusive",
                    confidence=0.85,
                    method="Target Intent Classifier",
                    is_uncertain=False,
                    confidence_type="heuristic",
                    metadata={
                        "actual_backend": "baseline",
                        "is_abusive": False,
                        "primary_category": "Not abusive",
                        "all_categories": active_categories,
                        "severity": "Mild" if active_categories else "None",
                        "is_product_criticism": True,
                        "detected_terms": all_detected_terms,
                        "explanation": "Frustration or complaint directed at software/service, not an abusive personal attack.",
                    },
                )

        # ---------------------------------------------------------------------
        # 3. NO ABUSE DETECTED
        # ---------------------------------------------------------------------
        if not active_categories:
            return PredictionResult(
                label="Not abusive",
                confidence=0.95,
                method="Multilingual Lexicon & Syntax Matcher",
                is_uncertain=False,
                confidence_type="heuristic",
                metadata={
                    "actual_backend": "baseline",
                    "is_abusive": False,
                    "primary_category": "Not abusive",
                    "all_categories": [],
                    "severity": "None",
                    "explanation": "No abusive language, profanity, or threats detected.",
                },
            )

        # ---------------------------------------------------------------------
        # 4. PRIMARY CATEGORY SELECTION & SEVERITY ASSESSMENT
        # ---------------------------------------------------------------------
        # Severity priority hierarchy: Threat > Sexualized Abuse > Severe Personal Insult > Harassment > Profanity
        priority_order = [
            "Threat",
            "Sexualized Abuse",
            "Severe Personal Insult",
            "Harassment",
            "Profanity / Vulgarity",
        ]

        primary_cat = next((c for c in priority_order if c in active_categories), active_categories[0])
        additional_cats = [c for c in active_categories if c != primary_cat]

        # Determine severity
        severity = "Moderate"
        conf = 0.90

        if primary_cat == "Threat":
            severity = "Severe"
            conf = 0.95
            explanation = "Explicit expression expressing an intent to cause serious harm or violence."
        elif primary_cat == "Sexualized Abuse":
            severity = "Severe"
            conf = 0.95
            explanation = "Explicit sexual profanity or sexualized family-directed abuse."
        elif primary_cat == "Severe Personal Insult":
            if flags["is_targeted_personal"]:
                severity = "Severe"
                conf = 0.92
                explanation = "Direct personal attack intended to demean, humiliate, or insult the individual."
            else:
                severity = "Moderate"
                conf = 0.85
                explanation = "Derogatory personal insult or disparaging remark."
        elif primary_cat == "Harassment":
            severity = "Moderate"
            conf = 0.88
            explanation = "Targeted, hostile, or harassing communication."
        else:  # Profanity / Vulgarity
            # Differentiate mild exclamations ('what the hell') from aggressive profanity
            lower_text = raw_text.lower()
            if any(m in lower_text for m in ["what the hell", "damn", "crap", "bakwaas"]):
                severity = "Mild"
                conf = 0.82
                explanation = "Mild profanity or colloquial vulgar expression used as an exclamation."
            else:
                severity = "Moderate"
                conf = 0.88
                explanation = "Vulgar or profane language detected."

        return PredictionResult(
            label=primary_cat,
            confidence=conf,
            method="Multilingual Abuse & Profanity Detector",
            is_uncertain=False,
            confidence_type="heuristic",
            metadata={
                "actual_backend": "baseline",
                "is_abusive": True,
                "primary_category": primary_cat,
                "additional_categories": additional_cats,
                "all_categories": active_categories,
                "severity": severity,
                "detected_terms": all_detected_terms,
                "is_threat": flags["is_threat"],
                "is_targeted_personal": flags["is_targeted_personal"],
                "explanation": explanation,
            },
        )
