"""
Language identification predictor for LinguaLabel.

Combines:
1. Fast, deterministic Unicode script heuristic (Devanagari U+0900..U+097F vs. Latin A..Za..z).
2. Character n-gram TF-IDF + linear classifier for ambiguous or transliterated text.
3. Explicit uncertainty boundary for confidence < 0.60.
"""

import re
from typing import Optional, Tuple
from ml.base import BasePredictor, PredictionResult

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class LanguageDetector(BasePredictor):
    """
    Detects language (English, Hindi, Other, or Uncertain).
    """

    def __init__(self, uncertainty_threshold: float = 0.60):
        self.uncertainty_threshold = uncertainty_threshold
        self.classifier: Optional[Any] = None
        self._init_statistical_model()

    def _init_statistical_model(self):
        """Initializes and pre-fits a lightweight character n-gram pipeline on seed data."""
        if not SKLEARN_AVAILABLE:
            return

        seed_data = [
            ("I am feeling happy today.", "English"),
            ("Please send the quarterly financial report by Monday.", "English"),
            ("Can you help me with this problem?", "English"),
            ("Good morning, have a great day ahead!", "English"),
            ("The quick brown fox jumps over the lazy dog.", "English"),
            ("This product quality is really bad and disappointing.", "English"),
            ("Where is the nearest railway station located?", "English"),
            ("Thank you very much for your kind support.", "English"),
            ("Hello team, please review the attached document.", "English"),
            ("What is your plan for tomorrow evening?", "English"),
            ("आज का मौसम बहुत सुहावना है।", "Hindi"),
            ("कृपया समय पर अपनी रिपोर्ट जमा करें।", "Hindi"),
            ("क्या आप मुझे इस सवाल का जवाब बता सकते हैं?", "Hindi"),
            ("नमस्ते, आपका दिन मंगलमय हो।", "Hindi"),
            ("यह खाना बहुत स्वादिष्ट और लजीज था।", "Hindi"),
            ("ट्रेन में बहुत भीड़ थी और यात्रा काफी कठिन रही।", "Hindi"),
            ("आपकी अगली परीक्षा कब शुरू हो रही है?", "Hindi"),
            ("कल हम सभी एक साथ पिकनिक पर जाने वाले हैं।", "Hindi"),
            ("कृपया खिड़की बंद कर दीजिए, बहुत ठंड है।", "Hindi"),
            ("इस समस्या का कोई आसान समाधान नहीं दिख रहा है।", "Hindi"),
            ("kya aap meri madad kar sakte hain?", "Hindi"),
            ("aaj ka din bahut accha tha", "Hindi"),
            ("Bonjour tout le monde, comment allez-vous?", "Other"),
            ("Hola amigo, como estas hoy?", "Other"),
            ("Guten Tag, wie geht es Ihnen?", "Other"),
            ("Ciao a tutti, buona giornata!", "Other"),
            ("12345 67890", "Other"),
        ]

        texts = [x[0] for x in seed_data]
        labels = [x[1] for x in seed_data]

        try:
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)),
                ("clf", MultinomialNB(alpha=0.5))
            ])
            pipe.fit(texts, labels)
            self.classifier = pipe
        except Exception:
            self.classifier = None

    def is_ready(self) -> bool:
        return True

    def _eval_script_heuristic(self, text: str) -> Tuple[Optional[str], float, dict]:
        """
        Calculates Unicode codepoint frequencies for Devanagari vs Latin characters.
        """
        if not text or not text.strip():
            return None, 0.0, {"devanagari_count": 0, "latin_count": 0, "total_alpha": 0}

        devanagari_count = 0
        latin_count = 0
        other_alpha_count = 0

        for char in text:
            code = ord(char)
            if 0x0900 <= code <= 0x097F:
                devanagari_count += 1
            elif (65 <= code <= 90) or (97 <= code <= 122):
                latin_count += 1
            elif char.isalpha():
                other_alpha_count += 1

        total_alpha = devanagari_count + latin_count + other_alpha_count
        meta = {
            "devanagari_count": devanagari_count,
            "latin_count": latin_count,
            "other_alpha_count": other_alpha_count,
            "total_alpha": total_alpha,
        }

        if total_alpha == 0:
            return "Other", 0.50, meta

        dev_ratio = devanagari_count / total_alpha
        latin_ratio = latin_count / total_alpha

        # Devanagari script presence is a strong, deterministic indicator of Hindi
        if dev_ratio >= 0.50:
            conf = min(0.99, 0.75 + (dev_ratio * 0.24))
            return "Hindi", conf, meta

        # Dominant Latin script: can be English or transliterated Hindi / Other
        if latin_ratio >= 0.75:
            # If we don't have statistical model, default to English with moderate confidence
            conf = min(0.95, 0.70 + (latin_ratio * 0.25))
            return "English", conf, meta

        if other_alpha_count / total_alpha > 0.40:
            return "Other", 0.80, meta

        return None, 0.45, meta

    def predict(self, text: str) -> PredictionResult:
        """
        Predicts language with confidence and provenance.
        """
        clean_text = (text or "").strip()
        if not clean_text:
            return PredictionResult(
                label="Uncertain",
                confidence=0.0,
                method="Fallback",
                is_uncertain=True,
                metadata={"reason": "Empty text"}
            )

        # 1. First run deterministic script heuristic
        script_label, script_conf, script_meta = self._eval_script_heuristic(clean_text)

        # If Devanagari script is clear, return Script Heuristic with high confidence
        if script_label == "Hindi" and script_conf >= 0.80:
            return PredictionResult(
                label="Hindi",
                confidence=script_conf,
                method="Script Heuristic",
                is_uncertain=False,
                metadata=script_meta
            )

        # 2. If statistical classifier is available, evaluate character n-grams
        if self.classifier is not None:
            try:
                probs = self.classifier.predict_proba([clean_text])[0]
                classes = list(self.classifier.classes_)
                best_idx = probs.argmax()
                best_label = classes[best_idx]
                best_conf = float(probs[best_idx])

                # Disambiguate Latin script: English vs transliterated Hindi vs Other
                if script_label == "English" and best_label == "Hindi":
                    # Transliterated Hindi detected by n-gram classifier
                    chosen_label = "Hindi"
                    chosen_conf = best_conf
                    method = "N-Gram Classifier (Transliterated)"
                elif script_label == "English" and best_label == "English":
                    chosen_label = "English"
                    chosen_conf = max(script_conf, best_conf)
                    method = "Script Heuristic"
                else:
                    chosen_label = best_label
                    chosen_conf = best_conf
                    method = "N-Gram Classifier"

                is_uncertain = chosen_conf < self.uncertainty_threshold
                return PredictionResult(
                    label="Uncertain" if is_uncertain else chosen_label,
                    confidence=chosen_conf,
                    method=method,
                    is_uncertain=is_uncertain,
                    metadata={"classes": classes, "probabilities": [round(float(p), 4) for p in probs], **script_meta}
                )
            except Exception:
                pass

        # 3. Fallback to script heuristic result
        if script_label:
            is_uncertain = script_conf < self.uncertainty_threshold
            return PredictionResult(
                label="Uncertain" if is_uncertain else script_label,
                confidence=script_conf,
                method="Script Heuristic",
                is_uncertain=is_uncertain,
                metadata=script_meta
            )

        return PredictionResult(
            label="Uncertain",
            confidence=0.40,
            method="Script Heuristic",
            is_uncertain=True,
            metadata=script_meta
        )
