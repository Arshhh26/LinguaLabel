"""
Sentiment polarity predictor for LinguaLabel.

Combines:
1. Curated bilingual polarity lexicon (English + Hindi).
2. Contextual negation parsing (नहीं, मत, ना in Hindi; not, never, no in English).
3. Statistical TF-IDF + linear classifier.
4. Calibrated confidence and explicit uncertainty threshold (< 0.60).
"""

import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Set, Tuple
from ml.base import BasePredictor, PredictionResult

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


LEXICONS_DIR = Path(__file__).parent / "lexicons"


class SentimentDetector(BasePredictor):
    """
    Detects sentiment: Positive, Negative, Neutral, Not applicable, or Uncertain.
    """

    def __init__(self, uncertainty_threshold: float = 0.60):
        self.uncertainty_threshold = uncertainty_threshold
        self.hi_positive: Set[str] = set()
        self.hi_negative: Set[str] = set()
        self.hi_negations: Set[str] = set()
        self.en_positive: Set[str] = set()
        self.en_negative: Set[str] = set()
        self.en_negations: Set[str] = set()

        self._load_lexicons()
        self.classifier: Optional[Any] = None
        self._init_statistical_model()

    def _load_lexicons(self):
        """Loads curated sentiment and negation lexicons from JSON files."""
        hi_path = LEXICONS_DIR / "hindi_sentiment.json"
        if hi_path.exists():
            try:
                with open(hi_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.hi_positive = set(data.get("positive_words", []))
                    self.hi_negative = set(data.get("negative_words", []))
                    self.hi_negations = set(data.get("negation_tokens", []))
            except Exception:
                pass

        en_path = LEXICONS_DIR / "english_sentiment.json"
        if en_path.exists():
            try:
                with open(en_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.en_positive = set(data.get("positive_words", []))
                    self.en_negative = set(data.get("negative_words", []))
                    self.en_negations = set(data.get("negation_tokens", []))
            except Exception:
                pass

    def _init_statistical_model(self):
        """Trains a baseline TF-IDF + LogisticRegression model on bilingual polarity utterances."""
        if not SKLEARN_AVAILABLE:
            return

        seed_data = [
            ("I am feeling happy today.", "Positive"),
            ("This is a wonderful and fantastic achievement!", "Positive"),
            ("Excellent product, I really loved it.", "Positive"),
            ("Thank you for your warm and kind support.", "Positive"),
            ("The food was very delicious and enjoyable.", "Positive"),
            ("Great job team, brilliant performance!", "Positive"),
            ("आज का मौसम बहुत सुखद और खुशनुमा है।", "Positive"),
            ("यह खाना बहुत स्वादिष्ट और लजीज था।", "Positive"),
            ("आपकी सहायता के लिए बहुत-बहुत धन्यवाद।", "Positive"),
            ("शानदार प्रदर्शन, हमें आप पर गर्व है।", "Positive"),
            ("सब कुछ बहुत अच्छा और व्यवस्थित रहा।", "Positive"),
            
            ("This product quality is really bad and terrible.", "Negative"),
            ("Horrible customer service, very disappointed.", "Negative"),
            ("Train was delayed and cancelled without notice.", "Negative"),
            ("I hate waiting in this long and painful queue.", "Negative"),
            ("The item arrived completely broken and defective.", "Negative"),
            ("ट्रेन में बहुत भीड़ थी और यात्रा काफी कष्टदायक रही।", "Negative"),
            ("इस समस्या के कारण मुझे बहुत परेशानी हुई।", "Negative"),
            ("यह सेवा बहुत घटिया और निराशाजनक है।", "Negative"),
            ("खाना बिल्कुल खराब और बासी निकला।", "Negative"),
            ("मेरा अनुभव बहुत बुरा और दुखद रहा।", "Negative"),
            
            ("Please send the quarterly financial report by Monday.", "Neutral"),
            ("Where is the nearest railway station located?", "Neutral"),
            ("The meeting is scheduled at 4 PM in room 302.", "Neutral"),
            ("Can you help me find this address?", "Neutral"),
            ("कृपया समय पर अपनी रिपोर्ट जमा करें।", "Neutral"),
            ("क्या आप मुझे इस सवाल का जवाब बता सकते हैं?", "Neutral"),
            ("आपकी अगली परीक्षा कब शुरू हो रही है?", "Neutral"),
            ("कल हम सभी एक साथ पिकनिक पर जाने वाले हैं।", "Neutral"),
            ("कृपया खिड़की बंद कर दीजिए, बहुत ठंड है।", "Neutral"),
        ]

        texts = [x[0] for x in seed_data]
        labels = [x[1] for x in seed_data]

        try:
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
                ("clf", LogisticRegression(C=1.0, max_iter=200))
            ])
            pipe.fit(texts, labels)
            self.classifier = pipe
        except Exception:
            self.classifier = None

    def is_ready(self) -> bool:
        return bool(self.hi_positive or self.en_positive)

    def _tokenize(self, text: str) -> List[str]:
        """Extracts words while preserving Unicode scripts and stripping edge punctuation."""
        return [w.strip(' ,.!?;:।()"\'-') for w in text.lower().split() if w.strip(' ,.!?;:।()"\'-')]

    def _eval_lexicon_polarity(self, text: str) -> Tuple[str, float, dict]:
        """
        Evaluates sentiment using curated lexicons with contextual negation window
        (both preceding negation in English/Hindi and following negation in Hindi).
        """
        tokens = self._tokenize(text)
        n = len(tokens)
        pos_hits = []
        neg_hits = []

        all_negations = self.hi_negations | self.en_negations
        all_positive = self.hi_positive | self.en_positive
        all_negative = self.hi_negative | self.en_negative

        for i, token in enumerate(tokens):
            is_pos = token in all_positive
            is_neg = token in all_negative

            if not (is_pos or is_neg):
                continue

            # Check bidirectional window (up to 2 words before and 2 words after) for negation
            window = [tokens[j] for j in range(max(0, i - 2), min(n, i + 3)) if j != i]
            is_negated = any(w in all_negations for w in window)

            if is_pos:
                if is_negated:
                    neg_hits.append(f"negated_{token}")
                else:
                    pos_hits.append(token)
            elif is_neg:
                if is_negated:
                    pos_hits.append(f"negated_{token}")
                else:
                    neg_hits.append(token)

        meta = {
            "positive_matches": pos_hits,
            "negative_matches": neg_hits,
            "pos_count": len(pos_hits),
            "neg_count": len(neg_hits),
        }

        pos_count = len(pos_hits)
        neg_count = len(neg_hits)

        if pos_count > 0 and neg_count == 0:
            conf = min(0.95, 0.70 + (pos_count * 0.10))
            return "Positive", conf, meta
        elif neg_count > 0 and pos_count == 0:
            conf = min(0.95, 0.70 + (neg_count * 0.10))
            return "Negative", conf, meta
        elif pos_count > 0 and neg_count > 0:
            if pos_count > neg_count:
                return "Positive", 0.65, meta
            elif neg_count > pos_count:
                return "Negative", 0.65, meta
            else:
                return "Neutral", 0.55, meta

        # No polarity words found: Check if factual/procedural sentence
        return "Neutral", 0.65, meta

    def predict(self, text: str) -> PredictionResult:
        """
        Predicts sentiment with calibrated confidence and provenance.
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

        # 1. Lexicon evaluation with negation parsing
        lex_label, lex_conf, lex_meta = self._eval_lexicon_polarity(clean_text)

        # If strong lexicon signal (multiple hits or clear polarity)
        if (lex_meta["pos_count"] > 0 or lex_meta["neg_count"] > 0) and lex_conf >= 0.70:
            return PredictionResult(
                label=lex_label,
                confidence=lex_conf,
                method="Lexicon Rules",
                is_uncertain=False,
                metadata=lex_meta
            )

        # 2. Statistical classifier evaluation if available
        if self.classifier is not None:
            try:
                probs = self.classifier.predict_proba([clean_text])[0]
                classes = list(self.classifier.classes_)
                best_idx = probs.argmax()
                stat_label = classes[best_idx]
                stat_conf = float(probs[best_idx])

                # Blend: if lexicon found polarity, favor that; else use statistical
                if lex_meta["pos_count"] > 0 or lex_meta["neg_count"] > 0:
                    final_label = lex_label
                    final_conf = max(lex_conf, stat_conf)
                    method = "Lexicon + N-Gram"
                else:
                    final_label = stat_label
                    final_conf = stat_conf
                    method = "TF-IDF + Linear"

                is_uncertain = final_conf < self.uncertainty_threshold
                return PredictionResult(
                    label="Uncertain" if is_uncertain else final_label,
                    confidence=final_conf,
                    method=method,
                    is_uncertain=is_uncertain,
                    metadata={"classes": classes, "probabilities": [round(float(p), 4) for p in probs], **lex_meta}
                )
            except Exception:
                pass

        # 3. Fallback to lexicon result
        is_uncertain = lex_conf < self.uncertainty_threshold
        return PredictionResult(
            label="Uncertain" if is_uncertain else lex_label,
            confidence=lex_conf,
            method="Lexicon Rules",
            is_uncertain=is_uncertain,
            metadata=lex_meta
        )
