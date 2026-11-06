"""
Text type and communicative intent predictor for LinguaLabel.

Combines:
1. Syntactic token matcher for interrogatives, requests, and greetings (English + Hindi).
2. Word n-gram intent classifier for ambiguous phrases and declarative statements.
3. Explicit uncertainty boundary (< 0.60).
"""

import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Set, Tuple
from ml.base import BasePredictor, PredictionResult

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

LEXICONS_DIR = Path(__file__).parent / "lexicons"


class TextTypeDetector(BasePredictor):
    """
    Detects text type: Statement, Question, Request / Instruction, Greeting, Other, or Uncertain.
    """

    def __init__(self, uncertainty_threshold: float = 0.60):
        self.uncertainty_threshold = uncertainty_threshold
        self.q_symbols: Set[str] = set()
        self.en_q_starters: Set[str] = set()
        self.hi_q_words: Set[str] = set()
        self.en_greetings: Set[str] = set()
        self.hi_greetings: Set[str] = set()
        self.en_requests: Set[str] = set()
        self.hi_requests: Set[str] = set()
        self.en_complaints: List[str] = []
        self.hi_complaints: List[str] = []
        self.hinglish_complaints: List[str] = []

        self._load_keywords()
        self.classifier: Optional[Any] = None
        self._init_statistical_model()

    def _load_keywords(self):
        kw_path = LEXICONS_DIR / "text_type_keywords.json"
        if kw_path.exists():
            try:
                with open(kw_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    qm = data.get("question_markers", {})
                    self.q_symbols = set(qm.get("symbols", ["?"]))
                    self.en_q_starters = set(qm.get("english_starters", []))
                    self.hi_q_words = set(qm.get("hindi_keywords", []))

                    gm = data.get("greeting_markers", {})
                    self.en_greetings = set(gm.get("english_keywords", []))
                    self.hi_greetings = set(gm.get("hindi_keywords", []))

                    rm = data.get("request_markers", {})
                    self.en_requests = set(rm.get("english_keywords", []))
                    self.hi_requests = set(rm.get("hindi_keywords", []))
            except Exception:
                pass

        # Load complaints lexicon
        comp_path = LEXICONS_DIR / "complaint_lexicon.json"
        if comp_path.exists():
            try:
                with open(comp_path, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                    cm = cdata.get("complaint_markers", {})
                    self.en_complaints = cm.get("english", [])
                    self.hinglish_complaints = cm.get("hinglish", [])
                    self.hi_complaints = cm.get("devanagari", [])
            except Exception:
                pass

    def _init_statistical_model(self):
        if not SKLEARN_AVAILABLE:
            return

        seed_data = [
            ("I am feeling happy today.", "Statement"),
            ("The meeting was held in room 405.", "Statement"),
            ("Today I am feeling sad.", "Statement"),
            ("कल हम सभी एक साथ पिकनिक पर जाने वाले हैं।", "Statement"),
            ("यह खाना बहुत स्वादिष्ट और लजीज था।", "Statement"),
            ("इस समस्या का कोई आसान समाधान नहीं दिख रहा है।", "Statement"),
            ("आज का मौसम बहुत सुहावना है।", "Statement"),
            ("The project deadline has been postponed to next week.", "Statement"),
            ("ट्रेन में बहुत भीड़ थी और यात्रा काफी कठिन रही।", "Statement"),

            ("Your application keeps crashing.", "Complaining"),
            ("I am very disappointed with your service.", "Complaining"),
            ("The delivery was extremely late.", "Complaining"),
            ("आज आपकी सेवा बहुत खराब है।", "Complaining"),
            ("Yaar tumhara app baar baar crash ho raha hai.", "Complaining"),
            ("Ye service bilkul bekar hai, paisa barbad ho gaya.", "Complaining"),
            ("My order never arrived and support is unresponsive.", "Complaining"),
            ("ऐप बार-बार बंद हो रहा है, बहुत खराब अनुभव।", "Complaining"),
            ("App is not working properly, keeps freezing.", "Complaining"),
            ("Worst customer service ever, nobody replies.", "Complaining"),
            ("पैसे कट गए लेकिन टिकट बुक नहीं हुआ।", "Complaining"),

            ("Can you help me with this problem?", "Question"),
            ("Where is the nearest railway station located?", "Question"),
            ("What is your plan for tomorrow evening?", "Question"),
            ("क्या आप मुझे इस सवाल का जवाब बता सकते हैं?", "Question"),
            ("आपकी अगली परीक्षा कब शुरू हो रही है?", "Question"),
            ("How do I reset my account password?", "Question"),
            ("Why was the morning train delayed?", "Question"),

            ("Please fix this problem immediately.", "Request / Instruction"),
            ("Please send the quarterly financial report by Monday.", "Request / Instruction"),
            ("कृपया समय पर अपनी रिपोर्ट जमा करें।", "Request / Instruction"),
            ("कृपया खिड़की बंद कर दीजिए, बहुत ठंड है।", "Request / Instruction"),
            ("Kindly review the attached contract and sign.", "Request / Instruction"),
            ("Make sure to save your work before logging out.", "Request / Instruction"),
            ("कृपया कतार में खड़े होकर अपनी बारी का इंतजार करें।", "Request / Instruction"),
            ("Do not touch the electrical wires.", "Request / Instruction"),

            ("Good morning, have a great day ahead!", "Greeting"),
            ("Hello team, welcome to the new quarter.", "Greeting"),
            ("नमस्ते, आपका दिन मंगलमय हो।", "Greeting"),
            ("Hi everyone, hope you are doing well.", "Greeting"),
            ("नमस्कार, आज के इस विशेष सत्र में आपका स्वागत है।", "Greeting"),
            ("Goodbye, see you all next week!", "Greeting"),
        ]

        texts = [x[0] for x in seed_data]
        labels = [x[1] for x in seed_data]

        try:
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
                ("clf", MultinomialNB(alpha=0.3))
            ])
            pipe.fit(texts, labels)
            self.classifier = pipe
        except Exception:
            self.classifier = None

    def is_ready(self) -> bool:
        return bool(self.en_q_starters or self.hi_q_words)

    def _eval_syntax_rules(self, text: str) -> Tuple[Optional[str], float, dict]:
        clean = text.strip()
        lower = clean.lower()
        words = re.findall(r"[\w']+|[^\s\w]", lower)
        meta = {"matched_pattern": None}

        # 1. Request / Instruction: High priority for explicit polite directives or imperative openers
        # e.g. "Please fix this problem immediately." -> Request / Instruction
        if words and words[0] in ("please", "kindly", "do", "don't"):
            meta["matched_pattern"] = f"en_request_starter_{words[0]}"
            return "Request / Instruction", 0.92, meta

        for r in self.hi_requests:
            if clean.startswith(r) or clean.endswith(r) or f" {r} " in f" {clean} ":
                meta["matched_pattern"] = f"hi_request_{r}"
                return "Request / Instruction", 0.90, meta

        for r in self.en_requests:
            if re.search(r'\b' + re.escape(r) + r'\b', lower):
                meta["matched_pattern"] = f"en_request_{r}"
                return "Request / Instruction", 0.88, meta

        # 2. Greetings: Matches greeting phrases at word boundaries
        for g in self.en_greetings:
            if re.search(r'\b' + re.escape(g) + r'\b', lower):
                meta["matched_pattern"] = f"en_greeting_{g}"
                return "Greeting", 0.92, meta

        for g in self.hi_greetings:
            if g in clean.split() or clean.startswith(g):
                meta["matched_pattern"] = f"hi_greeting_{g}"
                return "Greeting", 0.92, meta

        # 3. Complaining: Matches service, software, product, or delivery complaints
        # Supports English, Hindi, and Hinglish
        for c in self.en_complaints:
            if re.search(r'\b' + re.escape(c) + r'\b', lower) or c in lower:
                meta["matched_pattern"] = f"en_complaint_{c}"
                return "Complaining", 0.92, meta

        for c in self.hinglish_complaints:
            if c in lower or re.search(r'\b' + re.escape(c) + r'\b', lower):
                meta["matched_pattern"] = f"hinglish_complaint_{c}"
                return "Complaining", 0.92, meta

        for c in self.hi_complaints:
            if c in clean:
                meta["matched_pattern"] = f"hi_complaint_{c}"
                return "Complaining", 0.92, meta

        # 4. Question: Contains question mark or interrogative token
        has_q_symbol = any(sym in clean for sym in self.q_symbols)
        if has_q_symbol:
            meta["matched_pattern"] = "question_symbol"
            return "Question", 0.95, meta

        if words:
            first_word = words[0]
            if first_word in self.en_q_starters:
                meta["matched_pattern"] = f"en_starter_{first_word}"
                return "Question", 0.85, meta

        for hq in self.hi_q_words:
            if hq in clean:
                meta["matched_pattern"] = f"hi_interrogative_{hq}"
                return "Question", 0.90, meta

        # 5. Declarative Statement: Ends with period / purna viram or general declarative structure
        # e.g. "Today I am feeling sad." -> Statement (negative sentiment, but communicative intent is Statement)
        if clean.endswith(".") or clean.endswith("।") or len(words) > 3:
            meta["matched_pattern"] = "declarative_structure"
            return "Statement", 0.75, meta

        return None, 0.50, meta

    def predict(self, text: str) -> PredictionResult:
        clean_text = (text or "").strip()
        if not clean_text:
            return PredictionResult(
                label="Uncertain",
                confidence=0.0,
                method="Fallback",
                is_uncertain=True,
                confidence_type="heuristic",
                metadata={"actual_backend": "baseline", "reason": "Empty text"}
            )

        # 1. Syntactic rules
        syn_label, syn_conf, syn_meta = self._eval_syntax_rules(clean_text)
        syn_meta["actual_backend"] = "baseline"

        # High confidence syntactic matches (e.g. ? or explicit request/greeting keywords)
        if syn_label and syn_conf >= 0.85:
            return PredictionResult(
                label=syn_label,
                confidence=syn_conf,
                method="Syntactic Matcher",
                is_uncertain=False,
                confidence_type="heuristic",
                metadata=syn_meta
            )

        # 2. Statistical classifier evaluation
        if self.classifier is not None:
            try:
                probs = self.classifier.predict_proba([clean_text])[0]
                classes = list(self.classifier.classes_)
                best_idx = probs.argmax()
                stat_label = classes[best_idx]
                stat_conf = float(probs[best_idx])

                if syn_label and syn_label == stat_label:
                    final_label = syn_label
                    final_conf = max(syn_conf, stat_conf)
                    method = "Syntax + Intent Classifier"
                    conf_type = "heuristic"
                elif syn_label and syn_conf >= 0.75:
                    final_label = syn_label
                    final_conf = syn_conf
                    method = "Syntactic Matcher"
                    conf_type = "heuristic"
                else:
                    final_label = stat_label
                    final_conf = stat_conf
                    method = "Intent Classifier"
                    conf_type = "model_probability"

                is_uncertain = final_conf < self.uncertainty_threshold
                return PredictionResult(
                    label="Uncertain" if is_uncertain else final_label,
                    confidence=final_conf,
                    method=method,
                    is_uncertain=is_uncertain,
                    confidence_type=conf_type,
                    metadata={"actual_backend": "baseline", "classes": classes, "probabilities": [round(float(p), 4) for p in probs], **syn_meta}
                )
            except Exception:
                pass

        if syn_label:
            is_uncertain = syn_conf < self.uncertainty_threshold
            return PredictionResult(
                label="Uncertain" if is_uncertain else syn_label,
                confidence=syn_conf,
                method="Syntactic Matcher",
                is_uncertain=is_uncertain,
                confidence_type="heuristic",
                metadata=syn_meta
            )

        return PredictionResult(
            label="Uncertain",
            confidence=0.45,
            method="Syntactic Matcher",
            is_uncertain=True,
            confidence_type="heuristic",
            metadata=syn_meta
        )
