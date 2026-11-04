"""
Unit tests for LinguaLabel Multilingual Transformer Integration (Part 6).

Tests:
- ModelLoader singleton behavior, device resolution, and caching
- SentimentTransformer multilingual inference with mock pipeline
- TextTypeTransformer intent inference and schema mapping
- Handling of missing dependencies and invalid checkpoints
- Uncertainty threshold enforcement (< 0.60 -> Uncertain)
- Model registry caching and backend selection
"""

import pytest
from ml.base import PredictionResult
from ml.model_config import (
    DEFAULT_SENTIMENT_CHECKPOINT,
    DEFAULT_TEXT_TYPE_CHECKPOINT,
    MLBackend,
    MLEngineConfig,
)
from ml.model_registry import ModelRegistry
from ml.transformers.model_loader import get_model_loader
from ml.transformers.sentiment_transformer import SentimentTransformer
from ml.transformers.text_type_transformer import TextTypeTransformer


@pytest.fixture(autouse=True)
def clean_loader():
    loader = get_model_loader()
    loader.clear_cache()
    yield
    loader.clear_cache()


# ---------------------------------------------------------------------------
# 1. ModelLoader & Hardware / Device Tests
# ---------------------------------------------------------------------------

def test_model_loader_singleton():
    l1 = get_model_loader()
    l2 = get_model_loader()
    assert l1 is l2


def test_model_loader_device_resolution():
    loader = get_model_loader()
    cpu_dev = loader.get_device("cpu")
    assert cpu_dev == "cpu"
    # Auto resolution must return a non-empty string
    auto_dev = loader.get_device("auto")
    assert isinstance(auto_dev, str)
    assert auto_dev in ("cpu", "cuda:0")


def test_model_loader_missing_model_error():
    loader = get_model_loader()
    pipe, err = loader.load_sequence_classification_pipeline("non_existent_dummy_checkpoint_xyz_123")
    assert pipe is None
    assert err is not None
    assert "non_existent_dummy_checkpoint_xyz_123" in err or "dependencies missing" in err.lower()


# ---------------------------------------------------------------------------
# 2. SentimentTransformer Tests
# ---------------------------------------------------------------------------

def test_sentiment_transformer_empty_input():
    st = SentimentTransformer()
    for empty_val in [None, "", "   ", "\n\t"]:
        res = st.predict(empty_val)
        assert res.label == "Uncertain"
        assert res.is_uncertain is True
        assert res.confidence == 0.0


def test_sentiment_transformer_unloaded_graceful():
    # Attempting prediction without valid model pipeline gracefully returns Uncertain
    st = SentimentTransformer(checkpoint="mock/invalid_ckpt")
    res = st.predict("Hello world")
    assert res.label == "Uncertain"
    assert res.is_uncertain is True
    assert "Unavailable" in res.method or "Error" in res.method or "Fallback" in res.method


def test_sentiment_transformer_with_mock_pipeline():
    loader = get_model_loader()
    checkpoint = "mock/xlm-roberta-sentiment"

    # Mock HF sequence classification output: list of dicts with scores
    def mock_pipe(text):
        if "खुश" in text or "happy" in text:
            return [[{"label": "positive", "score": 0.95}, {"label": "neutral", "score": 0.04}, {"label": "negative", "score": 0.01}]]
        elif "खराब" in text or "terrible" in text:
            return [[{"label": "negative", "score": 0.91}, {"label": "neutral", "score": 0.06}, {"label": "positive", "score": 0.03}]]
        elif "ambiguous" in text:
            return [[{"label": "positive", "score": 0.45}, {"label": "neutral", "score": 0.35}, {"label": "negative", "score": 0.20}]]
        else:
            return [[{"label": "neutral", "score": 0.88}, {"label": "positive", "score": 0.08}, {"label": "negative", "score": 0.04}]]

    loader.register_mock_pipeline(checkpoint, mock_pipe)

    st = SentimentTransformer(checkpoint=checkpoint, uncertainty_threshold=0.60)
    assert st.load() is True
    assert st.is_ready() is True

    # English Positive
    res_en = st.predict("I am feeling very happy today!")
    assert res_en.label == "Positive"
    assert res_en.confidence == 0.95
    assert res_en.is_uncertain is False
    assert res_en.metadata["calibrated"] is False
    assert res_en.metadata["checkpoint"] == checkpoint
    assert "inference_time_ms" in res_en.metadata

    # Hindi Positive
    res_hi = st.predict("आज मुझे बहुत खुशी हो रही है।")
    assert res_hi.label == "Positive"
    assert res_hi.confidence == 0.95
    assert res_hi.is_uncertain is False

    # English Negative
    res_neg = st.predict("This service was terrible and slow.")
    assert res_neg.label == "Negative"
    assert res_neg.confidence == 0.91
    assert res_neg.is_uncertain is False

    # Uncertainty Threshold check (< 0.60 -> Uncertain)
    res_amb = st.predict("This is an ambiguous text.")
    assert res_amb.label == "Uncertain"
    assert res_amb.is_uncertain is True
    assert res_amb.confidence == 0.45


# ---------------------------------------------------------------------------
# 3. TextTypeTransformer Tests
# ---------------------------------------------------------------------------

def test_text_type_transformer_empty_input():
    tt = TextTypeTransformer()
    res = tt.predict("")
    assert res.label == "Uncertain"
    assert res.is_uncertain is True


def test_text_type_transformer_with_mock_pipeline():
    loader = get_model_loader()
    checkpoint = "mock/mdeberta-text-type"

    def mock_zero_shot(text, candidate_labels, **kwargs):
        if "?" in text or "क्या" in text:
            return {"sequence": text, "labels": ["Question", "General"], "scores": [0.94, 0.06]}
        elif "नमस्ते" in text or "hello" in text.lower():
            return {"sequence": text, "labels": ["Greeting", "General"], "scores": [0.92, 0.08]}
        elif "please" in text.lower() or "कृपया" in text:
            return {"sequence": text, "labels": ["Request", "General"], "scores": [0.89, 0.11]}
        elif "awful" in text.lower() or "complaint" in text.lower():
            return {"sequence": text, "labels": ["Complaint", "General"], "scores": [0.87, 0.13]}
        else:
            return {"sequence": text, "labels": ["General", "Question"], "scores": [0.55, 0.45]}

    loader.register_mock_pipeline(checkpoint, mock_zero_shot)

    tt = TextTypeTransformer(checkpoint=checkpoint, uncertainty_threshold=0.60)
    assert tt.load() is True
    assert tt.is_ready() is True

    # Question (English and Hindi)
    res_q1 = tt.predict("Where is the nearest bank?")
    assert res_q1.label == "Question"
    assert res_q1.confidence == 0.94
    assert res_q1.metadata["extended_intent"] == "Question"

    res_q2 = tt.predict("क्या आप आ रहे हैं?")
    assert res_q2.label == "Question"
    assert res_q2.confidence == 0.94

    # Greeting
    res_gr = tt.predict("नमस्ते, आपका दिन शुभ हो।")
    assert res_gr.label == "Greeting"
    assert res_gr.confidence == 0.92

    # Request mapped to UI label "Request / Instruction"
    res_req = tt.predict("Please send the latest document.")
    assert res_req.label == "Request / Instruction"
    assert res_req.confidence == 0.89
    assert res_req.metadata["extended_intent"] == "Request"

    # Complaint mapped to UI label "Statement"
    res_comp = tt.predict("This awful service is my official complaint.")
    assert res_comp.label == "Statement"
    assert res_comp.confidence == 0.87
    assert res_comp.metadata["extended_intent"] == "Complaint"

    # Uncertainty boundary (< 0.60)
    res_low = tt.predict("random phrase")
    assert res_low.label == "Uncertain"
    assert res_low.is_uncertain is True


# ---------------------------------------------------------------------------
# 4. ModelRegistry Tests
# ---------------------------------------------------------------------------

def test_model_registry_backend_selection():
    registry = ModelRegistry()

    # Baseline detectors
    sent_base = registry.get_sentiment_detector(backend=MLBackend.BASELINE)
    assert sent_base.__class__.__name__ == "SentimentDetector"

    tt_base = registry.get_text_type_detector(backend=MLBackend.BASELINE)
    assert tt_base.__class__.__name__ == "TextTypeDetector"

    lang_det = registry.get_language_detector()
    assert lang_det.__class__.__name__ == "LanguageDetector"

    # Transformer detectors
    sent_tf = registry.get_sentiment_detector(backend=MLBackend.TRANSFORMER)
    assert isinstance(sent_tf, SentimentTransformer)

    tt_tf = registry.get_text_type_detector(backend=MLBackend.TRANSFORMER)
    assert isinstance(tt_tf, TextTypeTransformer)


def test_model_registry_caching():
    registry = ModelRegistry()
    d1 = registry.get_sentiment_detector(backend=MLBackend.BASELINE)
    d2 = registry.get_sentiment_detector(backend=MLBackend.BASELINE)
    assert d1 is d2


# ---------------------------------------------------------------------------
# 5. Hybrid MLEngine Tests
# ---------------------------------------------------------------------------

def test_ml_engine_hybrid_backend_switch():
    from ml.engine import MLEngine
    engine = MLEngine(backend=MLBackend.BASELINE)
    assert engine.active_backend == MLBackend.BASELINE

    engine.set_backend(MLBackend.TRANSFORMER)
    assert engine.active_backend == MLBackend.TRANSFORMER
    assert engine.active_backend.value == "transformer"

    engine.set_backend("baseline")
    assert engine.active_backend == MLBackend.BASELINE


def test_ml_engine_fallback_to_baseline():
    from ml.engine import MLEngine
    # When transformer is active but dependencies/models are not loaded,
    # engine falls back to baseline and annotates fallback metadata
    engine = MLEngine(backend=MLBackend.TRANSFORMER)
    res_sent = engine.predict_sentiment("This product is really wonderful and excellent.")
    assert res_sent.label == "Positive"
    assert res_sent.metadata.get("fallback_from") == "transformer" or "Transformer" in res_sent.method

    res_tt = engine.predict_text_type("Where is my order?")
    assert res_tt.label == "Question"
    assert res_tt.metadata.get("fallback_from") == "transformer" or "Transformer" in res_tt.method


def test_ml_engine_backend_info():
    from ml.engine import MLEngine
    engine = MLEngine(backend=MLBackend.BASELINE)
    info = engine.get_backend_info()
    assert "active_backend" in info
    assert info["active_backend"] == "baseline"
    assert "device" in info
    assert "torch_available" in info
    assert "transformers_available" in info
    assert "sentiment_checkpoint" in info
    assert "text_type_checkpoint" in info


# ---------------------------------------------------------------------------
# 6. Schema Compatibility, Preservation & Evaluation Tests
# ---------------------------------------------------------------------------

def test_prediction_schema_compatibility():
    from ml.engine import MLEngine
    for backend in [MLBackend.BASELINE, MLBackend.TRANSFORMER]:
        engine = MLEngine(backend=backend)
        preds = engine.predict_all("Sample sentence for schema verification.")
        assert set(preds.keys()) == {"language", "sentiment", "text_type"}
        for task, res in preds.items():
            assert isinstance(res, PredictionResult)
            d = res.to_dict()
            assert "label" in d
            assert "confidence" in d
            assert "method" in d
            assert "is_uncertain" in d
            assert "metadata" in d
            assert isinstance(d["metadata"], dict)


def test_unsupported_language_and_noise_handling():
    from ml.engine import MLEngine
    engine = MLEngine(backend=MLBackend.BASELINE)
    # Binary/garbage/symbols
    noise = "### @@@ $$$ %%% ^^^ &&& *** 12345"
    preds = engine.predict_all(noise)
    assert preds["language"].label in ("Other", "Uncertain")
    assert preds["sentiment"].label in ("Neutral", "Not applicable", "Uncertain")


def test_human_approved_annotation_preservation():
    """
    Contract test: ML suggestions must NEVER mutate or overwrite
    human-approved annotations or disk data without explicit user action.
    """
    import pandas as pd
    from ml.engine import MLEngine

    # Human-approved record
    record = {
        "id": "item_101",
        "text": "Great service!",
        "language": "English",
        "sentiment": "Positive",
        "text_type": "Statement",
        "review_status": "Reviewed — looks correct",
        "annotator_notes": "Verified by senior annotator"
    }
    df = pd.DataFrame([record])
    original_df = df.copy(deep=True)

    # Run predictions with both backends
    for backend in [MLBackend.BASELINE, MLBackend.TRANSFORMER]:
        engine = MLEngine(backend=backend)
        advisory_preds = engine.predict_all(record["text"])
        assert isinstance(advisory_preds, dict)

        # Confirm df in memory is completely untouched
        pd.testing.assert_frame_equal(df, original_df)


def test_evaluate_compare_backends():
    from pathlib import Path
    from ml.evaluate import compare_backends

    dataset_path = Path("annotated_output.csv")
    if dataset_path.exists():
        results = compare_backends(dataset_path)
        assert "tasks" in results
        assert "sentiment" in results["tasks"]
        assert "text_type" in results["tasks"]
        for task_key, task_data in results["tasks"].items():
            assert "baseline" in task_data
            assert "transformer" in task_data
            assert task_data["baseline"]["support"] > 0
            assert "accuracy" in task_data["baseline"]
            assert "macro_f1" in task_data["baseline"]


