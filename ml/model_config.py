"""
Model Configuration and Constants for LinguaLabel ML Engine.

Defines supported backends, transformer checkpoint identifiers,
supported languages, task label mappings, and calibration settings.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class MLBackend(str, Enum):
    BASELINE = "baseline"
    TRANSFORMER = "transformer"


# Standard LinguaLabel target schemas
SENTIMENT_LABELS = ["Positive", "Negative", "Neutral", "Not applicable", "Uncertain"]
TEXT_TYPE_LABELS = [
    "Statement",
    "Question",
    "Request / Instruction",
    "Greeting",
    "Complaining",
    "Other",
    "Uncertain",
]
LANGUAGE_LABELS = ["English", "Hindi", "Hinglish", "Other", "Uncertain"]

# Abusive Language Detection target schemas (Part 11.2)
ABUSE_CATEGORIES = [
    "Not abusive",
    "Profanity / Vulgarity",
    "Severe Personal Insult",
    "Sexualized Abuse",
    "Harassment",
    "Threat",
    "Uncertain",
]

ABUSE_SEVERITY_LEVELS = [
    "None",
    "Mild",
    "Moderate",
    "Severe",
]

# Extended text-type categories for transformer intent modeling
EXTENDED_TEXT_TYPE_CATEGORIES = [
    "Question",
    "Complaint",
    "Review",
    "General",
    "Greeting",
    "Request",
]

# Mapping from extended transformer text types to LinguaLabel UI schema
EXTENDED_TO_UI_TEXT_TYPE_MAP: Dict[str, str] = {
    "Question": "Question",
    "Greeting": "Greeting",
    "Request": "Request / Instruction",
    "Complaint": "Complaining",
    "Review": "Statement",
    "General": "Statement",
}


@dataclass
class ModelMetadata:
    """Detailed metadata for a machine learning model or transformer checkpoint."""
    checkpoint: str
    task: str
    architecture: str
    license: str
    supported_languages: List[str]
    intended_task: str
    label_mapping: Dict[str, str] = field(default_factory=dict)
    default_device: str = "cpu"
    is_fine_tuned: bool = True
    notes: str = ""


# Verified Hugging Face transformer checkpoints
DEFAULT_SENTIMENT_CHECKPOINT = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
DEFAULT_TEXT_TYPE_CHECKPOINT = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

VERIFIED_MODELS: Dict[str, ModelMetadata] = {
    DEFAULT_SENTIMENT_CHECKPOINT: ModelMetadata(
        checkpoint=DEFAULT_SENTIMENT_CHECKPOINT,
        task="sentiment",
        architecture="XLM-RoBERTa (base)",
        license="MIT",
        supported_languages=["English", "Hindi", "Spanish", "French", "Italian", "Portuguese", "German", "Arabic"],
        intended_task="Multilingual 3-class sentiment analysis (negative, neutral, positive)",
        label_mapping={
            "negative": "Negative",
            "neutral": "Neutral",
            "positive": "Positive",
            "LABEL_0": "Negative",
            "LABEL_1": "Neutral",
            "LABEL_2": "Positive",
        },
        notes="Trained on ~198M multilingual tweets including English and Hindi.",
    ),
    DEFAULT_TEXT_TYPE_CHECKPOINT: ModelMetadata(
        checkpoint=DEFAULT_TEXT_TYPE_CHECKPOINT,
        task="text_type",
        architecture="mDeBERTa-v3 (base)",
        license="MIT",
        supported_languages=["English", "Hindi", "Multilingual (100+ languages)"],
        intended_task="Multilingual Zero-Shot NLI / Text Classification",
        label_mapping=EXTENDED_TO_UI_TEXT_TYPE_MAP,
        notes="Pretrained on 2.7M multilingual NLI pairs, evaluated on XNLI including Hindi.",
    ),
}


@dataclass
class MLEngineConfig:
    """Configuration container for MLEngine and its active backends."""
    active_backend: MLBackend = MLBackend.BASELINE
    uncertainty_threshold: float = 0.60
    device: str = "auto"  # "auto", "cpu", or "cuda"
    sentiment_checkpoint: str = DEFAULT_SENTIMENT_CHECKPOINT
    text_type_checkpoint: str = DEFAULT_TEXT_TYPE_CHECKPOINT
    enable_fallback_to_baseline: bool = True
    calibrate_confidence: bool = False

