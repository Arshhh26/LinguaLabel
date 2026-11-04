"""
Safe, cached loader for HuggingFace Transformers and PyTorch models in LinguaLabel.

Responsibilities:
- Detect presence of torch and transformers libraries gracefully.
- Device selection (GPU with CUDA, or CPU fallback).
- Thread-safe singleton model caching to prevent re-downloads and memory leaks.
- Clear error propagation and diagnostic status reporting.
"""

import logging
import threading
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Check dependency availability
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

try:
    import transformers
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    transformers = None  # type: ignore
    pipeline = None  # type: ignore
    AutoModelForSequenceClassification = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    TRANSFORMERS_AVAILABLE = False


class ModelLoader:
    """
    Singleton manager for loading, caching, and querying transformer models.
    """

    _instance: Optional["ModelLoader"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._cache: Dict[str, Any] = {}
        self._load_errors: Dict[str, str] = {}
        self._mock_pipelines: Dict[str, Any] = {}
        self._initialized = True

    @property
    def is_torch_available(self) -> bool:
        return TORCH_AVAILABLE

    @property
    def is_transformers_available(self) -> bool:
        return TRANSFORMERS_AVAILABLE

    def get_device(self, requested: str = "auto") -> str:
        """Determines computation device based on environment and request."""
        if requested in ("cpu", "cuda"):
            if requested == "cuda" and TORCH_AVAILABLE and torch.cuda.is_available():
                return "cuda:0"
            return "cpu"

        # Auto detection
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    def register_mock_pipeline(self, checkpoint: str, mock_fn: Any) -> None:
        """Enables unit testing without requiring active downloads or network."""
        self._mock_pipelines[checkpoint] = mock_fn

    def unregister_mock_pipeline(self, checkpoint: str) -> None:
        self._mock_pipelines.pop(checkpoint, None)

    def load_sequence_classification_pipeline(
        self,
        checkpoint: str,
        device: str = "auto",
        **kwargs
    ) -> Tuple[Optional[Any], Optional[str]]:
        """
        Loads a HuggingFace text-classification pipeline with caching.
        Returns: (pipeline_or_callable, error_message)
        """
        if checkpoint in self._mock_pipelines:
            return self._mock_pipelines[checkpoint], None

        if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
            err = (
                "Transformer dependencies missing. Please install torch and transformers: "
                "`pip install torch transformers`"
            )
            self._load_errors[checkpoint] = err
            return None, err

        cache_key = f"seq_{checkpoint}_{device}"
        if cache_key in self._cache:
            return self._cache[cache_key], None

        target_device = self.get_device(device)
        device_arg = 0 if "cuda" in target_device else -1

        try:
            pipe = pipeline(
                "text-classification",
                model=checkpoint,
                tokenizer=checkpoint,
                device=device_arg,
                top_k=None,  # Return all class probabilities for calibration/inspection
                **kwargs
            )
            self._cache[cache_key] = pipe
            self._load_errors.pop(checkpoint, None)
            return pipe, None
        except Exception as e:
            err_msg = f"Failed to load checkpoint '{checkpoint}': {str(e)}"
            logger.error(err_msg)
            self._load_errors[checkpoint] = err_msg
            return None, err_msg

    def load_zero_shot_pipeline(
        self,
        checkpoint: str,
        device: str = "auto",
        **kwargs
    ) -> Tuple[Optional[Any], Optional[str]]:
        """
        Loads a zero-shot classification pipeline with caching.
        Returns: (pipeline_or_callable, error_message)
        """
        if checkpoint in self._mock_pipelines:
            return self._mock_pipelines[checkpoint], None

        if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
            err = (
                "Transformer dependencies missing. Please install torch and transformers: "
                "`pip install torch transformers`"
            )
            self._load_errors[checkpoint] = err
            return None, err

        cache_key = f"zero_shot_{checkpoint}_{device}"
        if cache_key in self._cache:
            return self._cache[cache_key], None

        target_device = self.get_device(device)
        device_arg = 0 if "cuda" in target_device else -1

        try:
            pipe = pipeline(
                "zero-shot-classification",
                model=checkpoint,
                tokenizer=checkpoint,
                device=device_arg,
                **kwargs
            )
            self._cache[cache_key] = pipe
            self._load_errors.pop(checkpoint, None)
            return pipe, None
        except Exception as e:
            err_msg = f"Failed to load zero-shot checkpoint '{checkpoint}': {str(e)}"
            logger.error(err_msg)
            self._load_errors[checkpoint] = err_msg
            return None, err_msg

    def get_last_error(self, checkpoint: str) -> Optional[str]:
        return self._load_errors.get(checkpoint)

    def is_cached(self, checkpoint: str) -> bool:
        return any(k.startswith(f"seq_{checkpoint}") or k.startswith(f"zero_shot_{checkpoint}") for k in self._cache)

    def clear_cache(self) -> None:
        self._cache.clear()
        self._load_errors.clear()
        self._mock_pipelines.clear()


def get_model_loader() -> ModelLoader:
    return ModelLoader()

