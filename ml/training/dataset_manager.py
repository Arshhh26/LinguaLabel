"""
LinguaLabel — Training Dataset Ingestion, Validation & Audit Engine (Part 11.3)

Handles:
- Uploaded CSV & Excel dataset parsing and preservation.
- Pre-training validation report:
  * Missing text and label rows.
  * Validation against canonical schema label options.
  * Exact duplicate text detection.
  * Language distribution estimation (English, Hindi, Hinglish).
  * Class imbalance analysis and sample scarcity detection.
  * Incompatible/invalid label reporting (never silently discarded).
- Dataset cataloging and SHA-256 fingerprinting.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd

from data_manager import SUPPORTED_ENCODINGS, TEXT_CANDIDATE_NAMES
from ml.model_config import (
    ABUSE_CATEGORIES,
    ABUSE_SEVERITY_LEVELS,
    LANGUAGE_LABELS,
    SENTIMENT_LABELS,
    TEXT_TYPE_LABELS,
)
from ui_mappings import (
    DISPLAY_TO_INTERNAL_ABUSE,
    DISPLAY_TO_INTERNAL_COMPLAINT,
    DISPLAY_TO_INTERNAL_LANGUAGE,
    DISPLAY_TO_INTERNAL_SENTIMENT,
    DISPLAY_TO_INTERNAL_SEVERITY,
    DISPLAY_TO_INTERNAL_TEXT_TYPE,
)


CANONICAL_TASK_LABELS: Dict[str, Set[str]] = {
    "language": set(LANGUAGE_LABELS),
    "sentiment": set(SENTIMENT_LABELS),
    "text_type": set(TEXT_TYPE_LABELS),
    "complaint": {"Complaining", "Not complaining"},
    "complaining": {"Complaining", "Not complaining"},
    "abuse_category": set(ABUSE_CATEGORIES),
    "abuse": set(ABUSE_CATEGORIES),
    "abuse_severity": set(ABUSE_SEVERITY_LEVELS),
    "severity": set(ABUSE_SEVERITY_LEVELS),
}

DISPLAY_TO_CANONICAL_MAPS: Dict[str, Dict[str, str]] = {
    "language": DISPLAY_TO_INTERNAL_LANGUAGE,
    "sentiment": DISPLAY_TO_INTERNAL_SENTIMENT,
    "text_type": DISPLAY_TO_INTERNAL_TEXT_TYPE,
    "complaint": DISPLAY_TO_INTERNAL_COMPLAINT,
    "complaining": DISPLAY_TO_INTERNAL_COMPLAINT,
    "abuse_category": DISPLAY_TO_INTERNAL_ABUSE,
    "abuse": DISPLAY_TO_INTERNAL_ABUSE,
    "abuse_severity": DISPLAY_TO_INTERNAL_SEVERITY,
    "severity": DISPLAY_TO_INTERNAL_SEVERITY,
}


@dataclass
class ValidationReport:
    is_valid: bool
    total_rows: int
    usable_rows: int
    empty_text_rows: int
    missing_label_rows: int
    duplicate_rows: int
    class_distribution: Dict[str, int]
    invalid_labels: Dict[str, int]
    language_distribution: Dict[str, int]
    class_imbalance_ratio: float
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def duplicate_text_count(self) -> int:
        return self.duplicate_rows

    @property
    def missing_values_count(self) -> int:
        return self.empty_text_rows + self.missing_label_rows

    @property
    def valid_samples_count(self) -> int:
        return self.usable_rows

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingDatasetEntry:
    dataset_id: str
    name: str
    filename: str
    file_format: str
    row_count: int
    usable_rows: int
    task: str
    text_column: str
    label_column: str
    fingerprint: str
    created_at: str
    file_path: str
    validation_report: Dict[str, Any]

    @property
    def num_samples(self) -> int:
        return self.row_count

    @property
    def text_col(self) -> str:
        return self.text_column

    @property
    def label_col(self) -> str:
        return self.label_column

    @property
    def uploaded_at(self) -> str:
        return self.created_at

    @property
    def validation_summary(self) -> Dict[str, Any]:
        return self.validation_report

    @property
    def description(self) -> str:
        return self.name

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetManagementEngine:
    """
    Manages owner-uploaded training datasets, validation checks, and storage.
    """

    def __init__(self, root_dir: Optional[Path] = None):
        self.root_dir = root_dir or Path(__file__).resolve().parent.parent.parent
        self.datasets_dir = self.root_dir / "training_datasets"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.datasets_dir / "datasets_manifest.json"
        self._ensure_manifest()

    def _ensure_manifest(self):
        if not self.manifest_file.exists():
            with open(self.manifest_file, "w", encoding="utf-8") as f:
                json.dump({}, f, indent=2)

    def _load_manifest(self) -> Dict[str, Dict[str, Any]]:
        try:
            with open(self.manifest_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_manifest(self, data: Dict[str, Dict[str, Any]]):
        temp_path = self.datasets_dir / "datasets_manifest.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        temp_path.replace(self.manifest_file)

    @staticmethod
    def parse_uploaded_file(file_bytes: bytes, filename: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Parses CSV or Excel bytes into a Pandas DataFrame."""
        if not file_bytes:
            return None, "Uploaded file is empty."

        lower_fn = filename.lower()
        if lower_fn.endswith((".xlsx", ".xls")):
            try:
                stream = io.BytesIO(file_bytes)
                df = pd.read_excel(stream, dtype=str)
                df = df.fillna("")
                return df, None
            except Exception as e:
                return None, f"Failed to parse Excel file: {str(e)}"
        else:
            # CSV parsing across supported encodings
            last_err = ""
            for enc in SUPPORTED_ENCODINGS:
                try:
                    stream = io.BytesIO(file_bytes)
                    df = pd.read_csv(stream, encoding=enc, keep_default_na=False, dtype=str)
                    return df, None
                except Exception as e:
                    last_err = str(e)
            return None, f"Failed to decode CSV file: {last_err}"

    @staticmethod
    def detect_probable_columns(df: pd.DataFrame, task: str) -> Tuple[Optional[str], Optional[str]]:
        """Auto-detects probable text and label columns."""
        if df.empty:
            return None, None

        col_names = list(df.columns)
        col_lower = [str(c).strip().lower() for c in col_names]

        # 1. Detect Text Column
        text_col = None
        for cand in TEXT_CANDIDATE_NAMES:
            if cand in col_lower:
                text_col = col_names[col_lower.index(cand)]
                break
        if not text_col and len(col_names) > 0:
            text_col = col_names[0]

        # 2. Detect Label Column
        task_candidates = {
            "sentiment": ["sentiment", "sentiment_label", "feeling", "polarity", "sent_label", "label"],
            "text_type": ["text_type", "sentence_type", "type", "intent", "category", "label"],
            "language": ["language", "lang", "locale", "label"],
            "abuse_category": ["abuse_category", "abuse", "toxicity", "harassment", "label"],
        }
        cand_list = list(task_candidates.get(task, ["label", "category", "target"]))
        cand_list.extend([f"{task}_label", f"{task}_category", f"{task}_type"])
        label_col = None
        for cand in cand_list:
            if cand in col_lower:
                label_col = col_names[col_lower.index(cand)]
                break

        return text_col, label_col

    @staticmethod
    def estimate_language(text: str) -> str:
        """Heuristic language estimation for dataset language distribution analysis."""
        txt = str(text).strip()
        if re.search(r"[\u0900-\u097F]", txt):
            return "Hindi"
        lower = txt.lower()
        hinglish_markers = ["hai", "nahi", "kya", "bhai", "yaar", "karo", "raha", "gaya", "mein", "hota"]
        if any(re.search(r"\b" + m + r"\b", lower) for m in hinglish_markers):
            return "Hinglish"
        return "English"

    def validate_dataset(
        self,
        df: pd.DataFrame,
        text_col: str,
        label_col: str,
        task: str,
    ) -> ValidationReport:
        """
        Executes a rigorous, non-silent dataset validation audit against canonical schema.
        """
        if df.empty:
            return ValidationReport(
                is_valid=False,
                total_rows=0,
                usable_rows=0,
                empty_text_rows=0,
                missing_label_rows=0,
                duplicate_rows=0,
                class_distribution={},
                invalid_labels={},
                language_distribution={},
                class_imbalance_ratio=1.0,
                errors=["Dataset is empty."],
            )

        errors: List[str] = []
        warnings: List[str] = []

        if text_col not in df.columns:
            errors.append(f"Selected text column '{text_col}' does not exist in dataset.")
            return ValidationReport(False, len(df), 0, 0, 0, 0, {}, {}, {}, 1.0, warnings, errors)

        if label_col not in df.columns:
            errors.append(f"Selected label column '{label_col}' does not exist in dataset.")
            return ValidationReport(False, len(df), 0, 0, 0, 0, {}, {}, {}, 1.0, warnings, errors)

        total_rows = len(df)
        canonical_labels = CANONICAL_TASK_LABELS.get(task, set())
        display_map = DISPLAY_TO_CANONICAL_MAPS.get(task, {})

        empty_text_rows = 0
        missing_label_rows = 0
        invalid_labels: Dict[str, int] = {}
        class_distribution: Dict[str, int] = {}
        language_distribution: Dict[str, int] = {"English": 0, "Hindi": 0, "Hinglish": 0}
        seen_texts: Set[str] = set()
        duplicate_rows = 0
        usable_rows = 0

        for _, row in df.iterrows():
            val = row.get(text_col)
            lval = row.get(label_col)

            is_text_empty = pd.isna(val) or val is None or not str(val).strip() or str(val).strip().lower() in ("none", "nan")
            is_label_empty = pd.isna(lval) or lval is None or not str(lval).strip() or str(lval).strip().lower() in ("— select —", "none", "nan")

            if is_text_empty:
                empty_text_rows += 1
                continue

            if is_label_empty:
                missing_label_rows += 1
                continue

            raw_text = str(val).strip()
            raw_label = str(lval).strip()

            # Duplicate text check
            if raw_text in seen_texts:
                duplicate_rows += 1
            else:
                seen_texts.add(raw_text)

            # Map display label to internal canonical if needed
            norm_label = display_map.get(raw_label, raw_label)

            # Check validity against canonical schema
            if canonical_labels and norm_label not in canonical_labels:
                invalid_labels[raw_label] = invalid_labels.get(raw_label, 0) + 1
            else:
                usable_rows += 1
                class_distribution[norm_label] = class_distribution.get(norm_label, 0) + 1
                lang = self.estimate_language(raw_text)
                language_distribution[lang] = language_distribution.get(lang, 0) + 1

        # Warnings & Quality checks
        if empty_text_rows > 0:
            warnings.append(f"Found {empty_text_rows} row(s) with empty or blank sentence text.")
        if missing_label_rows > 0:
            warnings.append(f"Found {missing_label_rows} row(s) with missing or empty target labels.")
        if duplicate_rows > 0:
            warnings.append(f"Found {duplicate_rows} duplicate sentence(s).")
        if invalid_labels:
            invalid_summary = ", ".join([f"'{k}' ({v} rows)" for k, v in invalid_labels.items()])
            errors.append(f"Dataset contains invalid labels not in canonical schema for {task}: {invalid_summary}.")

        # Class imbalance calculation
        imbalance_ratio = 1.0
        if class_distribution:
            counts = list(class_distribution.values())
            max_c = max(counts)
            min_c = min(counts)
            imbalance_ratio = round(max_c / min_c, 2) if min_c > 0 else float("inf")
            if imbalance_ratio > 8.0:
                warnings.append(f"Severe class imbalance detected: ratio is {imbalance_ratio}:1 (largest class {max_c}, smallest {min_c}).")
            for lbl, cnt in class_distribution.items():
                if cnt < 3:
                    warnings.append(f"Class '{lbl}' has only {cnt} sample(s); model training may underperform on this class.")

        is_valid = (len(errors) == 0) and (usable_rows >= 3)
        if usable_rows < 3:
            errors.append(f"Dataset has only {usable_rows} usable sample(s); at least 3 labeled samples are required for training.")

        return ValidationReport(
            is_valid=is_valid,
            total_rows=total_rows,
            usable_rows=usable_rows,
            empty_text_rows=empty_text_rows,
            missing_label_rows=missing_label_rows,
            duplicate_rows=duplicate_rows,
            class_distribution=class_distribution,
            invalid_labels=invalid_labels,
            language_distribution=language_distribution,
            class_imbalance_ratio=imbalance_ratio,
            warnings=warnings,
            errors=errors,
        )

    def ingest_dataset(
        self,
        file_bytes: bytes,
        filename: str,
        name: Optional[str] = None,
        task: str = "sentiment",
        text_col: str = "text",
        label_col: str = "label",
        description: str = "",
        dataset_name: Optional[str] = None,
        **kwargs,
    ) -> Tuple[Optional[TrainingDatasetEntry], Optional[str]]:
        """
        Parses, validates, fingerprints, and saves an owner-uploaded training dataset.
        """
        if name in ("sentiment", "language", "text_type", "abuse", "abuse_category", "complaint") and task not in ("sentiment", "language", "text_type", "abuse", "abuse_category", "complaint"):
            real_task = name
            real_text_col = task
            real_label_col = text_col
            task = real_task
            text_col = real_text_col
            label_col = real_label_col
            name = filename

        effective_name = (name or dataset_name or description or filename).strip()

        df, parse_err = self.parse_uploaded_file(file_bytes, filename)
        if parse_err or df is None:
            return None, parse_err

        val_report = self.validate_dataset(df, text_col, label_col, task)
        if not val_report.is_valid:
            err_msg = "Validation failed: " + "; ".join(val_report.errors)
            return None, err_msg

        # Compute deterministic fingerprint over text + label pairs
        items = []
        for _, row in df.iterrows():
            t = str(row.get(text_col, "")).strip()
            l = str(row.get(label_col, "")).strip()
            if t and l:
                items.append(f"{t}:::{l}")
        items.sort()
        fp = hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()[:16]

        ts = datetime.now(timezone.utc)
        safe_fn = "".join(c for c in filename if c.isalnum() or c in ("-", "_", "."))
        dataset_id = f"ds_{ts.strftime('%Y%m%d_%H%M%S')}_{fp[:6]}"
        saved_file_name = f"{dataset_id}_{safe_fn}"
        saved_path = self.datasets_dir / saved_file_name

        with open(saved_path, "wb") as f:
            f.write(file_bytes)

        entry = TrainingDatasetEntry(
            dataset_id=dataset_id,
            name=effective_name,
            filename=filename,
            file_format="excel" if filename.lower().endswith((".xlsx", ".xls")) else "csv",
            row_count=len(df),
            usable_rows=val_report.usable_rows,
            task=task,
            text_column=text_col,
            label_column=label_col,
            fingerprint=fp,
            created_at=ts.isoformat(),
            file_path=str(saved_path),
            validation_report=val_report.to_dict(),
        )

        manifest = self._load_manifest()
        manifest[dataset_id] = entry.to_dict()
        self._save_manifest(manifest)

        return entry, None

    def list_datasets(self, task: Optional[str] = None) -> List[TrainingDatasetEntry]:
        manifest = self._load_manifest()
        entries = [TrainingDatasetEntry(**d) for d in manifest.values()]
        if task:
            entries = [e for e in entries if e.task == task]
        entries.sort(key=lambda e: e.created_at, reverse=True)
        return entries

    def get_dataset(self, dataset_id: str) -> Optional[TrainingDatasetEntry]:
        manifest = self._load_manifest()
        data = manifest.get(dataset_id)
        if not data:
            return None
        return TrainingDatasetEntry(**data)

    def load_dataset_dataframe(self, dataset_id: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        entry = self.get_dataset(dataset_id)
        if not entry:
            return None, f"Dataset '{dataset_id}' not found."

        p = Path(entry.file_path)
        if not p.exists():
            return None, f"Dataset file '{p}' does not exist on disk."

        try:
            with open(p, "rb") as f:
                content = f.read()
            return self.parse_uploaded_file(content, entry.filename)
        except Exception as e:
            return None, f"Failed to load dataset: {str(e)}"

    detect_columns = staticmethod(detect_probable_columns)
    validate_dataframe = validate_dataset

    def save_dataset(
        self,
        file_bytes: bytes,
        filename: str,
        task: str,
        text_col: str,
        label_col: str,
        description: str = "",
        dataset_name: Optional[str] = None,
    ) -> TrainingDatasetEntry:
        entry, err = self.ingest_dataset(
            file_bytes=file_bytes,
            filename=filename,
            task=task,
            text_col=text_col,
            label_col=label_col,
            description=description,
            dataset_name=dataset_name or filename,
        )
        if err or entry is None:
            raise ValueError(f"Failed to save dataset: {err}")
        return entry

    def delete_dataset(self, dataset_id: str) -> bool:
        manifest = self._load_manifest()
        if dataset_id not in manifest:
            return False
        data = manifest.pop(dataset_id)
        self._save_manifest(manifest)
        try:
            p = Path(data.get("file_path", ""))
            if p.exists():
                p.unlink()
        except Exception:
            pass
        return True
