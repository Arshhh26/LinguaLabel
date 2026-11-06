"""
Controlled Learning Preparation Engine for LinguaLabel.

Prepares human-verified annotations and active learning feedback for future model training:
1. Standardizes training dataset schema (text, label, task, record_id, dataset_key, timestamp).
2. Rigorous label schema validation (ensures labels belong to standard target classes).
3. Deterministic dataset fingerprinting (SHA-256 content hashes).
4. Train / Validation / Test split generation with deterministic seeding.
5. Cross-split data leakage detection and identical text collision analysis.

STRICT RULE: This module NEVER trains or fine-tunes models automatically.
It solely validates, audits, partitions, and exports clean datasets.
"""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd

from ml.active_learning.feedback import FeedbackRecord
from ml.model_config import LANGUAGE_LABELS, SENTIMENT_LABELS, TEXT_TYPE_LABELS


VALID_TASK_LABELS: Dict[str, Set[str]] = {
    "sentiment": set(SENTIMENT_LABELS),
    "text_type": set(TEXT_TYPE_LABELS),
    "language": set(LANGUAGE_LABELS),
}


@dataclass
class DatasetSplit:
    """Container for partitioned dataset splits."""
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    fingerprint: str
    total_samples: int
    train_count: int
    val_count: int
    test_count: int
    leakage_detected: bool
    duplicate_count: int
    validation_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "total_samples": self.total_samples,
            "train_count": self.train_count,
            "val_count": self.val_count,
            "test_count": self.test_count,
            "leakage_detected": self.leakage_detected,
            "duplicate_count": self.duplicate_count,
            "validation_warnings": self.validation_warnings,
        }


def compute_training_data_fingerprint(df: pd.DataFrame) -> str:
    """Computes a deterministic SHA-256 fingerprint over sorted texts and labels."""
    if df.empty or "text" not in df.columns or "label" not in df.columns:
        return "empty_dataset"

    items = []
    for _, row in df.iterrows():
        t = str(row.get("text", "")).strip()
        l = str(row.get("label", "")).strip()
        items.append(f"{t}:::{l}")

    items.sort()
    content = "\n".join(items)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def format_feedback_for_training(
    records: List[FeedbackRecord],
    task: Optional[str] = None,
    exclude_uncertain: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Transforms human feedback records into clean training samples.
    Validates labels and filters out uncertain or invalid entries.
    """
    warnings: List[str] = []
    clean_rows = []

    for r in records:
        if task and r.task != task:
            continue

        h_label = r.human_corrected_label.strip()
        if exclude_uncertain and h_label.lower() in ("uncertain", "— select —", ""):
            continue

        # Schema validation
        valid_set = VALID_TASK_LABELS.get(r.task, set())
        if valid_set and h_label not in valid_set:
            warnings.append(f"Invalid label '{h_label}' for task '{r.task}' in feedback {r.feedback_id}; skipped.")
            continue

        clean_rows.append({
            "record_id": r.record_id,
            "text": r.original_text,
            "label": h_label,
            "task": r.task,
            "dataset_key": r.dataset_key,
            "source": "human_feedback",
            "is_correction": r.is_correction,
            "timestamp": r.timestamp,
        })

    df = pd.DataFrame(clean_rows)
    return df, warnings


def check_data_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str = "text",
) -> Tuple[bool, List[str]]:
    """
    Checks for identical sentences shared between train, val, and test splits.
    Returns (has_leakage, list_of_conflicts).
    """
    conflicts: List[str] = []
    if train_df.empty:
        return False, []

    train_texts = set(train_df[text_col].astype(str).str.strip())
    val_texts = set(val_df[text_col].astype(str).str.strip()) if not val_df.empty else set()
    test_texts = set(test_df[text_col].astype(str).str.strip()) if not test_df.empty else set()

    train_val_overlap = train_texts.intersection(val_texts)
    train_test_overlap = train_texts.intersection(test_texts)
    val_test_overlap = val_texts.intersection(test_texts)

    if train_val_overlap:
        conflicts.append(f"Leakage: {len(train_val_overlap)} sentence(s) appear in both Train and Validation splits.")
    if train_test_overlap:
        conflicts.append(f"Leakage: {len(train_test_overlap)} sentence(s) appear in both Train and Test splits.")
    if val_test_overlap:
        conflicts.append(f"Leakage: {len(val_test_overlap)} sentence(s) appear in both Validation and Test splits.")

    return (len(conflicts) > 0), conflicts


def prepare_controlled_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify_by_label: bool = True,
) -> DatasetSplit:
    """
    Partitions a verified dataset into train, val, and test splits.
    Performs deterministic shuffling and cross-split leakage checks.
    """
    if df.empty:
        return DatasetSplit(
            train=pd.DataFrame(),
            val=pd.DataFrame(),
            test=pd.DataFrame(),
            fingerprint="empty",
            total_samples=0,
            train_count=0,
            val_count=0,
            test_count=0,
            leakage_detected=False,
            duplicate_count=0,
            validation_warnings=["Input dataset is empty."],
        )

    # Detect duplicates within df
    texts = df["text"].astype(str).str.strip()
    dup_count = int(texts.duplicated().sum())

    # Shuffling with fixed seed
    shuffled_df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled_df)

    if n < 3:
        # Small sample edge case: allocate all to train
        return DatasetSplit(
            train=shuffled_df,
            val=pd.DataFrame(columns=df.columns),
            test=pd.DataFrame(columns=df.columns),
            fingerprint=compute_training_data_fingerprint(shuffled_df),
            total_samples=n,
            train_count=n,
            val_count=0,
            test_count=0,
            leakage_detected=False,
            duplicate_count=dup_count,
            validation_warnings=["Sample size too small for 3-way partitioning; all allocated to train."],
        )

    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio))) if val_ratio > 0 else 0
    # Ensure remaining goes to test
    n_test = max(0, n - n_train - n_val)
    if n_test == 0 and test_ratio > 0 and n >= 3:
        n_train -= 1
        n_test = 1

    train_df = shuffled_df.iloc[:n_train].reset_index(drop=True)
    val_df = shuffled_df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df = shuffled_df.iloc[n_train + n_val:].reset_index(drop=True)

    has_leakage, leakage_msgs = check_data_leakage(train_df, val_df, test_df)
    fingerprint = compute_training_data_fingerprint(shuffled_df)

    warnings = []
    if dup_count > 0:
        warnings.append(f"Dataset contains {dup_count} duplicate sentence(s).")
    warnings.extend(leakage_msgs)

    return DatasetSplit(
        train=train_df,
        val=val_df,
        test=test_df,
        fingerprint=fingerprint,
        total_samples=n,
        train_count=len(train_df),
        val_count=len(val_df),
        test_count=len(test_df),
        leakage_detected=has_leakage,
        duplicate_count=dup_count,
        validation_warnings=warnings,
    )

