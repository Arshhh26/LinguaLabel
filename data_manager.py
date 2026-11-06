"""
LinguaLabel — Data Ingestion & Dataset Management Module (Part 1)

Handles:
- Robust multi-encoding CSV reading (UTF-8, UTF-8-sig, Latin-1, CP1252)
- Text column detection, validation, and mapping
- Stable deterministic ID validation and generation
- Dataset isolation to prevent overwriting sample_data.csv annotations
- Preservation of all original columns on export
"""

import csv
import hashlib
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

COLS = ["language", "sentiment", "text_type", "review_status", "annotator_notes"]

TEXT_CANDIDATE_NAMES = [
    "text", "sentence", "content", "utterance", "review", 
    "comment", "line", "data", "message", "query", "input",
    "prompt", "transcription", "statement"
]

SUPPORTED_ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]


def load_csv_bytes(
    file_bytes: bytes, filename: str = "uploaded.csv"
) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    """
    Attempts to read CSV bytes across supported encodings with automatic delimiter detection.
    Returns: (dataframe, encoding_used, error_message)
    """
    if not file_bytes or not file_bytes.strip():
        return None, None, "The uploaded file is empty."

    last_error = "Unable to decode file."

    for enc in SUPPORTED_ENCODINGS:
        try:
            # First attempt standard comma-separated with keep_default_na=False
            stream = io.BytesIO(file_bytes)
            df = pd.read_csv(stream, encoding=enc, keep_default_na=False, dtype=str)
            
            # If the CSV parsed into only 1 column but content looks like TSV or semicolon-delimited
            if len(df.columns) == 1 and len(df) > 0:
                first_col = str(df.columns[0])
                first_row = str(df.iloc[0, 0]) if not df.empty else ""
                if "\t" in first_col or "\t" in first_row:
                    stream.seek(0)
                    df = pd.read_csv(stream, encoding=enc, sep="\t", keep_default_na=False, dtype=str)
                elif ";" in first_col or ";" in first_row:
                    stream.seek(0)
                    df = pd.read_csv(stream, encoding=enc, sep=";", keep_default_na=False, dtype=str)

            if df.empty and len(df.columns) == 0:
                return None, None, "The uploaded CSV contains no headers or rows."

            return df, enc, None

        except UnicodeDecodeError as ude:
            last_error = f"Encoding decode error with {enc}: {str(ude)}"
            continue
        except pd.errors.EmptyDataError:
            return None, None, "The uploaded file contains no data."
        except pd.errors.ParserError as pe:
            last_error = f"Malformed CSV structure: {str(pe)}"
            continue
        except Exception as ex:
            last_error = f"Error reading CSV with {enc}: {str(ex)}"
            continue

    return None, None, f"Failed to parse CSV with supported encodings. Details: {last_error}"


def load_excel_bytes(
    file_bytes: bytes, filename: str = "uploaded.xlsx"
) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    """
    Attempts to read Excel workbook bytes (.xlsx, .xls) using openpyxl.
    Returns: (dataframe, 'excel', error_message)
    """
    if not file_bytes:
        return None, None, "The uploaded Excel file is empty."

    try:
        stream = io.BytesIO(file_bytes)
        df = pd.read_excel(stream, dtype=str)
        if df.empty or len(df.columns) == 0:
            return None, None, "The uploaded Excel sheet contains no rows or columns."
        # Fill NaN with empty string
        df = df.fillna("")
        return df, "excel", None
    except Exception as e:
        return None, None, f"Failed to read Excel file: {str(e)}"


def load_pasted_sentences(text_content: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Parses multiple lines of text into a sentence DataFrame.
    """
    if not text_content or not text_content.strip():
        return None, "Please paste one or more sentences."

    raw_lines = text_content.splitlines()
    clean_lines = [line.strip() for line in raw_lines if line.strip()]

    if not clean_lines:
        return None, "No valid non-empty sentences found in the pasted text."

    df = pd.DataFrame({"text": clean_lines})
    return df, None


def load_single_sentence(sentence: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Packages a single manual sentence into a standardized single-row DataFrame.
    """
    clean = str(sentence).strip()
    if not clean:
        return None, "Please enter a sentence to analyse."

    df = pd.DataFrame([{"text": clean}])
    return df, None


def export_excel_bytes(df: pd.DataFrame) -> Optional[bytes]:
    """
    Exports a DataFrame into an in-memory Excel workbook (.xlsx) as bytes.
    """
    if df is None or df.empty:
        return None

    try:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="LinguaLabel Annotations")
        return buffer.getvalue()
    except Exception:
        return None


def detect_text_column(df: pd.DataFrame) -> Optional[str]:
    """
    Detects the most probable column representing sentence text.
    """
    if df is None or df.empty:
        return None

    col_names = list(df.columns)
    col_names_lower = [str(c).strip().lower() for c in col_names]

    # 1. Exact candidate match
    for cand in TEXT_CANDIDATE_NAMES:
        if cand in col_names_lower:
            return col_names[col_names_lower.index(cand)]

    # 2. Substring candidate match
    for cand in TEXT_CANDIDATE_NAMES:
        for idx, col in enumerate(col_names_lower):
            if cand in col and not col.startswith("id"):
                return col_names[idx]

    # 3. String column with highest average character length
    best_col = None
    max_avg_len = -1
    for col in col_names:
        if str(col).lower() in ["id", "_mid", "review_status"]:
            continue
        sample_lens = df[col].astype(str).str.len()
        avg_len = sample_lens.mean() if not sample_lens.empty else 0
        if avg_len > max_avg_len:
            max_avg_len = avg_len
            best_col = col

    return best_col if best_col is not None else col_names[0]


def generate_deterministic_record_ids(texts: List[str], dataset_key: str = "") -> List[str]:
    """
    Generates deterministic, dataset-aware, occurrence-safe record IDs.

    Guarantees:
    - Content Determinism: Identifiers are derived from SHA-256 hash of text content.
    - Duplicate Text Preservation: An occurrence counter (_1, _2...) ensures identical
      sentences appearing multiple times receive distinct, collision-free IDs.
    - Dataset Isolation: Prefixed with dataset_key to prevent collisions across datasets.
    - Rerun Stability: Identical datasets produce identical IDs across runs without
      depending on volatile row position shifting.
    """
    seen_counts: Dict[str, int] = {}
    new_ids = []
    prefix = f"{dataset_key}_" if dataset_key else "rec_"

    for text in texts:
        norm_text = str(text).strip()
        h = hashlib.sha256(norm_text.encode("utf-8")).hexdigest()[:10]
        seen_counts[h] = seen_counts.get(h, 0) + 1
        occ = seen_counts[h]
        new_ids.append(f"{prefix}{h}_{occ}")

    return new_ids


def stabilize_dataset(
    df: pd.DataFrame, text_column: str, dataset_key: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validates and normalizes the dataset:
    - Preserves all original columns
    - Maps chosen text_column to active 'text' field
    - Validates or generates stable, deterministic, unique IDs
    - Ensures standard annotation schema columns exist
    """
    if df is None or df.empty:
        return pd.DataFrame(), {"error": "Dataset is empty"}

    clean_df = df.copy()
    info = {
        "dataset_key": dataset_key,
        "original_rows": len(clean_df),
        "original_columns": list(clean_df.columns),
        "text_column_selected": text_column,
        "id_generated": False,
        "id_reason": "Preserved original unique IDs",
        "empty_text_rows": 0,
    }

    if text_column not in clean_df.columns:
        raise ValueError(f"Selected text column '{text_column}' is not present in the dataset.")

    # 1. Map to working 'text' column while preserving original
    if text_column != "text":
        if "text" in clean_df.columns:
            # Avoid collision if another column was named 'text'
            clean_df["_original_text_col"] = clean_df["text"]
        clean_df["text"] = clean_df[text_column].astype(str)
    else:
        clean_df["text"] = clean_df["text"].astype(str)

    # Check for empty text values
    empty_mask = clean_df["text"].str.strip().eq("")
    info["empty_text_rows"] = int(empty_mask.sum())

    # 2. Stable ID handling
    has_id = "id" in clean_df.columns
    needs_new_id = False
    id_reason = ""

    if has_id:
        # Check non-null, non-empty, and uniqueness
        raw_ids = clean_df["id"].astype(str).str.strip().tolist()
        has_empties = any(x == "" or x.lower() == "nan" for x in raw_ids)
        is_unique = len(set(raw_ids)) == len(raw_ids)
        if not is_unique:
            needs_new_id = True
            id_reason = "Original 'id' column contained duplicates"
        elif has_empties:
            needs_new_id = True
            id_reason = "Original 'id' column contained blank or null entries"
        else:
            # Clean up potential floating-point formatting e.g. '1.0' -> '1'
            clean_ids = []
            for item in raw_ids:
                if item.endswith(".0") and item[:-2].isdigit():
                    clean_ids.append(item[:-2])
                else:
                    clean_ids.append(item)
            clean_df["id"] = clean_ids
    else:
        needs_new_id = True
        id_reason = "Dataset had no 'id' column"

    if needs_new_id:
        if has_id and "original_id" not in clean_df.columns:
            clean_df["original_id"] = clean_df["id"]
        
        # Generate deterministic, dataset-aware, occurrence-safe record IDs
        clean_df["id"] = generate_deterministic_record_ids(clean_df["text"].tolist(), dataset_key)
        info["id_generated"] = True
        info["id_reason"] = f"{id_reason} (generated deterministic content-occurrence IDs)"

    # Ensure 'id' is the first column for clean UI layout
    cols_order = ["id"] + [c for c in clean_df.columns if c != "id"]
    clean_df = clean_df[cols_order]

    # 3. Ensure all standard annotation columns exist
    for c in COLS:
        if c not in clean_df.columns:
            clean_df[c] = ""
        else:
            clean_df[c] = clean_df[c].fillna("").astype(str)

    return clean_df, info


def compute_dataset_key(file_bytes: Optional[bytes], default_key: str = "default_sample") -> str:
    """
    Computes a deterministic short key representing the dataset source.
    """
    if file_bytes is None:
        return default_key
    h = hashlib.sha256(file_bytes).hexdigest()[:12]
    return f"upload_{h}"


def get_dataset_storage_path(dataset_key: str, root_dir: Path) -> Path:
    """
    Returns the isolated persistence path for a given dataset key.
    The default dataset strictly uses annotated_output.csv.
    Custom datasets are persisted inside a dedicated 'saved_sessions' directory.
    """
    if dataset_key == "default_sample":
        return root_dir / "annotated_output.csv"
    
    sessions_dir = root_dir / "saved_sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    return sessions_dir / f"annotated_{dataset_key}.csv"


def merge_saved_annotations(df: pd.DataFrame, storage_path: Path) -> pd.DataFrame:
    """
    Merges saved annotations from storage_path onto df using stable 'id' string join.
    Preserves all other original columns.
    """
    if df.empty or not storage_path.exists():
        return df

    try:
        saved = pd.read_csv(storage_path, keep_default_na=False, dtype=str)
        if "id" not in saved.columns:
            return df

        saved_cols = [c for c in COLS if c in saved.columns]
        if not saved_cols:
            return df

        saved_subset = saved[["id"] + saved_cols].drop_duplicates("id", keep="last")

        working_df = df.copy()
        working_df["_mid"] = working_df["id"].astype(str)
        saved_subset["_mid"] = saved_subset["id"].astype(str)

        # Drop existing COLS in working_df that will be updated from saved
        drop_cols = [c for c in saved_cols if c in working_df.columns]
        working_df = working_df.drop(columns=drop_cols)

        merged = working_df.merge(saved_subset[["_mid"] + saved_cols], on="_mid", how="left").drop(columns=["_mid"])

        for c in COLS:
            if c not in merged.columns:
                merged[c] = ""
            merged[c] = merged[c].fillna("")

        return merged
    except Exception as e:
        # If reading saved annotations encounters an issue, return un-merged df rather than crash
        return df


def save_dataset_record(
    storage_path: Path, current_df: pd.DataFrame, row_id: str, vals: Dict[str, Any]
) -> bool:
    """
    Updates the record in current_df and persists the full dataset to storage_path with utf-8-sig encoding.
    Ensures isolated, dataset-specific persistence and preserves all original columns.
    """
    try:
        mask = current_df["id"].astype(str) == str(row_id)
        if not mask.any():
            return False

        for col, value in vals.items():
            if col in current_df.columns:
                current_df.loc[mask, col] = str(value)

        storage_path.parent.mkdir(parents=True, exist_ok=True)
        current_df.to_csv(storage_path, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8-sig")
        return True
    except Exception:
        return False


def validation_issues(values: Dict[str, Any]) -> List[str]:
    """
    Rigorously enforces the four LinguaLabel validation rules:
    1. All four schema fields (language, sentiment, text_type, review_status) must be selected.
    2. Mandatory annotator notes when any schema field is 'Uncertain'.
    3. Marking as 'Reviewed — looks correct' is blocked if any label is empty or uncertain.
    4. Explanatory notes are mandatory for triage statuses (Needs second review, Ambiguous, Skip).
    """
    issues = []
    required = ["language", "sentiment", "text_type", "review_status"]
    for column in required:
        val = str(values.get(column, "")).strip()
        if not val or val == "— Select —":
            issues.append(f"Choose a value for {column.replace('_', ' ')}.")

    notes = str(values.get("annotator_notes", "")).strip()
    for column in ["language", "sentiment", "text_type"]:
        if str(values.get(column, "")).strip() == "Uncertain" and not notes:
            issues.append(f"Add an annotator note explaining the uncertain {column.replace('_', ' ')}.")

    if str(values.get("review_status", "")).strip() == "Reviewed — looks correct":
        uncertain = [
            c.replace("_", " ") for c in ["language", "sentiment", "text_type"]
            if str(values.get(c, "")).strip() in ("Uncertain", "— Select —", "")
        ]
        if uncertain:
            issues.append("Cannot mark as reviewed while labels are missing or uncertain: " + ", ".join(uncertain) + ".")

    if str(values.get("review_status", "")).strip() in ("Needs second review", "Ambiguous", "Skip") and not notes:
        issues.append("Add a note explaining this follow-up, ambiguous, or skipped item.")

    return issues

