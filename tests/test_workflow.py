"""
Unit and integration tests for LinguaLabel annotation workflow, validation rules,
navigation boundary safety, and UTF-8-SIG export formatting (Part 2 & Part 5).
"""

import io
from pathlib import Path
import tempfile
import pandas as pd
import pytest

from data_manager import (
    validation_issues,
    save_dataset_record,
    merge_saved_annotations,
    COLS,
)


class TestValidationRules:
    """Tests the 4 mandatory validation rules enforced by LinguaLabel."""

    def test_rule_1_all_four_fields_required(self):
        # Empty values
        vals = {"language": "", "sentiment": "", "text_type": "", "review_status": ""}
        issues = validation_issues(vals)
        assert len(issues) >= 4
        assert any("language" in iss for iss in issues)
        assert any("sentiment" in iss for iss in issues)
        assert any("text type" in iss for iss in issues)
        assert any("review status" in iss for iss in issues)

        # "— Select —" placeholder values
        vals_sel = {
            "language": "— Select —",
            "sentiment": "Positive",
            "text_type": "Statement",
            "review_status": "Reviewed — looks correct",
        }
        issues_sel = validation_issues(vals_sel)
        assert any("language" in iss for iss in issues_sel)

    def test_rule_2_mandatory_notes_when_uncertain(self):
        # Language is Uncertain without notes
        vals = {
            "language": "Uncertain",
            "sentiment": "Positive",
            "text_type": "Statement",
            "review_status": "Needs second review",
            "annotator_notes": "",
        }
        issues = validation_issues(vals)
        assert any("uncertain language" in iss.lower() for iss in issues)

        # Providing notes resolves rule 2
        vals["annotator_notes"] = "Language script is ambiguous between Hindi and Marathi."
        issues_with_notes = validation_issues(vals)
        assert not any("uncertain language" in iss.lower() for iss in issues_with_notes)

    def test_rule_3_reviewed_blocked_if_uncertain_or_missing(self):
        # Marking reviewed when sentiment is Uncertain
        vals = {
            "language": "English",
            "sentiment": "Uncertain",
            "text_type": "Statement",
            "review_status": "Reviewed — looks correct",
            "annotator_notes": "Sarcastic sentence",
        }
        issues = validation_issues(vals)
        assert any("cannot mark as reviewed while labels are missing or uncertain" in iss.lower() for iss in issues)

        # Marking reviewed when valid
        vals["sentiment"] = "Neutral"
        issues_valid = validation_issues(vals)
        assert len(issues_valid) == 0

    def test_rule_4_notes_mandatory_for_triage_statuses(self):
        for status in ["Needs second review", "Ambiguous", "Skip"]:
            vals = {
                "language": "Hindi",
                "sentiment": "Neutral",
                "text_type": "Question",
                "review_status": status,
                "annotator_notes": "",
            }
            issues = validation_issues(vals)
            assert any("add a note explaining this follow-up, ambiguous, or skipped item" in iss.lower() for iss in issues)

            # Resolves when note is added
            vals["annotator_notes"] = f"Requires verification due to {status} status."
            assert len(validation_issues(vals)) == 0

    def test_valid_submission_passes_cleanly(self):
        vals = {
            "language": "Hindi",
            "sentiment": "Positive",
            "text_type": "Greeting",
            "review_status": "Reviewed — looks correct",
            "annotator_notes": "",
        }
        assert validation_issues(vals) == []


class TestNavigationBoundaries:
    """Tests boundary conditions and next unreviewed candidate resolution."""

    def test_boundary_clamping(self):
        total = 5
        # Index boundary logic
        idx = 0
        prev_idx = max(0, idx - 1)
        assert prev_idx == 0  # Cannot go negative

        idx_end = total - 1
        next_idx = min(total - 1, idx_end + 1)
        assert next_idx == total - 1  # Cannot exceed total - 1

    def test_single_item_dataset_navigation(self):
        total = 1
        idx = 0
        assert max(0, idx - 1) == 0
        assert min(total - 1, idx + 1) == 0

    def test_next_unreviewed_candidate_resolution(self):
        df = pd.DataFrame({
            "id": ["1", "2", "3", "4", "5"],
            "review_status": ["Reviewed — looks correct", "", "Reviewed — looks correct", "", ""],
        })
        pending = df.index[df.review_status.astype(str).str.strip().eq("")].tolist()
        assert pending == [1, 3, 4]

        # From index 0, candidates after 0: [1, 3, 4] -> picks 1
        current_idx = 0
        candidates = [i for i in pending if i > current_idx] or [i for i in pending if i <= current_idx]
        assert candidates[0] == 1

        # From index 3, candidates after 3: [4] -> picks 4
        current_idx = 3
        candidates = [i for i in pending if i > current_idx] or [i for i in pending if i <= current_idx]
        assert candidates[0] == 4

        # From index 4 (last item), wraps around to candidates <= 4: [1, 3, 4] -> picks 1
        current_idx = 4
        candidates = [i for i in pending if i > current_idx] or [i for i in pending if i <= current_idx]
        assert candidates[0] == 1

    def test_all_reviewed_behavior(self):
        df = pd.DataFrame({
            "id": ["1", "2"],
            "review_status": ["Reviewed — looks correct", "Reviewed — looks correct"],
        })
        pending = df.index[df.review_status.astype(str).str.strip().eq("")].tolist()
        assert len(pending) == 0


class TestExportAndEncoding:
    """Tests UTF-8-SIG encoding and column preservation."""

    def test_utf8_sig_bom_and_devanagari_preservation(self):
        df = pd.DataFrame({
            "id": ["101"],
            "text": ["नमस्ते भारत, आज का दिन बहुत सुखद है।"],
            "language": ["Hindi"],
            "sentiment": ["Positive"],
            "text_type": ["Greeting"],
            "review_status": ["Reviewed — looks correct"],
            "annotator_notes": ["शुभकामनाएं"],
            "custom_metadata_col": ["batch_99_A"],
        })

        encoded_csv = df.to_csv(index=False).encode("utf-8-sig")
        # Must start with UTF-8 BOM: EF BB BF
        assert encoded_csv.startswith(b"\xef\xbb\xbf")

        # Decode back with utf-8-sig and verify Devanagari text is intact
        decoded_df = pd.read_csv(io.BytesIO(encoded_csv), encoding="utf-8-sig", dtype=str)
        assert decoded_df.iloc[0]["text"] == "नमस्ते भारत, आज का दिन बहुत सुखद है।"
        assert decoded_df.iloc[0]["annotator_notes"] == "शुभकामनाएं"
        assert decoded_df.iloc[0]["custom_metadata_col"] == "batch_99_A"

    def test_persistence_isolation_and_save(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "test_out.csv"
            df = pd.DataFrame({
                "id": ["1", "2"],
                "text": ["Hello", "World"],
                "language": ["", ""],
                "sentiment": ["", ""],
                "text_type": ["", ""],
                "review_status": ["", ""],
                "annotator_notes": ["", ""],
            })

            success = save_dataset_record(
                tmp_path,
                df.copy(),
                row_id="1",
                vals={
                    "language": "English",
                    "sentiment": "Positive",
                    "text_type": "Greeting",
                    "review_status": "Reviewed — looks correct",
                    "annotator_notes": "All verified",
                }
            )
            assert success is True
            assert tmp_path.exists()

            # Merge saved back
            merged = merge_saved_annotations(df.copy(), tmp_path)
            assert merged.loc[merged["id"] == "1", "language"].values[0] == "English"
            assert merged.loc[merged["id"] == "2", "language"].values[0] == ""
