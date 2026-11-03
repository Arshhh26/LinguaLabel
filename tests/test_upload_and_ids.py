"""
Unit tests for LinguaLabel Part 1:
- CSV Upload & Decoding (UTF-8, UTF-8-sig, Latin-1, CP1252)
- Delimiter handling and malformed/empty data defense
- Text column detection and mapping
- Stable ID preservation and generation
- Dataset isolation & column preservation
"""

import os
import shutil
import tempfile
import unittest
import pandas as pd
from pathlib import Path

from data_manager import (
    load_csv_bytes,
    detect_text_column,
    stabilize_dataset,
    compute_dataset_key,
    get_dataset_storage_path,
    merge_saved_annotations,
    save_dataset_record,
    COLS,
)


class TestUploadAndStableIDs(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.root_path = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_empty_and_whitespace_bytes(self):
        df, enc, err = load_csv_bytes(b"")
        self.assertIsNone(df)
        self.assertIn("empty", err.lower())

        df, enc, err = load_csv_bytes(b"   \n  \n")
        self.assertIsNone(df)
        self.assertIn("empty", err.lower())

    def test_utf8_and_utf8_sig_loading(self):
        # Hindi and English in standard UTF-8
        content = "id,sentence\n1,Good morning\n2,नमस्ते भारत\n"
        df, enc, err = load_csv_bytes(content.encode("utf-8"))
        self.assertIsNotNone(df)
        self.assertIsNone(err)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[1]["sentence"], "नमस्ते भारत")

        # UTF-8 with BOM
        df_bom, enc_bom, err_bom = load_csv_bytes(content.encode("utf-8-sig"))
        self.assertIsNotNone(df_bom)
        self.assertIsNone(err_bom)
        self.assertEqual(df_bom.iloc[1]["sentence"], "नमस्ते भारत")

    def test_latin1_and_cp1252_encoding(self):
        # Accented characters common in Latin-1 / CP1252
        content = "id,text\n1,café au lait\n2,résumé review\n"
        raw_bytes = content.encode("latin-1")
        df, enc, err = load_csv_bytes(raw_bytes)
        self.assertIsNotNone(df)
        self.assertIsNone(err)
        self.assertEqual(df.iloc[0]["text"], "café au lait")

    def test_semicolon_and_tsv_delimiters(self):
        tsv_content = "id\ttext\tcategory\n1\tHello\tGreeting\n2\tWorld\tEntity\n"
        df_tsv, _, err = load_csv_bytes(tsv_content.encode("utf-8"))
        self.assertIsNotNone(df_tsv)
        self.assertEqual(len(df_tsv.columns), 3)

        semi_content = "id;text;category\n1;Hello;Greeting\n2;World;Entity\n"
        df_semi, _, err = load_csv_bytes(semi_content.encode("utf-8"))
        self.assertIsNotNone(df_semi)
        self.assertEqual(len(df_semi.columns), 3)

    def test_text_column_detection(self):
        # Common candidate names
        df1 = pd.DataFrame({"id": ["1"], "sentence": ["How are you?"], "meta": ["x"]})
        self.assertEqual(detect_text_column(df1), "sentence")

        df2 = pd.DataFrame({"id": ["1"], "customer_review": ["Great product!"], "score": [5]})
        self.assertEqual(detect_text_column(df2), "customer_review")

        # Fallback to column with highest average string length
        df3 = pd.DataFrame({"id": ["1"], "code": ["A"], "description": ["A long descriptive text sentence."]})
        self.assertEqual(detect_text_column(df3), "description")

    def test_preserve_valid_unique_ids(self):
        raw_df = pd.DataFrame({
            "id": ["101", "102", "103"],
            "sentence": ["One", "Two", "Three"],
            "custom_metadata": ["A", "B", "C"]
        })
        stable_df, info = stabilize_dataset(raw_df, text_column="sentence", dataset_key="test_key")
        self.assertFalse(info["id_generated"])
        self.assertEqual(list(stable_df["id"]), ["101", "102", "103"])
        # Original columns preserved
        self.assertIn("custom_metadata", stable_df.columns)
        self.assertIn("text", stable_df.columns)
        # Standard COLS initialized
        for col in COLS:
            self.assertIn(col, stable_df.columns)

    def test_generate_stable_ids_when_missing_or_duplicate(self):
        # Missing ID column
        no_id_df = pd.DataFrame({"sentence": ["First", "Second", "Third"]})
        res_no_id, info_no_id = stabilize_dataset(no_id_df, text_column="sentence", dataset_key="no_id")
        self.assertTrue(info_no_id["id_generated"])
        self.assertEqual(list(res_no_id["id"]), ["1", "2", "3"])

        # Duplicate IDs
        dup_id_df = pd.DataFrame({
            "id": ["1", "1", "2"],
            "sentence": ["First", "Second duplicate", "Third"]
        })
        res_dup, info_dup = stabilize_dataset(dup_id_df, text_column="sentence", dataset_key="dup_id")
        self.assertTrue(info_dup["id_generated"])
        self.assertEqual(list(res_dup["id"]), ["1", "2", "3"])
        # Preserved original ID
        self.assertIn("original_id", res_dup.columns)
        self.assertEqual(list(res_dup["original_id"]), ["1", "1", "2"])

    def test_dataset_isolation(self):
        # Default sample storage path
        default_path = get_dataset_storage_path("default_sample", self.root_path)
        self.assertEqual(default_path, self.root_path / "annotated_output.csv")

        # Custom uploaded dataset storage path
        custom_key = compute_dataset_key(b"sample_custom_content_bytes")
        custom_path = get_dataset_storage_path(custom_key, self.root_path)
        self.assertNotEqual(default_path, custom_path)
        self.assertIn("saved_sessions", str(custom_path))
        self.assertTrue(str(custom_path).endswith(f"annotated_{custom_key}.csv"))

    def test_save_and_merge_annotations(self):
        custom_key = "test_custom_1"
        storage_path = get_dataset_storage_path(custom_key, self.root_path)

        raw_df = pd.DataFrame({
            "id": ["1", "2"],
            "text": ["Hello", "World"],
            "extra_info": ["x1", "x2"]
        })
        stable_df, _ = stabilize_dataset(raw_df, "text", custom_key)

        # Save annotation for item 1
        success = save_dataset_record(
            storage_path=storage_path,
            current_df=stable_df,
            row_id="1",
            vals={
                "language": "English",
                "sentiment": "Positive",
                "text_type": "Greeting",
                "review_status": "Reviewed — looks correct",
                "annotator_notes": "All clear."
            }
        )
        self.assertTrue(success)
        self.assertTrue(storage_path.exists())

        # Reload and merge into fresh dataframe
        reloaded = merge_saved_annotations(stable_df, storage_path)
        self.assertEqual(reloaded.iloc[0]["language"], "English")
        self.assertEqual(reloaded.iloc[0]["sentiment"], "Positive")
        self.assertEqual(reloaded.iloc[0]["extra_info"], "x1")
        self.assertEqual(reloaded.iloc[1]["language"], "")


if __name__ == "__main__":
    unittest.main()
