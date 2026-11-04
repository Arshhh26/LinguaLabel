"""
Integration tests for Part 1:
- Default dataset vs. Uploaded CSV lifecycle
- Data isolation (sample_data.csv annotations never overwritten)
- Custom column preservation on export
- Stable ID generation when IDs are duplicate or absent
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


class TestPart1Integration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.root_path = Path(self.temp_dir)
        self.default_data_file = self.root_path / "sample_data.csv"
        self.default_out_file = self.root_path / "annotated_output.csv"

        # Create mock sample_data.csv
        sample_content = (
            "id,text,language,sentiment,text_type,review_status,annotator_notes\n"
            "1,I am feeling happy today.,,,,,\n"
            "2,मुझे आज बहुत खुशी हो रही है।,,,,,\n"
        )
        self.default_data_file.write_text(sample_content, encoding="utf-8")

        # Create mock annotated_output.csv
        annotated_content = (
            "id,text,language,sentiment,text_type,review_status,annotator_notes\n"
            "1,I am feeling happy today.,English,Positive,Statement,Reviewed — looks correct,\n"
        )
        self.default_out_file.write_text(annotated_content, encoding="utf-8-sig")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_default_sample_lifecycle(self):
        raw_df = pd.read_csv(self.default_data_file, keep_default_na=False, dtype=str)
        key = "default_sample"
        out_path = get_dataset_storage_path(key, self.root_path)
        self.assertEqual(out_path, self.default_out_file)

        stable_df, info = stabilize_dataset(raw_df, text_column="text", dataset_key=key)
        self.assertFalse(info["id_generated"])

        # Merge saved annotations
        merged = merge_saved_annotations(stable_df, out_path)
        self.assertEqual(merged.iloc[0]["review_status"], "Reviewed — looks correct")
        self.assertEqual(merged.iloc[0]["language"], "English")
        self.assertEqual(merged.iloc[1]["review_status"], "")

        # Save annotation for item 2
        success = save_dataset_record(
            storage_path=out_path,
            current_df=merged,
            row_id="2",
            vals={
                "language": "Hindi",
                "sentiment": "Positive",
                "text_type": "Statement",
                "review_status": "Reviewed — looks correct",
                "annotator_notes": "Annotated via test",
            }
        )
        self.assertTrue(success)

        # Re-read default output file and confirm update
        updated = pd.read_csv(self.default_out_file, keep_default_na=False, dtype=str)
        self.assertEqual(len(updated), 2)
        self.assertEqual(updated.iloc[1]["language"], "Hindi")

    def test_uploaded_csv_lifecycle_and_isolation(self):
        # Custom dataset with:
        # 1. Custom text column name ("user_query")
        # 2. No ID column
        # 3. Custom metadata ("channel", "timestamp")
        custom_csv = (
            "channel,timestamp,user_query\n"
            "web,2026-10-22,Where is my delivery?\n"
            "app,2026-10-22,क्या मेरी ट्रेन समय पर है?\n"
        ).encode("utf-8")

        parsed_df, enc, err = load_csv_bytes(custom_csv)
        self.assertIsNone(err)
        detected_col = detect_text_column(parsed_df)
        self.assertEqual(detected_col, "user_query")

        dataset_key = compute_dataset_key(custom_csv)
        self.assertNotEqual(dataset_key, "default_sample")

        storage_path = get_dataset_storage_path(dataset_key, self.root_path)
        self.assertNotEqual(storage_path, self.default_out_file)

        # Stabilize
        stable_df, info = stabilize_dataset(parsed_df, text_column=detected_col, dataset_key=dataset_key)
        self.assertTrue(info["id_generated"])
        self.assertEqual(len(stable_df["id"]), 2)
        self.assertTrue(all(x.startswith(f"{dataset_key}_") for x in stable_df["id"]))
        # Original columns preserved
        self.assertIn("channel", stable_df.columns)
        self.assertIn("timestamp", stable_df.columns)
        self.assertIn("text", stable_df.columns)

        # Save annotation for custom item 1
        item_1_id = stable_df.iloc[0]["id"]
        save_success = save_dataset_record(
            storage_path=storage_path,
            current_df=stable_df,
            row_id=item_1_id,
            vals={
                "language": "English",
                "sentiment": "Neutral",
                "text_type": "Question",
                "review_status": "Reviewed — looks correct",
                "annotator_notes": "Order query",
            }
        )
        self.assertTrue(save_success)

        # Verify custom file written
        self.assertTrue(storage_path.exists())
        custom_saved = pd.read_csv(storage_path, keep_default_na=False, dtype=str)
        self.assertEqual(custom_saved.iloc[0]["channel"], "web")
        self.assertEqual(custom_saved.iloc[0]["language"], "English")

        # CRITICAL TEST: Verify default sample output was NOT modified!
        default_after = pd.read_csv(self.default_out_file, keep_default_na=False, dtype=str)
        self.assertEqual(len(default_after), 1)
        self.assertEqual(default_after.iloc[0]["id"], "1")
        self.assertEqual(default_after.iloc[0]["text"], "I am feeling happy today.")


if __name__ == "__main__":
    unittest.main()
