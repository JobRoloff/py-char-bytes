"""Unit tests for CLEAR corpus loading and normalization."""

import pandas as pd
import pytest

from evals.datasets.clear import load_clear_dataframe


def test_load_clear_dataframe_normalizes_expected_columns(tmp_path):
    csv_path = tmp_path / "clear_sample.csv"
    pd.DataFrame(
        {
            "ID": [101, 102, 103],
            "Excerpt": ["Easy text.", None, "Harder technical text."],
            "BT Easiness": [0.5, 0.1, -1.25],
            "unused": ["x", "y", "z"],
        }
    ).to_csv(csv_path, index=False)

    df = load_clear_dataframe(csv_path)

    assert list(df.columns) == ["document_id", "raw_text", "clear_easiness"]
    assert df.to_dict(orient="records") == [
        {"document_id": "101", "raw_text": "Easy text.", "clear_easiness": 0.5},
        {
            "document_id": "103",
            "raw_text": "Harder technical text.",
            "clear_easiness": -1.25,
        },
    ]


def test_load_clear_dataframe_raises_for_missing_required_columns(tmp_path):
    csv_path = tmp_path / "bad_clear_sample.csv"
    pd.DataFrame({"id": [1], "excerpt": ["Missing score."]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_clear_dataframe(csv_path)
