import pandas as pd
import pytest

from datarecon import reconcile
from datarecon.cli import main


def frames():
    source = pd.DataFrame(
        {
            "Student_ID": ["001234", "005678", "009999", "000111", "000111", "000222"],
            "Name": ["John Smith", "Mary  Jones", "Ann Lee", "Bob Ray", "Bob Ray", "Cy Young"],
            "DOB": ["2026-09-01", "2010-01-15", "2011-03-03", "2012-04-04", "2012-04-04", "2013-05-05"],
            "Score": ["10.00", "20.5", "30", "40", "40", "50"],
            "Status": [None, "A", "B", "C", "C", "D"],
        }
    )
    target = pd.DataFrame(
        {
            "STUDENT_ID": [1234, 5678, 9999, 111, 333],
            "NAME": ["JOHN SMITH", "Mary Jones", "Anne Lee", "Bob Ray", "New Kid"],
            "DOB": ["09/01/2026", "2010-01-15", "2011-03-04", "2012-04-04", "2014-06-06"],
            "SCORE": [10.004, 20.5, 30, 40, 60],
            "STATUS": ["", "A", "B", "X", "E"],
        }
    )
    return source, target


def causes(result):
    return dict(zip(zip(result.differences["key"], result.differences["column"]), result.differences["cause"]))


def test_key_normalization_matches_leading_zeros():
    result = reconcile(*frames(), key="student_id")
    assert result.matched_keys == 4
    assert [k for k in result.missing_in_target["Student_ID"]] == ["000222"]
    assert list(result.missing_in_source["STUDENT_ID"]) == ["333"]


def test_causes_are_classified():
    result = reconcile(*frames(), key="student_id", rules={"score": {"tolerance": 0.01}}, fuzzy_threshold=0.8)
    found = causes(result)
    assert found[("001234", "Name")] == "case"
    assert found[("001234", "DOB")] == "date_format"
    assert found[("001234", "Status")] == "null_vs_blank"
    assert found[("005678", "Name")] == "whitespace"
    assert found[("009999", "Name")] == "similar_text"
    assert found[("009999", "DOB")] == "value_mismatch"
    assert found[("000111", "Status")] == "value_mismatch"
    assert ("001234", "Score") not in found  # within tolerance


def test_row_buckets_and_duplicates():
    result = reconcile(*frames(), key="student_id", rules={"score": {"tolerance": 0.01}})
    assert result.formatting_rows == 2
    assert result.mismatch_rows == 2
    assert result.exact_rows == 0
    assert result.duplicates["key"].tolist() == ["000111"]


def test_treat_as_equal():
    result = reconcile(*frames(), key="student_id", treat_as_equal={"case", "date_format", "null_vs_blank"})
    assert "case" not in set(result.differences["cause"])
    assert result.ignored_cells >= 3


def test_numeric_format_and_multi_key():
    a = pd.DataFrame({"a": ["1", "2"], "b": ["x", "y"], "v": ["0.0", "5"]})
    b = pd.DataFrame({"A": [1, 2], "B": ["X", "y"], "V": ["00", "5.0"]})
    result = reconcile(a, b, key=["a", "b"])
    assert result.matched_keys == 2
    assert set(result.differences["cause"]) == {"numeric_format"}


def test_missing_key_column_raises():
    a, b = frames()
    with pytest.raises(KeyError):
        reconcile(a, b, key="nope")


def test_summary_and_excel_and_cli(tmp_path):
    a, b = frames()
    a.to_csv(tmp_path / "a.csv", index=False)
    b.to_csv(tmp_path / "b.csv", index=False)
    report = tmp_path / "out.xlsx"
    code = main(["compare", str(tmp_path / "a.csv"), str(tmp_path / "b.csv"),
                 "--key", "student_id", "-o", str(report), "--fail-on-diff"])
    assert code == 1
    assert report.exists()
    sheets = pd.read_excel(report, sheet_name=None, dtype=str)
    assert {"Summary", "Differences", "Missing in target", "Duplicate keys"} <= set(sheets)
    assert "Reconciliation Summary" in reconcile(a, b, key="student_id").summary()


def test_identical_frames():
    a, _ = frames()
    result = reconcile(a, a, key="student_id")
    assert result.mismatch_rows == 0 and result.formatting_rows == 0
    assert result.duplicates["key"].tolist() == ["000111", "000111"]
