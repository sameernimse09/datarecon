# datarecon

Compare two datasets and get an explanation of *why* they differ, not just a list of mismatched cells.

```python
from datarecon import reconcile

result = reconcile(
    "source.csv", "warehouse_export.csv",
    key="student_id",
    rules={"score": {"tolerance": 0.01}},
    treat_as_equal={"case", "whitespace"},
)
print(result.summary())
result.to_excel("reconciliation.xlsx")
```

```
datarecon compare source.csv target.csv --key student_id -o report.xlsx --fail-on-diff
```

## What it detects

| Cause | Example |
|---|---|
| `leading_zeros` | `00123` vs `123` |
| `null_vs_blank` | `None` vs `""` vs `NULL` |
| `whitespace` / `case` | `Mary  Jones` vs `MARY JONES` |
| `numeric_format` | `0.0` vs `00`, `10` vs `10.00` |
| `numeric_tolerance` | `10.004` vs `10.00` with `tolerance=0.01` |
| `date_format` | `2026-09-01` vs `09/01/2026` (month-first) |
| `similar_text` | `Ann Lee` vs `Anne Lee` (opt-in via `fuzzy_threshold`) |
| `value_mismatch` | everything else |

Also reported: rows missing on either side, duplicate keys, null keys, columns present on only one side.

Key values are matched after trimming, case folding and stripping leading zeros. Files are read as strings so no information is lost before comparison.

## Development

```
pip install -e ".[dev]"
pytest
```
