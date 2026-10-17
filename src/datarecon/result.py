from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import pandas as pd


@dataclass
class ReconResult:
    source_rows: int
    target_rows: int
    matched_keys: int
    exact_rows: int
    formatting_rows: int
    mismatch_rows: int
    ignored_cells: int
    differences: pd.DataFrame
    missing_in_target: pd.DataFrame
    missing_in_source: pd.DataFrame
    duplicates: pd.DataFrame
    null_keys_source: int
    null_keys_target: int
    columns_only_in_source: list[str]
    columns_only_in_target: list[str]
    compared_columns: list[str]

    @property
    def has_differences(self) -> bool:
        return bool(
            len(self.differences)
            or len(self.missing_in_target)
            or len(self.missing_in_source)
            or len(self.duplicates)
            or self.null_keys_source
            or self.null_keys_target
            or self.columns_only_in_source
            or self.columns_only_in_target
        )

    def cause_summary(self) -> pd.DataFrame:
        return (
            self.differences["cause"]
            .value_counts()
            .rename_axis("cause")
            .reset_index(name="cells")
        )

    def column_summary(self) -> pd.DataFrame:
        if self.differences.empty:
            return pd.DataFrame(columns=["column", "differences", "top_cause"])
        grouped = self.differences.groupby("column")["cause"]
        out = pd.DataFrame(
            {"differences": grouped.size(), "top_cause": grouped.agg(lambda s: s.value_counts().index[0])}
        )
        return out.sort_values("differences", ascending=False, kind="stable").reset_index()

    def metrics(self) -> pd.DataFrame:
        dup_src = int((self.duplicates["side"] == "source").sum()) if len(self.duplicates) else 0
        dup_tgt = int((self.duplicates["side"] == "target").sum()) if len(self.duplicates) else 0
        rows = [
            ("Source rows", self.source_rows),
            ("Target rows", self.target_rows),
            ("Matched keys", self.matched_keys),
            ("Exact matches", self.exact_rows),
            ("Formatting differences only", self.formatting_rows),
            ("Value mismatches", self.mismatch_rows),
            ("Missing in target", len(self.missing_in_target)),
            ("Missing in source", len(self.missing_in_source)),
            ("Duplicate keys in source", dup_src),
            ("Duplicate keys in target", dup_tgt),
            ("Null keys in source", self.null_keys_source),
            ("Null keys in target", self.null_keys_target),
        ]
        return pd.DataFrame(rows, columns=["metric", "value"])

    def summary(self, top: int = 10) -> str:
        lines = ["Reconciliation Summary", "-" * 42]
        metrics = self.metrics()
        always = {"Source rows", "Target rows", "Matched keys", "Exact matches"}
        for label, value in metrics.itertuples(index=False):
            if label in always or value:
                indent = "  " if label in {"Exact matches", "Formatting differences only", "Value mismatches"} else ""
                lines.append(f"{indent}{label + ':':<30}{value:>10,}")
        if self.ignored_cells:
            lines.append(f"{'Cells treated as equal:':<32}{self.ignored_cells:>10,}")

        if len(self.differences):
            lines += ["", "Differences by cause:"]
            for cause, cells in self.cause_summary().itertuples(index=False):
                lines.append(f"  {cause:<28}{cells:>10,}")
            lines += ["", "Most problematic columns:"]
            for i, (col, n, cause) in enumerate(self.column_summary().head(top).itertuples(index=False), 1):
                lines.append(f"  {i:>2}. {col:<24}{n:>8,}  (mostly {cause})")

        if self.columns_only_in_source:
            lines += ["", "Columns only in source: " + ", ".join(self.columns_only_in_source)]
        if self.columns_only_in_target:
            lines += ["Columns only in target: " + ", ".join(self.columns_only_in_target)]
        return "\n".join(lines)

    def to_excel(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        schema = pd.DataFrame(
            [("compared", c) for c in self.compared_columns]
            + [("only_in_source", c) for c in self.columns_only_in_source]
            + [("only_in_target", c) for c in self.columns_only_in_target],
            columns=["status", "column"],
        )
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            self.metrics().to_excel(writer, sheet_name="Summary", index=False)
            if len(self.differences):
                start = len(self.metrics()) + 3
                self.cause_summary().to_excel(writer, sheet_name="Summary", index=False, startrow=start)
                self.column_summary().to_excel(
                    writer, sheet_name="Summary", index=False, startrow=start + len(self.cause_summary()) + 3
                )
            self.differences.to_excel(writer, sheet_name="Differences", index=False)
            self.missing_in_target.to_excel(writer, sheet_name="Missing in target", index=False)
            self.missing_in_source.to_excel(writer, sheet_name="Missing in source", index=False)
            self.duplicates.to_excel(writer, sheet_name="Duplicate keys", index=False)
            schema.to_excel(writer, sheet_name="Schema", index=False)
        return path
