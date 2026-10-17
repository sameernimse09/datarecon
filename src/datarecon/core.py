from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .normalize import FORMATTING_CAUSES, classify, normalize_key_part
from .result import ReconResult

Source = Union[pd.DataFrame, str, Path]

_DEFAULT_IGNORED = frozenset({"numeric_tolerance"})


def _norm_col(name: Any) -> str:
    return str(name).strip().casefold()


def _read(obj: Source) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    path = Path(obj)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path, dtype=str, keep_default_na=False)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path.suffix!r}")


def _is_na(value: Any) -> bool:
    return value is None or (not isinstance(value, str) and bool(pd.isna(value)))


def _prepare(obj: Source) -> tuple[pd.DataFrame, dict[str, str]]:
    """Convert to an all-string frame (None for nulls) with normalized column names."""
    df = _read(obj)
    data: dict[str, list] = {}
    display: dict[str, str] = {}
    for col in df.columns:
        name = _norm_col(col)
        if name in data:
            raise ValueError(f"Duplicate column after normalization: {col!r}")
        data[name] = [None if _is_na(v) else str(v) for v in df[col].tolist()]
        display[name] = str(col).strip()
    return pd.DataFrame(data), display


def _add_key(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    parts = [df[k].map(normalize_key_part).tolist() for k in keys]
    raw = [df[k].tolist() for k in keys]
    df = df.copy()
    df["__key__"] = [
        None if any(p is None for p in row) else "|".join(row) for row in zip(*parts)
    ]
    df["__label__"] = ["|".join("" if v is None else v for v in row) for row in zip(*raw)]
    return df


def _duplicates(df: pd.DataFrame, side: str) -> pd.DataFrame:
    counts = df.groupby("__key__").agg(key=("__label__", "first"), occurrences=("__label__", "size"))
    counts = counts[counts["occurrences"] > 1].reset_index(drop=True)
    counts.insert(0, "side", side)
    return counts


def _public(df: pd.DataFrame, display: Mapping[str, str]) -> pd.DataFrame:
    out = df.drop(columns=["__label__"], errors="ignore").reset_index(drop=True)
    return out.rename(columns=display)


def reconcile(
    source: Source,
    target: Source,
    key: Union[str, Sequence[str]],
    *,
    rules: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ignore_columns: Iterable[str] = (),
    treat_as_equal: Optional[Iterable[str]] = None,
    fuzzy_threshold: Optional[float] = None,
) -> ReconResult:
    """Compare two datasets and explain every difference.

    Parameters
    ----------
    key
        Column name (or list of names) that identifies a row. Names are matched
        case-insensitively. Key values are compared after trimming, case folding
        and removing leading zeros, so ``001234`` matches ``1234``.
    rules
        Per-column options: ``tolerance`` (absolute numeric tolerance),
        ``fuzzy`` (0-1 similarity threshold), ``date_formats`` (list of strptime formats).
    treat_as_equal
        Cause labels to treat as equal, for example ``{"case", "whitespace"}``.
        ``numeric_tolerance`` is always treated as equal.
    fuzzy_threshold
        Default similarity threshold for labelling near-identical text ``similar_text``.
    """
    keys = [key] if isinstance(key, str) else list(key)
    keys_n = [_norm_col(k) for k in keys]

    src, src_names = _prepare(source)
    tgt, tgt_names = _prepare(target)
    for original, name in zip(keys, keys_n):
        for side, frame in (("source", src), ("target", tgt)):
            if name not in frame.columns:
                raise KeyError(f"Key column {original!r} not found in {side}")

    ignore = {_norm_col(c) for c in ignore_columns}
    rules_n = {_norm_col(c): dict(r) for c, r in (rules or {}).items()}
    ignored_causes = set(_DEFAULT_IGNORED) | set(treat_as_equal or ())

    common = [c for c in src.columns if c in tgt.columns and c not in keys_n and c not in ignore]
    only_src = [src_names[c] for c in src.columns if c not in tgt.columns]
    only_tgt = [tgt_names[c] for c in tgt.columns if c not in src.columns]
    compared = [src_names[c] for c in common]

    src_rows, tgt_rows = len(src), len(tgt)
    src, tgt = _add_key(src, keys_n), _add_key(tgt, keys_n)
    null_src, null_tgt = int(src["__key__"].isna().sum()), int(tgt["__key__"].isna().sum())
    src, tgt = src[src["__key__"].notna()], tgt[tgt["__key__"].notna()]

    duplicates = pd.concat([_duplicates(src, "source"), _duplicates(tgt, "target")], ignore_index=True)

    s_idx = src.drop_duplicates("__key__", keep="first").set_index("__key__")
    t_idx = tgt.drop_duplicates("__key__", keep="first").set_index("__key__")

    s_in_t = s_idx.index.isin(t_idx.index)
    t_in_s = t_idx.index.isin(s_idx.index)
    missing_in_target = _public(s_idx[~s_in_t], src_names)
    missing_in_source = _public(t_idx[~t_in_s], tgt_names)

    both = s_idx.index[s_in_t]
    sa, ta = s_idx.loc[both, common], t_idx.loc[both, common]
    labels = s_idx.loc[both, "__label__"].tolist()

    equal = ((sa == ta) | (sa.isna() & ta.isna())).to_numpy()
    sv = sa.astype(object).where(sa.notna(), None).to_numpy()
    tv = ta.astype(object).where(ta.notna(), None).to_numpy()

    records, rows_any, rows_real, ignored_cells = [], set(), set(), 0
    for r, c in zip(*np.nonzero(~equal)):
        col = common[c]
        cause = classify(sv[r, c], tv[r, c], rules_n.get(col, {}), fuzzy_threshold)
        if cause in ignored_causes:
            ignored_cells += 1
            continue
        rows_any.add(r)
        if cause not in FORMATTING_CAUSES:
            rows_real.add(r)
        records.append((labels[r], src_names[col], sv[r, c], tv[r, c], cause))

    differences = pd.DataFrame(records, columns=["key", "column", "source", "target", "cause"])

    return ReconResult(
        source_rows=src_rows,
        target_rows=tgt_rows,
        matched_keys=len(both),
        exact_rows=len(both) - len(rows_any),
        formatting_rows=len(rows_any - rows_real),
        mismatch_rows=len(rows_real),
        ignored_cells=ignored_cells,
        differences=differences,
        missing_in_target=missing_in_target,
        missing_in_source=missing_in_source,
        duplicates=duplicates,
        null_keys_source=null_src,
        null_keys_target=null_tgt,
        columns_only_in_source=only_src,
        columns_only_in_target=only_tgt,
        compared_columns=compared,
    )
