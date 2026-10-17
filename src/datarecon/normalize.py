"""Cell-level comparison logic: decide *why* two values differ."""
from __future__ import annotations

import math
import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import Optional, Sequence

FORMATTING_CAUSES = frozenset(
    {"null_vs_blank", "whitespace", "case", "leading_zeros", "numeric_format", "date_format"}
)

DEFAULT_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%m/%d/%Y",
    "%m/%d/%Y %H:%M:%S",
    "%Y/%m/%d",
    "%m-%d-%Y",
    "%Y%m%d",
    "%d-%b-%Y",
    "%b %d, %Y",
)

_NULL_TOKENS = frozenset({"", "null", "none", "nan"})
_NUM_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
_KEY_NUM_RE = re.compile(r"^\d+(\.0+)?$")


def is_null(value: Optional[str]) -> bool:
    return value is None or value.strip().casefold() in _NULL_TOKENS


def normalize_key_part(value: Optional[str]) -> Optional[str]:
    """Canonical form of one key column value used only for matching rows."""
    if value is None:
        return None
    text = value.strip().casefold()
    if not text:
        return None
    if _KEY_NUM_RE.match(text):
        text = text.split(".")[0].lstrip("0") or "0"
    return text


def _to_float(text: str) -> Optional[float]:
    cleaned = text.replace(",", "")
    if not _NUM_RE.match(cleaned):
        return None
    number = float(cleaned)
    return number if math.isfinite(number) else None


def _to_datetime(text: str, formats: Sequence[str]) -> Optional[datetime]:
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def classify(
    source: Optional[str],
    target: Optional[str],
    rule: dict,
    fuzzy_default: Optional[float] = None,
) -> str:
    """Return the cause label for a pair of values already known to differ."""
    s_null, t_null = is_null(source), is_null(target)
    if s_null and t_null:
        return "null_vs_blank"
    if s_null or t_null:
        return "value_mismatch"

    if source.strip() == target.strip():
        return "whitespace"
    a, b = " ".join(source.split()), " ".join(target.split())
    if a == b:
        return "whitespace"
    if a.casefold() == b.casefold():
        return "case"
    if a.isdigit() and b.isdigit() and a.lstrip("0") == b.lstrip("0"):
        return "leading_zeros"

    fa, fb = _to_float(a), _to_float(b)
    if fa is not None and fb is not None:
        if fa == fb:
            return "numeric_format"
        tolerance = rule.get("tolerance")
        if tolerance is not None and abs(fa - fb) <= tolerance:
            return "numeric_tolerance"
        return "value_mismatch"

    formats = rule.get("date_formats", DEFAULT_DATE_FORMATS)
    da, db = _to_datetime(a, formats), _to_datetime(b, formats)
    if da is not None and db is not None:
        return "date_format" if da == db else "value_mismatch"

    threshold = rule.get("fuzzy", fuzzy_default)
    if threshold is not None:
        if SequenceMatcher(None, a.casefold(), b.casefold()).ratio() >= threshold:
            return "similar_text"
    return "value_mismatch"
