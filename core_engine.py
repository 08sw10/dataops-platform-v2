"""
core_engine.py
--------------
Pure Pandas transformation logic for the DataOps Transformation Platform.
No file I/O — all operations run on in-memory DataFrames.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Characters allowed after "Remove Special Characters" transformation.
# Keeps letters (a-z, A-Z), digits (0-9), and plain spaces.
# Adjust only this constant to change the allowed character set globally.
_ALLOWED_CHARS_PATTERN: re.Pattern = re.compile(r"[^a-zA-Z0-9 ]")

# Hard limit consistent with the API and UI paths.
_MAX_BYTES: int = 50 * 1024 * 1024  # 50 MB


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------


def validate_dataframe(df: pd.DataFrame) -> None:
    """
    Validate an incoming DataFrame before transformation.

    Raises:
        ValueError: with a clear, user-friendly message when:
          - The dataframe is empty (zero rows).
          - The dataframe has no column headers.
          - The dataframe exceeds 50 MB when serialised to memory.
    """
    if df.empty:
        raise ValueError(
            "The uploaded file is empty. Please upload a file that contains at least one row of data."
        )

    if len(df.columns) == 0:
        raise ValueError(
            "The uploaded file has no column headers. "
            "Please ensure the first row of your file contains column names."
        )

    # Approximate in-memory size check (covers most real-world cases without
    # requiring a full re-serialisation, which would double peak memory usage).
    estimated_bytes = df.memory_usage(deep=True).sum()
    if estimated_bytes > _MAX_BYTES:
        raise ValueError(
            f"The uploaded file is too large ({estimated_bytes / 1_048_576:.1f} MB). "
            "Please upload a file smaller than 50 MB."
        )


# ---------------------------------------------------------------------------
# Individual rule implementations (all vectorised — no row-by-row loops)
# ---------------------------------------------------------------------------


def _trim_whitespace(series: pd.Series) -> pd.Series:
    """Strip leading/trailing spaces and collapse interior runs of spaces to one."""
    # .str accessor works on object/string dtype; skip if numeric
    if pd.api.types.is_numeric_dtype(series):
        return series
    return series.str.strip().str.replace(r" {2,}", " ", regex=True)


def _to_uppercase(series: pd.Series) -> pd.Series:
    return series.str.upper()


def _to_lowercase(series: pd.Series) -> pd.Series:
    return series.str.lower()


def _to_titlecase(series: pd.Series) -> pd.Series:
    # Collapse interior whitespace runs before title-casing so that
    # e.g. "jane    smith" becomes "Jane Smith" rather than "Jane    Smith".
    collapsed = series.str.replace(r" {2,}", " ", regex=True)
    return collapsed.str.strip().str.title()


def _remove_special_chars(series: pd.Series) -> pd.Series:
    """Remove any character not matching [a-zA-Z0-9 ] using the named constant."""
    return series.str.replace(_ALLOWED_CHARS_PATTERN, "", regex=True)


def _fill_blanks_with_median(
    series: pd.Series, col_name: str, changes: List[str]
) -> pd.Series:
    """
    Replace NaN with the column median.
    Skips non-numeric columns and appends a warning instead of raising.
    Returns (transformed_series, blanks_filled_count).
    """
    if not pd.api.types.is_numeric_dtype(series):
        changes.append(
            f"Skipped 'Fill Blanks with Median' on non-numeric column '{col_name}'"
        )
        return series
    return series.fillna(series.median())


def _fill_blanks_with_zero(series: pd.Series) -> pd.Series:
    return series.fillna(0)


def _standardize_dates(
    series: pd.Series,
    col_name: str,
    date_parse_failures: Dict[str, int],
) -> pd.Series:
    """
    Parse mixed date formats into 'YYYY-MM-DD' strings via pd.to_datetime.
    Records the count of values that could not be parsed (became NaT).
    """
    converted = pd.to_datetime(series, errors="coerce")
    nat_count = int(converted.isna().sum())
    if nat_count > 0:
        date_parse_failures[col_name] = nat_count
    # Format successfully-parsed values; leave NaT as NaT (becomes NaN in object col)
    return converted.dt.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Master transformation function
# ---------------------------------------------------------------------------


def apply_transformation_rules(
    df: pd.DataFrame,
    column_rules: Dict[str, str],
    type_overrides: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply column-level transformation rules to a DataFrame.

    Execution order
    ---------------
    0. (Optional) Type casting via *type_overrides* — runs first, before any rule.
    1. All column-wise rules are applied in the order columns appear in *df*.
    2. After all column-wise rules finish, "Remove Duplicates" is applied
       once across the combined set of columns that had that rule selected.
       This guarantees deduplication operates on fully-cleaned values.

    Args:
        df:             Input DataFrame (never mutated — a copy is taken immediately).
        column_rules:   Mapping of column name -> rule name.
                        Example: {"Email": "Trim Whitespace", "Sales": "Fill Blanks with 0"}
        type_overrides: Optional mapping of column name -> target type string.
                        Supported values: "Text", "Numeric", "Datetime".
                        Example: {"NationalID": "Text", "Revenue": "Numeric"}
                        Text casting preserves leading zeros and safely clears
                        stringified null-like entries ("nan", "None", "<NA>").

    Returns:
        (cleaned_df, metadata)

    Metadata shape
    --------------
    {
        "rows_before":        int,
        "rows_after":         int,
        "changes_applied":    list[str],   # human-readable log lines
        "date_parse_failures":dict[str, int],  # {col: NaT count}
        "blanks_filled":      int,         # total NaN replaced across all columns
        "duplicates_removed": int,         # rows_before - rows_after
    }
    """
    # Work on a defensive copy so the caller's DataFrame is never mutated.
    result = df.copy()

    rows_before: int = len(result)
    changes_applied: List[str] = []
    date_parse_failures: Dict[str, int] = {}
    blanks_filled: int = 0
    dedup_columns: List[str] = []  # accumulate columns that want "Remove Duplicates"

    # ------------------------------------------------------------------
    # Phase 0: Manual type overrides — cast columns before any rule runs.
    # This is critical for fields like National IDs where Pandas infers
    # the wrong dtype and strips leading zeros on read.
    # ------------------------------------------------------------------
    if type_overrides:
        # Stringified null-like tokens produced by .astype(str) on NaN/NA values.
        _NULL_STRINGS: tuple = ("nan", "none", "<na>", "nat", "")

        for col, target_type in type_overrides.items():
            if col not in result.columns:
                # Column referenced in overrides but absent in the file — skip silently.
                continue
            if target_type == "No Override":
                continue

            try:
                if target_type == "Text":
                    # Convert to string first to preserve leading zeros (e.g. "007").
                    result[col] = result[col].astype(str)
                    # Re-introduce true NA for values that became null-like strings.
                    # Use vectorised .str.lower() for a case-insensitive comparison.
                    null_mask = result[col].str.strip().str.lower().isin(_NULL_STRINGS)
                    result[col] = result[col].where(~null_mask, other=pd.NA)
                    changes_applied.append(
                        f"Cast '{col}' to Text (leading zeros preserved)"
                    )

                elif target_type == "Numeric":
                    result[col] = pd.to_numeric(result[col], errors="coerce")
                    changes_applied.append(f"Cast '{col}' to Numeric")

                elif target_type == "Datetime":
                    result[col] = pd.to_datetime(result[col], errors="coerce")
                    changes_applied.append(f"Cast '{col}' to Datetime")

                else:
                    # Unknown override string — warn and skip.
                    changes_applied.append(
                        f"Skipped unknown type override '{target_type}' on '{col}'"
                    )

            except Exception as cast_exc:  # noqa: BLE001
                # Per-column resilience: a bad cast must never crash the whole job.
                changes_applied.append(
                    f"Skipped manual type cast on '{col}' due to incompatibility: {cast_exc}"
                )

    # ------------------------------------------------------------------
    # Phase 1: Apply column-wise rules in column order
    # ------------------------------------------------------------------
    for col in result.columns:
        rule = column_rules.get(col, "No Action")

        if rule == "No Action" or rule not in (
            "Trim Whitespace",
            "UPPERCASE",
            "lowercase",
            "Title Case",
            "Remove Special Characters",
            "Fill Blanks with Median",
            "Fill Blanks with 0",
            "Standardize Dates",
            "Remove Duplicates",
        ):
            # "Remove Duplicates" is handled in Phase 2; skip here.
            if rule == "Remove Duplicates":
                dedup_columns.append(col)
            continue

        if rule == "Remove Duplicates":
            dedup_columns.append(col)
            continue  # deferred to Phase 2

        series_before = result[col].copy()

        try:
            if rule == "Trim Whitespace":
                result[col] = _trim_whitespace(result[col])
                changes_applied.append(f"Trimmed whitespace in '{col}'")

            elif rule == "UPPERCASE":
                if pd.api.types.is_numeric_dtype(result[col]):
                    changes_applied.append(
                        f"Skipped 'UPPERCASE' on numeric column '{col}'"
                    )
                else:
                    result[col] = _to_uppercase(result[col])
                    changes_applied.append(f"Converted '{col}' to UPPERCASE")

            elif rule == "lowercase":
                if pd.api.types.is_numeric_dtype(result[col]):
                    changes_applied.append(
                        f"Skipped 'lowercase' on numeric column '{col}'"
                    )
                else:
                    result[col] = _to_lowercase(result[col])
                    changes_applied.append(f"Converted '{col}' to lowercase")

            elif rule == "Title Case":
                if pd.api.types.is_numeric_dtype(result[col]):
                    changes_applied.append(
                        f"Skipped 'Title Case' on numeric column '{col}'"
                    )
                else:
                    result[col] = _to_titlecase(result[col])
                    changes_applied.append(f"Converted '{col}' to Title Case")

            elif rule == "Remove Special Characters":
                if pd.api.types.is_numeric_dtype(result[col]):
                    changes_applied.append(
                        f"Skipped 'Remove Special Characters' on numeric column '{col}'"
                    )
                else:
                    result[col] = _remove_special_chars(result[col])
                    changes_applied.append(f"Removed special characters from '{col}'")

            elif rule == "Fill Blanks with Median":
                null_count_before = int(result[col].isna().sum())
                result[col] = _fill_blanks_with_median(result[col], col, changes_applied)
                null_count_after = int(result[col].isna().sum())
                filled = null_count_before - null_count_after
                if filled > 0:
                    blanks_filled += filled
                    changes_applied.append(
                        f"Filled {filled} blank(s) with median in '{col}'"
                    )

            elif rule == "Fill Blanks with 0":
                null_count_before = int(result[col].isna().sum())
                result[col] = _fill_blanks_with_zero(result[col])
                null_count_after = int(result[col].isna().sum())
                filled = null_count_before - null_count_after
                if filled > 0:
                    blanks_filled += filled
                    changes_applied.append(
                        f"Filled {filled} blank(s) with 0 in '{col}'"
                    )

            elif rule == "Standardize Dates":
                result[col] = _standardize_dates(result[col], col, date_parse_failures)
                changes_applied.append(f"Standardized dates in '{col}' to YYYY-MM-DD")

        except Exception as exc:  # noqa: BLE001
            # Safety net: skip the rule and record a warning rather than crashing.
            changes_applied.append(
                f"Skipped '{rule}' on '{col}' due to unexpected error: {exc}"
            )
            result[col] = series_before  # restore original values

    # ------------------------------------------------------------------
    # Phase 2: Remove Duplicates (runs once, after all column-wise rules)
    # ------------------------------------------------------------------
    if dedup_columns:
        # Filter to columns that actually exist in the DataFrame
        valid_dedup_cols = [c for c in dedup_columns if c in result.columns]
        if valid_dedup_cols:
            result = result.drop_duplicates(subset=valid_dedup_cols)
            subset_label = ", ".join(f"'{c}'" for c in valid_dedup_cols)
            changes_applied.append(
                f"Removed duplicate rows based on column(s): {subset_label}"
            )

    rows_after: int = len(result)
    duplicates_removed: int = rows_before - rows_after

    metadata: Dict[str, Any] = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "changes_applied": changes_applied,
        "date_parse_failures": date_parse_failures,
        "blanks_filled": blanks_filled,
        "duplicates_removed": duplicates_removed,
    }

    return result, metadata
