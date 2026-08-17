"""
main.py
-------
Streamlit UI for the DataOps Transformation Platform.
Upload messy CSV/XLSX files, configure per-column transformation rules,
preview the cleaned data, and download the result — all in-memory.

Run the Streamlit UI:
    streamlit run main.py
"""

import io
from typing import Dict, List

import pandas as pd
import streamlit as st

from core_engine import apply_transformation_rules, validate_dataframe

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DataOps Transformation Platform",
    page_icon="🔧",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🔧 DataOps Transformation Platform")
st.markdown(
    "Upload a messy spreadsheet, configure per-column cleaning rules, "
    "preview the result in real time, and download the transformed file — "
    "**all 100% in-memory, zero data stored.**"
)
st.divider()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_BYTES: int = 50 * 1024 * 1024  # 50 MB

TEXT_RULES: List[str] = [
    "No Action",
    "Trim Whitespace",
    "UPPERCASE",
    "lowercase",
    "Title Case",
    "Remove Special Characters",
    "Remove Duplicates",
]

NUMERIC_RULES: List[str] = [
    "No Action",
    "Fill Blanks with Median",
    "Fill Blanks with 0",
    "Remove Duplicates",
]

DATE_RULES: List[str] = [
    "No Action",
    "Standardize Dates",
    "Remove Duplicates",
]

# Options for the manual data-type override selectbox.
TYPE_OVERRIDE_OPTIONS: List[str] = ["No Override", "Text", "Numeric", "Datetime"]


# ---------------------------------------------------------------------------
# Helper: detect column type and return the appropriate rule list
# ---------------------------------------------------------------------------


def _detect_col_type(series: pd.Series) -> str:
    """Return 'numeric', 'datetime', or 'text' based on the Series dtype."""
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    # Boolean, mixed-type, object, category → treat as text (safe fallback)
    return "text"


def _rules_for_type(col_type: str) -> List[str]:
    if col_type == "numeric":
        return NUMERIC_RULES
    if col_type == "datetime":
        return DATE_RULES
    return TEXT_RULES


def _effective_col_type(inferred_type: str, override: str) -> str:
    """
    Resolve the *effective* column type used to select the rule dropdown options.

    If the user chose a Data Type override, that wins over Pandas inference.
    Supported override values map as follows:
      "Text"     -> "text"
      "Numeric"  -> "numeric"
      "Datetime" -> "datetime"
    Any other value (including "No Override") defers to *inferred_type*.
    """
    mapping = {"Text": "text", "Numeric": "numeric", "Datetime": "datetime"}
    return mapping.get(override, inferred_type)


# ---------------------------------------------------------------------------
# Helper: load uploaded file into a DataFrame (in-memory only)
# ---------------------------------------------------------------------------


def _load_dataframe(uploaded_file) -> pd.DataFrame:
    """
    Parse the uploaded file bytes into a DataFrame.
    Uses the file extension to choose between CSV and XLSX parsers.
    Raises ValueError for unsupported file types.
    """
    raw_bytes = uploaded_file.read()
    if len(raw_bytes) > MAX_BYTES:
        raise ValueError(
            f"File is too large ({len(raw_bytes) / 1_048_576:.1f} MB). "
            "Please upload a file smaller than 50 MB."
        )

    ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
    buffer = io.BytesIO(raw_bytes)

    if ext == "csv":
        return pd.read_csv(buffer)
    elif ext in ("xlsx", "xls"):
        return pd.read_excel(buffer, engine="openpyxl")
    else:
        raise ValueError(
            f"Unsupported file type '.{ext}'. Please upload a .csv or .xlsx file."
        )


# ---------------------------------------------------------------------------
# Helper: serialise cleaned DataFrame back to in-memory bytes
# ---------------------------------------------------------------------------


def _dataframe_to_bytes(df: pd.DataFrame, ext: str) -> tuple[bytes, str]:
    """
    Convert a cleaned DataFrame to bytes in the original file format.
    Returns (file_bytes, mime_type).
    """
    buffer = io.BytesIO()
    if ext == "csv":
        df.to_csv(buffer, index=False)
        mime = "text/csv"
    else:  # xlsx
        df.to_excel(buffer, index=False, engine="openpyxl")
        mime = (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    buffer.seek(0)
    return buffer.getvalue(), mime


# ---------------------------------------------------------------------------
# File uploader
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload your spreadsheet",
    type=["csv", "xlsx"],
    help="Accepts .csv and .xlsx files up to 50 MB.",
)

if uploaded_file is not None:
    # -----------------------------------------------------------------------
    # Load and validate the file
    # -----------------------------------------------------------------------
    try:
        df_original = _load_dataframe(uploaded_file)
        validate_dataframe(df_original)
    except ValueError as exc:
        st.error(f"❌ {exc}")
        st.stop()
    except Exception as exc:  # noqa: BLE001
        st.error(
            "❌ Could not parse the uploaded file. "
            "Please check that it is a valid CSV or Excel file."
        )
        st.stop()

    # Detect the file extension (needed later for download)
    file_ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
    # Normalise xls to xlsx for output
    output_ext = "xlsx" if file_ext in ("xlsx", "xls") else "csv"

    st.success(
        f"✅ Loaded **{uploaded_file.name}** — "
        f"{len(df_original):,} rows × {len(df_original.columns):,} columns"
    )
    st.divider()

    # -----------------------------------------------------------------------
    # Column rule configuration grid
    # -----------------------------------------------------------------------
    st.subheader("Configure Transformation Rules")

    columns = list(df_original.columns)

    # Keys tracked per file — used to scope the reset button cleanly.
    current_rule_keys = [f"rule_{col}" for col in columns]
    current_type_keys = [f"type_{col}" for col in columns]

    # Initialise session state defaults for this file's columns
    for col in columns:
        if f"rule_{col}" not in st.session_state:
            st.session_state[f"rule_{col}"] = "No Action"
        if f"type_{col}" not in st.session_state:
            st.session_state[f"type_{col}"] = "No Override"

    # "Reset all rules" button — resets BOTH rule and type keys for the current file only.
    if st.button("↺ Reset all rules", key="reset_btn"):
        for key in current_rule_keys:
            st.session_state[key] = "No Action"
        for key in current_type_keys:
            st.session_state[key] = "No Override"
        st.rerun()

    # Render selectboxes in rows of 4
    col_groups = [columns[i : i + 4] for i in range(0, len(columns), 4)]

    column_rules: Dict[str, str] = {}
    type_overrides: Dict[str, str] = {}

    for group in col_groups:
        st_cols = st.columns(4)
        for idx, col in enumerate(group):
            inferred_type = _detect_col_type(df_original[col])

            with st_cols[idx]:
                # --- Column header with inferred type badge ---
                inferred_label = {
                    "numeric": "🔢 Numeric",
                    "datetime": "📅 DateTime",
                    "text": "🔤 Text",
                }[inferred_type]
                st.markdown(f"**{col}**  \n`{inferred_label}`")

                # --- 1. Data Type override selectbox ---
                chosen_override = st.selectbox(
                    label="Data Type",
                    options=TYPE_OVERRIDE_OPTIONS,
                    key=f"type_{col}",
                )
                type_overrides[col] = chosen_override

                # --- 2. Transformation Rule selectbox ---
                # Rule options are driven by the *effective* type: override wins
                # over Pandas inference so the user sees only compatible rules.
                effective_type = _effective_col_type(inferred_type, chosen_override)
                available_rules = _rules_for_type(effective_type)

                # If the currently-stored rule is no longer valid for the new
                # effective type, silently fall back to "No Action".
                current_rule = st.session_state.get(f"rule_{col}", "No Action")
                safe_default = current_rule if current_rule in available_rules else "No Action"
                if safe_default != current_rule:
                    st.session_state[f"rule_{col}"] = safe_default

                chosen_rule = st.selectbox(
                    label="Transformation Rule",
                    options=available_rules,
                    key=f"rule_{col}",
                )
                column_rules[col] = chosen_rule

    st.divider()

    # -----------------------------------------------------------------------
    # Apply transformations and render preview
    # -----------------------------------------------------------------------
    st.subheader("Transformed Preview")

    try:
        with st.spinner("Applying transformations…"):
            cleaned_df, metadata = apply_transformation_rules(
                df_original,
                column_rules,
                type_overrides={
                    col: ov
                    for col, ov in type_overrides.items()
                    if ov != "No Override"
                },
            )
    except Exception as exc:  # noqa: BLE001
        st.error(f"❌ Transformation failed: {exc}")
        st.stop()

    # Live preview — first 20 rows
    st.dataframe(cleaned_df.head(20), use_container_width=True)

    # -----------------------------------------------------------------------
    # Metadata summary (built from structured counters, not parsed strings)
    # -----------------------------------------------------------------------
    st.subheader("Transformation Summary")

    rows_before = metadata["rows_before"]
    rows_after = metadata["rows_after"]
    dups_removed = metadata["duplicates_removed"]
    blanks_filled = metadata["blanks_filled"]
    date_failures = metadata["date_parse_failures"]
    changes_applied = metadata["changes_applied"]

    # Count date standardisation successes from the changes_applied log
    dates_standardized = sum(
        1 for line in changes_applied if "Standardized dates" in line
    )

    # Primary metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows Before", f"{rows_before:,}")
    col2.metric("Rows After", f"{rows_after:,}")
    col3.metric("Duplicates Removed", f"{dups_removed:,}")
    col4.metric("Blanks Filled", f"{blanks_filled:,}")

    # Summary sentence
    summary_parts = []
    if dups_removed > 0:
        summary_parts.append(f"✅ **{dups_removed:,}** duplicates removed")
    if blanks_filled > 0:
        summary_parts.append(f"✅ **{blanks_filled:,}** blanks filled")
    if dates_standardized > 0:
        summary_parts.append(f"✅ **{dates_standardized}** date column(s) standardized")

    if summary_parts:
        st.markdown(" · ".join(summary_parts))

    # Date parse failure warnings
    if date_failures:
        for col_name, nat_count in date_failures.items():
            st.warning(
                f"⚠ **{nat_count}** value(s) in **'{col_name}'** "
                "could not be parsed and were left blank."
            )

    # Skipped-rule warnings
    skipped_lines = [ln for ln in changes_applied if ln.startswith("Skipped")]
    if skipped_lines:
        with st.expander(f"⚠ {len(skipped_lines)} rule(s) were skipped — click to expand"):
            for line in skipped_lines:
                st.markdown(f"- {line}")

    # Full changes log (optional detail)
    with st.expander("📋 Full transformation log"):
        if changes_applied:
            for line in changes_applied:
                st.markdown(f"- {line}")
        else:
            st.markdown("_No transformations were applied._")

    st.divider()

    # -----------------------------------------------------------------------
    # Download
    # -----------------------------------------------------------------------
    st.subheader("Download Cleaned File")

    file_bytes, mime_type = _dataframe_to_bytes(cleaned_df, output_ext)
    base_name = uploaded_file.name.rsplit(".", 1)[0]
    download_name = f"{base_name}-cleaned.{output_ext}"

    st.download_button(
        label=f"⬇️ Download {download_name}",
        data=file_bytes,
        file_name=download_name,
        mime=mime_type,
        use_container_width=True,
    )

else:
    # Placeholder state when no file is uploaded
    st.info(
        "👆 Upload a .csv or .xlsx file above to get started. "
        "Your data never leaves your browser session — nothing is stored on disk."
    )


# ---------------------------------------------------------------------------
# Run the Streamlit UI:
#     streamlit run main.py
#
# Run the FastAPI backend (separate process, separate port):
#     uvicorn api:app --reload --port 8000
# ---------------------------------------------------------------------------
