"""
api.py
------
FastAPI backend for the DataOps Transformation Platform.
Exposes a single POST /v1/transform endpoint that accepts a multipart
file upload plus a JSON-encoded rules dict, applies the transformation
engine, and streams the cleaned file back — all in-memory.

Run the FastAPI backend:
    uvicorn api:app --reload --port 8000

Run the Streamlit UI (separate process, separate port):
    streamlit run main.py
"""

import io
import json
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

from core_engine import apply_transformation_rules, validate_dataframe

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_BYTES: int = 50 * 1024 * 1024  # 50 MB
MAX_CHANGES_IN_HEADER: int = 20     # Cap on changes_applied in response header

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------
app = FastAPI(
    title="DataOps Transformation Platform API",
    description=(
        "Upload messy CSV/XLSX spreadsheets and apply column-level cleaning rules "
        "via a simple JSON payload. All processing is 100% in-memory — "
        "no data is ever written to disk."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for all origins (required for browser-based clients)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Transform-Metadata"],  # allow clients to read custom header
)


# ---------------------------------------------------------------------------
# GET /  — Welcome page
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root() -> HTMLResponse:
    """Return a simple HTML welcome page linking to the interactive API docs."""
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>DataOps Transformation Platform API</title>
        <style>
            body { font-family: system-ui, sans-serif; max-width: 640px;
                   margin: 80px auto; padding: 0 24px; color: #1a1a2e; }
            h1   { font-size: 1.8rem; margin-bottom: 0.25rem; }
            p    { color: #555; line-height: 1.6; }
            a    { color: #4f46e5; font-weight: 600; text-decoration: none; }
            a:hover { text-decoration: underline; }
            .badge { display: inline-block; background: #eef2ff; color: #4f46e5;
                     padding: 2px 10px; border-radius: 999px; font-size: 0.8rem;
                     margin-left: 8px; }
        </style>
    </head>
    <body>
        <h1>🔧 DataOps Transformation Platform <span class="badge">v1.0</span></h1>
        <p>
            A lightweight, stateless API for cleaning messy spreadsheets.<br/>
            Upload a <code>.csv</code> or <code>.xlsx</code> file, pass a JSON dict of
            column → rule mappings, and receive the transformed file back — all
            processed in RAM, nothing stored on disk.
        </p>
        <p>
            👉 <a href="/docs">Interactive API Documentation (Swagger UI)</a><br/>
            👉 <a href="/redoc">ReDoc Documentation</a>
        </p>
        <p style="font-size:0.85rem; color:#999;">
            POST <code>/v1/transform</code> · multipart/form-data ·
            file + rules (JSON string)
        </p>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


# ---------------------------------------------------------------------------
# POST /v1/transform  — Main transformation endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/v1/transform",
    summary="Transform a spreadsheet",
    response_description=(
        "The cleaned file as a binary stream. "
        "Transformation metadata is returned in the X-Transform-Metadata response header."
    ),
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "examples": {
                        "Trim email, fill sales blanks, cast NationalID to Text": {
                            "summary": "Trim Email, fill Sales with 0, preserve NationalID leading zeros",
                            "value": {
                                "rules": '{"Email": "Trim Whitespace", "Sales": "Fill Blanks with 0", "CustomerID": "Remove Duplicates"}',
                                "data_types": '{"NationalID": "Text"}',
                            },
                        }
                    }
                }
            }
        }
    },
)
async def transform(
    file: UploadFile = File(
        ...,
        description="A .csv or .xlsx spreadsheet to transform (max 50 MB).",
    ),
    rules: str = Form(
        ...,
        description=(
            'JSON object mapping column names to rule names. '
            'Example: \'{"Email": "Trim Whitespace", "Sales": "Fill Blanks with 0"}\''
        ),
    ),
    data_types: str = Form(
        default="{}",
        description=(
            'Optional JSON object mapping column names to target type strings. '
            'Supported values: "Text", "Numeric", "Datetime". '
            'Example: \'{"NationalID": "Text", "Revenue": "Numeric"}\'  '
            'Text casting preserves leading zeros (e.g. National IDs).'
        ),
    ),
) -> StreamingResponse:
    """
    Apply column-level transformation rules to a spreadsheet and return the
    cleaned file as a binary download.

    **Supported rules per column type:**

    - *Text columns:* Trim Whitespace, UPPERCASE, lowercase, Title Case,
      Remove Special Characters, Remove Duplicates
    - *Numeric columns:* Fill Blanks with Median, Fill Blanks with 0, Remove Duplicates
    - *Date columns:* Standardize Dates, Remove Duplicates

    **Form fields:**

    | Field | Required | Description |
    |-------|----------|-------------|
    | `file` | ✅ | CSV or XLSX file (max 50 MB) |
    | `rules` | ✅ | JSON dict: column → rule name |
    | `data_types` | ❌ | JSON dict: column → "Text" / "Numeric" / "Datetime" |

    **Response headers:**

    | Header | Description |
    |--------|-------------|
    | `X-Transform-Metadata` | JSON string with rows_before, rows_after, changes_applied (first 20), date_parse_failures, blanks_filled, duplicates_removed |

    **Error codes:**

    | Code | Meaning |
    |------|---------|
    | 400  | Malformed or invalid rules / data_types JSON |
    | 413  | File exceeds 50 MB |
    | 422  | Unsupported file type (not .csv/.xlsx) |
    | 500  | Unexpected processing failure |
    """
    # ------------------------------------------------------------------
    # 1. Read raw bytes & enforce 50 MB limit BEFORE any parsing
    # ------------------------------------------------------------------
    file_bytes: bytes = await file.read()
    if len(file_bytes) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large ({len(file_bytes) / 1_048_576:.1f} MB). "
                "Maximum allowed size is 50 MB."
            ),
        )

    # ------------------------------------------------------------------
    # 2. Determine file type from the original filename
    # ------------------------------------------------------------------
    original_filename: str = file.filename or "upload.csv"
    ext = original_filename.rsplit(".", 1)[-1].lower() if "." in original_filename else ""

    if ext not in ("csv", "xlsx", "xls"):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported file type '.{ext}'. "
                "Please upload a .csv or .xlsx file."
            ),
        )
    output_ext = "xlsx" if ext in ("xlsx", "xls") else "csv"

    # ------------------------------------------------------------------
    # 3. Parse bytes into a DataFrame
    # ------------------------------------------------------------------
    try:
        buffer = io.BytesIO(file_bytes)
        if ext == "csv":
            df: pd.DataFrame = pd.read_csv(buffer)
        else:
            df = pd.read_excel(buffer, engine="openpyxl")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not parse the uploaded file. "
                "Please check that it is a valid CSV or Excel file."
            ),
        ) from exc

    # ------------------------------------------------------------------
    # 4. Validate the DataFrame
    # ------------------------------------------------------------------
    try:
        validate_dataframe(df)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # ------------------------------------------------------------------
    # 5. Parse rules JSON
    # ------------------------------------------------------------------
    try:
        column_rules: Dict[str, str] = json.loads(rules)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid rules JSON: {exc.msg} at position {exc.pos}.",
        ) from exc

    if not isinstance(column_rules, dict):
        raise HTTPException(
            status_code=400,
            detail="The 'rules' field must be a JSON object (dict) mapping column names to rule names.",
        )

    # ------------------------------------------------------------------
    # 5b. Parse data_types JSON (optional — defaults to '{}')
    # ------------------------------------------------------------------
    try:
        types_dict: Dict[str, str] = json.loads(data_types) if data_types.strip() else {}
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid data_types JSON: {exc.msg} at position {exc.pos}.",
        ) from exc

    if not isinstance(types_dict, dict):
        raise HTTPException(
            status_code=400,
            detail="The 'data_types' field must be a JSON object (dict) mapping column names to type strings.",
        )

    # ------------------------------------------------------------------
    # 6. Apply transformation rules
    # ------------------------------------------------------------------
    try:
        cleaned_df, metadata = apply_transformation_rules(
            df,
            column_rules,
            type_overrides=types_dict if types_dict else None,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing the file. Please try again.",
        ) from exc

    # ------------------------------------------------------------------
    # 7. Serialise cleaned DataFrame back to bytes (in-memory)
    # ------------------------------------------------------------------
    try:
        out_buffer = io.BytesIO()
        if output_ext == "csv":
            cleaned_df.to_csv(out_buffer, index=False)
            mime_type = "text/csv"
        else:
            cleaned_df.to_excel(out_buffer, index=False, engine="openpyxl")
            mime_type = (
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        out_buffer.seek(0)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail="Failed to serialise the cleaned file. Please try again.",
        ) from exc

    # ------------------------------------------------------------------
    # 8. Build X-Transform-Metadata response header
    # ------------------------------------------------------------------
    # Cap changes_applied to keep the header compact (proxies may truncate large headers).
    header_metadata: Dict[str, Any] = dict(metadata)
    header_metadata["changes_applied"] = metadata["changes_applied"][:MAX_CHANGES_IN_HEADER]

    # ensure_ascii=True escapes accented/non-Latin characters — safe for HTTP headers.
    metadata_json: str = json.dumps(header_metadata, ensure_ascii=True)

    # ------------------------------------------------------------------
    # 9. Stream the response
    # ------------------------------------------------------------------
    base_name = original_filename.rsplit(".", 1)[0]
    download_filename = f"{base_name}-cleaned.{output_ext}"

    return StreamingResponse(
        content=out_buffer,
        media_type=mime_type,
        headers={
            "Content-Disposition": f'attachment; filename="{download_filename}"',
            "X-Transform-Metadata": metadata_json,
        },
    )


# ---------------------------------------------------------------------------
# Run the FastAPI backend:
#     uvicorn api:app --reload --port 8000
#
# Run the Streamlit UI (separate process, separate port):
#     streamlit run main.py
# ---------------------------------------------------------------------------
