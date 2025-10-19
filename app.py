"""
Corrected Streamlit app (gntapp_corrected.py)
- Adds a robust safe_read_csv helper to avoid pandas.errors.EmptyDataError
- Handles UploadedFile pointer resets and empty-file checks
- Parses dates after reading as a fallback
- Provides helpful UI messages and a downloadable CSV template

Run: streamlit run gntapp_corrected.py
"""

import io
import streamlit as st
import pandas as pd
import pandas.errors as pd_errors
from typing import Optional, List

st.set_page_config(page_title='Growth & Transformation Tracker', layout='wide')

# ---------------- Helper: safe CSV reader ---------------------------------

def safe_read_csv(uploaded_file,
                  parse_dates: Optional[List[str]] = None,
                  dayfirst: bool = True,
                  keep_default_na: bool = False,
                  **pd_kwargs) -> Optional[pd.DataFrame]:
    """
    Safely read a CSV from a Streamlit UploadedFile, bytes, or file path.
    Returns DataFrame on success, or None on failure (and shows Streamlit messages).

    - Guards against empty uploads (EmptyDataError)
    - Resets file pointer when possible
    - Attempts fast C engine then falls back to python engine on ParserError
    - If parse_dates columns are not present in header, parsing is skipped and done post-read
    """
    # Normalize raw content
    if uploaded_file is None:
        st.error("No file provided to safe_read_csv")
        return None

    try:
        if hasattr(uploaded_file, "read"):
            raw = uploaded_file.read()
            # Reset pointer for future reads
            try:
                uploaded_file.seek(0)
            except Exception:
                pass
        elif isinstance(uploaded_file, (bytes, bytearray)):
            raw = bytes(uploaded_file)
        elif isinstance(uploaded_file, str):
            with open(uploaded_file, "rb") as f:
                raw = f.read()
        else:
            st.error("Unsupported uploaded_file type for safe_read_csv")
            return None
    except Exception as e:
        st.error(f"Failed reading uploaded file bytes: {e}")
        return None

    # Basic empty file check
    if not raw or raw.strip() == b'':
        st.error("Uploaded file is empty. Please upload a valid CSV file.")
        return None

    # Inspect header columns (nrows=0) to check for date columns
    header_cols = []
    try:
        header_cols = pd.read_csv(io.BytesIO(raw), nrows=0).columns.tolist()
    except Exception:
        header_cols = []

    actual_parse_dates = [c for c in (parse_dates or []) if c in header_cols]

    # Attempt using default (C) engine first
    try:
        df = pd.read_csv(
            io.BytesIO(raw),
            parse_dates=actual_parse_dates if actual_parse_dates else None,
            dayfirst=dayfirst,
            keep_default_na=keep_default_na,
            **pd_kwargs,
        )
        return df

    except pd_errors.EmptyDataError:
        st.error("CSV parsing failed: file appears empty or has no valid rows.")
        return None

    except pd_errors.ParserError as pe:
        st.warning(f"C engine failed ({pe}). Retrying with python engine...")
        try:
            df = pd.read_csv(
                io.BytesIO(raw),
                engine="python",
                parse_dates=actual_parse_dates if actual_parse_dates else None,
                dayfirst=dayfirst,
                keep_default_na=keep_default_na,
                **pd_kwargs,
            )
            return df
        except Exception as e2:
            st.error(f"Failed to parse CSV with python engine: {e2}")
            return None

    except Exception as e:
        st.error(f"Unexpected error reading CSV: {e}")
        return None


# ---------------- Small CSV template generator ---------------------------------

def csv_template_bytes() -> bytes:
    template = (
        "Project,Owner,Date,expected_start,Status,Notes\n"
        "Migration ABC,alice,01/10/2025,05/10/2025,Open,Initial intake\n"
        "Upgrade XYZ,bob,15/09/2025,20/09/2025,Planned,Vendor coordination\n"
    )
    return template.encode("utf-8")


# ---------------- Streamlit UI -----------------------------------------------

st.title("Growth & Transformation Tracker")
st.write("Upload your opportunities/tracker CSV. Columns: Project,Owner,Date,expected_start,Status,Notes")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Upload CSV")
    st.download_button("Download CSV template", data=csv_template_bytes(), file_name="gnt_template.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload tracker CSV", type=["csv", "txt"], accept_multiple_files=False)

    df_in = None
    if uploaded is not None:
        # Use helper to safely read CSV and avoid EmptyDataError
        df_in = safe_read_csv(uploaded, parse_dates=["Date", "expected_start"], keep_default_na=False)

        # If parse_dates columns were not present or parsing not done above, try coercing here
        if df_in is not None:
            for col in ["Date", "expected_start"]:
                if col in df_in.columns:
                    try:
                        df_in[col] = pd.to_datetime(df_in[col], dayfirst=True, errors="coerce")
                    except Exception:
                        # If conversion fails, leave as-is and warn
                        st.warning(f"Could not coerce column '{col}' to datetime. Values may be inconsistent.")

            st.success(f"Loaded {len(df_in)} rows")
            st.dataframe(df_in)

            # Simple summary
            st.markdown("**Summary**")
            st.write(df_in.describe(include="all"))

with col2:
    st.subheader("Quick actions")
    if st.button("Show uploaded file info"):
        if uploaded is None:
            st.info("No file uploaded yet.")
        else:
            try:
                uploaded_bytes = uploaded.read()
                st.write("filename:", uploaded.name)
                st.write("size bytes:", len(uploaded_bytes))
                st.write("first 300 bytes preview:")
                st.code(uploaded_bytes[:300])
                # reset pointer
                try:
                    uploaded.seek(0)
                except Exception:
                    pass
            except Exception as e:
                st.error(f"Failed to read uploaded file for info: {e}")


# ---------------- Optional: allow manual editing / download ---------------------
if 'df_in' in locals() and df_in is not None:
    st.markdown("---")
    st.subheader("Edit / Export")
    if st.button("Download loaded CSV"):
        try:
            out = df_in.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", data=out, file_name="gnt_loaded.csv", mime='text/csv')
        except Exception as e:
            st.error(f"Failed to prepare download: {e}")


st.markdown("---")
st.caption("This app includes robust CSV handling to avoid pandas EmptyDataError and provide clear user feedback.")