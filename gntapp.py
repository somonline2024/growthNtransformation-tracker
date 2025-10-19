# gntapp.py
import streamlit as st
import sqlite3
from datetime import datetime, date
import pandas as pd
import io
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ---------- Config ----------
DB_PATH = "opportunities.db"
st.set_page_config(page_title="Growth & Transformation Tracker", layout="wide")

# ---------- DB helpers ----------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS opportunities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sr_no INTEGER,
        date_created TEXT,
        opportunity_name TEXT,
        description TEXT,
        wo_number TEXT,
        project_type TEXT,
        action_owner TEXT,
        expected_start TEXT,
        value_gbp REAL,
        status TEXT,
        expected_quarter TEXT,
        period TEXT,
        remarks TEXT
    )
    """)
    conn.commit()
    conn.close()

def run_query(query, params=(), fetch=False):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(query, params)
    data = cur.fetchall() if fetch else None
    conn.commit()
    conn.close()
    return data

init_db()

# ---------- Quarter & Period logic ----------
def quarter_from_date(dt: date, method: str = "calendar"):
    """Return quarter label like 'Q1' given a date.
       method: 'calendar' (Jan-Mar=Q1) or 'uk_fiscal' (Apr-Jun=Q1)."""
    if dt is None:
        return ""
    m = dt.month
    if method == "uk_fiscal":
        fiscal_month = ((m - 4) % 12) + 1
        q = (fiscal_month - 1) // 3 + 1
    else:
        q = (m - 1) // 3 + 1
    return f"Q{q}"

def period_from_date(dt: date, method: str = "calendar"):
    """Return period label (month + mini period inside quarter)."""
    if dt is None:
        return ""
    m = dt.month
    mon = dt.strftime("%b")
    if method == "uk_fiscal":
        fiscal_month = ((m - 4) % 12) + 1
        q = (fiscal_month - 1) // 3 + 1
        within_q = ((fiscal_month - 1) % 3) + 1
        return f"FY-M{fiscal_month} (Q{q}-P{within_q})"
    else:
        q = (m - 1) // 3 + 1
        within_q = ((m - 1) % 3) + 1
        return f"{mon} (Q{q}-P{within_q})"

# ---------- UI ----------
st.title("Growth & Transformation Tracker")
st.markdown("Fields: Sr No, Date, Opportunity Name, Description, WO Number, Project type, Action Owner, Expected Date to Start, Value (GBP), Status, Expected Quarter, Period, Remarks")

# Sidebar: settings
with st.sidebar:
    st.header("Settings & Import")
    quarter_method = st.selectbox("Quarter/Period calculation method", ["calendar","uk_fiscal"],
                                  help="calendar: Jan-Mar=Q1. uk_fiscal: Apr-Mar fiscal year (Q1=Apr-Jun).")
    uploaded = st.file_uploader("Import CSV (optional) - use same column names", type=["csv"])
    if st.button("Import CSV"):
        if uploaded is None:
            st.warning("Choose a CSV file first.")
        else:
            try:
                df_in = pd.read_csv(uploaded, parse_dates=["Date","expected_start"], dayfirst=False, keep_default_na=False)
            except Exception:
                df_in = pd.read_csv(uploaded, parse_dates=["Date","expected_start"], dayfirst=True, keep_default_na=False)
            # Expect column names from UI; try to be flexible
            for _, row in df_in.iterrows():
                # try safe reads
                sr = int(row.get("Sr No") or row.get("sr_no") or 0)
                dt = row.get("Date") or row.get("date_created") or datetime.utcnow().date()
                opp = row.get("Opportunity Name") or row.get("opportunity_name") or ""
                desc = row.get("Description") or ""
                wo = row.get("WO Number") or row.get("wo_number") or ""
                ptype = row.get("Project type") or row.get("project_type") or "Fixed Price"
                owner = row.get("Action Owner") or row.get("action_owner") or ""
                est_start = row.get("Expected Date to Start") or row.get("expected_start") or dt
                if isinstance(est_start, str):
                    try:
                        est_start = pd.to_datetime(est_start).date()
                    except:
                        est_start = pd.to_datetime(dt).date()
                value = float(row.get("Value of WO (in GBP)") or row.get("value_gbp") or 0.0)
                status = row.get("Status") or "Candidate"
                # compute quarter and period
                q = quarter_from_date(est_start, method=quarter_method)
                p = period_from_date(est_start, method=quarter_method)
                run_query("""INSERT INTO opportunities (sr_no, date_created, opportunity_name, description, wo_number, project_type, action_owner, expected_start, value_gbp, status, expected_quarter, period, remarks)
                             VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                          (sr, pd.to_datetime(dt).date().isoformat(), opp, desc, wo, ptype, owner, est_start.isoformat(), value, status, q, p, row.get("Remarks") or ""))
            st.success("CSV imported. Reloading.")
            st.experimental_rerun()

# Add new entry
st.header("Create a new opportunity")
with st.form("add_form", clear_on_submit=True):
    c1, c2, c3 = st.columns([1,2,2])
    sr_no = c1.number_input("Sr No", min_value=1, value=1, step=1)
    date_created = c1.date_input("Date", value=datetime.utcnow().date())
    opp_name = c2.text_input("Opportunity Name")
    wo_number = c2.text_input("WO Number")
    project_type = c2.selectbox("Project type", ["Fixed Price","Pseudo FP","T&M"])
    action_owner = c3.text_input("Action Owner")
    expected_start = c3.date_input("Expected Date to Start", value=datetime.utcnow().date())
    value_gbp = c3.number_input("Value of WO (in GBP)", min_value=0.0, value=0.0, format="%.2f")
    status = st.selectbox("Status", ["Candidate","Active","Won","Lost"])
    description = st.text_area("Description", height=80)
    remarks = st.text_area("Remarks", height=60)
    submitted = st.form_submit_button("Create")
    if submitted:
        q = quarter_from_date(expected_start, method=quarter_method)
        p = period_from_date(expected_start, method=quarter_method)
        run_query("""INSERT INTO opportunities (sr_no, date_created, opportunity_name, description, wo_number, project_type, action_owner, expected_start, value_gbp, status, expected_quarter, period, remarks)
                     VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                  (sr_no, date_created.isoformat(), opp_name, description, wo_number, project_type, action_owner, expected_start.isoformat(), float(value_gbp), status, q, p, remarks))
        st.success("Opportunity created.")
        st.experimental_rerun()

st.markdown("---")
# Load data
rows = run_query("SELECT id, sr_no, date_created, opportunity_name, description, wo_number, project_type, action_owner, expected_start, value_gbp, status, expected_quarter, period, remarks FROM opportunities ORDER BY id DESC", fetch=True)
df = pd.DataFrame(rows, columns=["id","Sr No","Date","Opportunity Name","Description","WO Number","Project type","Action Owner","Expected Date to Start","Value (GBP)","Status","Expected Quarter","Period","Remarks"])
if df.empty:
    st.info("No opportunities yet. Add one above or import CSV.")
else:
    st.subheader("Opportunity list")
    # Filters
    f_col1, f_col2, f_col3 = st.columns([2,2,2])
    with f_col1:
        f_owner = st.text_input("Filter by Action Owner")
    with f_col2:
        f_status = st.selectbox("Filter by Status", ["(any)","Candidate","Active","Won","Lost"])
    with f_col3:
        f_quarter = st.selectbox("Filter by Quarter", ["(any)","Q1","Q2","Q3","Q4"])
    search = st.text_input("Search Opportunity Name or WO")
    # apply filters
    if f_owner:
        df = df[df["Action Owner"].str.contains(f_owner, case=False, na=False)]
    if f_status != "(any)":
        df = df[df["Status"] == f_status]
    if f_quarter != "(any)":
        df = df[df["Expected Quarter"] == f_quarter]
    if search:
        df = df[df["Opportunity Name"].str.contains(search, case=False, na=False) | df["WO Number"].str.contains(search, case=False, na=False)]
    st.dataframe(df.drop(columns=["id"]), use_container_width=True)

    # CSV export
    csv = df.to_csv(index=False)
    st.download_button("Export CSV", data=csv, file_name=f"opportunities_{datetime.utcnow().date()}.csv", mime="text/csv")

    # Edit/Delete
    st.markdown("#### Edit / Delete an entry")
    edit_id = st.number_input("Enter internal ID to edit/delete (column 'id' from the table above)", min_value=0, value=0, step=1)
    if edit_id:
        rec = run_query("SELECT id, sr_no, date_created, opportunity_name, description, wo_number, project_type, action_owner, expected_start, value_gbp, status, expected_quarter, period, remarks FROM opportunities WHERE id=?", (edit_id,), fetch=True)
        if not rec:
            st.error("ID not found")
        else:
            r = rec[0]
            with st.form("edit_form"):
                e_sr = st.number_input("Sr No", min_value=1, value=r[1], step=1)
                e_date = st.date_input("Date", value=datetime.fromisoformat(r[2]).date())
                e_name = st.text_input("Opportunity Name", value=r[3])
                e_desc = st.text_area("Description", value=r[4])
                e_wo = st.text_input("WO Number", value=r[5])
                e_ptype = st.selectbox("Project type", ["Fixed Price","Pseudo FP","T&M"], index=["Fixed Price","Pseudo FP","T&M"].index(r[6]))
                e_owner = st.text_input("Action Owner", value=r[7])
                e_expected = st.date_input("Expected Date to Start", value=datetime.fromisoformat(r[8]).date())
                e_value = st.number_input("Value of WO (in GBP)", min_value=0.0, value=r[9] or 0.0, format="%.2f")
                e_status = st.selectbox("Status", ["Candidate","Active","Won","Lost"], index=["Candidate","Active","Won","Lost"].index(r[10]))
                e_remarks = st.text_area("Remarks", value=r[13] or "")
                save = st.form_submit_button("Save changes")
                if save:
                    q = quarter_from_date(e_expected, method=quarter_method)
                    p = period_from_date(e_expected, method=quarter_method)
                    run_query("""UPDATE opportunities SET sr_no=?, date_created=?, opportunity_name=?, description=?, wo_number=?, project_type=?, action_owner=?, expected_start=?, value_gbp=?, status=?, expected_quarter=?, period=?, remarks=? WHERE id=?""",
                              (e_sr, e_date.isoformat(), e_name, e_desc, e_wo, e_ptype, e_owner, e_expected.isoformat(), float(e_value), e_status, q, p, e_remarks, edit_id))
                    st.success("Saved.")
                    st.experimental_rerun()
            if st.button("Delete this record"):
                run_query("DELETE FROM opportunities WHERE id=?", (edit_id,))
                st.success("Deleted.")
                st.experimental_rerun()

    # ---------- Growth matrix visuals ----------
    st.markdown("---")
    st.header("Growth matrix & Forecasts")

    # Pie chart: distribution by Status
    st.subheader("Distribution by Status (pie)")
    pie_df = df.groupby("Status")["Value (GBP)"].sum().reset_index()
    if pie_df.empty:
        st.info("No data to show in charts.")
    else:
        fig1, ax1 = plt.subplots()
        ax1.pie(pie_df["Value (GBP)"], labels=pie_df["Status"], autopct="%1.1f%%")
        ax1.set_title("Pipeline value by Status")
        st.pyplot(fig1)

    # Bar chart: value by quarter (historical)
    st.subheader("Value by Quarter (historical)")
    # ensure expected_start is datetime
    df["Expected Date to Start"] = pd.to_datetime(df["Expected Date to Start"])
    df["YearQuarter"] = df["Expected Date to Start"].dt.to_period("Q").astype(str)
    bar_df = df.groupby("YearQuarter")["Value (GBP)"].sum().reset_index().sort_values("YearQuarter")
    fig2, ax2 = plt.subplots()
    ax2.bar(bar_df["YearQuarter"], bar_df["Value (GBP)"])
    ax2.set_xlabel("Quarter")
    ax2.set_ylabel("Value (GBP)")
    ax2.set_title("Historical pipeline by quarter")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig2)

    # ---------- Forecast ----------
    st.subheader("Forecast (next 4 quarters) — Linear regression on quarter totals")
    # Prepare data for regression: convert YearQuarter to numeric index
    if len(bar_df) >= 2:
        # numeric X: 0,1,2,...
        bar_df = bar_df.reset_index(drop=True)
        X = np.arange(len(bar_df)).reshape(-1,1)
        y = bar_df["Value (GBP)"].values
        model = LinearRegression()
        model.fit(X, y)
        # forecast next 4
        future_idx = np.arange(len(bar_df), len(bar_df)+4).reshape(-1,1)
        y_pred = model.predict(future_idx)
        # prepare combined plot
        all_quarters = list(bar_df["YearQuarter"]) + [f"FQ{i+1}" for i in range(4)]
        all_values = list(bar_df["Value (GBP)"]) + list(y_pred)
        fig3, ax3 = plt.subplots()
        # historical bars
        ax3.bar(range(len(bar_df)), bar_df["Value (GBP)"])
        # forecast bars (patterned by hatch)
        ax3.bar(range(len(bar_df), len(bar_df)+4), y_pred, hatch='//', alpha=0.7)
        ax3.set_xticks(range(len(all_quarters)))
        ax3.set_xticklabels(all_quarters, rotation=45, ha="right")
        ax3.set_ylabel("Value (GBP)")
        ax3.set_title("Historical + Forecast (next 4 quarters)")
        plt.tight_layout()
        st.pyplot(fig3)

        # show forecast table
        forecast_df = pd.DataFrame({
            "Quarter": [f"FQ{i+1}" for i in range(4)],
            "Forecast Value (GBP)": y_pred
        })
        st.table(forecast_df.style.format({"Forecast Value (GBP)": "£{:,.2f}".format}))
        st.markdown("**Forecast method:** simple linear regression on quarter totals (lightweight baseline). Use a richer model if you have many quarters or seasonal patterns.")
    else:
        st.info("Not enough historical quarters (need >=2) to produce a forecast. Add more data or import historical CSV.")

st.markdown("---")
st.caption("Save or export CSV to use in other tools. Forecast is a simple baseline for quick planning.")
