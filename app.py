# FlightWise AI — corrected navy / purple / white theme
# Only table and chart surface styling has been reinforced.

from __future__ import annotations

from pathlib import Path
from datetime import date, time

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# FLIGHTWISE AI — MULTI-PAGE STREAMLIT APPLICATION
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "flightwise_deployment_bundle.joblib"

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


st.set_page_config(
    page_title="FlightWise AI | Flight Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');

:root {
    --navy: #102746;
    --purple: #6845DE;
    --purple-dark: #4C27C7;
    --ink: #17233D;
    --muted: #71809A;
    --page: #F3F5FA;
    --line: #E1E7F0;
    --green: #16845B;
    --red: #D92D3A;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: var(--page);
    color: var(--ink);
}

.block-container {
    max-width: 1580px;
    padding: 1.35rem 2rem 3rem;
}

h1, h2, h3, h4 {
    font-family: 'Manrope', sans-serif !important;
    color: var(--ink) !important;
}

p {
    color: var(--muted);
}

#MainMenu, footer {
    visibility: hidden;
}

[data-testid="stHeader"] {
    background: transparent;
}

/* SIDEBAR — PREMIUM FLIGHT INTELLIGENCE */

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #102743 0%, #102746 58%, #0D203A 100%);
    border-right: 1px solid rgba(255,255,255,.08);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.1rem;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #EAF0FA;
}

[data-testid="stSidebar"] p {
    color: #AFC3DF;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(173,195,222,.20);
    margin: 1.15rem 0;
}

/* Brand block */

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 2px 18px;
}

.sidebar-brand-icon {
    width: 46px;
    height: 46px;
    flex: 0 0 46px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 13px;
    color: #FFFFFF;
    font-size: 25px;
    background: linear-gradient(135deg, #7657F2, #5032C7);
    box-shadow: 0 7px 18px rgba(76,48,190,.30);
}

.sidebar-brand-name {
    color: #FFFFFF;
    font-family: 'Manrope', sans-serif;
    font-size: 24px;
    line-height: 1.15;
    font-weight: 800;
    letter-spacing: -.6px;
}

.sidebar-brand-subtitle {
    color: #AFC3DF;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 2px;
    margin-top: 5px;
}

/* NAVIGATION — CLEAN EDITORIAL STYLE */

.sidebar-section-label {
    color: #91A9C6;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 2.4px;
    margin: 5px 0 13px 4px;
}

/* Text navigation buttons */
[data-testid="stSidebar"] [data-testid="stButton"] {
    width: 100%;
}

[data-testid="stSidebar"] [data-testid="stButton"] > button {
    width: 100%;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    padding: 8px 4px !important;
    min-height: 42px;
    text-align: left !important;
    transition: color .18s ease;
}

[data-testid="stSidebar"] [data-testid="stButton"] > button > div {
    width: 100%;
    display: flex !important;
    justify-content: flex-start !important;
    text-align: left !important;
}

[data-testid="stSidebar"] [data-testid="stButton"] > button span,
[data-testid="stSidebar"] [data-testid="stButton"] > button p {
    width: 100%;
    display: block;
    text-align: left !important;
}

[data-testid="stSidebar"] [data-testid="stButton"] > button p {
    color: #D4E1F0 !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    text-align: left !important;
}

[data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
    background: transparent !important;
    border: none !important;
}

[data-testid="stSidebar"] [data-testid="stButton"] > button:hover p {
    color: #FFFFFF !important;
}

/* Selected navigation item */
[data-testid="stSidebar"] [data-testid="stButton"] > button[kind="primary"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

[data-testid="stSidebar"] [data-testid="stButton"] > button[kind="primary"] p {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    text-decoration: underline;
    text-decoration-color: #A994FF;
    text-decoration-thickness: 1px;
    text-underline-offset: 5px;
}

/* Model status card */

.sidebar-status {
    background: linear-gradient(135deg, rgba(23,104,91,.38), rgba(20,77,79,.55));
    border: 1px solid rgba(91,224,178,.20);
    border-radius: 12px;
    padding: 13px 14px;
    margin-top: 4px;
}

.sidebar-status-heading {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #FFFFFF;
    font-size: 12px;
    font-weight: 800;
    margin-bottom: 9px;
}

.sidebar-status-dot {
    width: 8px;
    height: 8px;
    display: inline-block;
    border-radius: 50%;
    background: #5CE0B2;
    box-shadow: 0 0 0 4px rgba(92,224,178,.12);
}

.sidebar-status-main {
    color: #B9EBDD;
    font-size: 12px;
    font-weight: 700;
}

.sidebar-status-detail {
    color: #A9CFC6;
    font-size: 10px;
    margin-top: 4px;
}

.sidebar-status-error {
    background: rgba(217,45,58,.13);
    border: 1px solid rgba(255,125,135,.22);
    border-radius: 12px;
    padding: 13px 14px;
    margin-top: 4px;
}

.sidebar-status-error .sidebar-status-main {
    color: #FFC3C7;
}

.sidebar-status-error .sidebar-status-detail {
    color: #E6B8BD;
}

/* Sidebar footer */

.sidebar-footer {
    padding-top: 8px;
    line-height: 1.7;
}

.sidebar-footer-title {
    color: #DCE7F5;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.3px;
}

.sidebar-footer-copy {
    color: #91A9C6;
    font-size: 10px;
    margin-top: 5px;
}

.sidebar-footer-version {
    color: #718CA9;
    font-size: 9px;
    margin-top: 8px;
}

/* KPI CARDS */

[data-testid="stMetric"] {
    background: white;
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 17px 19px;
    min-height: 104px;
    box-shadow: 0 4px 16px rgba(20,40,75,.045);
}

[data-testid="stMetricLabel"] p {
    color: var(--muted) !important;
    font-size: 12px !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    color: #263957 !important;
    font-family: 'Manrope',sans-serif;
    font-size: 26px !important;
    font-weight: 800 !important;
}

/* CONTAINERS */

[data-testid="stVerticalBlockBorderWrapper"] {
    background: white;
    border: 1px solid var(--line);
    border-radius: 15px;
    box-shadow: 0 4px 16px rgba(20,40,75,.035);
}

/* DATA TABLES — FLIGHTWISE THEME */

[data-testid="stDataFrame"] {
    border: 1px solid #DCE3EF;
    border-radius: 12px;
    overflow: hidden;
}

/* Pandas Styler table rendered by st.table */
[data-testid="stTable"] table {
    width: 100% !important;
    border-collapse: separate !important;
    border-spacing: 0 !important;
    background: #FFFFFF !important;
    border: 1px solid #DCE3EF !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

[data-testid="stTable"] thead th {
    background: #102746 !important;
    color: #FFFFFF !important;
}

[data-testid="stTable"] tbody td {
    color: #263957 !important;
}

[data-testid="stDataFrame"] [role="columnheader"] {
    background: #102746 !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

[data-testid="stDataFrame"] [role="gridcell"] {
    color: #263957 !important;
    background: #FFFFFF !important;
    border-bottom: 1px solid #E1E7F0 !important;
}

/* Alternating table rows — white and FlightWise light grey */
[data-testid="stDataFrame"] [role="rowgroup"] [role="row"]:nth-child(odd) [role="gridcell"] {
    background: #F3F5FA !important;
}

[data-testid="stDataFrame"] [role="rowgroup"] [role="row"]:nth-child(even) [role="gridcell"] {
    background: #FFFFFF !important;
}

[data-testid="stDataFrame"] [role="rowgroup"] [role="row"]:hover [role="gridcell"] {
    background: #EEE9FF !important;
}

/* Plotly chart panels — white surface with subtle FlightWise border */
div[data-testid="stPlotlyChart"] {
    background: #FFFFFF !important;
    border: 1px solid #E1E7F0 !important;
    border-radius: 14px !important;
    padding: 10px !important;
}

/* INPUTS */

.stTextInput label,
.stSelectbox label,
.stDateInput label,
.stTimeInput label,
.stNumberInput label {
    color: #34445F !important;
    font-size: 12px !important;
    font-weight: 700 !important;
}

/* WHITE FORM CONTAINER */

[data-testid="stForm"] {
    background: #FFFFFF !important;
    border: 1px solid #E1E7F0 !important;
    border-radius: 14px !important;
    padding: 24px !important;
}

/* WHITE DROPDOWN FIELDS */

div[data-testid="stSelectbox"] div[data-baseweb="select"],
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
div[data-testid="stSelectbox"] div[data-baseweb="select"] [role="combobox"] {
    background: #FFFFFF !important;
    color: #000000 !important;
    border-color: #D7DEEA !important;
    border-radius: 9px !important;
    min-height: 43px;
}

/* BLACK SELECTED DROPDOWN VALUES */

div[data-baseweb="select"] span {
    color: #000000 !important;
}

/* WHITE TEXT, NUMBER, DATE AND TIME INPUTS WITH BLACK VALUES */

div[data-testid="stTextInput"] [data-baseweb="input"],
div[data-testid="stNumberInput"] [data-baseweb="input"],
div[data-testid="stDateInput"] [data-baseweb="input"],
div[data-testid="stTimeInput"] [data-baseweb="input"] {
    background: #FFFFFF !important;
    border: 1px solid #D7DEEA !important;
    border-radius: 9px !important;
}

div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input,
div[data-testid="stDateInput"] input,
div[data-testid="stTimeInput"] input {
    background: #FFFFFF !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    border: 0 !important;
    border-radius: 9px !important;
    min-height: 43px;
    color-scheme: light !important;
}

div[data-testid="stDateInput"] button,
div[data-testid="stTimeInput"] button,
div[data-testid="stNumberInput"] button {
    background: #FFFFFF !important;
    color: #102746 !important;
}

/* WHITE DROPDOWN MENU WITH BLACK OPTIONS */

div[data-baseweb="popover"],
ul[role="listbox"] {
    background: #FFFFFF !important;
}

li[role="option"] {
    background: #FFFFFF !important;
    color: #000000 !important;
}

li[role="option"]:hover {
    background: #F0ECFF !important;
    color: #4C27C7 !important;
}

/* BUTTONS */

.stButton > button,
.stFormSubmitButton > button {
    border-radius: 10px;
    font-weight: 700;
    min-height: 42px;
}

/* WHITE PREDICT FLIGHT DELAY BUTTON */

.stFormSubmitButton > button {
    width: 100%;
    min-height: 48px;
    background: #FFFFFF !important;
    color: #5B35CF !important;
    border: 1.5px solid #6845DE !important;
    border-radius: 10px;
    font-weight: 800;
}

.stFormSubmitButton > button p {
    color: #5B35CF !important;
}

.stFormSubmitButton > button:hover {
    background: #F2EEFF !important;
    border-color: #4C27C7 !important;
}

.stFormSubmitButton > button:hover p {
    color: #4C27C7 !important;
}

/* HERO */

.hero {
    min-height: 230px;
    border-radius: 18px;
    padding: 34px 42px;
    margin-bottom: 21px;

    display: flex;
    flex-direction: column;
    justify-content: center;

    background:
      linear-gradient(
        90deg,
        rgba(9,27,55,.97),
        rgba(16,39,75,.83) 57%,
        rgba(16,39,75,.15)
      ),
      url("https://images.unsplash.com/photo-1436491865332-7a61a109cc05?auto=format&fit=crop&w=1800&q=85");

    background-size: cover;
    background-position: center;
}

.hero-tag {
    width: fit-content;
    color: white;
    background: rgba(255,255,255,.13);
    border: 1px solid rgba(255,255,255,.2);
    border-radius: 30px;
    padding: 7px 13px;
    font-size: 11px;
    font-weight: 800;
}

.hero-title {
    color: white !important;
    font-size: clamp(34px,4vw,49px);
    font-weight: 800;
    line-height: 1.12;
    margin: 12px 0 7px;
}

.hero-title span {
    color: #A994FF;
}

.hero-copy {
    color: #E6EDF9;
    font-size: 14px;
    max-width: 650px;
    line-height: 1.7;
}

/* SECTION HEADINGS */

.eyebrow {
    color: #7357C9;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.7px;
    text-transform: uppercase;
}

.section-intro {
    color: var(--muted);
    font-size: 13px;
    margin-top: -8px;
    margin-bottom: 20px;
}

.section-heading {
    margin: 30px 0 14px;
}

.section-heading h2 {
    font-size: 23px;
    font-weight: 800;
    margin: 0 0 5px;
}

.section-heading p {
    font-size: 13px;
    margin: 0;
}

/* PREDICTION RESULT */

.result-card {
    border-radius: 14px;
    padding: 22px 23px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 18px;
    margin: 8px 0 15px;
}

.result-status {
    font-size: 27px;
    font-weight: 800;
    margin: 0;
}

.result-description {
    font-size: 12px;
    line-height: 1.6;
    margin: 5px 0 0;
}

.prob-label {
    color: #65738B;
    font-size: 11px;
    text-align: right;
}

.prob-value {
    font-family: 'Manrope',sans-serif;
    font-size: 35px;
    line-height: 1.2;
    font-weight: 800;
    text-align: right;
}

/* SUMMARY */

.summary-strip {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    background: #F1EDFF;
    color: #30215F;
    border-radius: 11px;
    padding: 16px;
    font-size: 12px;
    font-weight: 700;
}

/* WORKFLOW */

.workflow-card {
    min-height: 140px;
    padding: 5px 2px;
}

.workflow-num {
    display: flex;
    width: 36px;
    height: 36px;
    border-radius: 50%;
    align-items: center;
    justify-content: center;
    background: var(--purple);
    color: white;
    font-weight: 800;
    margin-bottom: 12px;
}

.workflow-card h4 {
    font-size: 16px;
    margin: 0 0 7px;
}

.workflow-card p {
    font-size: 12px;
    line-height: 1.6;
}

/* FOOTER */

.footer {
    border-top: 1px solid #DFE5EF;
    margin-top: 38px;
    padding-top: 18px;
    color: #8490A4;
    font-size: 11px;
    line-height: 1.8;
}

.small-note {
    color: #8794A8;
    font-size: 11px;
}

@media(max-width:900px) {

    .block-container {
        padding: 1rem 1rem 2rem;
    }

    .hero {
        padding: 25px;
        min-height: 200px;
    }

    .hero-title {
        font-size: 35px;
    }

    .result-card {
        flex-direction: column;
        align-items: flex-start;
    }

    .prob-label,
    .prob-value {
        text-align: left;
    }
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL FEATURES — MATCH THE CURRENT MASTER NOTEBOOK
# ============================================================

NUMERIC_FEATURES = [
    "Distance", "ScheduledDuration",
    "Year", "Month", "Day", "DayOfWeek", "DayOfYear", "IsWeekend",
    "DepHour", "DepMinute", "ArrHour", "ArrMinute",
    "DepHourSin", "DepHourCos", "MonthSin", "MonthCos"
]

CATEGORICAL_FEATURES = [
    "Airline", "Origin", "Dest", "DepPeriod"
]

FEATURES = [
    "Airline", "Origin", "Dest",
    "Distance", "ScheduledDuration",
    "Year", "Month", "Day", "DayOfWeek", "DayOfYear", "IsWeekend",
    "DepHour", "DepMinute", "ArrHour", "ArrMinute",
    "DepHourSin", "DepHourCos", "MonthSin", "MonthCos",
    "DepPeriod"
]


# ============================================================
# HISTORICAL DATA
# ============================================================

YEARLY_DATA = pd.DataFrame({

    "Year": [
        2018, 2019, 2020, 2021, 2022
    ],

    "Completed flights": [
        5586619,
        7917264,
        4712930,
        6185870,
        3944916
    ],

    "Delayed flights": [
        1088342,
        1527455,
        468955,
        1068059,
        853962
    ],

    "Delay rate (%)": [
        19.48,
        19.29,
        9.95,
        17.27,
        21.65
    ]

})


# ============================================================
# MODEL EVALUATION RESULTS
# Values are loaded from the current notebook's deployment bundle.
# ============================================================

TEST_METRICS = {}
TEST_RECORDS = 0
DEFAULT_THRESHOLD = 0.50


# ============================================================
# LOAD DEPLOYMENT BUNDLE CREATED BY THE MASTER NOTEBOOK
# ============================================================

@st.cache_resource
def load_bundle(path_string):
    path = Path(path_string)

    if not path.exists():
        return None, f"Model bundle not found: {path}"

    try:
        obj = joblib.load(path)

        if not isinstance(obj, dict):
            return None, "Deployment bundle must be a dictionary."

        required_keys = [
            "model", "model_name", "threshold",
            "numeric_imputer", "scaler", "categorical_imputer",
            "training_columns", "categorical_features", "numerical_features"
        ]

        missing = [key for key in required_keys if key not in obj]
        if missing:
            return None, "Bundle is missing required items: " + ", ".join(missing)

        return obj, None

    except Exception as exc:
        return None, str(exc)


bundle, bundle_error = load_bundle(str(MODEL_PATH))


def get_bundle_value(keys, default=None):
    if not isinstance(bundle, dict):
        return default

    for key in keys:
        if key in bundle and bundle[key] is not None:
            return bundle[key]

    return default


model = get_bundle_value(["model"])
model_name = get_bundle_value(["model_name"], "Trained classifier")
threshold = float(get_bundle_value(["threshold"], DEFAULT_THRESHOLD))

numeric_imputer = get_bundle_value(["numeric_imputer"])
scaler = get_bundle_value(["scaler"])
categorical_imputer = get_bundle_value(["categorical_imputer"])
training_columns = get_bundle_value(["training_columns"], [])
bundle_numeric_features = get_bundle_value(["numerical_features"], NUMERIC_FEATURES)
bundle_categorical_features = get_bundle_value(["categorical_features"], CATEGORICAL_FEATURES)

TEST_METRICS = get_bundle_value(["final_metrics"], {})
TEST_RECORDS = int(get_bundle_value(["test_records"], 0) or 0)


# ============================================================
# MODEL STATUS
# ============================================================

def model_is_ready():
    return (
        bundle is not None
        and model is not None
        and numeric_imputer is not None
        and scaler is not None
        and categorical_imputer is not None
        and bool(training_columns)
    )


# ============================================================
# FEATURE ENGINEERING — SAME LOGIC AS THE MASTER NOTEBOOK
# ============================================================

def get_period(hour):
    if pd.isna(hour):
        return "Unknown"
    elif hour < 6:
        return "Night"
    elif hour < 12:
        return "Morning"
    elif hour < 18:
        return "Afternoon"
    return "Evening"


def engineer_single_flight(
    flight_date,
    airline,
    origin,
    destination,
    departure_time,
    arrival_time,
    distance
):
    dep_hour = departure_time.hour
    dep_minute = departure_time.minute
    arr_hour = arrival_time.hour
    arr_minute = arrival_time.minute

    departure_minutes = dep_hour * 60 + dep_minute
    arrival_minutes = arr_hour * 60 + arr_minute
    scheduled_duration = arrival_minutes - departure_minutes

    if scheduled_duration < 0:
        scheduled_duration += 1440

    month = flight_date.month
    day_of_week = flight_date.weekday()

    row = {
        "Airline": airline,
        "Origin": origin,
        "Dest": destination,
        "Distance": float(distance),
        "ScheduledDuration": float(scheduled_duration),
        "Year": flight_date.year,
        "Month": month,
        "Day": flight_date.day,
        "DayOfWeek": day_of_week,
        "DayOfYear": flight_date.timetuple().tm_yday,
        "IsWeekend": int(day_of_week >= 5),
        "DepHour": dep_hour,
        "DepMinute": dep_minute,
        "ArrHour": arr_hour,
        "ArrMinute": arr_minute,
        "DepHourSin": np.sin(2 * np.pi * dep_hour / 24),
        "DepHourCos": np.cos(2 * np.pi * dep_hour / 24),
        "MonthSin": np.sin(2 * np.pi * month / 12),
        "MonthCos": np.cos(2 * np.pi * month / 12),
        "DepPeriod": get_period(dep_hour)
    }

    return pd.DataFrame([row]).reindex(columns=FEATURES)


# ============================================================
# PREDICTION FUNCTION — REUSE NOTEBOOK IMPUTERS, SCALER AND DUMMIES
# ============================================================

def prepare_prediction_features(frame):
    numeric_columns = list(bundle_numeric_features)
    categorical_columns = list(bundle_categorical_features)

    numeric_values = pd.DataFrame(
        numeric_imputer.transform(frame[numeric_columns]),
        columns=numeric_columns
    )

    numeric_values = pd.DataFrame(
        scaler.transform(numeric_values),
        columns=numeric_columns
    )

    categorical_values = pd.DataFrame(
        categorical_imputer.transform(frame[categorical_columns]),
        columns=categorical_columns
    )

    ready_frame = pd.concat(
        [numeric_values, categorical_values],
        axis=1
    )

    encoded_frame = pd.get_dummies(
        ready_frame,
        columns=categorical_columns
    )

    encoded_frame = encoded_frame.reindex(
        columns=training_columns,
        fill_value=0
    )

    return encoded_frame


def predict_flight(frame):
    if not model_is_ready():
        raise RuntimeError("Trained model bundle is missing or incomplete.")

    transformed = prepare_prediction_features(frame)

    if not hasattr(model, "predict_proba"):
        raise RuntimeError("The selected model does not support predict_proba().")

    probabilities = model.predict_proba(transformed)[0]
    classes = list(getattr(model, "classes_", [0, 1]))

    positive_idx = classes.index(1) if 1 in classes else len(probabilities) - 1
    probability = float(probabilities[positive_idx])
    prediction = int(probability >= threshold)

    return prediction, probability


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def section(title, subtitle=None):

    st.markdown(
        f"""
        <div class="section-heading">
            <h2>{title}</h2>
            <p>{subtitle or ""}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_footer():

    st.markdown(
        """
        <div class="footer">

        FLIGHTWISE AI · MACHINE LEARNING FLIGHT DELAY PREDICTION

        <br>

        Historical U.S. flight data, 2018–2022.

        <br>

        Predictions are estimates based on historical patterns.
        This application does not connect to live airline,
        airport or weather feeds.

        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# DROPDOWN OPTIONS
# ============================================================

# Keep the existing dropdown experience/design. These options cover the
# common airports and airlines represented in the training dataset.
AIRLINE_OPTIONS = [
    "Southwest Airlines Co.",
    "American Airlines Inc.",
    "Delta Air Lines Inc.",
    "United Air Lines Inc.",
    "SkyWest Airlines Inc.",
    "Republic Airways",
    "JetBlue Airways",
    "Alaska Airlines Inc.",
    "Spirit Air Lines",
    "Frontier Airlines Inc.",
    "Hawaiian Airlines Inc.",
    "Allegiant Air"
]

AIRPORT_OPTIONS = [
    "ATL", "DFW", "DEN", "ORD", "LAX", "CLT", "MCO", "LAS",
    "PHX", "MIA", "SEA", "JFK", "EWR", "SFO", "BOS", "IAH",
    "MSP", "DTW", "FLL", "BWI"
]

# Prefer categories actually seen by the fitted categorical imputer where available.
def get_training_categories(column, fallback):
    if bundle is None:
        return fallback

    try:
        idx = list(bundle_categorical_features).index(column)
        values = categorical_imputer.statistics_[idx]
        # The imputer stores one most-frequent value, not the full category list.
        # Fallback list is therefore retained for user selection.
    except Exception:
        pass

    return fallback


AIRLINE_OPTIONS = get_training_categories("Airline", AIRLINE_OPTIONS)
ORIGIN_OPTIONS = get_training_categories("Origin", AIRPORT_OPTIONS)
DEST_OPTIONS = get_training_categories("Dest", AIRPORT_OPTIONS)


# ============================================================
# COMMON FUNCTIONS
# ============================================================

def prediction_result(frame, airline, origin, destination,
                      flight_date, departure_time, arrival_time,
                      distance):

    if not model_is_ready():
        st.error(
            "The trained model is not available. "
            "Check models/flightwise_deployment_bundle.joblib."
        )
        return

    try:
        prediction, probability = predict_flight(frame)

        status = "Delayed" if prediction == 1 else "Likely On Time"
        color = "#D92D3A" if prediction == 1 else "#16845B"
        background = "#FFF0F0" if prediction == 1 else "#EAF8F1"

        st.markdown(
            f"""
            <div class="result-card"
                 style="background:{background};">
                <div>
                    <div style="font-size:12px;color:#65738B;">
                        PREDICTION RESULT
                    </div>
                    <div class="result-status"
                         style="color:{color};">
                        {status}
                    </div>
                    <div class="result-description"
                         style="color:#65738B;">
                        {"The model estimates an elevated chance of a delay of 15 minutes or more."
                         if prediction == 1
                         else "The model estimates a lower chance of a delay of 15 minutes or more."}
                    </div>
                </div>
                <div>
                    <div class="prob-label">
                        Probability of delay
                    </div>
                    <div class="prob-value"
                         style="color:{color};">
                        {probability:.1%}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="summary-strip">
                <span>{airline}</span>
                <span>{origin} → {destination}</span>
                <span>{flight_date.strftime('%d %b %Y')}</span>
                <span>{departure_time.strftime('%H:%M')}</span>
                <span>{distance:,.0f} miles</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption(
            f"Decision threshold: {threshold:.3f}. "
            "This is a model estimate, not a guaranteed flight outcome. "
            "The model was trained on historical data ending in July 2022. "
            "Predictions for later dates use learned historical patterns; "
            "they are not updated with newer flight, weather, airline or "
            "airport information and may be less reliable as conditions change."
        )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")


def flight_form(form_key):

    with st.form(form_key):

        col1, col2 = st.columns(2)

        with col1:
            flight_date = st.date_input(
                "Flight date",
                value=date.today(),
                max_value=date(2035, 12, 31)
            )

            airline = st.selectbox(
                "Airline",
                AIRLINE_OPTIONS
            )

            origin = st.selectbox(
                "Origin airport",
                ORIGIN_OPTIONS
            )

            departure_time = st.time_input(
                "Scheduled departure",
                value=time(9, 0)
            )

        with col2:
            destination = st.selectbox(
                "Destination airport",
                DEST_OPTIONS
            )

            arrival_time = st.time_input(
                "Scheduled arrival",
                value=time(12, 0)
            )

            distance = st.number_input(
                "Flight distance (miles)",
                min_value=1.0,
                max_value=6000.0,
                value=850.0,
                step=10.0
            )

        submitted = st.form_submit_button(
            "Predict Flight Delay",
            use_container_width=True
        )

    if submitted:

        if origin == destination:
            st.error("Origin and destination must be different.")
            return

        frame = engineer_single_flight(
            flight_date,
            airline,
            origin,
            destination,
            departure_time,
            arrival_time,
            distance
        )

        prediction_result(
            frame,
            airline,
            origin,
            destination,
            flight_date,
            departure_time,
            arrival_time,
            distance
        )


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

with st.sidebar:

    # BRAND
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-icon">✈</div>
            <div>
                <div class="sidebar-brand-name">FlightWise</div>
                <div class="sidebar-brand-subtitle">
                    AI FLIGHT INTELLIGENCE
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # NAVIGATION
    st.markdown(
        '<div class="sidebar-section-label">EXPLORE</div>',
        unsafe_allow_html=True
    )

    navigation_pages = [
        "Home",
        "Predict Delay",
        "Model Insights",
        "Exploratory Analysis",
        "Project Overview"
    ]

    for nav_page in navigation_pages:

        is_active = st.session_state.page == nav_page

        if st.button(
            nav_page,
            key=f"nav_{nav_page}",
            type="primary" if is_active else "secondary",
            use_container_width=True
        ):
            if st.session_state.page != nav_page:
                st.session_state.page = nav_page
                st.rerun()

    page = st.session_state.page

    st.divider()

    # MODEL STATUS
    if model_is_ready():
        st.markdown(
            f"""
            <div class="sidebar-status">
                <div class="sidebar-status-heading">
                    <span class="sidebar-status-dot"></span>
                    SYSTEM STATUS
                </div>
                <div class="sidebar-status-main">
                    ML Model Loaded
                </div>
                <div class="sidebar-status-detail">
                    {model_name} · Ready for prediction
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            f"""
            <div class="sidebar-status-error">
                <div class="sidebar-status-heading">
                    MODEL STATUS
                </div>
                <div class="sidebar-status-main">
                    ML Model Unavailable
                </div>
                <div class="sidebar-status-detail">
                    {bundle_error or "Check your model bundle."}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # FOOTER
    st.markdown(
        """
        <div class="sidebar-footer">
            <div class="sidebar-footer-title">FLIGHTWISE AI</div>
            <div class="sidebar-footer-copy">
                Machine Learning Project<br>
                Historical data · 2018–2022
            </div>
            <div class="sidebar-footer-version">
                VERSION 1.0
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PAGE 1 — HOME
# ============================================================

if page == "Home":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-tag">
                ✦ MACHINE LEARNING POWERED
            </div>
            <div class="hero-title">
                Smarter Flights.<br>
                <span>Better Predictions.</span>
            </div>
            <div class="hero-copy">
                Predict potential flight delays using historical
                aviation data and machine learning.
                Explore flight intelligence and make more
                informed travel decisions.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    section(
        "Flight Intelligence Overview",
        "A snapshot of the historical dataset and final model performance."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Eligible Flights", "28.35M")
    c2.metric("Overall Delay Rate", "17.7%")
    c3.metric("Model Features", str(len(FEATURES)))
    c4.metric(
        "Test ROC-AUC",
        f"{TEST_METRICS.get('ROC-AUC', 0):.4f}" if TEST_METRICS else "Pending"
    )

    section(
        "Predict Your Flight",
        "Enter your scheduled flight information to estimate its delay risk."
    )

    flight_form("home_prediction_form")

    section(
        "How FlightWise AI Works",
        "From historical flight records to an individual prediction."
    )

    steps = st.columns(4)

    workflow = [
        ("01", "Flight Details",
         "Enter airline, airports, schedule and distance."),
        ("02", "Feature Engineering",
         "Extract date, time and cyclic features."),
        ("03", "Machine Learning",
         "Apply the trained Random Forest model."),
        ("04", "Prediction",
         "View estimated delay probability and status.")
    ]

    for col, (number, heading, description) in zip(steps, workflow):

        with col:
            st.markdown(
                f"""
                <div class="workflow-card">
                    <div class="workflow-num">{number}</div>
                    <h4>{heading}</h4>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    show_footer()


# ============================================================
# PAGE 2 — PREDICT DELAY
# ============================================================

elif page == "Predict Delay":

    st.title("Flight Delay Prediction")

    st.markdown(
        "Enter your flight details below to calculate the estimated "
        "probability of an arrival delay of 15 minutes or more."
    )

    if not model_is_ready():
        st.error(
            "Model bundle not found or incomplete. "
            "Place flightwise_deployment_bundle.joblib inside models/."
        )
    else:
        flight_form("dedicated_prediction_form")

    show_footer()


# ============================================================
# PAGE 3 — MODEL INSIGHTS
# ============================================================

elif page == "Model Insights":

    st.title("Model Performance & Insights")

    st.markdown(
        "Evaluation results from the held-out 2022 test dataset."
    )

    if TEST_METRICS:
        c1, c2, c3, c4, c5 = st.columns(5)

        metric_keys = [
            ("Accuracy", "Accuracy"),
            ("Precision", "Precision"),
            ("Recall", "Recall"),
            ("F1-score", "F1 Score"),
            ("ROC-AUC", "ROC-AUC")
        ]

        for col, (label, key) in zip([c1, c2, c3, c4, c5], metric_keys):
            value = TEST_METRICS.get(key)
            col.metric(label, f"{float(value):.2%}" if value is not None else "N/A")
    else:
        st.info("Final test metrics will appear here after you run the updated master notebook and save its deployment bundle.")

    section(
        "Test Dataset",
        "Final model evaluation on unseen 2022 flight records."
    )

    c1, c2 = st.columns(2)

    c1.metric("Evaluation Records", f"{TEST_RECORDS:,}" if TEST_RECORDS else "Not saved")
    c2.metric("Decision Threshold", f"{threshold:.6f}")

    section(
        "Understanding the Metrics",
        "What each evaluation measure tells us."
    )

    explanations = {
        "Accuracy": "Overall proportion of correct predictions.",
        "Precision": "Proportion of predicted delays that were actual delays.",
        "Recall": "Proportion of actual delays identified by the model.",
        "F1-score": "Harmonic mean of precision and recall.",
        "ROC-AUC": "Ability to distinguish delayed and non-delayed flights."
    }

    for name, explanation in explanations.items():
        st.markdown(f"**{name}** — {explanation}")

    section(
        "Final Model Configuration",
        "The selected model and its evaluation approach."
    )

    st.markdown(
        f"""
        - **Selected algorithm:** {model_name}
        - **Training period:** 2018–2020
        - **Validation period:** 2021
        - **Held-out test period:** 2022
        - **Target:** Arrival delay of 15 minutes or more
        - **Model selection:** Validation accuracy and F1-score
        - **Threshold selection:** 2021 validation data
        """
    )

    st.warning(
        "The model has moderate predictive performance. "
        "Accuracy alone is not sufficient because the dataset "
        "has an imbalanced target. Predictions should be treated "
        "as estimates, not guarantees."
    )

    show_footer()


# ============================================================
# PAGE 4 — EXPLORATORY ANALYSIS
# ============================================================

elif page == "Exploratory Analysis":

    st.title("Exploratory Data Analysis")

    st.markdown(
        "Explore historical flight volumes and delay patterns "
        "across the five dataset years."
    )

    section(
        "Yearly Flight Performance",
        "Completed flights, delayed flights and delay rates."
    )

    styled_yearly_data = (
        YEARLY_DATA.style
        .format({
            "Completed flights": "{:,.0f}",
            "Delayed flights": "{:,.0f}",
            "Delay rate (%)": "{:.2f}%"
        })
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#102746"),
                    ("color", "#FFFFFF"),
                    ("font-weight", "700"),
                    ("border-bottom", "2px solid #6845DE"),
                    ("padding", "12px")
                ]
            },
            {
                "selector": "td",
                "props": [
                    ("color", "#263957"),
                    ("border-bottom", "1px solid #E1E7F0"),
                    ("padding", "10px")
                ]
            },
            {
                "selector": "tbody tr:nth-child(odd)",
                "props": [("background-color", "#F3F5FA")]
            },
            {
                "selector": "tbody tr:nth-child(even)",
                "props": [("background-color", "#FFFFFF")]
            },
            {
                "selector": "tbody tr:hover",
                "props": [("background-color", "#EEE9FF")]
            }
        ])
    )

    st.table(styled_yearly_data)

    section(
        "Delay Rate by Year",
        "Percentage of completed flights delayed by at least 15 minutes."
    )

    fig = px.line(
        YEARLY_DATA,
        x="Year",
        y="Delay rate (%)",
        markers=True,
        text="Delay rate (%)",
        title="Yearly Flight Delay Rate"
    )

    fig.update_traces(
        line=dict(color="#6845DE", width=3),
        marker=dict(
            color="#102746",
            size=10,
            line=dict(color="#FFFFFF", width=2)
        ),
        texttemplate="%{text:.2f}%",
        textposition="top center",
        textfont=dict(color="#4C27C7", size=12)
    )

    fig.update_layout(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(
            family="DM Sans, sans-serif",
            color="#263957",
            size=12
        ),
        title_font=dict(
            family="Manrope, sans-serif",
            color="#102746",
            size=18
        ),
        xaxis=dict(
            tickmode="linear",
            dtick=1,
            color="#52627C",
            gridcolor="#E8EAF3",
            linecolor="#D9DFEA",
            zerolinecolor="#D9DFEA"
        ),
        yaxis=dict(
            title="Delay rate (%)",
            color="#52627C",
            gridcolor="#E8EAF3",
            linecolor="#D9DFEA",
            zerolinecolor="#D9DFEA"
        ),
        margin=dict(l=20, r=20, t=55, b=20),
        hoverlabel=dict(
            bgcolor="#102746",
            bordercolor="#6845DE",
            font=dict(color="#FFFFFF")
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    section(
        "Completed vs Delayed Flights",
        "Annual flight volume comparison."
    )

    chart_data = YEARLY_DATA.melt(
        id_vars="Year",
        value_vars=[
            "Completed flights",
            "Delayed flights"
        ],
        var_name="Flight category",
        value_name="Number of flights"
    )

    bar = px.bar(
        chart_data,
        x="Year",
        y="Number of flights",
        color="Flight category",
        barmode="group",
        title="Yearly Flight Volume"
    )

    bar.update_traces(
        marker_line_color="#FFFFFF",
        marker_line_width=1.2,
        opacity=0.92
    )

    bar.update_layout(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        colorway=["#6845DE", "#102746"],
        font=dict(
            family="DM Sans, sans-serif",
            color="#263957",
            size=12
        ),
        title_font=dict(
            family="Manrope, sans-serif",
            color="#102746",
            size=18
        ),
        xaxis=dict(
            color="#52627C",
            gridcolor="#E8EAF3",
            linecolor="#D9DFEA",
            zerolinecolor="#D9DFEA"
        ),
        yaxis=dict(
            title="Number of flights",
            color="#52627C",
            gridcolor="#E8EAF3",
            linecolor="#D9DFEA",
            zerolinecolor="#D9DFEA"
        ),
        legend=dict(
            title_font=dict(color="#102746"),
            font=dict(color="#263957")
        ),
        margin=dict(l=20, r=20, t=55, b=20),
        hoverlabel=dict(
            bgcolor="#102746",
            bordercolor="#6845DE",
            font=dict(color="#FFFFFF")
        )
    )

    st.plotly_chart(bar, use_container_width=True)

    section(
        "Saved EDA Visualizations",
        "Additional figures generated during your EDA notebooks."
    )

    saved_figures = [
        ("yearly_delay_rate.png", "Yearly Delay Rate"),
        ("monthly_delay_rate.png", "Monthly Delay Rate"),
        ("airline_delay_rates.png", "Airline Delay Rates")
    ]

    for filename, title in saved_figures:

        image_path = FIGURES_DIR / filename

        if image_path.exists():
            st.subheader(title)
            st.image(str(image_path), use_container_width=True)

    show_footer()


# ============================================================
# PAGE 5 — PROJECT OVERVIEW
# ============================================================

elif page == "Project Overview":

    st.title("About FlightWise AI")

    st.markdown(
        """
        FlightWise AI is a machine-learning-based flight delay
        prediction application developed using historical
        U.S. flight records from 2018 to 2022.
        """
    )

    section(
        "Problem Statement",
        "Why flight delay prediction matters."
    )

    st.markdown(
        """
        Flight delays affect passengers, airlines and airport
        operations. This project investigates historical flight
        patterns to estimate the likelihood of an arrival delay.

        The application uses scheduled flight information and
        a trained classification model to generate predictions.
        """
    )

    section(
        "Dataset Information",
        "Historical flight records used in model development."
    )

    st.markdown(
        """
        - **Source:** Kaggle — Flight Delay Dataset 2018–2022
        - **Coverage:** 2018 through July 2022
        - **Original records:** 28+ million flights
        - **Target variable:** Delayed by 15 minutes or more
        - **Cancelled and diverted flights:** Excluded from
          completed-flight target modelling
        """
    )

    section(
        "Machine Learning Pipeline",
        "The major stages of the FlightWise AI project."
    )

    pipeline = [
        "01 — Data Understanding",
        "02 — Exploratory Data Analysis",
        "03 — Data Preprocessing",
        "04 — Feature Engineering",
        "05 — Model Training",
        "06 — Model Evaluation",
        "07 — Final Model Selection",
        "08 — Streamlit Deployment"
    ]

    for item in pipeline:
        st.markdown(f"**{item}**")

    section(
        "Model Input Features",
        "The 25 engineered features used by the final model."
    )

    with st.expander("View numerical features"):
        st.write(NUMERIC_FEATURES)

    with st.expander("View categorical features"):
        st.write(CATEGORICAL_FEATURES)

    section(
        "Technology Stack",
        "Tools and libraries used in the project."
    )

    st.markdown(
        """
        - Python
        - Pandas and NumPy
        - Scikit-learn
        - Random Forest Classifier
        - Joblib
        - Streamlit
        - Plotly
        - Matplotlib and Seaborn for EDA
        """
    )

    st.info(
        "FlightWise AI is a historical-data-based ML application. "
        "It does not currently connect to live airline, airport "
        "or weather APIs."
    )

    show_footer()
