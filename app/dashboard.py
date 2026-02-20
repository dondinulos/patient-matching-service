"""
Patient Matching Dashboard - Streamlit UI

A comprehensive dashboard for viewing and managing patient match results.

Run with: streamlit run app/dashboard.py
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import json
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Graph visualization
try:
    from streamlit_agraph import agraph, Node, Edge, Config
    AGRAPH_AVAILABLE = True
except ImportError:
    AGRAPH_AVAILABLE = False

from azure.identity import DefaultAzureCredential, AzureCliCredential
from azure.cosmos import CosmosClient
from gremlin_python.driver import client, serializer

# Agent Framework (optional)
try:
    import asyncio
    from agent_framework.azure import AzureAIClient
    from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential
    from src.patient_matching.agent import create_foundry_agent
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Patient Matching Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme ────────────────────────────────────────────────────────────────────
_is_dark = True

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Design System — Preclinic-inspired tokens                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

_LIGHT = {
    # Brand
    "primary": "#2E6FF3",
    "primary_hover": "#1B5CD9",
    "primary_light": "#EBF1FE",
    "secondary": "#0DBFA9",
    "secondary_light": "#E6F9F6",
    "danger": "#E74C3C",
    "danger_light": "#FDF0EF",
    "warning": "#F59E0B",
    "warning_light": "#FFF8E6",
    "success": "#10B981",
    "success_light": "#ECFDF5",
    # Surfaces
    "bg": "#F5F6FA",
    "surface": "#FFFFFF",
    "sidebar_bg": "#FFFFFF",
    "sidebar_border": "#E8ECF1",
    # Text
    "text": "#1E293B",
    "text_secondary": "#64748B",
    "text_muted": "#94A3B8",
    # Borders & Shadows
    "border": "#E2E8F0",
    "shadow_sm": "0 1px 2px rgba(0,0,0,0.05)",
    "shadow": "0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06)",
    "shadow_md": "0 4px 6px rgba(0,0,0,0.07), 0 2px 4px rgba(0,0,0,0.06)",
    "shadow_lg": "0 10px 15px rgba(0,0,0,0.1), 0 4px 6px rgba(0,0,0,0.05)",
    # Misc
    "radius": "10px",
    "radius_sm": "6px",
    "radius_lg": "14px",
    "nav_active_bg": "#EBF1FE",
    "nav_active_text": "#2E6FF3",
    "nav_hover_bg": "#F1F5F9",
    "input_bg": "#FFFFFF",
    "badge_bg": "#F1F5F9",
    "table_header": "#F8FAFC",
    "table_stripe": "#FAFBFD",
    "scrollbar_track": "#F5F6FA",
    "scrollbar_thumb": "#CBD5E1",
    "divider": "#E2E8F0",
    "chart_grid": "#F1F5F9",
}

_DARK = {
    "primary": "#5B8DEF",
    "primary_hover": "#7BA4F7",
    "primary_light": "#1E2A3E",
    "secondary": "#34D4B8",
    "secondary_light": "#142E29",
    "danger": "#F87171",
    "danger_light": "#2D1F1F",
    "warning": "#FBBF24",
    "warning_light": "#2D2A1A",
    "success": "#34D399",
    "success_light": "#152E24",
    "bg": "#0F1117",
    "surface": "#1A1D26",
    "sidebar_bg": "#141720",
    "sidebar_border": "#262A36",
    "text": "#E2E8F0",
    "text_secondary": "#94A3B8",
    "text_muted": "#64748B",
    "border": "#2D3344",
    "shadow_sm": "0 1px 2px rgba(0,0,0,0.3)",
    "shadow": "0 1px 3px rgba(0,0,0,0.4), 0 1px 2px rgba(0,0,0,0.3)",
    "shadow_md": "0 4px 6px rgba(0,0,0,0.4), 0 2px 4px rgba(0,0,0,0.3)",
    "shadow_lg": "0 10px 15px rgba(0,0,0,0.5), 0 4px 6px rgba(0,0,0,0.3)",
    "radius": "10px",
    "radius_sm": "6px",
    "radius_lg": "14px",
    "nav_active_bg": "#1E2A3E",
    "nav_active_text": "#5B8DEF",
    "nav_hover_bg": "#1F2233",
    "input_bg": "#1A1D26",
    "badge_bg": "#262A36",
    "table_header": "#1F2233",
    "table_stripe": "#161922",
    "scrollbar_track": "#0F1117",
    "scrollbar_thumb": "#3B4252",
    "divider": "#2D3344",
    "chart_grid": "#262A36",
}

T = _DARK if _is_dark else _LIGHT

# ── Inject global CSS ────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* === Reset & Global === */
    html, body, .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        background: {T["bg"]} !important;
        color: {T["text"]} !important;
    }}
    .stApp > header {{ background: transparent !important; }}
    p, li, td, th, label, summary, .stMarkdown {{
        color: {T["text"]} !important;
        font-family: 'Inter', sans-serif !important;
    }}
    /* Text spans — but exclude icon fonts used by Streamlit sidebar collapse */
    span:not([class*="material"]):not([data-testid]) {{
        color: {T["text"]} !important;
    }}
    div:not([class*="st"]):not([data-testid]) {{
        font-family: 'Inter', sans-serif !important;
    }}

    /* === Sidebar === */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div {{
        background: {T["sidebar_bg"]} !important;
        border-right: 1px solid {T["sidebar_border"]} !important;
        box-shadow: none !important;
    }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
        color: {T["text"]} !important;
    }}

    /* Hide radio group label */
    [data-testid="stSidebar"] .stRadio > label {{
        display: none !important;
    }}
    /* Nav list layout */
    [data-testid="stSidebar"] .stRadio > div {{
        display: flex !important;
        flex-direction: column !important;
        gap: 2px !important;
    }}
    /* Hide radio dot */
    [data-testid="stSidebar"] .stRadio > div > label > div:first-child {{
        display: none !important;
    }}
    /* Nav items */
    [data-testid="stSidebar"] .stRadio > div > label {{
        background: transparent !important;
        border-radius: {T["radius_sm"]} !important;
        padding: 9px 14px !important;
        margin: 0 !important;
        box-shadow: none !important;
        transition: background 0.15s ease, color 0.15s ease !important;
        cursor: pointer !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        color: {T["text_secondary"]} !important;
        border: none !important;
    }}
    [data-testid="stSidebar"] .stRadio > div > label p,
    [data-testid="stSidebar"] .stRadio > div > label span,
    [data-testid="stSidebar"] .stRadio > div > label div {{
        color: {T["text_secondary"]} !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
    }}
    [data-testid="stSidebar"] .stRadio > div > label:hover {{
        background: {T["nav_hover_bg"]} !important;
    }}
    [data-testid="stSidebar"] .stRadio > div > label:hover p,
    [data-testid="stSidebar"] .stRadio > div > label:hover span,
    [data-testid="stSidebar"] .stRadio > div > label:hover div {{
        color: {T["text"]} !important;
    }}
    /* Active nav */
    [data-testid="stSidebar"] .stRadio > div > label[data-checked="true"],
    [data-testid="stSidebar"] .stRadio > div > label[aria-checked="true"] {{
        background: {T["nav_active_bg"]} !important;
        box-shadow: none !important;
    }}
    [data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] p,
    [data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] span,
    [data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] div,
    [data-testid="stSidebar"] .stRadio > div > label[aria-checked="true"] p,
    [data-testid="stSidebar"] .stRadio > div > label[aria-checked="true"] span,
    [data-testid="stSidebar"] .stRadio > div > label[aria-checked="true"] div {{
        color: {T["nav_active_text"]} !important;
        font-weight: 600 !important;
    }}

    /* === Headings === */
    h1, h2, h3, .stTitle, [data-testid="stHeading"] {{
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        color: {T["text"]} !important;
    }}
    h1 {{ font-size: 1.75rem !important; letter-spacing: -0.02em !important; }}
    h2 {{ font-size: 1.35rem !important; letter-spacing: -0.01em !important; }}
    h3 {{ font-size: 1.1rem !important; }}

    /* === Metric Cards === */
    [data-testid="stMetric"] {{
        background: {T["surface"]} !important;
        border: 1px solid {T["border"]} !important;
        border-radius: {T["radius"]} !important;
        padding: 20px 22px !important;
        box-shadow: {T["shadow_sm"]} !important;
        transition: box-shadow 0.2s ease, transform 0.2s ease !important;
    }}
    [data-testid="stMetric"]:hover {{
        box-shadow: {T["shadow_md"]} !important;
        transform: translateY(-1px) !important;
    }}
    [data-testid="stMetricLabel"] {{
        font-weight: 500 !important;
        color: {T["text_secondary"]} !important;
        font-size: 0.8rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }}
    [data-testid="stMetricValue"] {{
        font-weight: 700 !important;
        color: {T["text"]} !important;
        font-size: 1.85rem !important;
    }}

    /* === Buttons === */
    .stButton > button {{
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        border-radius: {T["radius_sm"]} !important;
        border: 1px solid {T["border"]} !important;
        background: {T["surface"]} !important;
        color: {T["text"]} !important;
        box-shadow: {T["shadow_sm"]} !important;
        padding: 8px 16px !important;
        transition: all 0.15s ease !important;
    }}
    .stButton > button:hover {{
        box-shadow: {T["shadow"]} !important;
        border-color: {T["primary"]} !important;
        color: {T["primary"]} !important;
        background: {T["primary_light"]} !important;
    }}
    .stButton > button:active {{
        transform: scale(0.98) !important;
    }}

    /* Primary buttons */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {{
        background: {T["primary"]} !important;
        color: #FFFFFF !important;
        border-color: {T["primary"]} !important;
    }}
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {{
        background: {T["primary_hover"]} !important;
    }}

    /* === Tabs === */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0 !important;
        border-bottom: 1px solid {T["border"]} !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        border-radius: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        padding: 10px 18px !important;
        color: {T["text_secondary"]} !important;
        transition: color 0.15s ease, border-color 0.15s ease !important;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        color: {T["text"]} !important;
    }}
    .stTabs [aria-selected="true"] {{
        color: {T["primary"]} !important;
        border-bottom-color: {T["primary"]} !important;
        font-weight: 600 !important;
        background: transparent !important;
    }}
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {{ display: none !important; }}

    /* === Expanders === */
    [data-testid="stExpander"] {{
        background: {T["surface"]} !important;
        border: 1px solid {T["border"]} !important;
        border-radius: {T["radius"]} !important;
        box-shadow: none !important;
        margin-bottom: 8px !important;
        overflow: hidden !important;
    }}
    [data-testid="stExpander"] summary {{
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }}

    /* === Inputs === */
    .stSelectbox > div > div,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    [data-testid="stChatInput"] > div {{
        border-radius: {T["radius_sm"]} !important;
        border: 1px solid {T["border"]} !important;
        background: {T["input_bg"]} !important;
        box-shadow: none !important;
        font-family: 'Inter', sans-serif !important;
        color: {T["text"]} !important;
        transition: border-color 0.15s ease !important;
    }}
    .stSelectbox > div > div:focus-within,
    .stTextInput > div > div > input:focus,
    [data-testid="stChatInput"] > div:focus-within {{
        border-color: {T["primary"]} !important;
        box-shadow: 0 0 0 3px {T["primary"]}22 !important;
    }}

    /* === Dataframes & Tables === */
    [data-testid="stDataFrame"], .stDataFrame {{
        border-radius: {T["radius"]} !important;
        overflow: hidden !important;
        border: 1px solid {T["border"]} !important;
    }}

    /* === Chat Messages === */
    [data-testid="stChatMessage"] {{
        background: {T["surface"]} !important;
        border: 1px solid {T["border"]} !important;
        border-radius: {T["radius"]} !important;
        padding: 14px 16px !important;
        margin-bottom: 8px !important;
        box-shadow: none !important;
    }}

    /* === Charts === */
    [data-testid="stVegaLiteChart"],
    [data-testid="stArrowVegaLiteChart"],
    .vega-embed, .vega-embed .chart-wrapper {{
        background: {T["surface"]} !important;
        border-radius: {T["radius"]} !important;
        border: 1px solid {T["border"]} !important;
        padding: 8px !important;
    }}
    /* Force Vega-Lite text color for dark mode */
    .vega-embed .vega-bind,
    .vega-embed .vega-bind label,
    .vega-embed text {{
        fill: {T["text"]} !important;
        color: {T["text"]} !important;
    }}
    .vega-embed .role-axis-title text,
    .vega-embed .role-axis-label text {{
        fill: {T["text_secondary"]} !important;
    }}
    .vega-embed line.role-axis-grid {{
        stroke: {T["border"]} !important;
    }}

    /* === Alerts === */
    .stAlert, [data-testid="stAlert"] {{
        border-radius: {T["radius_sm"]} !important;
        font-size: 0.875rem !important;
    }}

    /* === Progress === */
    .stProgress > div > div > div > div {{
        border-radius: 6px !important;
        background: linear-gradient(90deg, {T["primary"]} 0%, {T["secondary"]} 100%) !important;
    }}

    /* === Dividers === */
    hr {{
        border: none !important;
        height: 1px !important;
        background: {T["divider"]} !important;
        margin: 20px 0 !important;
    }}

    /* === Scrollbar === */
    ::-webkit-scrollbar {{ width: 6px; }}
    ::-webkit-scrollbar-track {{ background: {T["scrollbar_track"]}; }}
    ::-webkit-scrollbar-thumb {{ background: {T["scrollbar_thumb"]}; border-radius: 6px; }}

    /* === Helper classes === */
    .match-card {{
        background: {T["surface"]} !important;
        border: 1px solid {T["border"]} !important;
        border-radius: {T["radius"]} !important;
        padding: 18px !important;
        margin: 8px 0 !important;
        box-shadow: {T["shadow_sm"]} !important;
        transition: box-shadow 0.2s ease !important;
    }}
    .match-card:hover {{
        box-shadow: {T["shadow_md"]} !important;
    }}
    .score-high {{ color: {T["success"]}; font-weight: 700; }}
    .score-medium {{ color: {T["warning"]}; font-weight: 700; }}
    .score-low {{ color: {T["danger"]}; font-weight: 700; }}
    .patient-card {{
        background: {T["surface"]} !important;
        border: 1px solid {T["border"]} !important;
        border-radius: {T["radius"]} !important;
        padding: 16px !important;
        margin: 6px !important;
        box-shadow: {T["shadow_sm"]} !important;
        transition: box-shadow 0.2s ease !important;
    }}
    .patient-card:hover {{
        box-shadow: {T["shadow_md"]} !important;
    }}

    /* Stat card */
    .stat-card {{
        background: {T["surface"]};
        border: 1px solid {T["border"]};
        border-radius: {T["radius"]};
        padding: 20px 22px;
        box-shadow: {T["shadow_sm"]};
        display: flex;
        align-items: flex-start;
        gap: 16px;
        transition: box-shadow 0.2s ease;
    }}
    .stat-card:hover {{
        box-shadow: {T["shadow_md"]};
    }}
    .stat-icon {{
        width: 48px;
        height: 48px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.4rem;
        flex-shrink: 0;
    }}
    .stat-content {{ flex: 1; }}
    .stat-label {{
        font-size: 0.78rem;
        font-weight: 500;
        color: {T["text_secondary"]};
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
    }}
    .stat-value {{
        font-size: 1.75rem;
        font-weight: 700;
        color: {T["text"]};
        line-height: 1.2;
    }}
    .stat-delta {{
        font-size: 0.78rem;
        font-weight: 600;
        margin-top: 4px;
    }}
    .stat-delta.up {{ color: {T["success"]}; }}
    .stat-delta.down {{ color: {T["danger"]}; }}

    /* Badge / pill */
    .badge {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 600;
        line-height: 1.5;
    }}
    .badge-success {{ background: {T["success_light"]}; color: {T["success"]}; }}
    .badge-warning {{ background: {T["warning_light"]}; color: {T["warning"]}; }}
    .badge-danger  {{ background: {T["danger_light"]};  color: {T["danger"]};  }}
    .badge-primary {{ background: {T["primary_light"]}; color: {T["primary"]}; }}

    /* Section title */
    .section-title {{
        font-size: 1.1rem;
        font-weight: 700;
        color: {T["text"]};
        margin-bottom: 12px;
    }}

    /* Legend badge (graph) */
    .legend-badge {{
        display: inline-block;
        border-radius: 50px;
        padding: 4px 12px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #fff;
    }}

    /* Page header */
    .page-header {{
        background: {T["surface"]};
        border: 1px solid {T["border"]};
        border-radius: {T["radius_lg"]};
        padding: 24px 28px;
        margin-bottom: 20px;
        box-shadow: {T["shadow_sm"]};
    }}
    .page-header h1 {{
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        margin: 0 0 4px 0 !important;
        color: {T["text"]} !important;
    }}
    .page-header p {{
        font-size: 0.9rem !important;
        color: {T["text_secondary"]} !important;
        margin: 0 !important;
    }}
</style>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Helpers                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def get_gremlin_property(vertex: dict, prop_name: str, default: str = "") -> str:
    """Extract a property value from a Cosmos DB Gremlin vertex."""
    if prop_name not in vertex:
        return default
    prop = vertex[prop_name]
    if isinstance(prop, list) and len(prop) > 0:
        first_item = prop[0]
        if isinstance(first_item, dict):
            if "_value" in first_item:
                return str(first_item["_value"])
            if "value" in first_item:
                return str(first_item["value"])
        return str(first_item)
    if isinstance(prop, dict):
        if "_value" in prop:
            return str(prop["_value"])
        if "value" in prop:
            return str(prop["value"])
        return str(prop)
    if prop is None:
        return default
    return str(prop)


def _stat_card(icon: str, label: str, value, bg_color: str, delta: str = "", delta_dir: str = "up"):
    """Render a Preclinic-style stat card using Streamlit columns."""
    # Use native Streamlit metric instead of raw HTML to avoid sanitization issues
    st.metric(label=label, value=value, delta=delta if delta else None)


def _page_header(title: str, subtitle: str):
    """Render a page header banner."""
    st.markdown(f"""
    <div class="page-header">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def _badge(text: str, variant: str = "primary"):
    """Return HTML for a pill badge."""
    return f'<span class="badge badge-{variant}">{text}</span>'


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Cosmos DB Clients & Data Fetching                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@st.cache_resource
def get_cosmos_clients():
    """Initialize Cosmos DB clients for both Gremlin and NoSQL APIs."""
    try:
        credential = DefaultAzureCredential()
        account_name = os.environ.get("COSMOS_ACCOUNT_NAME")
        if not account_name:
            st.error("COSMOS_ACCOUNT_NAME environment variable is required")
            return None, None, None
        nosql_endpoint = f"https://{account_name}.documents.azure.com:443/"
        nosql_client = CosmosClient(nosql_endpoint, credential=credential)
        gremlin_endpoint = f"wss://{account_name}.gremlin.cosmos.azure.com:443/"
        gremlin_client = client.Client(
            gremlin_endpoint, 'g',
            username="/dbs/patient-matching-db/colls/patients",
            password="",
            message_serializer=serializer.GraphSONSerializersV2d0()
        )
        return nosql_client, gremlin_client, account_name
    except Exception as e:
        st.error(f"Failed to connect to Cosmos DB: {e}")
        return None, None, None


@st.cache_data(ttl=60)
def fetch_match_results(_nosql_client, account_name):
    try:
        database = _nosql_client.get_database_client("patient-matching-db")
        container = database.get_container_client("match_results")
        query = "SELECT * FROM c ORDER BY c.score DESC"
        return list(container.query_items(query=query, enable_cross_partition_query=True))
    except Exception as e:
        st.error(f"Error fetching match results: {e}")
        return []


@st.cache_data(ttl=60)
def fetch_patients(_nosql_client, account_name):
    try:
        database = _nosql_client.get_database_client("patient-matching-db")
        container = database.get_container_client("patients")
        query = "SELECT * FROM c WHERE c.label = 'Patient'"
        return list(container.query_items(query=query, enable_cross_partition_query=True))
    except Exception as e:
        st.error(f"Error fetching patients: {e}")
        return []


@st.cache_data(ttl=60)
def fetch_patient_clinical_data(_nosql_client, account_name, patient_id):
    try:
        database = _nosql_client.get_database_client("patient-matching-db")
        container = database.get_container_client("patients")
        patient_query = f"SELECT * FROM c WHERE c.id = '{patient_id}' AND c.label = 'Patient'"
        patients = list(container.query_items(query=patient_query, enable_cross_partition_query=True))
        patient = patients[0] if patients else None
        clinical_data = {
            "patient": patient, "encounters": [], "observations": [],
            "conditions": [], "procedures": [], "immunizations": [],
            "medications": [], "identifiers": [], "potential_matches": []
        }
        if not patient:
            return clinical_data
        edges_query = f"""
            SELECT c._sink, c.label as edge_label
            FROM c
            WHERE c._isEdge = true AND c._vertexId = '{patient_id}'
        """
        edges = list(container.query_items(query=edges_query, enable_cross_partition_query=True))
        buckets = {
            "HAS_ENCOUNTER": [], "HAS_OBSERVATION": [], "HAS_CONDITION": [],
            "HAS_PROCEDURE": [], "HAS_IMMUNIZATION": [], "HAS_MEDICATION": [],
            "HAS_IDENTIFIER": []
        }
        for edge in edges:
            lbl = edge.get("edge_label", "")
            sink = edge.get("_sink", "")
            if sink and lbl in buckets:
                buckets[lbl].append(sink)
        label_map = {
            "HAS_ENCOUNTER": ("encounters", "Encounter"),
            "HAS_OBSERVATION": ("observations", "Observation"),
            "HAS_CONDITION": ("conditions", "Condition"),
            "HAS_PROCEDURE": ("procedures", "Procedure"),
            "HAS_IMMUNIZATION": ("immunizations", "Immunization"),
            "HAS_MEDICATION": ("medications", "MedicationRequest"),
            "HAS_IDENTIFIER": ("identifiers", "Identifier"),
        }
        for edge_lbl, (key, vlabel) in label_map.items():
            ids = buckets[edge_lbl][:50]
            if ids:
                ids_str = ", ".join([f"'{v}'" for v in ids])
                q = f"SELECT * FROM c WHERE c.id IN ({ids_str}) AND c.label = '{vlabel}'"
                clinical_data[key] = list(container.query_items(query=q, enable_cross_partition_query=True))
        return clinical_data
    except Exception as e:
        st.error(f"Error fetching patient clinical data: {e}")
        return {
            "patient": None, "encounters": [], "observations": [],
            "conditions": [], "procedures": [], "immunizations": [],
            "medications": [], "identifiers": [], "potential_matches": []
        }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Score / Confidence helpers                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def get_confidence_color(confidence):
    if confidence == "auto_merge":
        return "🟢", T["success"]
    elif confidence == "human_review":
        return "🟡", T["warning"]
    else:
        return "🔴", T["danger"]


def get_confidence_badge(confidence):
    if confidence == "auto_merge":
        return _badge("Auto Merge", "success")
    elif confidence == "human_review":
        return _badge("Human Review", "warning")
    else:
        return _badge("No Match", "danger")


def get_score_class(score):
    if score >= 0.85:
        return "score-high"
    elif score >= 0.50:
        return "score-medium"
    else:
        return "score-low"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Main App                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    with st.sidebar:
        # Brand
        st.markdown(f"""
        <div style="text-align: center; padding: 16px 0 8px 0;">
            <div style="display: inline-flex; align-items: center; justify-content: center;
                        width: 52px; height: 52px; border-radius: 12px;
                        background: linear-gradient(135deg, {T['primary']} 0%, {T['secondary']} 100%);">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"
                          fill="rgba(255,255,255,0.2)"/>
                    <path d="M11 7h2v4h4v2h-4v4h-2v-4H7v-2h4V7z" fill="#FFFFFF"/>
                </svg>
            </div>
            <div style="font-weight: 700; font-size: 1.15rem; color: {T['text']}; margin-top: 8px;">
                Contoso Health
            </div>
            <div style="font-weight: 400; font-size: 0.72rem; color: {T['text_muted']}; margin-top: 2px;
                        letter-spacing: 0.03em;">
                Patient Matching Service
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        # Navigation
        page = st.radio(
            "Navigation",
            ["📊 Dashboard", "🔍 Match Results", "👥 Patients",
             "🕸️ Patient Graph", "📋 Review Queue", "🤖 Patient Matching Agent",
             "⚙️ Settings"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Connection status
        nosql_client, gremlin_client, account_name = get_cosmos_clients()
        if nosql_client:
            st.markdown(
                f'<div style="padding: 8px 12px; background: {T["success_light"]}; '
                f'color: {T["success"]}; border-radius: 6px; font-size: 0.82rem; '
                f'font-weight: 600; text-align: center;">✅ Connected to Cosmos DB</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div style="padding: 8px 12px; background: {T["danger_light"]}; '
                f'color: {T["danger"]}; border-radius: 6px; font-size: 0.82rem; '
                f'font-weight: 600; text-align: center;">❌ Not connected</div>',
                unsafe_allow_html=True
            )

        st.markdown("")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # Page Router
    if page == "📊 Dashboard":
        render_dashboard(nosql_client, account_name)
    elif page == "🔍 Match Results":
        render_match_results(nosql_client, account_name)
    elif page == "👥 Patients":
        render_patients(nosql_client, account_name)
    elif page == "🕸️ Patient Graph":
        render_patient_graph(nosql_client, gremlin_client, account_name)
    elif page == "📋 Review Queue":
        render_review_queue(nosql_client, account_name)
    elif page == "🤖 Patient Matching Agent":
        render_agent_chat()
    elif page == "⚙️ Settings":
        render_settings()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Dashboard Page                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def render_dashboard(nosql_client, account_name):
    _page_header("Admin Dashboard", "Overview of patient matching operations and statistics")

    if not nosql_client:
        st.warning("Please configure Cosmos DB connection to view data.")
        return

    match_results = fetch_match_results(nosql_client, account_name)
    patients = fetch_patients(nosql_client, account_name)
    auto_merge = len([m for m in match_results if m.get('confidence') == 'auto_merge'])
    human_review = len([m for m in match_results if m.get('confidence') == 'human_review'])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _stat_card("👥", "Total Patients", len(patients), T["primary"])
    with c2:
        _stat_card("🔗", "Match Results", len(match_results), T["secondary"])
    with c3:
        _stat_card("✅", "Auto Merge", auto_merge, T["success"])
    with c4:
        _stat_card("⏳", "Pending Review", human_review, T["warning"])

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Match Confidence Distribution</div>', unsafe_allow_html=True)
        if match_results:
            conf_counts = {}
            for m in match_results:
                c = m.get('confidence', 'unknown')
                conf_counts[c] = conf_counts.get(c, 0) + 1
            df = pd.DataFrame([{"Confidence": k, "Count": v} for k, v in conf_counts.items()])
            st.bar_chart(df.set_index("Confidence"))
        else:
            st.info("No match results available")
    with c2:
        st.markdown('<div class="section-title">Score Distribution</div>', unsafe_allow_html=True)
        if match_results:
            scores = [m.get('score', 0) for m in match_results]
            st.line_chart(pd.DataFrame({"Score": scores}))
        else:
            st.info("No match results available")

    st.markdown("---")
    st.markdown('<div class="section-title">Recent Matches</div>', unsafe_allow_html=True)
    if match_results:
        recent = match_results[:10]
        rows = []
        for m in recent:
            p1 = m.get('patient1_name') or (m.get('patient1_id') or 'N/A')[:12]
            p2 = m.get('patient2_name') or (m.get('patient2_id') or 'N/A')[:12]
            rows.append({
                "Patient 1": p1, "Patient 2": p2,
                "Score": f"{m.get('score', 0):.3f}",
                "Confidence": m.get('confidence', 'N/A'),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown("")
        for idx, match in enumerate(recent[:5]):
            with st.expander(
                f"{match.get('patient1_name', 'Patient 1')} ↔ "
                f"{match.get('patient2_name', 'Patient 2')} · "
                f"Score: {match.get('score', 0):.2f}"
            ):
                render_match_details(match)
    else:
        st.info("No match results available. Run the matching service to generate results.")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Match Results Page                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def render_match_results(nosql_client, account_name):
    _page_header("Match Results", "Browse and filter all patient matching outcomes")

    if not nosql_client:
        st.warning("Please configure Cosmos DB connection to view data.")
        return

    match_results = fetch_match_results(nosql_client, account_name)
    c1, c2, c3 = st.columns(3)
    with c1:
        min_score = st.slider("Minimum Score", 0.0, 1.0, 0.0, 0.05)
    with c2:
        confidence_filter = st.multiselect(
            "Confidence Level",
            ["auto_merge", "human_review", "no_match"],
            default=["auto_merge", "human_review", "no_match"]
        )
    with c3:
        sort_by = st.selectbox("Sort By", ["Score (High→Low)", "Score (Low→High)", "Date"])

    filtered = [
        m for m in match_results
        if m.get('score', 0) >= min_score and m.get('confidence') in confidence_filter
    ]
    if sort_by == "Score (Low→High)":
        filtered = sorted(filtered, key=lambda x: x.get('score', 0))
    elif sort_by == "Date":
        filtered = sorted(filtered, key=lambda x: x.get('created_at', ''), reverse=True)

    st.caption(f"Showing {len(filtered)} of {len(match_results)} results")
    st.markdown("---")

    if filtered:
        rows = []
        for m in filtered:
            p1 = m.get('patient1_name') or (m.get('patient1_id') or 'N/A')[:12]
            p2 = m.get('patient2_name') or (m.get('patient2_id') or 'N/A')[:12]
            rows.append({
                "Patient 1": p1, "Patient 2": p2,
                "Score": f"{m.get('score', 0):.3f}",
                "Confidence": m.get('confidence', 'N/A'),
                "Details": str(m.get('details', {}))[:50] + "..."
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown("")
        st.markdown('<div class="section-title">Detailed Match View</div>', unsafe_allow_html=True)
        selected_idx = st.selectbox(
            "Select a match to view details", range(len(filtered)),
            format_func=lambda i: (
                f"{filtered[i].get('patient1_name', 'Patient 1')} ↔ "
                f"{filtered[i].get('patient2_name', 'Patient 2')}"
            )
        )
        if selected_idx is not None:
            render_match_details(filtered[selected_idx])
    else:
        st.info("No matches found with the current filters.")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Match Details (shared component)                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def render_match_details(match: dict):
    st.markdown("---")
    p1_name = match.get('patient1_name') or 'N/A'
    p1_id = match.get('patient1_id') or 'N/A'
    p1_source = match.get('patient1_source') or 'N/A'
    p2_name = match.get('patient2_name') or 'N/A'
    p2_id = match.get('patient2_id') or 'N/A'
    p2_source = match.get('patient2_source') or 'N/A'

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        st.markdown("##### 👤 Patient 1")
        st.write(f"**Name:** {p1_name}")
        st.write(f"**ID:** `{p1_id[:20]}...`")
        st.write(f"**Source:** {p1_source}")
    with c2:
        st.markdown("##### 👤 Patient 2")
        st.write(f"**Name:** {p2_name}")
        st.write(f"**ID:** `{p2_id[:20]}...`")
        st.write(f"**Source:** {p2_source}")
    with c3:
        score = match.get('score', 0)
        _, color = get_confidence_color(match.get('confidence'))
        st.markdown("##### Score")
        st.markdown(
            f'<div style="text-align:center; font-size:2.2rem; font-weight:800; '
            f'color:{color};">{score:.2f}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div style="text-align:center;">{get_confidence_badge(match.get("confidence"))}</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    details = match.get('details', {})
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except Exception:
            details = {}
    is_flat = 'deterministic_score' in details or 'name_score' in details

    tabs = st.tabs(["Summary", "Deterministic", "Probabilistic", "AI / Embeddings", "LLM Analysis", "Raw Data"])
    with tabs[0]:
        render_score_summary(match, details)
    with tabs[1]:
        det = details.get('deterministic_details', {}) if is_flat else details.get('deterministic', {})
        render_deterministic_details(det)
    with tabs[2]:
        render_probabilistic_details(details, is_flat)
    with tabs[3]:
        emb = details.get('embedding_details', {}) if is_flat else details.get('embedding', {})
        emb_s = details.get('embedding_score', 0) if is_flat else 0
        render_embedding_details(emb, emb_s)
    with tabs[4]:
        llm = details.get('llm_details', {}) if is_flat else details.get('llm', {})
        llm_s = details.get('llm_score', 0) if is_flat else 0
        render_llm_details(llm, match, llm_s)
    with tabs[5]:
        st.json(match)


def render_score_summary(match: dict, details: dict):
    is_flat = 'deterministic_score' in details or 'name_score' in details
    if is_flat:
        det_score = details.get('deterministic_score', 0)
        name_score = details.get('name_score', 0)
        addr_score = details.get('address_score', 0)
        emb_score = details.get('embedding_score', 0)
        llm_score = details.get('llm_score', 0)
        dob_score = 1.0 if details.get('gender_match') else 0.0
    else:
        det_d = details.get('deterministic', {})
        name_d = details.get('name', {})
        addr_d = details.get('address', {})
        dob_d = details.get('dob', {})
        emb_d = details.get('embedding', {})
        llm_d = details.get('llm', {})
        det_score = 0.9 if det_d.get('matched_identifiers') else (0.35 if det_d.get('dob_match') else 0.0)
        name_score = name_d.get('final_score', 0) if name_d else 0
        addr_score = addr_d.get('final_score', 0) if addr_d else 0
        emb_score = emb_d.get('cosine_similarity', 0) if emb_d else 0
        llm_score = llm_d.get('score', 0) if llm_d else 0
        dob_score = 1.0 if dob_d.get('exact_match') else (0.8 if dob_d.get('transposition_detected') else (0.5 if dob_d.get('year_match') else 0.0))

    scores = [
        ("🎯 Deterministic", det_score, "SSN, MRN, Enterprise ID"),
        ("📝 Name Similarity", name_score, "Jaro-Winkler, Soundex, Metaphone"),
        ("🏠 Address", addr_score, "Token matching"),
        ("📅 DOB", dob_score, "Exact / transposition"),
        ("🧠 Embedding", emb_score, "Cosine similarity"),
        ("💬 LLM", llm_score, "GPT-4o analysis"),
    ]
    for label, score, desc in scores:
        c1, c2, c3 = st.columns([2, 4, 1])
        with c1:
            st.markdown(f"**{label}**")
            st.caption(desc)
        with c2:
            st.progress(min(1.0, max(0.0, score)))
        with c3:
            if score > 0:
                clr = T["success"] if score >= 0.7 else T["warning"] if score >= 0.4 else T["danger"]
                st.markdown(f'<div style="text-align:right; font-size:1.15rem; font-weight:700; color:{clr};">{score:.2f}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="text-align:right; color:{T["text_muted"]};">N/A</div>', unsafe_allow_html=True)
    shared = match.get('shared_identifiers', [])
    if shared:
        st.markdown("**🔗 Shared Identifiers**")
        for s in shared:
            st.success(f"✓ {s}")


def render_deterministic_details(det_details: dict):
    if not det_details:
        st.info("No deterministic matching data available.")
        return
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Matched Identifiers**")
        matched_ids = det_details.get('matched_identifiers', [])
        if matched_ids:
            for info in matched_ids:
                if len(info) >= 2:
                    id_type, id_val = info[0], info[1]
                    system = info[2] if len(info) > 2 else ""
                    extra = f" (System: {system})" if system else ""
                    st.success(f"✅ **{id_type}:** {id_val}{extra}")
        else:
            st.warning("No matching identifiers found")
        st.markdown("**Matched Contacts**")
        contacts = det_details.get('matched_contacts', [])
        if contacts:
            for ct in contacts:
                if len(ct) >= 2:
                    st.success(f"✅ **{ct[0].title()}:** {ct[1]}")
        else:
            st.info("None")
    with c2:
        st.markdown("**Boolean Matches**")
        if det_details.get('dob_match'):
            st.success("✅ Date of Birth: EXACT MATCH")
        else:
            st.warning("❌ Date of Birth: No match")
        if det_details.get('gender_match'):
            st.success("✅ Gender: MATCH")
        else:
            st.info("➖ Gender: N/A")


def render_probabilistic_details(details: dict, is_flat: bool = False):
    st.markdown("**📝 Name Similarity**")
    name_details = details.get('name_details', {}) if is_flat else details.get('name', {})
    name_score = details.get('name_score', 0) if is_flat else 0
    if name_details and name_details != {"reason": "missing_name"}:
        if is_flat and name_score > 0:
            st.metric("Overall Name Score", f"{name_score:.3f}")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Full Name (JW)", f"{name_details.get('full_name_jaro_winkler', 0):.3f}")
        with c2:
            st.metric("First Name (JW)", f"{name_details.get('first_name_jaro_winkler', 0):.3f}")
        with c3:
            st.metric("Last Name (JW)", f"{name_details.get('last_name_jaro_winkler', 0):.3f}")
        c1, c2, c3 = st.columns(3)
        with c1:
            v = name_details.get('first_name_soundex_match')
            if v is True:
                st.success("✅ First Name Soundex")
            elif v is False:
                st.error("❌ No Match")
            else:
                st.info("➖ N/A")
        with c2:
            v = name_details.get('first_name_metaphone_match')
            if v is True:
                st.success("✅ First Name Metaphone")
            elif v is False:
                st.error("❌ No Match")
            else:
                st.info("➖ N/A")
        with c3:
            v = name_details.get('last_name_soundex_match')
            if v is True:
                st.success("✅ Last Name Soundex")
            elif v is False:
                st.error("❌ No Match")
            else:
                st.info("➖ N/A")
        st.metric("Levenshtein", f"{name_details.get('levenshtein_normalized', 0):.3f}")
        st.metric("Final Name Score", f"{name_details.get('final_score', 0):.3f}")
    else:
        st.warning("Name similarity data not available")

    st.markdown("---")
    st.markdown("**🏠 Address Similarity**")
    addr_details = details.get('address_details', {}) if is_flat else details.get('address', {})
    addr_score = details.get('address_score', 0) if is_flat else 0
    if addr_details and addr_details != {"reason": "missing_address"}:
        if is_flat and addr_score > 0:
            st.metric("Overall Address Score", f"{addr_score:.3f}")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Addr 1:**")
            st.code(addr_details.get('normalized_addr1', 'N/A'))
        with c2:
            st.write("**Addr 2:**")
            st.code(addr_details.get('normalized_addr2', 'N/A'))
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Jaro-Winkler", f"{addr_details.get('jaro_winkler', 0):.3f}")
        with c2:
            st.metric("Token Jaccard", f"{addr_details.get('token_jaccard', 0):.3f}")
        with c3:
            st.metric("Final Score", f"{addr_details.get('final_score', 0):.3f}")
        c1, c2, c3 = st.columns(3)
        with c1:
            if addr_details.get('postal_code_match'):
                st.success("✅ Postal Code")
            else:
                st.info("➖ Postal Code")
        with c2:
            if addr_details.get('city_match'):
                st.success("✅ City")
            else:
                st.info("➖ City")
        with c3:
            if addr_details.get('state_match'):
                st.success("✅ State")
            else:
                st.info("➖ State")
        shared = addr_details.get('shared_tokens', [])
        if shared:
            st.write(f"**Shared tokens:** {', '.join(shared)}")
    else:
        st.warning("Address similarity data not available")

    st.markdown("---")
    st.markdown("**📅 Date of Birth**")
    dob_details = details.get('dob', {})
    if dob_details and dob_details != {"reason": "missing_dob"}:
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**DOB 1:** {dob_details.get('dob1', 'N/A')}")
        with c2:
            st.write(f"**DOB 2:** {dob_details.get('dob2', 'N/A')}")
        if dob_details.get('exact_match'):
            st.success("✅ EXACT MATCH")
        elif dob_details.get('transposition_detected'):
            st.warning("⚠️ TRANSPOSITION DETECTED")
        elif dob_details.get('year_match'):
            st.info("📅 Year Match")
        else:
            st.error("❌ No DOB match")
    else:
        st.warning("DOB data not available")


def render_embedding_details(emb_details: dict, emb_score: float = 0):
    if not emb_details and emb_score == 0:
        st.info("Embedding matching was not used. Enable with `--use-embeddings`.")
        return
    if emb_details.get('reason') == 'embeddings_not_available':
        st.warning("Embeddings not available. Check OpenAI configuration.")
        return
    c1, c2 = st.columns(2)
    with c1:
        cosine = emb_score if emb_score > 0 else emb_details.get('cosine_similarity', 0)
        st.metric("Embedding Similarity", f"{cosine:.4f}")
        st.progress(min(1.0, max(0.0, cosine)))
        if cosine >= 0.95:
            st.success("Very High — Likely same patient")
        elif cosine >= 0.85:
            st.success("High — Strong match candidate")
        elif cosine >= 0.70:
            st.warning("Moderate — Review recommended")
        elif cosine > 0:
            st.error("Low — Likely different patients")
    with c2:
        st.markdown("""
**How it works:**
1. Demographics → text representation
2. Embedded via Azure OpenAI
3. Cosine similarity measures closeness
4. Score ≥ 0.85 → strong semantic match
""")


def render_llm_details(llm_details: dict, match: dict, llm_score_param: float = 0):
    llm_score = llm_score_param
    llm_rec = llm_details.get('recommendation', '')
    if llm_score == 0:
        md = match.get('details', {})
        if isinstance(md, str):
            try:
                md = json.loads(md)
            except Exception:
                md = {}
        llm_score = md.get('llm_score', 0)
        if not llm_rec:
            llm_rec = md.get('llm_recommendation', '')
    llm_blended = False
    if isinstance(match.get('details'), dict):
        llm_blended = match['details'].get('llm_blended', False)

    if llm_score == 0 and not llm_details:
        st.info("LLM analysis not used. Enable with `--use-llm`.")
        return
    c1, c2 = st.columns(2)
    with c1:
        st.metric("LLM Confidence", f"{llm_score:.2f}")
        st.progress(min(1.0, max(0.0, llm_score)))
        if llm_rec:
            if llm_rec.lower() in ('match', 'merge', 'same_patient'):
                st.success(f"🎯 **{llm_rec.upper()}**")
            elif llm_rec.lower() in ('review', 'uncertain', 'manual_review'):
                st.warning(f"⚠️ **{llm_rec.upper()}**")
            else:
                st.error(f"❌ **{llm_rec.upper()}**")
    with c2:
        if llm_blended:
            st.success("✅ LLM score blended (80/20 traditional/LLM)")
        st.markdown("**Process:** Demographics → GPT-4o → confidence + recommendation")
    if llm_details.get('reasoning'):
        st.markdown("**AI Reasoning**")
        st.info(llm_details['reasoning'])
    if llm_details.get('analysis'):
        st.markdown("**Analysis**")
        st.write(llm_details['analysis'])


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Patients Page                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def render_patients(nosql_client, account_name):
    _page_header("Patients", "Browse and search all patient records")
    if not nosql_client:
        st.warning("Please configure Cosmos DB connection to view data.")
        return
    patients = fetch_patients(nosql_client, account_name)
    search_term = st.text_input("🔎 Search patients", placeholder="Name, ID, or source system...")
    if search_term:
        patients = [p for p in patients if search_term.lower() in str(p).lower()]
    st.caption(f"{len(patients)} patients found")
    st.markdown("---")
    if patients:
        for i, patient in enumerate(patients[:20]):
            fn = get_gremlin_property(patient, 'firstName', '')
            ln = get_gremlin_property(patient, 'lastName', '')
            bd = get_gremlin_property(patient, 'birthDate', 'N/A')
            src = get_gremlin_property(patient, 'sourceSystem', 'N/A')
            with st.expander(
                f"👤 {fn} {ln}  ·  DOB: {bd}  ·  Source: {src}"
            ):
                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"**ID:** `{patient.get('id', 'N/A')}`")
                    st.write(f"**Gender:** {get_gremlin_property(patient, 'gender', 'N/A')}")
                    st.write(f"**Source ID:** {get_gremlin_property(patient, 'sourceId', 'N/A')}")
                with c2:
                    created = get_gremlin_property(patient, 'createdAt', 'N/A')
                    st.write(f"**Created:** {created[:19] if created and created != 'N/A' else 'N/A'}")
                    ssn = get_gremlin_property(patient, 'ssn', '')
                    if ssn:
                        st.write(f"**SSN:** ***-**-{ssn[-4:]}")
        if len(patients) > 20:
            st.info(f"Showing first 20 of {len(patients)}. Use search to narrow.")
    else:
        st.info("No patients found.")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Review Queue Page                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def render_review_queue(nosql_client, account_name):
    _page_header("Review Queue", "Matches requiring human verification before merging")
    if not nosql_client:
        st.warning("Please configure Cosmos DB connection to view data.")
        return
    match_results = fetch_match_results(nosql_client, account_name)
    pending = [m for m in match_results if m.get('confidence') == 'human_review']
    _stat_card("📋", "Pending Reviews", len(pending), T["warning"])
    st.markdown("---")
    if pending:
        for i, match in enumerate(pending):
            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                st.markdown(f"**Patient 1:** {match.get('patient1_name', 'N/A')}")
                st.caption(f"ID: {match.get('patient1_id', 'N/A')[:12]}...")
            with c2:
                st.markdown(f"**Patient 2:** {match.get('patient2_name', 'N/A')}")
                st.caption(f"ID: {match.get('patient2_id', 'N/A')[:12]}...")
            with c3:
                score = match.get('score', 0)
                color = T["success"] if score >= 0.85 else T["warning"] if score >= 0.5 else T["danger"]
                st.markdown(
                    f'<div style="text-align:center; font-size:1.6rem; font-weight:700; color:{color};">{score:.2f}</div>',
                    unsafe_allow_html=True
                )
            ac1, ac2, ac3 = st.columns(3)
            with ac1:
                if st.button("✅ Approve", key=f"approve_{i}"):
                    st.success("Approved (demo)")
            with ac2:
                if st.button("❌ Reject", key=f"reject_{i}"):
                    st.warning("Rejected (demo)")
            with ac3:
                if st.button("🔍 Details", key=f"details_{i}"):
                    st.session_state[f"show_details_{i}"] = True
            if st.session_state.get(f"show_details_{i}", False):
                render_match_details(match)
                if st.button("Hide", key=f"hide_{i}"):
                    st.session_state[f"show_details_{i}"] = False
                    st.rerun()
            st.markdown("---")
    else:
        st.success("🎉 No pending reviews — all matches processed.")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Settings Page                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def render_settings():
    _page_header("Settings", "Configure match weights, thresholds, and integrations")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Deterministic Matching**")
        st.slider("Enterprise ID Weight", 0.0, 1.0, 1.0, 0.1)
        st.slider("MRN Weight", 0.0, 1.0, 0.8, 0.1)
        st.slider("SSN Weight", 0.0, 1.0, 0.9, 0.1)
        st.slider("DOB Weight", 0.0, 1.0, 0.35, 0.05)
    with c2:
        st.markdown("**Probabilistic Matching**")
        st.slider("Name Similarity Weight", 0.0, 1.0, 0.35, 0.05)
        st.slider("Address Similarity Weight", 0.0, 1.0, 0.15, 0.05)
        st.slider("Embedding Similarity Weight", 0.0, 1.0, 0.1, 0.05)
    st.markdown("---")
    st.markdown("**Thresholds**")
    c1, c2 = st.columns(2)
    with c1:
        st.slider("Auto-Merge Threshold", 0.5, 1.0, 0.85, 0.05)
    with c2:
        st.slider("Human Review Threshold", 0.3, 0.9, 0.65, 0.05)
    st.markdown("---")
    st.markdown("**Azure OpenAI**")
    st.text_input("Endpoint", value=os.environ.get("AZURE_OPENAI_ENDPOINT", ""), type="password")
    st.text_input("Embedding Deployment", value=os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"))
    st.text_input("Chat Deployment", value=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o"))
    if st.button("💾 Save Settings"):
        st.success("Settings saved (demo — not persisted)")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Patient Graph Page                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def render_patient_graph(nosql_client, gremlin_client, account_name):
    _page_header("Patient Clinical Data Graph", "Visualize relationships between a patient and their clinical records")
    if not AGRAPH_AVAILABLE:
        st.error("Install `streamlit-agraph` to enable graph visualization.")
        st.code("pip install streamlit-agraph")
        return
    if not nosql_client:
        st.warning("Please configure Cosmos DB connection to view data.")
        return
    patients = fetch_patients(nosql_client, account_name)
    if not patients:
        st.info("No patients found.")
        return

    patient_options = {}
    for p in patients[:100]:
        fn = get_gremlin_property(p, 'firstName', '')
        ln = get_gremlin_property(p, 'lastName', '')
        bd = get_gremlin_property(p, 'birthDate', 'N/A')
        pid = p.get('id', '')
        patient_options[f"{fn} {ln} (DOB: {bd}) — {pid[:12]}..."] = pid

    selected_label = st.selectbox("🔎 Select patient", list(patient_options.keys()), index=0)
    selected_id = patient_options.get(selected_label)
    if not selected_id:
        return

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        show_enc = st.checkbox("🏥 Encounters", True)
        show_obs = st.checkbox("🔬 Observations", True)
    with c2:
        show_cond = st.checkbox("💊 Conditions", True)
        show_proc = st.checkbox("🔧 Procedures", True)
    with c3:
        show_imm = st.checkbox("💉 Immunizations", True)
        show_med = st.checkbox("💊 Medications", True)
    show_matched = st.checkbox("🔗 Show Matched Patients", True)

    clinical = fetch_patient_clinical_data(nosql_client, account_name, selected_id)
    if not clinical.get("patient"):
        st.warning("Could not load patient data.")
        return

    nodes, edges_list = [], []
    patient = clinical["patient"]
    pname = f"{get_gremlin_property(patient, 'firstName', '')} {get_gremlin_property(patient, 'lastName', '')}".strip() or "Unknown"
    bd = get_gremlin_property(patient, 'birthDate', 'N/A')
    gender = get_gremlin_property(patient, 'gender', 'N/A')

    _node_font = {"size": 14, "color": "#FFFFFF", "strokeWidth": 3, "strokeColor": "#000000"}
    _node_font_lg = {"size": 16, "color": "#FFFFFF", "strokeWidth": 4, "strokeColor": "#000000"}

    nodes.append(Node(
        id=selected_id, label=pname, size=40, color="#2E6FF3",
        shape="circularImage", image="https://img.icons8.com/color/96/user.png",
        font=_node_font_lg,
        title=f"Patient: {pname}\nDOB: {bd}\nGender: {gender}"
    ))

    node_colors = {
        "Encounter": "#3B82F6", "Observation": "#8B5CF6", "Condition": "#EF4444",
        "Procedure": "#F97316", "Immunization": "#06B6D4", "Medication": "#EC4899",
        "MatchedPatient": "#F59E0B",
    }

    # Emoji labels for node types (shown inside dot)
    node_symbols = {
        "Encounter": "🏥", "Observation": "🔬", "Condition": "❤️",
        "Procedure": "⚕️", "Immunization": "💉", "Medication": "💊",
    }

    def _add_nodes(items, label, prop_name, edge_label, color_key, date_prop="", limit=20):
        for item in items[:limit]:
            nid = item.get("id", "")
            code = get_gremlin_property(item, prop_name, label)
            dt = get_gremlin_property(item, date_prop, "")[:10] if date_prop else ""
            status = get_gremlin_property(item, "status", "")
            emoji = node_symbols.get(color_key, '')
            nodes.append(Node(
                id=nid, label=f"{emoji} {str(code)[:16]}", size=18,
                color=node_colors[color_key], shape="dot",
                font=_node_font,
                title=f"{emoji} {label}: {code}\nDate: {dt}\nStatus: {status}"
            ))
            edges_list.append(Edge(source=selected_id, target=nid, color=node_colors[color_key]))

    if show_enc:
        _add_nodes(clinical.get("encounters", []), "Encounter", "encounterType", "HAS_ENCOUNTER", "Encounter", "periodStart")
    if show_obs:
        _add_nodes(clinical.get("observations", []), "Observation", "codeDisplay", "HAS_OBSERVATION", "Observation", "effectiveDateTime")
    if show_cond:
        _add_nodes(clinical.get("conditions", []), "Condition", "codeDisplay", "HAS_CONDITION", "Condition", "onsetDateTime")
    if show_proc:
        _add_nodes(clinical.get("procedures", []), "Procedure", "codeDisplay", "HAS_PROCEDURE", "Procedure", "performedDateTime")
    if show_imm:
        _add_nodes(clinical.get("immunizations", []), "Immunization", "vaccineDisplay", "HAS_IMMUNIZATION", "Immunization", "occurrenceDateTime")
    if show_med:
        _add_nodes(clinical.get("medications", []), "Medication", "medicationDisplay", "HAS_MEDICATION", "Medication")

    if show_matched:
        mr = fetch_match_results(nosql_client, account_name)
        added = set()
        for m in mr:
            p1, p2 = m.get('patient1_id', ''), m.get('patient2_id', '')
            score = m.get('score', 0)
            conf = m.get('confidence', 'N/A')
            _, cc = get_confidence_color(conf)
            mid, mn = None, None
            if p1 == selected_id:
                mid, mn = p2, m.get('patient2_name', 'Unknown')
            elif p2 == selected_id:
                mid, mn = p1, m.get('patient1_name', 'Unknown')
            if mid and mid not in added:
                added.add(mid)
                nodes.append(Node(
                    id=mid, label=mn, size=35, color=node_colors["MatchedPatient"],
                    shape="circularImage", image="https://img.icons8.com/color/96/user.png",
                    font=_node_font_lg,
                    title=f"Matched: {mn}\nScore: {score:.2f}\nConfidence: {conf}"
                ))
                edges_list.append(Edge(source=selected_id, target=mid, label=f"{score:.2f}",
                                       color=cc, width=3,
                                       font={"size": 11, "color": "#FFFFFF", "strokeWidth": 2, "strokeColor": "#000000"}))

    config = Config(
        width=1200, height=900, directed=True,
        physics={"enabled": True, "barnesHut": {
            "gravitationalConstant": -30000, "centralGravity": 0.05,
            "springLength": 500, "springConstant": 0.005,
            "damping": 0.12, "avoidOverlap": 1.0},
            "minVelocity": 0.75,
            "stabilization": {"enabled": True, "iterations": 300}},
        hierarchical=False, nodeHighlightBehavior=True,
        highlightColor="#F7A7A6", collapsible=False,
        node={"labelProperty": "label", "renderLabel": True},
        link={"labelProperty": "label", "renderLabel": True}
    )

    st.markdown("---")
    st.markdown('<div class="section-title">📊 Clinical Data Summary</div>', unsafe_allow_html=True)
    cols = st.columns(6)
    labels = ["Encounters", "Observations", "Conditions", "Procedures", "Immunizations", "Medications"]
    keys = ["encounters", "observations", "conditions", "procedures", "immunizations", "medications"]
    for col, lbl, key in zip(cols, labels, keys):
        with col:
            st.metric(lbl, len(clinical.get(key, [])))

    st.markdown("---")
    if len(nodes) > 1:
        st.markdown('<div class="section-title">🕸️ Interactive Graph</div>', unsafe_allow_html=True)
        st.caption("Drag nodes · Hover for details · Scroll to zoom")
        agraph(nodes=nodes, edges=edges_list, config=config)
        st.markdown("---")
        legend_cols = st.columns(7)
        legend_items = [
            ("👤 Patient", "#2E6FF3"), ("🔗 Matched", "#F59E0B"),
            ("🏥 Encounter", "#3B82F6"), ("🔬 Observation", "#8B5CF6"),
            ("❤️ Condition", "#EF4444"), ("⚕️ Procedure", "#F97316"),
            ("💉 Immunization", "#06B6D4"),
        ]
        for col, (lbl, clr) in zip(legend_cols, legend_items):
            with col:
                st.markdown(f'<div class="legend-badge" style="background:{clr};">{lbl}</div>', unsafe_allow_html=True)
    else:
        st.info("No clinical data found for this patient.")

    with st.expander("📋 Raw Clinical Data"):
        tab_labels = ["Encounters", "Observations", "Conditions", "Procedures", "Immunizations", "Medications"]
        tabs = st.tabs(tab_labels)
        for tab, key in zip(tabs, keys):
            with tab:
                d = clinical.get(key, [])
                if d:
                    st.json(d[:10])
                else:
                    st.info(f"No {key}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Patient Matching Agent Page                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _run_agent_query(query: str, project_endpoint: str, deployment: str) -> str:
    async def _invoke():
        async with create_foundry_agent(
            project_endpoint=project_endpoint,
            deployment_name=deployment,
        ) as agent:
            result = await agent.run(query)
            return result.text
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_invoke())
    finally:
        loop.close()


def render_agent_chat():
    _page_header("🤖 Patient Matching Agent",
                 "Chat with the AI agent to search patients, find duplicates, compare records, and manage matches.")
    if not AGENT_AVAILABLE:
        st.error("Agent Framework not installed. `pip install agent-framework-azure-ai --pre`")
        return

    project_endpoint = os.environ.get("AZURE_AI_FOUNDRY_PROJECT_ENDPOINT", "")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    if not project_endpoint:
        st.warning("Set `AZURE_AI_FOUNDRY_PROJECT_ENDPOINT` to connect to Foundry Agent Service.")
        with st.expander("Configure Foundry Endpoint"):
            project_endpoint = st.text_input(
                "Foundry Project Endpoint",
                placeholder="https://<resource>.services.ai.azure.com/api/projects/<project>",
            )
            deployment = st.text_input("Model Deployment", value=deployment)
            if project_endpoint:
                os.environ["AZURE_AI_FOUNDRY_PROJECT_ENDPOINT"] = project_endpoint
                os.environ["AZURE_OPENAI_DEPLOYMENT"] = deployment
                st.success("Configured.")
                st.rerun()
        return

    st.markdown("**Quick queries:**")
    suggestions = [
        "What are current statistics?",
        "Search patients named Aaron",
        "Find matches for a patient",
        "How many pending reviews?",
    ]
    cols = st.columns(len(suggestions))
    for col, sug in zip(cols, suggestions):
        with col:
            if st.button(sug, key=f"sug_{sug[:10]}", use_container_width=True):
                st.session_state["agent_input"] = sug

    st.markdown("---")
    if "agent_messages" not in st.session_state:
        st.session_state["agent_messages"] = []
    for msg in st.session_state["agent_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask the Patient Matching Agent...")
    if "agent_input" in st.session_state:
        prompt = st.session_state.pop("agent_input")
    if prompt:
        st.session_state["agent_messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = _run_agent_query(prompt, project_endpoint, deployment)
                    st.markdown(response)
                    st.session_state["agent_messages"].append({"role": "assistant", "content": response})
                except Exception as e:
                    err = f"Error: {e}"
                    st.error(err)
                    st.session_state["agent_messages"].append({"role": "assistant", "content": err})
    if st.session_state["agent_messages"]:
        if st.button("🗑️ Clear Chat"):
            st.session_state["agent_messages"] = []
            st.rerun()


if __name__ == "__main__":
    main()
