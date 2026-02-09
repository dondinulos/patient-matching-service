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

# Page configuration
st.set_page_config(
    page_title="Patient Matching Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .match-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .score-high { color: #28a745; font-weight: bold; }
    .score-medium { color: #ffc107; font-weight: bold; }
    .score-low { color: #dc3545; font-weight: bold; }
    .patient-card {
        background-color: #ffffff;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin: 5px;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 20px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def get_gremlin_property(vertex: dict, prop_name: str, default: str = "") -> str:
    """
    Extract a property value from a Cosmos DB Gremlin vertex.
    
    Gremlin stores properties as arrays of objects: [{"id": "...", "value": "actual_value"}]
    This function handles both Gremlin format and simple key-value format.
    """
    if prop_name not in vertex:
        return default
    
    prop = vertex[prop_name]
    
    # If it's a list (Gremlin format), extract the first value
    if isinstance(prop, list) and len(prop) > 0:
        first_item = prop[0]
        if isinstance(first_item, dict):
            if "_value" in first_item:
                return str(first_item["_value"])
            if "value" in first_item:
                return str(first_item["value"])
        return str(first_item)
    
    # If it's a dict with _value or value key (single Gremlin property)
    if isinstance(prop, dict):
        if "_value" in prop:
            return str(prop["_value"])
        if "value" in prop:
            return str(prop["value"])
        return str(prop)
    
    # If it's already a simple value, return it
    if prop is None:
        return default
    
    return str(prop)


@st.cache_resource
def get_cosmos_clients():
    """Initialize Cosmos DB clients for both Gremlin and NoSQL APIs."""
    try:
        credential = AzureCliCredential()
        
        # Get Cosmos DB account name from environment
        account_name = os.environ.get("COSMOS_ACCOUNT_NAME")
        if not account_name:
            st.error("COSMOS_ACCOUNT_NAME environment variable is required")
            return None, None, None
        
        # NoSQL client for match_results
        nosql_endpoint = f"https://{account_name}.documents.azure.com:443/"
        nosql_client = CosmosClient(nosql_endpoint, credential=credential)
        
        # Gremlin client for graph data
        gremlin_endpoint = f"wss://{account_name}.gremlin.cosmos.azure.com:443/"
        gremlin_client = client.Client(
            gremlin_endpoint,
            'g',
            username=f"/dbs/patient-matching-db/colls/patients",
            password="",  # Will use Azure AD
            message_serializer=serializer.GraphSONSerializersV2d0()
        )
        
        return nosql_client, gremlin_client, account_name
    except Exception as e:
        st.error(f"Failed to connect to Cosmos DB: {e}")
        return None, None, None


@st.cache_data(ttl=60)
def fetch_match_results(_nosql_client, account_name):
    """Fetch match results from Cosmos DB NoSQL container."""
    try:
        database = _nosql_client.get_database_client("patient-matching-db")
        container = database.get_container_client("match_results")
        
        query = "SELECT * FROM c ORDER BY c.score DESC"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        return items
    except Exception as e:
        st.error(f"Error fetching match results: {e}")
        return []


@st.cache_data(ttl=60)
def fetch_patients(_nosql_client, account_name):
    """Fetch patients from Cosmos DB."""
    try:
        database = _nosql_client.get_database_client("patient-matching-db")
        container = database.get_container_client("patients")
        
        query = "SELECT * FROM c WHERE c.label = 'Patient'"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        return items
    except Exception as e:
        st.error(f"Error fetching patients: {e}")
        return []


@st.cache_data(ttl=60)
def fetch_patient_clinical_data(_nosql_client, account_name, patient_id):
    """Fetch all clinical data related to a patient from Cosmos DB Graph (via SQL API on graph container)."""
    try:
        database = _nosql_client.get_database_client("patient-matching-db")
        container = database.get_container_client("patients")
        
        # In Cosmos DB Gremlin, vertices are stored as documents with a 'label' property
        # Edges are stored separately with _isEdge=true and connect vertices via _sink/_vertexId
        
        # Fetch patient vertex
        patient_query = f"SELECT * FROM c WHERE c.id = '{patient_id}' AND c.label = 'Patient'"
        patients = list(container.query_items(query=patient_query, enable_cross_partition_query=True))
        patient = patients[0] if patients else None
        
        clinical_data = {
            "patient": patient,
            "encounters": [],
            "observations": [],
            "conditions": [],
            "procedures": [],
            "immunizations": [],
            "medications": [],
            "identifiers": [],
            "potential_matches": []
        }
        
        if not patient:
            return clinical_data
        
        # Get the patient's source_system (partition key) for efficient queries
        source_system = patient.get("source_system", "synthea")
        
        # Query edges from patient to find connected clinical vertices
        # In Cosmos DB Gremlin, edges have _isEdge=true, _sink (target vertex id), and _vertexId (source vertex id)
        
        # Find all edges originating from this patient
        edges_query = f"""
            SELECT c._sink, c.label as edge_label 
            FROM c 
            WHERE c._isEdge = true AND c._vertexId = '{patient_id}'
        """
        edges = list(container.query_items(query=edges_query, enable_cross_partition_query=True))
        
        # Group edge targets by type
        encounter_ids = []
        observation_ids = []
        condition_ids = []
        procedure_ids = []
        immunization_ids = []
        medication_ids = []
        identifier_ids = []
        
        for edge in edges:
            edge_label = edge.get("edge_label", "")
            sink_id = edge.get("_sink", "")
            if sink_id:
                if edge_label == "HAS_ENCOUNTER":
                    encounter_ids.append(sink_id)
                elif edge_label == "HAS_OBSERVATION":
                    observation_ids.append(sink_id)
                elif edge_label == "HAS_CONDITION":
                    condition_ids.append(sink_id)
                elif edge_label == "HAS_PROCEDURE":
                    procedure_ids.append(sink_id)
                elif edge_label == "HAS_IMMUNIZATION":
                    immunization_ids.append(sink_id)
                elif edge_label == "HAS_MEDICATION":
                    medication_ids.append(sink_id)
                elif edge_label == "HAS_IDENTIFIER":
                    identifier_ids.append(sink_id)
        
        # Fetch vertices by their IDs (batch queries for efficiency)
        def fetch_vertices_by_ids(vertex_ids, label):
            if not vertex_ids:
                return []
            # Limit to first 50 for performance
            vertex_ids = vertex_ids[:50]
            ids_str = ", ".join([f"'{vid}'" for vid in vertex_ids])
            query = f"SELECT * FROM c WHERE c.id IN ({ids_str}) AND c.label = '{label}'"
            return list(container.query_items(query=query, enable_cross_partition_query=True))
        
        clinical_data["encounters"] = fetch_vertices_by_ids(encounter_ids, "Encounter")
        clinical_data["observations"] = fetch_vertices_by_ids(observation_ids, "Observation")
        clinical_data["conditions"] = fetch_vertices_by_ids(condition_ids, "Condition")
        clinical_data["procedures"] = fetch_vertices_by_ids(procedure_ids, "Procedure")
        clinical_data["immunizations"] = fetch_vertices_by_ids(immunization_ids, "Immunization")
        clinical_data["medications"] = fetch_vertices_by_ids(medication_ids, "MedicationRequest")
        clinical_data["identifiers"] = fetch_vertices_by_ids(identifier_ids, "Identifier")
        
        return clinical_data
    except Exception as e:
        st.error(f"Error fetching patient clinical data: {e}")
        return {"patient": None, "encounters": [], "observations": [], "conditions": [], 
                "procedures": [], "immunizations": [], "medications": [], "identifiers": [], "potential_matches": []}


def get_confidence_color(confidence):
    """Return color based on confidence level."""
    if confidence == "auto_merge":
        return "🟢", "#28a745"
    elif confidence == "human_review":
        return "🟡", "#ffc107"
    else:
        return "🔴", "#dc3545"


def get_score_class(score):
    """Return CSS class based on score."""
    if score >= 0.85:
        return "score-high"
    elif score >= 0.50:
        return "score-medium"
    else:
        return "score-low"


def render_patient_card(patient_data, title="Patient"):
    """Render a patient information card."""
    with st.container():
        st.markdown(f"**{title}**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Name:** {patient_data.get('patient1_name', patient_data.get('name', 'N/A'))}")
            st.write(f"**DOB:** {patient_data.get('birth_date', 'N/A')}")
            st.write(f"**Gender:** {patient_data.get('gender', 'N/A')}")
        with col2:
            st.write(f"**ID:** {patient_data.get('id', 'N/A')[:8]}...")
            st.write(f"**Source:** {patient_data.get('source_system', 'N/A')}")


def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/hospital.png", width=80)
        st.title("Patient Matching")
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["📊 Dashboard", "🔍 Match Results", "👥 Patients", "🕸️ Patient Graph", "📋 Review Queue", "⚙️ Settings"]
        )
        
        st.markdown("---")
        
        # Connection status
        nosql_client, gremlin_client, account_name = get_cosmos_clients()
        if nosql_client:
            st.success("✅ Connected to Cosmos DB")
        else:
            st.error("❌ Not connected")
        
        # Refresh button
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    # Main content
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
    elif page == "⚙️ Settings":
        render_settings()


def render_dashboard(nosql_client, account_name):
    """Render the main dashboard with statistics."""
    st.title("🏥 Patient Matching Dashboard")
    st.markdown("Overview of patient matching operations and statistics")
    
    if not nosql_client:
        st.warning("Please configure Cosmos DB connection to view data.")
        return
    
    # Fetch data
    match_results = fetch_match_results(nosql_client, account_name)
    patients = fetch_patients(nosql_client, account_name)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Patients",
            value=len(patients),
            delta=None
        )
    
    with col2:
        st.metric(
            label="Match Results",
            value=len(match_results),
            delta=None
        )
    
    with col3:
        auto_merge = len([m for m in match_results if m.get('confidence') == 'auto_merge'])
        st.metric(
            label="Auto Merge",
            value=auto_merge,
            delta=None
        )
    
    with col4:
        human_review = len([m for m in match_results if m.get('confidence') == 'human_review'])
        st.metric(
            label="Pending Review",
            value=human_review,
            delta=None
        )
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Match Confidence Distribution")
        if match_results:
            confidence_counts = {}
            for m in match_results:
                conf = m.get('confidence', 'unknown')
                confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
            
            df_conf = pd.DataFrame([
                {"Confidence": k, "Count": v} 
                for k, v in confidence_counts.items()
            ])
            st.bar_chart(df_conf.set_index("Confidence"))
        else:
            st.info("No match results available")
    
    with col2:
        st.subheader("Score Distribution")
        if match_results:
            scores = [m.get('score', 0) for m in match_results]
            df_scores = pd.DataFrame({"Score": scores})
            st.line_chart(df_scores)
        else:
            st.info("No match results available")
    
    st.markdown("---")
    
    # Recent matches
    st.subheader("Recent Matches")
    if match_results:
        recent = match_results[:5]
        for idx, match in enumerate(recent):
            with st.expander(
                f"{match.get('patient1_name', 'Patient 1')} ↔ {match.get('patient2_name', 'Patient 2')} "
                f"| Score: {match.get('score', 0):.2f}"
            ):
                col1, col2, col3 = st.columns(3)
                with col1:
                    emoji, color = get_confidence_color(match.get('confidence'))
                    st.markdown(f"**Confidence:** {emoji} {match.get('confidence', 'N/A')}")
                with col2:
                    st.write(f"**Created:** {match.get('created_at', 'N/A')[:19]}")
                with col3:
                    if st.button("View Details", key=f"view_recent_{idx}"):
                        st.session_state['show_recent_details'] = idx
                
                # Show detailed view if this match is selected
                if st.session_state.get('show_recent_details') == idx:
                    render_match_details(match)
    else:
        st.info("No match results available. Run the matching service to generate results.")


def render_match_results(nosql_client, account_name):
    """Render the match results page."""
    st.title("🔍 Match Results")
    
    if not nosql_client:
        st.warning("Please configure Cosmos DB connection to view data.")
        return
    
    match_results = fetch_match_results(nosql_client, account_name)
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        min_score = st.slider("Minimum Score", 0.0, 1.0, 0.0, 0.05)
    with col2:
        confidence_filter = st.multiselect(
            "Confidence Level",
            ["auto_merge", "human_review", "no_match"],
            default=["auto_merge", "human_review", "no_match"]
        )
    with col3:
        sort_by = st.selectbox("Sort By", ["Score (High to Low)", "Score (Low to High)", "Date"])
    
    # Filter results
    filtered = [
        m for m in match_results 
        if m.get('score', 0) >= min_score and m.get('confidence') in confidence_filter
    ]
    
    # Sort results
    if sort_by == "Score (Low to High)":
        filtered = sorted(filtered, key=lambda x: x.get('score', 0))
    elif sort_by == "Date":
        filtered = sorted(filtered, key=lambda x: x.get('created_at', ''), reverse=True)
    
    st.markdown(f"**Showing {len(filtered)} of {len(match_results)} results**")
    st.markdown("---")
    
    # Results table
    if filtered:
        # Convert to DataFrame for display
        df_data = []
        for m in filtered:
            emoji, _ = get_confidence_color(m.get('confidence'))
            # Safely get patient names/IDs with None handling
            p1_name = m.get('patient1_name') or (m.get('patient1_id') or 'N/A')[:12]
            p2_name = m.get('patient2_name') or (m.get('patient2_id') or 'N/A')[:12]
            df_data.append({
                "Patient 1": p1_name,
                "Patient 2": p2_name,
                "Score": f"{m.get('score', 0):.3f}",
                "Confidence": f"{emoji} {m.get('confidence', 'N/A')}",
                "Details": str(m.get('details', {}))[:50] + "..."
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Detailed view
        st.subheader("Detailed Match View")
        selected_idx = st.selectbox(
            "Select a match to view details",
            range(len(filtered)),
            format_func=lambda i: f"{filtered[i].get('patient1_name', 'Patient 1')} ↔ {filtered[i].get('patient2_name', 'Patient 2')}"
        )
        
        if selected_idx is not None:
            match = filtered[selected_idx]
            render_match_details(match)
    else:
        st.info("No matches found with the current filters.")


def render_match_details(match: dict):
    """Render detailed match breakdown with all matching components."""
    st.markdown("---")
    
    # Header with overall score
    col1, col2, col3 = st.columns([2, 2, 1])
    
    # Safely get patient info with None handling
    p1_name = match.get('patient1_name') or 'N/A'
    p1_id = match.get('patient1_id') or 'N/A'
    p1_source = match.get('patient1_source') or 'N/A'
    p2_name = match.get('patient2_name') or 'N/A'
    p2_id = match.get('patient2_id') or 'N/A'
    p2_source = match.get('patient2_source') or 'N/A'
    
    with col1:
        st.markdown("### 👤 Patient 1")
        st.write(f"**Name:** {p1_name}")
        st.write(f"**ID:** `{p1_id[:20] if len(p1_id) > 20 else p1_id}...`")
        st.write(f"**Source:** {p1_source}")
    
    with col2:
        st.markdown("### 👤 Patient 2")
        st.write(f"**Name:** {p2_name}")
        st.write(f"**ID:** `{p2_id[:20] if len(p2_id) > 20 else p2_id}...`")
        st.write(f"**Source:** {p2_source}")
    
    with col3:
        st.markdown("### Overall Score")
        score = match.get('score', 0)
        emoji, color = get_confidence_color(match.get('confidence'))
        st.markdown(f"<h1 style='text-align: center; color: {color};'>{score:.2f}</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>{emoji} {match.get('confidence', 'N/A')}</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Get match details
    details = match.get('details', {})
    
    # If details is a string, try to parse it as JSON
    if isinstance(details, str):
        try:
            import json
            details = json.loads(details)
        except:
            details = {}
    
    # Check if we have the flat structure (from run_matching.py) vs nested structure
    is_flat = 'deterministic_score' in details or 'name_score' in details
    
    # Create tabs for different matching types
    tabs = st.tabs(["📊 Summary", "🎯 Deterministic", "📈 Probabilistic", "🧠 AI/Embeddings", "💬 LLM Analysis", "📋 Raw Data"])
    
    # Tab 1: Summary
    with tabs[0]:
        render_score_summary(match, details)
    
    # Tab 2: Deterministic Matching
    with tabs[1]:
        det_details = details.get('deterministic_details', {}) if is_flat else details.get('deterministic', {})
        render_deterministic_details(det_details)
    
    # Tab 3: Probabilistic Matching
    with tabs[2]:
        render_probabilistic_details(details, is_flat)
    
    # Tab 4: AI/Embeddings
    with tabs[3]:
        emb_details = details.get('embedding_details', {}) if is_flat else details.get('embedding', {})
        emb_score = details.get('embedding_score', 0) if is_flat else 0
        render_embedding_details(emb_details, emb_score)
    
    # Tab 5: LLM Analysis
    with tabs[4]:
        llm_details = details.get('llm_details', {}) if is_flat else details.get('llm', {})
        llm_score = details.get('llm_score', 0) if is_flat else 0
        render_llm_details(llm_details, match, llm_score)
    
    # Tab 6: Raw Data
    with tabs[5]:
        st.markdown("### Raw Match Data")
        st.json(match)


def render_score_summary(match: dict, details: dict):
    """Render a visual summary of all score components."""
    st.markdown("### Score Breakdown")
    
    # Extract scores from details - handle both nested and flat structures
    # Flat structure: deterministic_score, deterministic_details, name_score, name_details, etc.
    # Nested structure: deterministic: {...}, name: {...}, etc.
    
    # Check if we have the flat structure (from run_matching.py)
    is_flat = 'deterministic_score' in details or 'name_score' in details
    
    if is_flat:
        det_score = details.get('deterministic_score', 0)
        det_details = details.get('deterministic_details', {})
        name_score = details.get('name_score', 0)
        name_details = details.get('name_details', {})
        addr_score = details.get('address_score', 0)
        addr_details = details.get('address_details', {})
        emb_score = details.get('embedding_score', 0)
        emb_details = details.get('embedding_details', {})
        llm_score = details.get('llm_score', 0)
        llm_details = details.get('llm_details', {})
        dob_score = 1.0 if details.get('gender_match') else 0.0  # Approximation
    else:
        # Nested structure (from PatientMatcher.match())
        det_details = details.get('deterministic', {})
        name_details = details.get('name', {})
        addr_details = details.get('address', {})
        dob_details = details.get('dob', {})
        emb_details = details.get('embedding', {})
        llm_details = details.get('llm', {})
        
        # Calculate component scores from nested details
        det_score = 0.0
        if det_details.get('matched_identifiers'):
            det_score = 0.9
        elif det_details.get('dob_match'):
            det_score = 0.35
        
        name_score = name_details.get('final_score', 0) if name_details else 0
        addr_score = addr_details.get('final_score', 0) if addr_details else 0
        emb_score = emb_details.get('cosine_similarity', 0) if emb_details else 0
        llm_score = llm_details.get('score', 0) if llm_details else 0
        
        # DOB score
        if dob_details.get('exact_match'):
            dob_score = 1.0
        elif dob_details.get('transposition_detected'):
            dob_score = 0.8
        elif dob_details.get('year_match'):
            dob_score = 0.5
        else:
            dob_score = 0.0
    
    # Build scores list
    scores = [
        ("🎯 Deterministic", det_score, "SSN, MRN, Enterprise ID matches"),
        ("📝 Name Similarity", name_score, "Jaro-Winkler, Soundex, Metaphone"),
        ("🏠 Address Similarity", addr_score, "Address token matching"),
        ("📅 DOB Match", dob_score, "Exact match, transposition detection"),
        ("🧠 Embedding Similarity", emb_score, "OpenAI text-embedding-ada-002"),
        ("💬 LLM Analysis", llm_score, "GPT-4o match analysis"),
    ]
    
    # Display as aligned rows with columns for each item
    for label, score, description in scores:
        col1, col2, col3 = st.columns([2, 4, 1])
        
        with col1:
            st.markdown(f"**{label}**")
            st.caption(description)
        
        with col2:
            st.progress(min(1.0, max(0.0, score)))
        
        with col3:
            if score > 0:
                color = "green" if score >= 0.7 else "orange" if score >= 0.4 else "red"
                st.markdown(f"<h3 style='text-align: right; color: {color}; margin: 0;'>{score:.2f}</h3>", unsafe_allow_html=True)
            else:
                st.markdown("<h3 style='text-align: right; color: gray; margin: 0;'>N/A</h3>", unsafe_allow_html=True)
    
    # Shared identifiers
    shared_ids = match.get('shared_identifiers', [])
    if shared_ids:
        st.markdown("### 🔗 Shared Identifiers")
        for sid in shared_ids:
            st.success(f"✓ {sid}")


def render_deterministic_details(det_details: dict):
    """Render deterministic matching details."""
    st.markdown("### 🎯 Deterministic Matching Results")
    st.markdown("*Exact matches on unique identifiers provide highest confidence*")
    
    if not det_details:
        st.info("No deterministic matching data available. Run matching with verbose output to see details.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Matched Identifiers")
        matched_ids = det_details.get('matched_identifiers', [])
        if matched_ids:
            for id_info in matched_ids:
                if len(id_info) >= 2:
                    id_type = id_info[0]
                    id_value = id_info[1]
                    system = id_info[2] if len(id_info) > 2 else "N/A"
                    
                    if id_type == "SSN":
                        st.success(f"✅ **SSN Match:** {id_value}")
                    elif id_type == "ENTERPRISE_ID":
                        st.success(f"✅ **Enterprise ID Match:** {id_value}")
                    elif id_type == "MRN":
                        st.success(f"✅ **MRN Match:** {id_value} (System: {system})")
                    else:
                        st.success(f"✅ **{id_type} Match:** {id_value}")
        else:
            st.warning("No matching identifiers found")
        
        st.markdown("#### Matched Contacts")
        matched_contacts = det_details.get('matched_contacts', [])
        if matched_contacts:
            for contact in matched_contacts:
                if len(contact) >= 2:
                    st.success(f"✅ **{contact[0].title()} Match:** {contact[1]}")
        else:
            st.info("No matching contacts found")
    
    with col2:
        st.markdown("#### Boolean Matches")
        
        dob_match = det_details.get('dob_match', False)
        gender_match = det_details.get('gender_match', False)
        
        if dob_match:
            st.success("✅ Date of Birth: EXACT MATCH")
        else:
            st.warning("❌ Date of Birth: No match")
        
        if gender_match:
            st.success("✅ Gender: MATCH")
        else:
            st.info("➖ Gender: Not matched or N/A")


def render_probabilistic_details(details: dict, is_flat: bool = False):
    """Render probabilistic matching details."""
    st.markdown("### 📈 Probabilistic Matching Results")
    st.markdown("*Fuzzy matching using similarity algorithms*")
    
    # Name similarity
    st.markdown("#### 📝 Name Similarity")
    name_details = details.get('name_details', {}) if is_flat else details.get('name', {})
    name_score = details.get('name_score', 0) if is_flat else 0
    
    if name_details and name_details != {"reason": "missing_name"}:
        # Show overall name score if available
        if is_flat and name_score > 0:
            st.metric("Overall Name Score", f"{name_score:.3f}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Full Name (Jaro-Winkler)",
                f"{name_details.get('full_name_jaro_winkler', 0):.3f}"
            )
        with col2:
            st.metric(
                "First Name (Jaro-Winkler)",
                f"{name_details.get('first_name_jaro_winkler', 0):.3f}"
            )
        with col3:
            st.metric(
                "Last Name (Jaro-Winkler)",
                f"{name_details.get('last_name_jaro_winkler', 0):.3f}"
            )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            soundex = name_details.get('first_name_soundex_match')
            if soundex is True:
                st.success("✅ First Name Soundex: Match")
            elif soundex is False:
                st.error("❌ First Name Soundex: No Match")
            else:
                st.info("➖ First Name Soundex: N/A")
        
        with col2:
            metaphone = name_details.get('first_name_metaphone_match')
            if metaphone is True:
                st.success("✅ First Name Metaphone: Match")
            elif metaphone is False:
                st.error("❌ First Name Metaphone: No Match")
            else:
                st.info("➖ First Name Metaphone: N/A")
        
        with col3:
            last_soundex = name_details.get('last_name_soundex_match')
            if last_soundex is True:
                st.success("✅ Last Name Soundex: Match")
            elif last_soundex is False:
                st.error("❌ Last Name Soundex: No Match")
            else:
                st.info("➖ Last Name Soundex: N/A")
        
        st.metric("Levenshtein (Normalized)", f"{name_details.get('levenshtein_normalized', 0):.3f}")
        st.metric("Final Name Score", f"{name_details.get('final_score', 0):.3f}")
    else:
        st.warning("Name similarity data not available")
    
    st.markdown("---")
    
    # Address similarity
    st.markdown("#### 🏠 Address Similarity")
    addr_details = details.get('address_details', {}) if is_flat else details.get('address', {})
    addr_score = details.get('address_score', 0) if is_flat else 0
    
    if addr_details and addr_details != {"reason": "missing_address"}:
        # Show overall address score if available
        if is_flat and addr_score > 0:
            st.metric("Overall Address Score", f"{addr_score:.3f}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Normalized Address 1:**")
            st.code(addr_details.get('normalized_addr1', 'N/A'))
        
        with col2:
            st.write("**Normalized Address 2:**")
            st.code(addr_details.get('normalized_addr2', 'N/A'))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Jaro-Winkler", f"{addr_details.get('jaro_winkler', 0):.3f}")
        with col2:
            st.metric("Token Jaccard", f"{addr_details.get('token_jaccard', 0):.3f}")
        with col3:
            st.metric("Final Score", f"{addr_details.get('final_score', 0):.3f}")
        
        # Boolean matches
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if addr_details.get('postal_code_match'):
                st.success("✅ Postal Code: Match")
            else:
                st.info("➖ Postal Code: No Match")
        
        with col2:
            if addr_details.get('city_match'):
                st.success("✅ City: Match")
            else:
                st.info("➖ City: No Match")
        
        with col3:
            if addr_details.get('state_match'):
                st.success("✅ State: Match")
            else:
                st.info("➖ State: No Match")
        
        # Shared tokens
        shared_tokens = addr_details.get('shared_tokens', [])
        if shared_tokens:
            st.write("**Shared Address Tokens:**")
            st.write(", ".join(shared_tokens))
    else:
        st.warning("Address similarity data not available")
    
    st.markdown("---")
    
    # DOB similarity
    st.markdown("#### 📅 Date of Birth Similarity")
    dob_details = details.get('dob', {})
    
    if dob_details and dob_details != {"reason": "missing_dob"}:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**DOB 1:** {dob_details.get('dob1', 'N/A')}")
        with col2:
            st.write(f"**DOB 2:** {dob_details.get('dob2', 'N/A')}")
        
        if dob_details.get('exact_match'):
            st.success("✅ **EXACT MATCH** - Dates are identical")
        elif dob_details.get('transposition_detected'):
            st.warning("⚠️ **TRANSPOSITION DETECTED** - Month/Day may be swapped (80% confidence)")
        elif dob_details.get('year_match'):
            st.info("📅 **Year Match** - Same year, different month/day")
        else:
            st.error("❌ No DOB match detected")
    else:
        st.warning("DOB similarity data not available")


def render_embedding_details(emb_details: dict, emb_score: float = 0):
    """Render embedding/AI matching details."""
    st.markdown("### 🧠 AI-Enhanced Matching (Embeddings)")
    st.markdown("*Using OpenAI text-embedding-ada-002 for semantic similarity*")
    
    if not emb_details and emb_score == 0:
        st.info("Embedding matching was not used for this match. Enable with `--use-embeddings` flag.")
        st.markdown("""
        **To enable embeddings:**
        ```bash
        python scripts/run_matching.py --use-embeddings --verbose
        ```
        
        **Environment variables required:**
        - `AZURE_OPENAI_ENDPOINT`
        - `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`
        """)
        return
    
    if emb_details.get('reason') == 'embeddings_not_available':
        st.warning("Embeddings were requested but not available. Check your OpenAI configuration.")
        return
    
    # Display embedding scores
    col1, col2 = st.columns(2)
    
    with col1:
        # Use emb_score if provided (flat structure), otherwise get from details
        cosine_sim = emb_score if emb_score > 0 else emb_details.get('cosine_similarity', 0)
        st.metric("Embedding Similarity Score", f"{cosine_sim:.4f}")
        
        # Visual representation
        st.progress(min(1.0, max(0.0, cosine_sim)))
        
        if cosine_sim >= 0.95:
            st.success("🎯 Very High Similarity - Likely same patient")
        elif cosine_sim >= 0.85:
            st.success("✅ High Similarity - Strong match candidate")
        elif cosine_sim >= 0.70:
            st.warning("⚠️ Moderate Similarity - Review recommended")
        elif cosine_sim > 0:
            st.error("❌ Low Similarity - Likely different patients")
        else:
            st.info("No embedding score available")
    
    with col2:
        st.markdown("**How it works:**")
        st.markdown("""
        1. Patient demographics are converted to text
        2. Text is embedded using Azure OpenAI
        3. Cosine similarity measures semantic closeness
        4. Score > 0.85 indicates strong semantic match
        """)


def render_llm_details(llm_details: dict, match: dict, llm_score_param: float = 0):
    """Render LLM analysis details."""
    st.markdown("### 💬 LLM-Based Analysis (GPT-4o)")
    st.markdown("*AI reasoning about match probability*")
    
    # Check if LLM was used - get score from parameter first (flat structure)
    llm_score = llm_score_param
    llm_recommendation = llm_details.get('recommendation', '')
    
    # If score not from param, try to get from match details
    if llm_score == 0:
        match_details = match.get('details', {})
        if isinstance(match_details, str):
            try:
                import json
                match_details = json.loads(match_details)
            except:
                match_details = {}
        
        llm_score = match_details.get('llm_score', 0)
        if not llm_recommendation:
            llm_recommendation = match_details.get('llm_recommendation', '')
    
    llm_blended = False
    if isinstance(match.get('details'), dict):
        llm_blended = match.get('details', {}).get('llm_blended', False)
    
    if llm_score == 0 and not llm_details:
        st.info("LLM analysis was not used for this match. Enable with `--use-llm` flag.")
        st.markdown("""
        **To enable LLM analysis:**
        ```bash
        python scripts/run_matching.py --use-llm --verbose
        ```
        
        **Environment variables required:**
        - `AZURE_OPENAI_ENDPOINT`
        - `AZURE_OPENAI_CHAT_DEPLOYMENT`
        """)
        return
    
    # Display LLM results
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("LLM Confidence Score", f"{llm_score:.2f}")
        st.progress(min(1.0, max(0.0, llm_score)))
        
        if llm_recommendation:
            if llm_recommendation.lower() in ['match', 'merge', 'same_patient']:
                st.success(f"🎯 Recommendation: **{llm_recommendation.upper()}**")
            elif llm_recommendation.lower() in ['review', 'uncertain', 'manual_review']:
                st.warning(f"⚠️ Recommendation: **{llm_recommendation.upper()}**")
            else:
                st.error(f"❌ Recommendation: **{llm_recommendation.upper()}**")
    
    with col2:
        if llm_blended:
            st.success("✅ LLM score was blended with traditional matching")
            st.markdown("""
            **Blending weights:**
            - Traditional matching: 80%
            - LLM analysis: 20%
            """)
        
        st.markdown("**LLM Analysis Process:**")
        st.markdown("""
        1. Patient demographics sent to GPT-4o
        2. AI analyzes name variants, typos, transpositions
        3. Considers context and common data entry errors
        4. Returns confidence score and recommendation
        """)
    
    # Show reasoning if available
    if llm_details.get('reasoning'):
        st.markdown("#### AI Reasoning")
        st.info(llm_details.get('reasoning'))
    
    if llm_details.get('analysis'):
        st.markdown("#### Detailed Analysis")
        st.write(llm_details.get('analysis'))


def render_patients(nosql_client, account_name):
    """Render the patients list page."""
    st.title("👥 Patients")
    
    if not nosql_client:
        st.warning("Please configure Cosmos DB connection to view data.")
        return
    
    patients = fetch_patients(nosql_client, account_name)
    
    st.markdown(f"**Total Patients: {len(patients)}**")
    
    # Search
    search_term = st.text_input("🔎 Search patients", placeholder="Enter name, ID, or source system...")
    
    # Filter patients
    if search_term:
        patients = [
            p for p in patients 
            if search_term.lower() in str(p).lower()
        ]
    
    st.markdown("---")
    
    if patients:
        # Display as cards
        for i, patient in enumerate(patients[:20]):  # Limit to 20 for performance
            with st.expander(
                f"👤 {patient.get('firstName', '')} {patient.get('lastName', '')} | "
                f"DOB: {patient.get('birthDate', 'N/A')} | "
                f"Source: {patient.get('sourceSystem', 'N/A')}"
            ):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**ID:** `{patient.get('id', 'N/A')}`")
                    st.write(f"**Gender:** {patient.get('gender', 'N/A')}")
                    st.write(f"**Source ID:** {patient.get('sourceId', 'N/A')}")
                with col2:
                    st.write(f"**Created:** {patient.get('createdAt', 'N/A')[:19] if patient.get('createdAt') else 'N/A'}")
                    if patient.get('ssn'):
                        st.write(f"**SSN:** ***-**-{patient.get('ssn', '')[-4:]}")
        
        if len(patients) > 20:
            st.info(f"Showing first 20 of {len(patients)} patients. Use search to find specific patients.")
    else:
        st.info("No patients found.")


def render_review_queue(nosql_client, account_name):
    """Render the human review queue."""
    st.title("📋 Review Queue")
    st.markdown("Matches requiring human review before merging")
    
    if not nosql_client:
        st.warning("Please configure Cosmos DB connection to view data.")
        return
    
    match_results = fetch_match_results(nosql_client, account_name)
    
    # Filter for human review
    pending = [m for m in match_results if m.get('confidence') == 'human_review']
    
    st.metric("Pending Reviews", len(pending))
    st.markdown("---")
    
    if pending:
        for i, match in enumerate(pending):
            st.markdown(f"### Review #{i+1}")
            
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.markdown("**Patient 1**")
                st.write(f"Name: {match.get('patient1_name', 'N/A')}")
                st.write(f"ID: `{match.get('patient1_id', 'N/A')[:12]}...`")
            
            with col2:
                st.markdown("**Patient 2**")
                st.write(f"Name: {match.get('patient2_name', 'N/A')}")
                st.write(f"ID: `{match.get('patient2_id', 'N/A')[:12]}...`")
            
            with col3:
                st.markdown("**Score**")
                st.markdown(f"### {match.get('score', 0):.2f}")
            
            # Review actions
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("✅ Approve Merge", key=f"approve_{i}"):
                    st.success("Match approved! (Demo - not actually saved)")
            with col2:
                if st.button("❌ Reject", key=f"reject_{i}"):
                    st.warning("Match rejected! (Demo - not actually saved)")
            with col3:
                if st.button("🔍 View Details", key=f"details_{i}"):
                    st.session_state[f"show_details_{i}"] = True
            
            # Show detailed view if button was clicked
            if st.session_state.get(f"show_details_{i}", False):
                render_match_details(match)
                if st.button("Hide Details", key=f"hide_{i}"):
                    st.session_state[f"show_details_{i}"] = False
                    st.rerun()
            
            st.markdown("---")
    else:
        st.success("🎉 No pending reviews! All matches have been processed.")


def render_settings():
    """Render the settings page."""
    st.title("⚙️ Settings")
    
    st.subheader("Match Weights Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Deterministic Matching**")
        enterprise_id = st.slider("Enterprise ID Weight", 0.0, 1.0, 1.0, 0.1)
        mrn = st.slider("MRN Weight", 0.0, 1.0, 0.8, 0.1)
        ssn = st.slider("SSN Weight", 0.0, 1.0, 0.9, 0.1)
        dob = st.slider("DOB Weight", 0.0, 1.0, 0.35, 0.05)
    
    with col2:
        st.markdown("**Probabilistic Matching**")
        name_weight = st.slider("Name Similarity Weight", 0.0, 1.0, 0.35, 0.05)
        address_weight = st.slider("Address Similarity Weight", 0.0, 1.0, 0.15, 0.05)
        embedding_weight = st.slider("Embedding Similarity Weight", 0.0, 1.0, 0.1, 0.05)
    
    st.markdown("---")
    
    st.subheader("Thresholds")
    col1, col2 = st.columns(2)
    with col1:
        auto_merge = st.slider("Auto-Merge Threshold", 0.5, 1.0, 0.85, 0.05)
    with col2:
        human_review = st.slider("Human Review Threshold", 0.3, 0.9, 0.65, 0.05)
    
    st.markdown("---")
    
    st.subheader("Azure OpenAI Configuration")
    openai_endpoint = st.text_input(
        "Azure OpenAI Endpoint",
        value=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
        type="password"
    )
    embedding_deployment = st.text_input(
        "Embedding Deployment",
        value=os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
    )
    chat_deployment = st.text_input(
        "Chat Deployment",
        value=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
    )
    
    if st.button("💾 Save Settings"):
        st.success("Settings saved! (Demo - not actually persisted)")
        st.info("To persist settings, update environment variables or configuration files.")


def render_patient_graph(nosql_client, gremlin_client, account_name):
    """Render an interactive graph visualization of patient and clinical data relationships."""
    st.title("🕸️ Patient Clinical Data Graph")
    st.markdown("Visualize the relationships between a patient and their clinical data")
    
    if not AGRAPH_AVAILABLE:
        st.error("Graph visualization requires the `streamlit-agraph` package. Install it with:")
        st.code("pip install streamlit-agraph")
        st.info("After installing, restart the Streamlit server.")
        return
    
    if not nosql_client:
        st.warning("Please configure Cosmos DB connection to view data.")
        return
    
    # Fetch patients for selection
    patients = fetch_patients(nosql_client, account_name)
    
    if not patients:
        st.info("No patients found in the database.")
        return
    
    # Patient selector - use helper to extract Gremlin properties
    patient_options = {}
    for p in patients[:100]:  # Limit to first 100 for performance
        first_name = get_gremlin_property(p, 'firstName', '')
        last_name = get_gremlin_property(p, 'lastName', '')
        birth_date = get_gremlin_property(p, 'birthDate', 'N/A')
        patient_id = p.get('id', '')
        label = f"{first_name} {last_name} (DOB: {birth_date}) - {patient_id[:12]}..."
        patient_options[label] = patient_id
    
    selected_patient_label = st.selectbox(
        "🔎 Select a patient to view their clinical data graph",
        options=list(patient_options.keys()),
        index=0
    )
    
    selected_patient_id = patient_options.get(selected_patient_label)
    
    if not selected_patient_id:
        return
    
    # Graph configuration options
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_encounters = st.checkbox("🏥 Encounters", value=True)
        show_observations = st.checkbox("🔬 Observations", value=True)
    with col2:
        show_conditions = st.checkbox("💊 Conditions", value=True)
        show_procedures = st.checkbox("🔧 Procedures", value=True)
    with col3:
        show_immunizations = st.checkbox("💉 Immunizations", value=True)
        show_medications = st.checkbox("💊 Medications", value=True)
    
    show_matched_patients = st.checkbox("🔗 Show Matched Patients", value=True, help="Show other patient records that have been matched to this patient")
    
    # Fetch clinical data
    clinical_data = fetch_patient_clinical_data(nosql_client, account_name, selected_patient_id)
    
    if not clinical_data.get("patient"):
        st.warning("Could not load patient data.")
        return
    
    # Build the graph
    nodes = []
    edges = []
    
    patient = clinical_data["patient"]
    # Use helper to extract Gremlin property values
    first_name = get_gremlin_property(patient, 'firstName', '')
    last_name = get_gremlin_property(patient, 'lastName', '')
    patient_name = f"{first_name} {last_name}".strip() or "Unknown"
    birth_date = get_gremlin_property(patient, 'birthDate', 'N/A')
    gender = get_gremlin_property(patient, 'gender', 'N/A')
    
    # Patient node (center)
    nodes.append(Node(
        id=selected_patient_id,
        label=patient_name,
        size=40,
        color="#4CAF50",  # Green for patient
        shape="circularImage",
        image="https://img.icons8.com/color/96/user.png",
        font={"size": 16, "color": "#333"},
        title=f"Patient: {patient_name}\nDOB: {birth_date}\nGender: {gender}"
    ))
    
    # Color scheme for different node types
    node_colors = {
        "Encounter": "#2196F3",      # Blue
        "Observation": "#9C27B0",    # Purple
        "Condition": "#F44336",      # Red
        "Procedure": "#FF9800",      # Orange
        "Immunization": "#00BCD4",   # Cyan
        "Medication": "#E91E63",     # Pink
        "Identifier": "#607D8B",     # Gray
        "MatchedPatient": "#FFD700", # Gold for matched patients
    }
    
    node_icons = {
        "Encounter": "https://img.icons8.com/color/48/hospital.png",
        "Observation": "https://img.icons8.com/color/48/microscope.png",
        "Condition": "https://img.icons8.com/color/48/heart-with-pulse.png",
        "Procedure": "https://img.icons8.com/color/48/surgical-scissors.png",
        "Immunization": "https://img.icons8.com/color/48/syringe.png",
        "Medication": "https://img.icons8.com/color/48/pill.png",
        "Identifier": "https://img.icons8.com/color/48/identification-documents.png",
    }
    
    # Add encounter nodes
    if show_encounters:
        for enc in clinical_data.get("encounters", [])[:20]:  # Limit for performance
            enc_id = enc.get("id", "")
            enc_type = get_gremlin_property(enc, "encounterType", get_gremlin_property(enc, "type", "Encounter"))
            period_start = get_gremlin_property(enc, "periodStart", "")
            enc_date = period_start[:10] if period_start else "N/A"
            enc_status = get_gremlin_property(enc, "status", "N/A")
            nodes.append(Node(
                id=enc_id,
                label=f"{enc_type[:20]}",
                size=25,
                color=node_colors["Encounter"],
                shape="circularImage",
                image=node_icons["Encounter"],
                title=f"Encounter: {enc_type}\nDate: {enc_date}\nStatus: {enc_status}"
            ))
            edges.append(Edge(source=selected_patient_id, target=enc_id, label="HAS_ENCOUNTER", color="#2196F3"))
    
    # Add observation nodes
    if show_observations:
        for obs in clinical_data.get("observations", [])[:20]:
            obs_id = obs.get("id", "")
            obs_code = get_gremlin_property(obs, "codeDisplay", get_gremlin_property(obs, "code", "Observation"))
            obs_value = get_gremlin_property(obs, "valueString", get_gremlin_property(obs, "value", ""))
            eff_date = get_gremlin_property(obs, "effectiveDateTime", "")
            obs_date = eff_date[:10] if eff_date else "N/A"
            nodes.append(Node(
                id=obs_id,
                label=f"{str(obs_code)[:15]}",
                size=20,
                color=node_colors["Observation"],
                shape="circularImage",
                image=node_icons["Observation"],
                title=f"Observation: {obs_code}\nValue: {obs_value}\nDate: {obs_date}"
            ))
            edges.append(Edge(source=selected_patient_id, target=obs_id, label="HAS_OBSERVATION", color="#9C27B0"))
    
    # Add condition nodes
    if show_conditions:
        for cond in clinical_data.get("conditions", [])[:20]:
            cond_id = cond.get("id", "")
            cond_code = get_gremlin_property(cond, "codeDisplay", get_gremlin_property(cond, "code", "Condition"))
            cond_status = get_gremlin_property(cond, "clinicalStatus", "N/A")
            onset = get_gremlin_property(cond, "onsetDateTime", "")
            cond_onset = onset[:10] if onset else "N/A"
            nodes.append(Node(
                id=cond_id,
                label=f"{str(cond_code)[:15]}",
                size=22,
                color=node_colors["Condition"],
                shape="circularImage",
                image=node_icons["Condition"],
                title=f"Condition: {cond_code}\nStatus: {cond_status}\nOnset: {cond_onset}"
            ))
            edges.append(Edge(source=selected_patient_id, target=cond_id, label="HAS_CONDITION", color="#F44336"))
    
    # Add procedure nodes
    if show_procedures:
        for proc in clinical_data.get("procedures", [])[:20]:
            proc_id = proc.get("id", "")
            proc_code = get_gremlin_property(proc, "codeDisplay", get_gremlin_property(proc, "code", "Procedure"))
            performed = get_gremlin_property(proc, "performedDateTime", "")
            proc_date = performed[:10] if performed else "N/A"
            proc_status = get_gremlin_property(proc, "status", "N/A")
            nodes.append(Node(
                id=proc_id,
                label=f"{str(proc_code)[:15]}",
                size=22,
                color=node_colors["Procedure"],
                shape="circularImage",
                image=node_icons["Procedure"],
                title=f"Procedure: {proc_code}\nDate: {proc_date}\nStatus: {proc_status}"
            ))
            edges.append(Edge(source=selected_patient_id, target=proc_id, label="HAS_PROCEDURE", color="#FF9800"))
    
    # Add immunization nodes
    if show_immunizations:
        for imm in clinical_data.get("immunizations", [])[:20]:
            imm_id = imm.get("id", "")
            imm_code = get_gremlin_property(imm, "vaccineDisplay", get_gremlin_property(imm, "vaccineCode", "Immunization"))
            occurrence = get_gremlin_property(imm, "occurrenceDateTime", "")
            imm_date = occurrence[:10] if occurrence else "N/A"
            imm_status = get_gremlin_property(imm, "status", "N/A")
            nodes.append(Node(
                id=imm_id,
                label=f"{str(imm_code)[:15]}",
                size=20,
                color=node_colors["Immunization"],
                shape="circularImage",
                image=node_icons["Immunization"],
                title=f"Immunization: {imm_code}\nDate: {imm_date}\nStatus: {imm_status}"
            ))
            edges.append(Edge(source=selected_patient_id, target=imm_id, label="HAS_IMMUNIZATION", color="#00BCD4"))
    
    # Add matched patient nodes
    if show_matched_patients:
        match_results = fetch_match_results(nosql_client, account_name)
        matched_added = set()  # Track added matched patient IDs to avoid duplicates
        for m in match_results:
            p1_id = m.get('patient1_id', '')
            p2_id = m.get('patient2_id', '')
            score = m.get('score', 0)
            confidence = m.get('confidence', 'N/A')
            emoji, conf_color = get_confidence_color(confidence)
            
            # Check if this match involves the selected patient
            matched_id = None
            matched_name = None
            if p1_id == selected_patient_id:
                matched_id = p2_id
                matched_name = m.get('patient2_name', 'Unknown')
            elif p2_id == selected_patient_id:
                matched_id = p1_id
                matched_name = m.get('patient1_name', 'Unknown')
            
            if matched_id and matched_id not in matched_added:
                matched_added.add(matched_id)
                nodes.append(Node(
                    id=matched_id,
                    label=f"{matched_name}",
                    size=35,
                    color=node_colors["MatchedPatient"],
                    shape="circularImage",
                    image="https://img.icons8.com/color/96/user.png",
                    font={"size": 14, "color": "#333"},
                    title=f"Matched Patient: {matched_name}\nID: {matched_id[:20]}...\nMatch Score: {score:.2f}\nConfidence: {confidence}"
                ))
                edges.append(Edge(
                    source=selected_patient_id,
                    target=matched_id,
                    label=f"MATCHED ({score:.2f})",
                    color=conf_color,
                    width=3,
                    dashes=False
                ))

    # Add medication nodes
    if show_medications:
        for med in clinical_data.get("medications", [])[:20]:
            med_id = med.get("id", "")
            med_code = get_gremlin_property(med, "medicationDisplay", get_gremlin_property(med, "medicationCode", "Medication"))
            med_status = get_gremlin_property(med, "status", "N/A")
            med_intent = get_gremlin_property(med, "intent", "N/A")
            nodes.append(Node(
                id=med_id,
                label=f"{str(med_code)[:15]}",
                size=20,
                color=node_colors["Medication"],
                shape="circularImage",
                image=node_icons["Medication"],
                title=f"Medication: {med_code}\nStatus: {med_status}\nIntent: {med_intent}"
            ))
            edges.append(Edge(source=selected_patient_id, target=med_id, label="HAS_MEDICATION", color="#E91E63"))
    
    # Graph configuration
    config = Config(
        width="100%",
        height=800,
        directed=True,
        physics={
            "enabled": True,
            "barnesHut": {
                "gravitationalConstant": -8000,
                "centralGravity": 0.15,
                "springLength": 250,
                "springConstant": 0.02,
                "damping": 0.09,
                "avoidOverlap": 1.0,
            },
            "minVelocity": 0.75,
            "stabilization": {
                "enabled": True,
                "iterations": 200,
            },
        },
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=False,
        node={
            "labelProperty": "label",
            "renderLabel": True,
        },
        link={
            "labelProperty": "label",
            "renderLabel": False,
        }
    )
    
    # Display statistics
    st.markdown("---")
    st.subheader("📊 Clinical Data Summary")
    
    cols = st.columns(6)
    with cols[0]:
        st.metric("Encounters", len(clinical_data.get("encounters", [])))
    with cols[1]:
        st.metric("Observations", len(clinical_data.get("observations", [])))
    with cols[2]:
        st.metric("Conditions", len(clinical_data.get("conditions", [])))
    with cols[3]:
        st.metric("Procedures", len(clinical_data.get("procedures", [])))
    with cols[4]:
        st.metric("Immunizations", len(clinical_data.get("immunizations", [])))
    with cols[5]:
        st.metric("Medications", len(clinical_data.get("medications", [])))
    
    st.markdown("---")
    
    # Render the graph
    if len(nodes) > 1:
        st.subheader("🕸️ Interactive Graph")
        st.caption("Click and drag nodes to rearrange. Hover for details. Scroll to zoom.")
        
        return_value = agraph(nodes=nodes, edges=edges, config=config)
        
        # Legend
        st.markdown("---")
        st.subheader("🎨 Legend")
        legend_cols = st.columns(8)
        legend_items = [
            ("👤 Patient", "#4CAF50"),
            ("🔗 Matched", "#FFD700"),
            ("🏥 Encounter", "#2196F3"),
            ("🔬 Observation", "#9C27B0"),
            ("❤️ Condition", "#F44336"),
            ("🔧 Procedure", "#FF9800"),
            ("💉 Immunization", "#00BCD4"),
            ("💊 Medication", "#E91E63"),
        ]
        for col, (label, color) in zip(legend_cols, legend_items):
            with col:
                st.markdown(f"<div style='background-color: {color}; color: white; padding: 5px 10px; border-radius: 5px; text-align: center;'>{label}</div>", unsafe_allow_html=True)
    else:
        st.info("No clinical data found for this patient. The graph will appear when clinical data is available.")
    
    # Expandable detailed data
    with st.expander("📋 View Raw Clinical Data"):
        tabs = st.tabs(["Encounters", "Observations", "Conditions", "Procedures", "Immunizations", "Medications"])
        
        with tabs[0]:
            if clinical_data.get("encounters"):
                st.json(clinical_data["encounters"][:10])
            else:
                st.info("No encounters found")
        
        with tabs[1]:
            if clinical_data.get("observations"):
                st.json(clinical_data["observations"][:10])
            else:
                st.info("No observations found")
        
        with tabs[2]:
            if clinical_data.get("conditions"):
                st.json(clinical_data["conditions"][:10])
            else:
                st.info("No conditions found")
        
        with tabs[3]:
            if clinical_data.get("procedures"):
                st.json(clinical_data["procedures"][:10])
            else:
                st.info("No procedures found")
        
        with tabs[4]:
            if clinical_data.get("immunizations"):
                st.json(clinical_data["immunizations"][:10])
            else:
                st.info("No immunizations found")
        
        with tabs[5]:
            if clinical_data.get("medications"):
                st.json(clinical_data["medications"][:10])
            else:
                st.info("No medications found")


if __name__ == "__main__":
    main()
