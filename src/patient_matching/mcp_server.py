"""
Patient Matching MCP Server

Exposes the Patient Matching Service as a Model Context Protocol (MCP) server,
making patient matching capabilities available to any MCP-compatible client
(VS Code Copilot, Claude Desktop, custom agents, etc.).

Features:
- Tools: Find matches, compare patients, approve/reject, batch operations
- Resources: Patient records, service statistics, pending reviews
- Prompts: Templated workflows for common matching scenarios

Usage:
    # stdio transport (local dev, VS Code)
    python -m src.patient_matching.mcp_server

    # Or via MCP CLI
    mcp run src/patient_matching/mcp_server.py
"""

import os
import sys
import json
import logging
from typing import Optional

# Ensure the project root is on sys.path so that both `mcp dev` (standalone)
# and `python -m` (package) execution modes can resolve imports.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mcp.server.fastmcp import FastMCP

from src.patient_matching.service import PatientMatchingService
from src.patient_matching.matching import MatchWeights
from src.patient_matching.models import MatchConfidence

logger = logging.getLogger(__name__)

# ==================== Server Setup ====================

mcp = FastMCP(
    "Patient Matching Service",
    instructions="""You are connected to a Patient Matching Service (Master Patient Index).
This server provides tools to find duplicate patient records, compare patients,
manage match decisions, and query service statistics.

Confidence levels:
- AUTO_MERGE (score >= 0.85): High confidence - very likely the same person
- HUMAN_REVIEW (score 0.65-0.85): Medium confidence - requires human verification
- NO_MATCH (score < 0.65): Low confidence - likely different patients"""
)

# ==================== Service Singleton ====================

_service: Optional[PatientMatchingService] = None


def _get_service() -> PatientMatchingService:
    """Get or create the global PatientMatchingService instance."""
    global _service
    if _service is None:
        db_type = os.getenv("PM_DB_TYPE", "cosmos")

        _service = PatientMatchingService(
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
            cosmos_endpoint=os.getenv("COSMOS_GREMLIN_ENDPOINT"),
            cosmos_database=os.getenv("COSMOS_DATABASE", "PatientMatching"),
            cosmos_container=os.getenv("COSMOS_CONTAINER", "PatientGraph"),
            cosmos_key=os.getenv("COSMOS_KEY"),
            db_type=db_type,
            use_embeddings=os.getenv("USE_EMBEDDINGS", "false").lower() == "true",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        _service.initialize()
        logger.info("Patient Matching Service initialized for MCP server (%s backend)", db_type)

    return _service


def _get_recommendation(confidence: MatchConfidence) -> str:
    """Human-readable recommendation from a confidence level."""
    if confidence == MatchConfidence.AUTO_MERGE:
        return "High confidence match - recommended for automatic merge"
    elif confidence == MatchConfidence.HUMAN_REVIEW:
        return "Medium confidence - requires human review before merge"
    return "Low confidence - likely different patients"


# ==================== MCP Tools ====================


@mcp.tool()
def find_patient_matches(
    patient_id: str,
    min_score: float = 0.3,
    max_results: int = 20,
    search_entire_database: bool = False,
) -> str:
    """Find potential duplicate patient matches for a given patient ID.

    Uses deterministic matching (exact identifier matches) and probabilistic
    matching (name similarity, address similarity, etc.) to compute match scores.

    Args:
        patient_id: The unique ID of the patient to find matches for.
        min_score: Minimum match score threshold (0.0-1.0). Default 0.3.
        max_results: Maximum number of matches to return. Default 20.
        search_entire_database: If True, search against ALL patients instead
            of using graph-based candidate retrieval. Default False.

    Returns:
        JSON with patient_id, matches_found count, and list of matches
        including scores and confidence levels.
    """
    service = _get_service()

    if search_entire_database:
        matches = service.find_all_matches_for_patient(
            patient_id=patient_id,
            min_score=min_score,
            limit=max_results,
        )
    else:
        matches = service.find_matches_for_patient(
            patient_id=patient_id,
            min_score=min_score,
        )[:max_results]

    if not matches:
        return f"No matches found for patient {patient_id} with minimum score {min_score}."

    results = [
        {
            "matched_patient_id": m.patient2_id,
            "score": round(m.score, 3),
            "confidence": m.confidence.value,
            "deterministic_score": round(m.deterministic_score, 3),
            "name_similarity": round(m.name_similarity, 3),
            "address_similarity": round(m.address_similarity, 3),
            "shared_identifiers": m.shared_identifiers,
        }
        for m in matches
    ]

    return json.dumps(
        {"patient_id": patient_id, "matches_found": len(results), "matches": results},
        indent=2,
    )


@mcp.tool()
def get_patient_details(patient_id: str) -> str:
    """Get detailed demographic information about a specific patient.

    Args:
        patient_id: The unique ID of the patient to retrieve.

    Returns:
        JSON with patient demographics (name, DOB, gender, identifiers,
        addresses, contact points, EMPI ID).
    """
    service = _get_service()
    patient = service.db.get_patient(patient_id)

    if not patient:
        return f"Patient with ID {patient_id} not found."

    result = {
        "id": patient.id,
        "source_id": patient.source_id,
        "source_system": patient.source_system,
        "name": patient.name.full_name if patient.name else None,
        "birth_date": patient.birth_date.isoformat() if patient.birth_date else None,
        "gender": patient.gender.value if patient.gender else None,
        "identifiers": [
            {"type": i.type.value, "value": i.value, "system": i.system}
            for i in patient.identifiers
        ],
        "addresses": [addr.full_address for addr in patient.addresses],
        "contact_points": [
            {
                "system": c.system.value if hasattr(c.system, "value") else c.system,
                "value": c.value,
            }
            for c in patient.contact_points
        ],
        "empi_id": patient.empi_id,
    }

    return json.dumps(result, indent=2)


@mcp.tool()
def compare_two_patients(patient1_id: str, patient2_id: str) -> str:
    """Compare two specific patients and compute a detailed match score.

    Performs deterministic, probabilistic, and optionally AI-enhanced matching.

    Args:
        patient1_id: The ID of the first patient.
        patient2_id: The ID of the second patient.

    Returns:
        JSON with both patients' summaries, detailed score breakdown,
        and a merge recommendation.
    """
    service = _get_service()

    patient1 = service.db.get_patient(patient1_id)
    patient2 = service.db.get_patient(patient2_id)

    if not patient1:
        return f"Patient with ID {patient1_id} not found."
    if not patient2:
        return f"Patient with ID {patient2_id} not found."

    result = service.matcher.match(patient1, patient2)

    comparison = {
        "patient1": {
            "id": patient1_id,
            "name": patient1.name.full_name if patient1.name else None,
            "birth_date": patient1.birth_date.isoformat() if patient1.birth_date else None,
        },
        "patient2": {
            "id": patient2_id,
            "name": patient2.name.full_name if patient2.name else None,
            "birth_date": patient2.birth_date.isoformat() if patient2.birth_date else None,
        },
        "match_result": {
            "overall_score": round(result.score, 3),
            "confidence": result.confidence.value,
            "deterministic_score": round(result.deterministic_score, 3),
            "name_similarity": round(result.name_similarity, 3),
            "address_similarity": round(result.address_similarity, 3),
            "embedding_similarity": round(result.embedding_similarity, 3),
            "shared_identifiers": result.shared_identifiers,
            "recommendation": _get_recommendation(result.confidence),
        },
    }

    return json.dumps(comparison, indent=2)


@mcp.tool()
def run_batch_matching(min_score: float = 0.3) -> str:
    """Run matching for all patients in the database.

    Comprehensive matching operation that finds all potential duplicate records.
    Use for initial MPI setup or periodic comprehensive matching.

    WARNING: Can take a long time for large databases.

    Args:
        min_score: Minimum match score threshold. Default 0.3.

    Returns:
        JSON with operation status and statistics
        (patients processed, matches found, auto-merge/human-review counts).
    """
    service = _get_service()
    stats = service.run_global_matching(min_score=min_score)

    return json.dumps(
        {"operation": "global_matching", "status": "completed", "statistics": stats},
        indent=2,
    )


@mcp.tool()
def approve_patient_match(
    patient1_id: str,
    patient2_id: str,
    reviewed_by: str,
    notes: str | None = None,
) -> str:
    """Approve a potential match and merge two patients into a single EMPI record.

    Args:
        patient1_id: The ID of the first patient in the match.
        patient2_id: The ID of the second patient in the match.
        reviewed_by: The identifier of the person approving the match.
        notes: Optional notes about the approval decision.

    Returns:
        JSON confirmation with EMPI record ID.
    """
    service = _get_service()
    empi_id = service.approve_match(
        patient1_id=patient1_id,
        patient2_id=patient2_id,
        reviewed_by=reviewed_by,
        notes=notes,
    )

    return json.dumps(
        {
            "action": "match_approved",
            "patient1_id": patient1_id,
            "patient2_id": patient2_id,
            "empi_id": empi_id,
            "reviewed_by": reviewed_by,
            "message": f"Patients successfully merged into EMPI record {empi_id}",
        },
        indent=2,
    )


@mcp.tool()
def reject_patient_match(
    patient1_id: str,
    patient2_id: str,
    reviewed_by: str,
    reason: str,
) -> str:
    """Reject a potential match between two patients.

    Marks the match as rejected, preventing future auto-merge suggestions.

    Args:
        patient1_id: The ID of the first patient in the match.
        patient2_id: The ID of the second patient in the match.
        reviewed_by: The identifier of the person rejecting the match.
        reason: The reason for rejecting the match.

    Returns:
        JSON confirmation of rejection.
    """
    service = _get_service()
    service.reject_match(
        patient1_id=patient1_id,
        patient2_id=patient2_id,
        reviewed_by=reviewed_by,
        reason=reason,
    )

    return json.dumps(
        {
            "action": "match_rejected",
            "patient1_id": patient1_id,
            "patient2_id": patient2_id,
            "reviewed_by": reviewed_by,
            "reason": reason,
            "message": "Match rejection recorded",
        },
        indent=2,
    )


@mcp.tool()
def get_pending_reviews(limit: int = 20) -> str:
    """Get patient matches that require human review.

    Returns matches with medium confidence (0.65-0.85) that need a human
    to decide whether to approve or reject.

    Args:
        limit: Maximum number of pending reviews to return. Default 20.

    Returns:
        JSON with review count and list of match pairs awaiting review.
    """
    service = _get_service()
    reviews = service.get_pending_reviews(limit=limit)

    if not reviews:
        return "No pending reviews found."

    results = [
        {
            "patient1": r.get("patient1", {}),
            "patient2": r.get("patient2", {}),
            "match_score": r.get("match", {}).get("score"),
            "confidence": r.get("match", {}).get("confidence"),
        }
        for r in reviews
    ]

    return json.dumps({"pending_reviews": len(results), "reviews": results}, indent=2)


@mcp.tool()
def get_service_statistics() -> str:
    """Get statistics about the Patient Matching Service.

    Returns:
        JSON with total patients, EMPI records, and pending review counts.
    """
    service = _get_service()
    stats = service.get_stats()

    return json.dumps({"service": "Patient Matching Service", "statistics": stats}, indent=2)


@mcp.tool()
def search_patients(
    name: str | None = None,
    birth_date: str | None = None,
    identifier_value: str | None = None,
    limit: int = 20,
) -> str:
    """Search for patients by name, birth date, or identifier.

    At least one search criterion should be provided.

    Args:
        name: Patient name to search for (partial match supported).
        birth_date: Patient birth date in YYYY-MM-DD format.
        identifier_value: Identifier value (MRN, SSN, etc.) to search for.
        limit: Maximum number of results. Default 20.

    Returns:
        JSON list of matching patients with id, name, birth_date, gender.
    """
    service = _get_service()
    all_patients = service.db.get_all_patients(limit=500)

    results = []
    for patient in all_patients:
        match = True

        if name:
            patient_name = patient.name.full_name.lower() if patient.name else ""
            if name.lower() not in patient_name:
                match = False

        if birth_date and match:
            if patient.birth_date:
                if patient.birth_date.isoformat() != birth_date:
                    match = False
            else:
                match = False

        if identifier_value and match:
            found_id = any(
                identifier_value.lower() in ident.value.lower()
                for ident in patient.identifiers
            )
            if not found_id:
                match = False

        if match:
            results.append(
                {
                    "id": patient.id,
                    "name": patient.name.full_name if patient.name else None,
                    "birth_date": patient.birth_date.isoformat() if patient.birth_date else None,
                    "gender": patient.gender.value if patient.gender else None,
                }
            )

        if len(results) >= limit:
            break

    if not results:
        return "No patients found matching the search criteria."

    return json.dumps({"patients_found": len(results), "patients": results}, indent=2)


# ==================== MCP Resources ====================


@mcp.resource("patient://{patient_id}")
def patient_resource(patient_id: str) -> str:
    """Retrieve a patient record as an MCP resource.

    Provides the full demographic data for a given patient ID so that
    MCP clients can read patient context without explicitly calling a tool.
    """
    return get_patient_details(patient_id)


@mcp.resource("stats://service")
def service_stats_resource() -> str:
    """Current service statistics (patients, EMPI records, pending reviews)."""
    return get_service_statistics()


@mcp.resource("reviews://pending")
def pending_reviews_resource() -> str:
    """List of patient matches currently awaiting human review."""
    return get_pending_reviews(limit=50)


# ==================== MCP Prompts ====================


@mcp.prompt()
def analyze_patient(patient_id: str) -> str:
    """Comprehensive analysis workflow for a single patient.

    Retrieves patient details, finds all potential matches, and asks the
    model to summarise findings with recommendations.
    """
    return f"""Perform a comprehensive patient matching analysis:

1. First, use the get_patient_details tool to retrieve full demographics for patient {patient_id}.
2. Then use find_patient_matches with search_entire_database=true to find all potential duplicates.
3. For any matches with confidence "human_review" or "auto_merge", use compare_two_patients
   to get a detailed score breakdown.
4. Summarise your findings:
   - Patient overview (name, DOB, identifiers)
   - Number of potential matches found
   - For each significant match: score breakdown and your recommendation
   - Overall assessment of duplicate risk for this patient
"""


@mcp.prompt()
def review_pending_matches() -> str:
    """Guided workflow for reviewing all pending match decisions."""
    return """Review all patient matches that are pending human review:

1. Use get_pending_reviews to retrieve the current review queue.
2. For each pending match, use compare_two_patients to get the detailed score breakdown.
3. For each pair, provide:
   - Both patients' names, DOB, and key identifiers
   - Score breakdown (deterministic, name similarity, address similarity)
   - Your recommendation: approve (merge) or reject, with reasoning
4. Summarise: how many to approve, how many to reject, and any that need
   additional investigation.
"""


@mcp.prompt()
def batch_matching_report(min_score: str = "0.3") -> str:
    """Run batch matching and generate a summary report."""
    return f"""Run a comprehensive batch matching operation and report results:

1. First use get_service_statistics for a baseline snapshot.
2. Run run_batch_matching with min_score={min_score}.
3. After completion, use get_service_statistics again to see updated counts.
4. Use get_pending_reviews to list any new matches needing human review.
5. Produce a report with:
   - Total patients processed
   - Matches found (auto-merge vs human-review vs no-match)
   - Top duplicate clusters (if any auto-merge matches)
   - Items queued for human review
"""


# ==================== Entry Point ====================

def main():
    """Run the MCP server (stdio transport)."""
    mcp.run()


if __name__ == "__main__":
    main()
