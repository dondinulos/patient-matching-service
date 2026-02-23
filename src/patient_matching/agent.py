"""
Patient Matching AI Agent

Implements the Patient Matching Service as a Microsoft Foundry Agent
using the Microsoft Agent Framework. The agent exposes patient matching
capabilities as tools that can be called conversationally.

Features:
- Find matches for a patient against all records in the database
- Get patient details by ID
- Run batch matching operations
- Approve or reject match decisions
- Get service statistics

Usage:
    from patient_matching.agent import create_patient_matching_agent
    
    async with create_patient_matching_agent() as agent:
        result = await agent.run("Find all potential matches for patient P123")
        print(result.text)
"""

import os
import logging
import asyncio
from typing import Annotated, Optional, List, Dict, Any
from datetime import date, datetime
import json

# Apply nest_asyncio to allow nested event loops (required for gremlin_python with agent framework)
import nest_asyncio
nest_asyncio.apply()

from .service import PatientMatchingService
from .matching import MatchWeights
from .models import (
    Patient, MatchResult, MatchConfidence, HumanName, 
    Gender, Identifier, IdentifierType, Address
)

logger = logging.getLogger(__name__)

# Global service instance for tool access
_service: Optional[PatientMatchingService] = None


def get_service() -> PatientMatchingService:
    """Get or create the global patient matching service instance."""
    global _service
    if _service is None:
        # Initialize service with configuration from environment
        db_type = os.getenv("PM_DB_TYPE", "cosmos")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        use_azure = bool(azure_endpoint)
        # Enable embeddings/LLM if Azure OpenAI is configured or explicit env vars set
        use_embeddings = os.getenv("USE_EMBEDDINGS", str(use_azure)).lower() == "true"
        use_llm = os.getenv("USE_LLM", str(use_azure)).lower() == "true"
        
        _service = PatientMatchingService(
            # Neo4j configuration
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
            # Cosmos DB configuration
            cosmos_endpoint=os.getenv("COSMOS_GREMLIN_ENDPOINT"),
            cosmos_database=os.getenv("COSMOS_DATABASE", "PatientMatching"),
            cosmos_container=os.getenv("COSMOS_CONTAINER", "PatientGraph"),
            cosmos_key=os.getenv("COSMOS_KEY"),
            # Database selection
            db_type=db_type,
            # Matching configuration
            use_embeddings=use_embeddings,
            use_llm=use_llm,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            # Azure OpenAI configuration
            use_azure_openai=use_azure,
            azure_openai_endpoint=azure_endpoint,
            azure_openai_key=azure_key,
            azure_openai_embedding_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"),
            azure_openai_gpt_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        )
        _service.initialize()
        logger.info(f"Patient Matching Service initialized with {db_type} backend"
                    f" (embeddings={use_embeddings}, llm={use_llm}, azure={use_azure})")
    
    return _service


# ==================== Agent Tools ====================

def find_patient_matches(
    patient_id: Annotated[str, "The unique ID of the patient to find matches for"],
    min_score: Annotated[float, "Minimum match score threshold (0.0-1.0). Default 0.3"] = 0.3,
    max_results: Annotated[int, "Maximum number of matches to return. Default 20"] = 20,
    search_entire_database: Annotated[bool, "If True, search against ALL patients. If False, use graph-based candidate retrieval. Default False"] = False
) -> str:
    """
    Find potential patient matches for a given patient ID.
    
    This tool searches for patients that may be duplicates or the same person
    as the specified patient. It uses a combination of deterministic matching
    (exact identifier matches) and probabilistic matching (name similarity,
    address similarity, etc.) to compute match scores.
    
    Returns a list of potential matches with their scores and confidence levels.
    """
    try:
        service = get_service()
        
        if search_entire_database:
            matches = service.find_all_matches_for_patient(
                patient_id=patient_id,
                min_score=min_score,
                limit=max_results
            )
        else:
            matches = service.find_matches_for_patient(
                patient_id=patient_id,
                min_score=min_score
            )[:max_results]
        
        if not matches:
            return f"No matches found for patient {patient_id} with minimum score {min_score}."
        
        results = []
        for match in matches:
            result = {
                "matched_patient_id": match.patient2_id,
                "score": round(match.score, 3),
                "confidence": match.confidence.value,
                "deterministic_score": round(match.deterministic_score, 3),
                "name_similarity": round(match.name_similarity, 3),
                "address_similarity": round(match.address_similarity, 3),
                "embedding_similarity": round(match.embedding_similarity, 3),
                "shared_identifiers": match.shared_identifiers
            }
            # Include LLM analysis if present
            if match.match_details.get("llm_analysis"):
                result["llm_analysis"] = match.match_details["llm_analysis"]
            results.append(result)
        
        return json.dumps({
            "patient_id": patient_id,
            "matches_found": len(results),
            "matches": results
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Error finding matches for patient {patient_id}: {e}")
        return f"Error finding matches: {str(e)}"


def get_patient_details(
    patient_id: Annotated[str, "The unique ID of the patient to retrieve"]
) -> str:
    """
    Get detailed information about a specific patient.
    
    Returns the patient's demographic information including name, date of birth,
    gender, identifiers, addresses, and contact information.
    """
    try:
        service = get_service()
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
            "addresses": [
                addr.full_address for addr in patient.addresses
            ],
            "contact_points": [
                {"system": c.system.value if hasattr(c.system, 'value') else c.system, "value": c.value}
                for c in patient.contact_points
            ],
            "empi_id": patient.empi_id
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Error getting patient {patient_id}: {e}")
        return f"Error retrieving patient: {str(e)}"


def compare_two_patients(
    patient1_id: Annotated[str, "The ID of the first patient"],
    patient2_id: Annotated[str, "The ID of the second patient"]
) -> str:
    """
    Compare two specific patients and compute a detailed match score.
    
    This tool performs a comprehensive comparison between two patients using
    deterministic matching, probabilistic matching, and optionally AI-enhanced
    analysis. Returns detailed scoring breakdown and match recommendation.
    """
    try:
        service = get_service()
        
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
                "birth_date": patient1.birth_date.isoformat() if patient1.birth_date else None
            },
            "patient2": {
                "id": patient2_id,
                "name": patient2.name.full_name if patient2.name else None,
                "birth_date": patient2.birth_date.isoformat() if patient2.birth_date else None
            },
            "match_result": {
                "overall_score": round(result.score, 3),
                "confidence": result.confidence.value,
                "deterministic_score": round(result.deterministic_score, 3),
                "name_similarity": round(result.name_similarity, 3),
                "address_similarity": round(result.address_similarity, 3),
                "embedding_similarity": round(result.embedding_similarity, 3),
                "shared_identifiers": result.shared_identifiers,
                "recommendation": _get_recommendation(result.confidence)
            }
        }
        
        return json.dumps(comparison, indent=2)
        
    except Exception as e:
        logger.error(f"Error comparing patients: {e}")
        return f"Error comparing patients: {str(e)}"


def _get_recommendation(confidence: MatchConfidence) -> str:
    """Get a human-readable recommendation based on confidence level."""
    if confidence == MatchConfidence.AUTO_MERGE:
        return "High confidence match - recommended for automatic merge"
    elif confidence == MatchConfidence.HUMAN_REVIEW:
        return "Medium confidence - requires human review before merge"
    else:
        return "Low confidence - likely different patients"


def run_batch_matching(
    min_score: Annotated[float, "Minimum match score threshold. Default 0.3"] = 0.3
) -> str:
    """
    Run matching for all patients in the database.
    
    This is a comprehensive matching operation that finds all potential
    duplicate patient records. Use for initial MPI setup or periodic
    comprehensive matching.
    
    WARNING: This can take a long time for large databases.
    """
    try:
        service = get_service()
        stats = service.run_global_matching(min_score=min_score)
        
        return json.dumps({
            "operation": "global_matching",
            "status": "completed",
            "statistics": stats
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Error running batch matching: {e}")
        return f"Error running batch matching: {str(e)}"


def approve_patient_match(
    patient1_id: Annotated[str, "The ID of the first patient in the match"],
    patient2_id: Annotated[str, "The ID of the second patient in the match"],
    reviewed_by: Annotated[str, "The identifier of the person approving the match"],
    notes: Annotated[str, "Optional notes about the approval decision"] = None
) -> str:
    """
    Approve a potential match and merge two patients into a single EMPI record.
    
    This creates or updates an Enterprise Master Patient Index (EMPI) record
    linking the two patients as the same person.
    """
    try:
        service = get_service()
        empi_id = service.approve_match(
            patient1_id=patient1_id,
            patient2_id=patient2_id,
            reviewed_by=reviewed_by,
            notes=notes
        )
        
        return json.dumps({
            "action": "match_approved",
            "patient1_id": patient1_id,
            "patient2_id": patient2_id,
            "empi_id": empi_id,
            "reviewed_by": reviewed_by,
            "message": f"Patients successfully merged into EMPI record {empi_id}"
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Error approving match: {e}")
        return f"Error approving match: {str(e)}"


def reject_patient_match(
    patient1_id: Annotated[str, "The ID of the first patient in the match"],
    patient2_id: Annotated[str, "The ID of the second patient in the match"],
    reviewed_by: Annotated[str, "The identifier of the person rejecting the match"],
    reason: Annotated[str, "The reason for rejecting the match"]
) -> str:
    """
    Reject a potential match between two patients.
    
    This marks the match as reviewed and rejected, preventing future
    automatic merge suggestions for this pair.
    """
    try:
        service = get_service()
        service.reject_match(
            patient1_id=patient1_id,
            patient2_id=patient2_id,
            reviewed_by=reviewed_by,
            reason=reason
        )
        
        return json.dumps({
            "action": "match_rejected",
            "patient1_id": patient1_id,
            "patient2_id": patient2_id,
            "reviewed_by": reviewed_by,
            "reason": reason,
            "message": "Match rejection recorded"
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Error rejecting match: {e}")
        return f"Error rejecting match: {str(e)}"


def get_pending_reviews(
    limit: Annotated[int, "Maximum number of pending reviews to return. Default 20"] = 20
) -> str:
    """
    Get a list of patient matches that require human review.
    
    These are matches with medium confidence scores that need a human
    to decide whether to approve or reject the match.
    """
    try:
        service = get_service()
        reviews = service.get_pending_reviews(limit=limit)
        
        if not reviews:
            return "No pending reviews found."
        
        results = []
        for review in reviews:
            results.append({
                "patient1": review.get("patient1", {}),
                "patient2": review.get("patient2", {}),
                "match_score": review.get("match", {}).get("score"),
                "confidence": review.get("match", {}).get("confidence")
            })
        
        return json.dumps({
            "pending_reviews": len(results),
            "reviews": results
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Error getting pending reviews: {e}")
        return f"Error getting pending reviews: {str(e)}"


def get_service_statistics() -> str:
    """
    Get statistics about the Patient Matching Service.
    
    Returns counts of total patients, EMPI records, and pending reviews.
    """
    try:
        service = get_service()
        stats = service.get_stats()
        
        return json.dumps({
            "service": "Patient Matching Service",
            "statistics": stats
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return f"Error getting statistics: {str(e)}"


def search_patients(
    name: Annotated[str, "Patient name to search for (partial match supported)"] = None,
    birth_date: Annotated[str, "Patient birth date in YYYY-MM-DD format"] = None,
    identifier_value: Annotated[str, "Identifier value (MRN, SSN, etc.) to search for"] = None,
    limit: Annotated[int, "Maximum number of results. Default 20"] = 20
) -> str:
    """
    Search for patients by name, birth date, or identifier.
    
    Returns a list of patients matching the search criteria.
    """
    try:
        service = get_service()
        
        # Get all patients and filter (simple implementation)
        # In production, this would use database-level search
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
                found_id = False
                for identifier in patient.identifiers:
                    if identifier_value.lower() in identifier.value.lower():
                        found_id = True
                        break
                if not found_id:
                    match = False
            
            if match:
                results.append({
                    "id": patient.id,
                    "name": patient.name.full_name if patient.name else None,
                    "birth_date": patient.birth_date.isoformat() if patient.birth_date else None,
                    "gender": patient.gender.value if patient.gender else None
                })
            
            if len(results) >= limit:
                break
        
        if not results:
            return "No patients found matching the search criteria."
        
        return json.dumps({
            "patients_found": len(results),
            "patients": results
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Error searching patients: {e}")
        return f"Error searching patients: {str(e)}"


# ==================== Agent Factory ====================

def get_patient_matching_tools() -> list:
    """Get the list of tools for the Patient Matching Agent."""
    return [
        find_patient_matches,
        get_patient_details,
        compare_two_patients,
        run_batch_matching,
        approve_patient_match,
        reject_patient_match,
        get_pending_reviews,
        get_service_statistics,
        search_patients
    ]


class RateLimitRetryMiddleware:
    """Chat middleware that retries on 429 (Too Many Requests) errors with exponential backoff."""

    def __init__(self, max_retries: int = 5, base_delay: float = 2.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    async def process(self, context, call_next):
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                await call_next()
                return  # Success
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "too_many_requests" in error_str or "Too Many Requests" in error_str:
                    last_error = e
                    if attempt < self.max_retries:
                        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                        logger.warning(
                            f"Rate limited (429). Retrying in {delay:.1f}s "
                            f"(attempt {attempt + 1}/{self.max_retries})..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Rate limited (429). All {self.max_retries} retries exhausted.")
                        raise
                else:
                    raise  # Non-rate-limit error, don't retry

    # Mark as chat middleware so the framework routes it correctly
    _middleware_type = None  # Will be set below

# Set the middleware type marker after imports are available
def _mark_middleware():
    try:
        from agent_framework._middleware import MiddlewareType
        RateLimitRetryMiddleware._middleware_type = MiddlewareType.CHAT
    except (ImportError, AttributeError):
        pass  # Framework version may not need this marker

_mark_middleware()


AGENT_INSTRUCTIONS = """You are a Patient Matching Assistant that helps healthcare administrators 
manage patient identity matching and Master Patient Index (MPI) operations.

Your capabilities include:
1. Finding potential duplicate patient records by comparing against all patients in the database
2. Getting detailed patient information
3. Comparing two specific patients to determine if they're the same person
4. Running batch matching operations across the entire patient database
5. Approving or rejecting match decisions
6. Viewing pending match reviews
7. Getting service statistics

When users ask about patient matching:
- Use find_patient_matches to search for potential duplicates
- Use compare_two_patients for detailed comparison between specific patients
- Use search_patients to find patients by name, birth date, or identifier

Always explain the match confidence levels:
- AUTO_MERGE (score >= 0.85): High confidence - these are very likely the same person
- HUMAN_REVIEW (score 0.65-0.85): Medium confidence - requires human verification
- NO_MATCH (score < 0.65): Low confidence - likely different people

When presenting match results, ALWAYS show the full score breakdown for each match including:
- Overall score and confidence level
- Deterministic score (identifier matches)
- Name similarity score
- Address similarity score
- Embedding similarity score (AI semantic matching)
- LLM analysis (if available)
- Shared identifiers

Present each match with all these details in a structured format. Do not summarize or abbreviate
the score details even when multiple matches have similar scores.

Be helpful and explain the matching results clearly to help users make informed decisions
about patient identity management."""


def create_patient_matching_agent(
    endpoint: str = None,
    deployment_name: str = None,
    agent_name: str = "PatientMatchingAgent"
):
    """
    Create a Patient Matching Agent using Microsoft Agent Framework.
    
    The agent provides conversational access to patient matching capabilities
    including finding duplicates, comparing patients, and managing match decisions.
    
    Args:
        endpoint: Azure OpenAI endpoint (or AZURE_OPENAI_ENDPOINT env var)
        deployment_name: Model deployment name (or AZURE_OPENAI_DEPLOYMENT env var)
        agent_name: Name for the agent
    
    Returns:
        Configured ChatAgent
    
    Usage:
        agent = create_patient_matching_agent()
        result = await agent.run("Find matches for patient P123")
        print(result.text)
    """
    from agent_framework import ChatAgent
    from agent_framework.azure import AzureOpenAIChatClient
    from azure.identity import DefaultAzureCredential
    
    aoai_endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    
    if not aoai_endpoint:
        raise ValueError(
            "Azure OpenAI endpoint required. "
            "Set AZURE_OPENAI_ENDPOINT environment variable or pass endpoint parameter."
        )
    
    # Use DefaultAzureCredential for both local (CLI) and deployed (managed identity) scenarios
    credential = DefaultAzureCredential()
    
    # Create the chat client
    chat_client = AzureOpenAIChatClient(
        endpoint=aoai_endpoint,
        deployment_name=deployment,
        credential=credential,
        middleware=[RateLimitRetryMiddleware()]
    )
    
    # Create agent using as_agent() pattern
    return chat_client.as_agent(
        name=agent_name,
        instructions=AGENT_INSTRUCTIONS,
        tools=get_patient_matching_tools()
    )


def _build_openapi_tools(api_base_url: str) -> list:
    """Build OpenAPI tool definitions for the Foundry Agent Service.
    
    These tools point to the Container App REST API so the Foundry service
    can execute them server-side without a local Python client.
    """
    from azure.ai.projects.models import (
        OpenApiAgentTool,
        OpenApiFunctionDefinition,
        OpenApiAnonymousAuthDetails,
    )

    # Strip trailing slash
    base = api_base_url.rstrip("/")

    # Build an OpenAPI spec covering the patient matching API endpoints
    openapi_spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Patient Matching Service",
            "version": "1.0.0",
            "description": "MPI with deterministic, probabilistic, and AI-enhanced matching"
        },
        "servers": [{"url": base}],
        "paths": {
            "/patients/search": {
                "get": {
                    "operationId": "searchPatients",
                    "summary": "Search for patients by name, birth date, or identifier",
                    "parameters": [
                        {"name": "name", "in": "query", "schema": {"type": "string"}, "description": "Patient name to search for (partial match)"},
                        {"name": "birth_date", "in": "query", "schema": {"type": "string"}, "description": "Birth date in YYYY-MM-DD format"},
                        {"name": "identifier_value", "in": "query", "schema": {"type": "string"}, "description": "Identifier value (MRN, SSN, etc.)"},
                        {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 20}, "description": "Max results"}
                    ],
                    "responses": {"200": {"description": "Search results", "content": {"application/json": {"schema": {"type": "object"}}}}}
                }
            },
            "/patients/{patient_id}": {
                "get": {
                    "operationId": "getPatientDetails",
                    "summary": "Get detailed information about a specific patient",
                    "parameters": [
                        {"name": "patient_id", "in": "path", "required": True, "schema": {"type": "string"}, "description": "The patient ID"}
                    ],
                    "responses": {"200": {"description": "Patient details", "content": {"application/json": {"schema": {"type": "object"}}}}}
                }
            },
            "/match/all": {
                "post": {
                    "operationId": "findPatientMatches",
                    "summary": "Find potential duplicate matches for a patient against all records",
                    "parameters": [
                        {"name": "patient_id", "in": "query", "required": True, "schema": {"type": "string"}, "description": "Patient ID to find matches for"},
                        {"name": "min_score", "in": "query", "schema": {"type": "number", "default": 0.3}, "description": "Minimum match score (0.0-1.0)"},
                        {"name": "max_results", "in": "query", "schema": {"type": "integer", "default": 20}, "description": "Maximum matches to return"}
                    ],
                    "responses": {"200": {"description": "Match results", "content": {"application/json": {"schema": {"type": "array"}}}}}
                }
            },
            "/match/compare": {
                "post": {
                    "operationId": "compareTwoPatients",
                    "summary": "Compare two specific patients for a detailed match score",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["patient1_id", "patient2_id"],
                                    "properties": {
                                        "patient1_id": {"type": "string", "description": "First patient ID"},
                                        "patient2_id": {"type": "string", "description": "Second patient ID"}
                                    }
                                }
                            }
                        }
                    },
                    "responses": {"200": {"description": "Comparison result", "content": {"application/json": {"schema": {"type": "object"}}}}}
                }
            },
            "/reviews/pending": {
                "get": {
                    "operationId": "getPendingReviews",
                    "summary": "Get patient matches requiring human review",
                    "parameters": [
                        {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 20}, "description": "Max reviews to return"}
                    ],
                    "responses": {"200": {"description": "Pending reviews", "content": {"application/json": {"schema": {"type": "object"}}}}}
                }
            },
            "/reviews/decision": {
                "post": {
                    "operationId": "submitReviewDecision",
                    "summary": "Approve or reject a patient match",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["patient1_id", "patient2_id", "decision", "reviewed_by"],
                                    "properties": {
                                        "patient1_id": {"type": "string"},
                                        "patient2_id": {"type": "string"},
                                        "decision": {"type": "string", "enum": ["approve", "reject"]},
                                        "reviewed_by": {"type": "string"},
                                        "reason": {"type": "string"}
                                    }
                                }
                            }
                        }
                    },
                    "responses": {"200": {"description": "Decision result", "content": {"application/json": {"schema": {"type": "object"}}}}}
                }
            },
            "/stats": {
                "get": {
                    "operationId": "getServiceStatistics",
                    "summary": "Get patient matching service statistics",
                    "responses": {"200": {"description": "Service statistics", "content": {"application/json": {"schema": {"type": "object"}}}}}
                }
            }
        }
    }

    tool = OpenApiAgentTool(
        openapi=OpenApiFunctionDefinition(
            name="patient_matching_api",
            description="Patient Matching Service API for searching patients, finding duplicates, comparing records, and managing match reviews",
            spec=openapi_spec,
            auth=OpenApiAnonymousAuthDetails(),
        )
    )

    return [tool]


def create_foundry_agent(
    project_endpoint: str = None,
    deployment_name: str = None,
    agent_name: str = "PatientMatchingAgent",
    api_base_url: str = None
):
    """
    Create a Patient Matching Agent deployed to Azure AI Foundry Agent Service.
    
    When api_base_url is provided, uses OpenAPI tools that call the Container App
    API server-side (required for Foundry playground). Otherwise falls back to
    local Python function tools (for CLI usage).
    
    Args:
        project_endpoint: Foundry project endpoint (or AZURE_AI_FOUNDRY_PROJECT_ENDPOINT env var)
        deployment_name: Model deployment name (or AZURE_OPENAI_DEPLOYMENT env var)
        agent_name: Name for the agent
        api_base_url: Base URL for the Container App API (or PM_API_BASE_URL env var)
    
    Returns:
        Async context manager yielding a configured Foundry ChatAgent
    """
    from agent_framework.azure import AzureAIClient
    from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential

    project_ep = project_endpoint or os.getenv("AZURE_AI_FOUNDRY_PROJECT_ENDPOINT")
    deployment = deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    base_url = api_base_url or os.getenv("PM_API_BASE_URL")

    if not project_ep:
        raise ValueError(
            "Foundry project endpoint required. "
            "Set AZURE_AI_FOUNDRY_PROJECT_ENDPOINT environment variable or pass project_endpoint parameter. "
            "Format: https://<resource>.services.ai.azure.com/api/projects/<project-name>"
        )

    credential = AsyncDefaultAzureCredential()

    client = AzureAIClient(
        project_endpoint=project_ep,
        model_deployment_name=deployment,
        credential=credential,
        middleware=[RateLimitRetryMiddleware()],
    )

    # Use OpenAPI tools (server-side execution) when API URL is available,
    # otherwise fall back to local Python function tools
    if base_url:
        tools = _build_openapi_tools(base_url)
        logger.info(f"Using OpenAPI tools pointing to {base_url}")
    else:
        tools = get_patient_matching_tools()
        logger.info("Using local Python function tools")

    return client.as_agent(
        name=agent_name,
        instructions=AGENT_INSTRUCTIONS,
        tools=tools,
    )


# ==================== CLI Entry Point ====================

async def main():
    """Run the Patient Matching Agent in interactive mode.
    
    Uses Foundry Agent Service if AZURE_AI_FOUNDRY_PROJECT_ENDPOINT is set,
    otherwise falls back to direct Azure OpenAI.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Patient Matching Agent")
    parser.add_argument(
        "--foundry", action="store_true",
        help="Use Azure AI Foundry Agent Service (requires AZURE_AI_FOUNDRY_PROJECT_ENDPOINT)"
    )
    args = parser.parse_args()

    use_foundry = args.foundry or os.getenv("AZURE_AI_FOUNDRY_PROJECT_ENDPOINT")

    print("Patient Matching Agent")
    print("=" * 50)
    if use_foundry:
        print("Mode: Azure AI Foundry Agent Service")
    else:
        print("Mode: Direct Azure OpenAI")
    print("Type your questions about patient matching, or 'quit' to exit.")
    print()

    if use_foundry:
        async with create_foundry_agent() as agent:
            await _interactive_loop(agent)
    else:
        agent = create_patient_matching_agent()
        await _interactive_loop(agent)


async def _interactive_loop(agent):
    """Run the interactive conversation loop."""
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            print("Agent: ", end="", flush=True)
            result = await agent.run(user_input)
            print(result.text)
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
