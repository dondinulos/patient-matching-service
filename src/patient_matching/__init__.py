"""
Patient Matching Service

A comprehensive MPI (Master Patient Index) solution using:
- Graph Database (Neo4j or Cosmos DB Gremlin) for patient data and relationships
- Deterministic matching (exact identifiers)
- Probabilistic matching (similarity algorithms)
- OpenAI embeddings for enhanced matching
- AI Foundry Agent for conversational patient matching

Components:
- PatientMatchingService: Main service for patient matching operations
- PatientMatcher: Core matching algorithms
- create_patient_matching_agent: AI agent for conversational access
"""

__version__ = "0.1.0"

from .models import (
    Patient, MatchResult, MatchConfidence, EmpiRecord,
    HumanName, Address, ContactPoint, Identifier,
    Gender, IdentifierType, ContactPointSystem
)
from .service import PatientMatchingService, MatchingPipeline
from .matching import PatientMatcher, MatchWeights

# Agent imports (optional - requires agent-framework-azure-ai)
try:
    from .agent import create_patient_matching_agent, get_patient_matching_tools
    __all__ = [
        # Core models
        "Patient", "MatchResult", "MatchConfidence", "EmpiRecord",
        "HumanName", "Address", "ContactPoint", "Identifier",
        "Gender", "IdentifierType", "ContactPointSystem",
        # Services
        "PatientMatchingService", "MatchingPipeline",
        "PatientMatcher", "MatchWeights",
        # Agent
        "create_patient_matching_agent", "get_patient_matching_tools"
    ]
except ImportError:
    # Agent framework not installed
    __all__ = [
        # Core models
        "Patient", "MatchResult", "MatchConfidence", "EmpiRecord",
        "HumanName", "Address", "ContactPoint", "Identifier",
        "Gender", "IdentifierType", "ContactPointSystem",
        # Services
        "PatientMatchingService", "MatchingPipeline",
        "PatientMatcher", "MatchWeights"
    ]
