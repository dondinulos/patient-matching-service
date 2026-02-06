"""
REST API for Patient Matching Service

Provides HTTP endpoints for:
- Patient management
- Matching operations
- EMPI Record management
- Human review workflow
"""

from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import date, datetime
from enum import Enum
import logging
import os

from .service import PatientMatchingService, MatchingPipeline
from .models import (
    Patient, MatchResult, MatchConfidence, Gender,
    HumanName, Address, ContactPoint, Identifier,
    IdentifierType, ContactPointSystem
)
from .matching import MatchWeights

logger = logging.getLogger(__name__)

# ========== Pydantic Models for API ==========

class IdentifierRequest(BaseModel):
    value: str
    type: str = "OTHER"
    system: Optional[str] = None
    assigner: Optional[str] = None

class AddressRequest(BaseModel):
    line: List[str] = []
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None
    country: Optional[str] = None

class ContactPointRequest(BaseModel):
    system: str = "phone"
    value: str
    use: Optional[str] = None

class NameRequest(BaseModel):
    family: Optional[str] = None
    given: List[str] = []
    prefix: List[str] = []
    suffix: List[str] = []

class PatientRequest(BaseModel):
    source_id: str
    source_system: str = "API"
    name: Optional[NameRequest] = None
    birth_date: Optional[date] = None
    gender: str = "unknown"
    identifiers: List[IdentifierRequest] = []
    addresses: List[AddressRequest] = []
    contact_points: List[ContactPointRequest] = []

class PatientResponse(BaseModel):
    id: str
    source_id: str
    source_system: str
    name: Optional[Dict[str, Any]] = None
    birth_date: Optional[date] = None
    gender: str
    identifiers: List[Dict[str, Any]] = []
    addresses: List[Dict[str, Any]] = []
    contact_points: List[Dict[str, Any]] = []
    empi_id: Optional[str] = None  # Enterprise Master Patient Index ID
    created_at: datetime
    updated_at: datetime

class MatchResultResponse(BaseModel):
    patient1_id: str
    patient2_id: str
    score: float
    confidence: str
    deterministic_score: float
    name_similarity: float
    address_similarity: float
    embedding_similarity: float
    shared_identifiers: List[str]
    match_details: Dict[str, Any] = {}

class MatchRequest(BaseModel):
    patient_id: str
    min_score: float = 0.3
    use_embeddings: bool = False

class ReviewDecisionRequest(BaseModel):
    patient1_id: str
    patient2_id: str
    decision: str  # "approve" or "reject"
    reviewed_by: str
    reason: Optional[str] = None

class BatchLoadRequest(BaseModel):
    directory: str
    limit: Optional[int] = None
    source_system: str = "FHIR"

class StatsResponse(BaseModel):
    total_patients: int
    total_empi_records: int
    pending_reviews: int

class MatchWeightsRequest(BaseModel):
    enterprise_id: float = 1.0
    mrn: float = 0.8
    ssn: float = 0.9
    dob_exact: float = 0.35
    phone_exact: float = 0.3
    email_exact: float = 0.3
    deterministic_weight: float = 0.4
    name_weight: float = 0.35
    address_weight: float = 0.15
    embedding_weight: float = 0.1
    auto_merge_threshold: float = 0.85
    human_review_threshold: float = 0.65

# ========== FastAPI App ==========

app = FastAPI(
    title="Patient Matching Service",
    description="MPI (Master Patient Index) with deterministic, probabilistic, and AI-enhanced matching",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service instance (initialized on startup)
_service: Optional[PatientMatchingService] = None

def get_service() -> PatientMatchingService:
    """Dependency to get the service instance"""
    global _service
    if _service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return _service

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    global _service
    
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    use_embeddings = os.getenv("USE_EMBEDDINGS", "false").lower() == "true"
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    _service = PatientMatchingService(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        use_embeddings=use_embeddings,
        openai_api_key=openai_api_key
    )
    
    try:
        _service.initialize()
        logger.info("Patient Matching Service started")
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        _service = None

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global _service
    if _service:
        _service.close()
        _service = None
    logger.info("Patient Matching Service stopped")

# ========== Health Check ==========

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "patient-matching"}

@app.get("/ready")
async def readiness_check(service: PatientMatchingService = Depends(get_service)):
    """Readiness check - verifies database connection"""
    try:
        stats = service.get_stats()
        return {
            "status": "ready",
            "database": "connected",
            "patients": stats["total_patients"]
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Not ready: {str(e)}")

# ========== Patient Endpoints ==========

@app.post("/patients", response_model=PatientResponse, tags=["Patients"])
async def create_patient(
    request: PatientRequest,
    service: PatientMatchingService = Depends(get_service)
):
    """Create a new patient record"""
    # Convert request to Patient model
    name = None
    if request.name:
        name = HumanName(
            family=request.name.family,
            given=request.name.given,
            prefix=request.name.prefix,
            suffix=request.name.suffix
        )
    
    identifiers = []
    for i in request.identifiers:
        try:
            ident_type = IdentifierType[i.type.upper()]
        except KeyError:
            ident_type = IdentifierType.OTHER
        identifiers.append(Identifier(
            value=i.value,
            type=ident_type,
            system=i.system,
            assigner=i.assigner
        ))
    
    addresses = [
        Address(
            line=a.line,
            city=a.city,
            state=a.state,
            postal_code=a.postal_code,
            country=a.country
        )
        for a in request.addresses
    ]
    
    contact_points = []
    for c in request.contact_points:
        try:
            system = ContactPointSystem[c.system.upper()]
        except KeyError:
            system = ContactPointSystem.OTHER
        contact_points.append(ContactPoint(
            system=system,
            value=c.value,
            use=c.use
        ))
    
    try:
        gender = Gender[request.gender.upper()]
    except KeyError:
        gender = Gender.UNKNOWN
    
    import uuid
    patient = Patient(
        id=str(uuid.uuid4()),
        source_id=request.source_id,
        source_system=request.source_system,
        name=name,
        birth_date=request.birth_date,
        gender=gender,
        identifiers=identifiers,
        addresses=addresses,
        contact_points=contact_points
    )
    
    patient_id = service.load_patient(patient)
    patient.id = patient_id
    
    return _patient_to_response(patient)

@app.get("/patients/{patient_id}", response_model=PatientResponse, tags=["Patients"])
async def get_patient(
    patient_id: str,
    service: PatientMatchingService = Depends(get_service)
):
    """Get a patient by ID"""
    patient = service.db.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    return _patient_to_response(patient)

@app.post("/patients/batch", tags=["Patients"])
async def load_patients_batch(
    request: BatchLoadRequest,
    background_tasks: BackgroundTasks,
    service: PatientMatchingService = Depends(get_service)
):
    """Load patients from FHIR files in a directory"""
    # Run in background for large loads
    def load_task():
        count = service.load_patients_from_fhir(
            request.directory,
            limit=request.limit,
            source_system=request.source_system
        )
        logger.info(f"Batch load complete: {count} patients")
    
    background_tasks.add_task(load_task)
    
    return {
        "status": "started",
        "message": f"Loading patients from {request.directory}"
    }

# ========== Matching Endpoints ==========

@app.post("/match", response_model=List[MatchResultResponse], tags=["Matching"])
async def find_matches(
    request: MatchRequest,
    service: PatientMatchingService = Depends(get_service)
):
    """Find potential matches for a patient"""
    results = service.find_matches_for_patient(
        request.patient_id,
        min_score=request.min_score
    )
    
    return [_match_to_response(r) for r in results]


@app.post("/match/all", response_model=List[MatchResultResponse], tags=["Matching"])
async def find_all_matches(
    patient_id: str,
    min_score: float = Query(default=0.3, ge=0.0, le=1.0),
    max_results: int = Query(default=50, le=500),
    service: PatientMatchingService = Depends(get_service)
):
    """
    Find potential matches for a patient against ALL patients in the database.
    
    This performs a comprehensive search through the entire patient database,
    comparing the specified patient against every other patient record.
    Use for thorough duplicate detection.
    """
    results = service.find_all_matches_for_patient(
        patient_id=patient_id,
        min_score=min_score,
        limit=max_results
    )
    
    return [_match_to_response(r) for r in results]


@app.post("/match/global", tags=["Matching"])
async def run_global_matching(
    min_score: float = Query(default=0.3, ge=0.0, le=1.0),
    background_tasks: BackgroundTasks = None,
    service: PatientMatchingService = Depends(get_service)
):
    """
    Run matching for ALL patients against ALL other patients.
    
    This is a comprehensive matching operation that finds all potential
    duplicate patient records in the database. Runs in background for
    large databases.
    
    WARNING: This can be computationally expensive for large databases.
    """
    def global_match_task():
        stats = service.run_global_matching(min_score=min_score)
        logger.info(f"Global matching complete: {stats}")
    
    background_tasks.add_task(global_match_task)
    
    return {
        "status": "started",
        "message": "Global matching started in background",
        "min_score": min_score
    }


@app.post("/match/batch", tags=["Matching"])
async def run_batch_matching(
    patient_ids: List[str] = None,
    min_score: float = Query(default=0.3, ge=0.0, le=1.0),
    background_tasks: BackgroundTasks = None,
    service: PatientMatchingService = Depends(get_service)
):
    """
    Run matching for a batch of patients (or all if no IDs provided).
    
    If patient_ids is not provided, matches all patients in the database.
    Runs in background for efficiency.
    """
    def batch_match_task():
        stats = service.run_batch_matching(
            patient_ids=patient_ids,
            min_score=min_score
        )
        logger.info(f"Batch matching complete: {stats}")
    
    background_tasks.add_task(batch_match_task)
    
    return {
        "status": "started",
        "message": f"Batch matching started for {len(patient_ids) if patient_ids else 'all'} patients",
        "min_score": min_score
    }


@app.post("/match/compare", response_model=MatchResultResponse, tags=["Matching"])
async def compare_patients(
    patient1_id: str,
    patient2_id: str,
    service: PatientMatchingService = Depends(get_service)
):
    """Compare two specific patients"""
    p1 = service.db.get_patient(patient1_id)
    p2 = service.db.get_patient(patient2_id)
    
    if not p1:
        raise HTTPException(status_code=404, detail=f"Patient {patient1_id} not found")
    if not p2:
        raise HTTPException(status_code=404, detail=f"Patient {patient2_id} not found")
    
    result = service.matcher.match(p1, p2)
    return _match_to_response(result)

@app.get("/match/graph/{patient_id}", tags=["Matching"])
async def get_match_graph(
    patient_id: str,
    service: PatientMatchingService = Depends(get_service)
):
    """Get graph-based match results using Cypher query"""
    results = service.db.run_match_query(patient_id)
    return {"patient_id": patient_id, "candidates": results}

# ========== EMPI Record Endpoints ==========

@app.post("/empi-records", tags=["EMPI Records"])
async def create_empi_record(
    patient_id: str,
    created_by: str = "API",
    service: PatientMatchingService = Depends(get_service)
):
    """Create an EMPI Record from a patient"""
    patient = service.db.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    empi_id = service.create_empi_record_from_patient(patient, created_by)
    return {"empi_id": empi_id, "patient_id": patient_id}

@app.post("/empi-records/merge", tags=["EMPI Records"])
async def merge_patients(
    patient_ids: List[str],
    merged_by: str = "API",
    service: PatientMatchingService = Depends(get_service)
):
    """Merge multiple patients into an EMPI Record"""
    try:
        empi_id = service.merge_patients(patient_ids, merged_by=merged_by)
        return {"empi_id": empi_id, "patient_ids": patient_ids}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/empi-records/{empi_id}/patients", tags=["EMPI Records"])
async def get_empi_record_patients(
    empi_id: str,
    service: PatientMatchingService = Depends(get_service)
):
    """Get all patients linked to an EMPI Record"""
    patient_ids = service.db.get_empi_record_patients(empi_id)
    return {"empi_id": empi_id, "patient_ids": patient_ids}

@app.delete("/empi-records/{empi_id}/patients/{patient_id}", tags=["EMPI Records"])
async def unmerge_patient(
    empi_id: str,
    patient_id: str,
    unmerged_by: str,
    reason: str,
    service: PatientMatchingService = Depends(get_service)
):
    """Remove a patient from an EMPI Record"""
    service.unmerge_patient(patient_id, unmerged_by, reason)
    return {"status": "unmerged", "patient_id": patient_id}

# ========== Review Workflow Endpoints ==========

@app.get("/reviews/pending", tags=["Reviews"])
async def get_pending_reviews(
    limit: int = Query(default=100, le=1000),
    service: PatientMatchingService = Depends(get_service)
):
    """Get matches pending human review"""
    reviews = service.get_pending_reviews(limit=limit)
    return {"count": len(reviews), "reviews": reviews}

@app.post("/reviews/decision", tags=["Reviews"])
async def submit_review_decision(
    request: ReviewDecisionRequest,
    service: PatientMatchingService = Depends(get_service)
):
    """Submit a review decision (approve or reject)"""
    if request.decision == "approve":
        empi_id = service.approve_match(
            request.patient1_id,
            request.patient2_id,
            request.reviewed_by,
            request.reason
        )
        return {
            "status": "approved",
            "empi_id": empi_id,
            "patient_ids": [request.patient1_id, request.patient2_id]
        }
    
    elif request.decision == "reject":
        service.reject_match(
            request.patient1_id,
            request.patient2_id,
            request.reviewed_by,
            request.reason or "Rejected by reviewer"
        )
        return {
            "status": "rejected",
            "patient_ids": [request.patient1_id, request.patient2_id]
        }
    
    else:
        raise HTTPException(
            status_code=400,
            detail="Decision must be 'approve' or 'reject'"
        )

# ========== Configuration Endpoints ==========

@app.get("/config/weights", response_model=MatchWeightsRequest, tags=["Configuration"])
async def get_match_weights(
    service: PatientMatchingService = Depends(get_service)
):
    """Get current match weights configuration"""
    w = service.weights
    return MatchWeightsRequest(
        enterprise_id=w.enterprise_id,
        mrn=w.mrn,
        ssn=w.ssn,
        dob_exact=w.dob_exact,
        phone_exact=w.phone_exact,
        email_exact=w.email_exact,
        deterministic_weight=w.deterministic_weight,
        name_weight=w.name_weight,
        address_weight=w.address_weight,
        embedding_weight=w.embedding_weight,
        auto_merge_threshold=w.auto_merge_threshold,
        human_review_threshold=w.human_review_threshold
    )

@app.put("/config/weights", tags=["Configuration"])
async def update_match_weights(
    request: MatchWeightsRequest,
    service: PatientMatchingService = Depends(get_service)
):
    """Update match weights configuration"""
    new_weights = MatchWeights(
        enterprise_id=request.enterprise_id,
        mrn=request.mrn,
        ssn=request.ssn,
        dob_exact=request.dob_exact,
        phone_exact=request.phone_exact,
        email_exact=request.email_exact,
        deterministic_weight=request.deterministic_weight,
        name_weight=request.name_weight,
        address_weight=request.address_weight,
        embedding_weight=request.embedding_weight,
        auto_merge_threshold=request.auto_merge_threshold,
        human_review_threshold=request.human_review_threshold
    )
    
    service.weights = new_weights
    service.matcher.weights = new_weights
    service.matcher.deterministic.weights = new_weights
    service.matcher.probabilistic.weights = new_weights
    
    return {"status": "updated", "weights": request}

# ========== Statistics Endpoints ==========

@app.get("/stats", response_model=StatsResponse, tags=["Statistics"])
async def get_statistics(
    service: PatientMatchingService = Depends(get_service)
):
    """Get service statistics"""
    stats = service.get_stats()
    return StatsResponse(**stats)

# ========== Helper Functions ==========

def _patient_to_response(patient: Patient) -> PatientResponse:
    """Convert Patient model to API response"""
    name_dict = None
    if patient.name:
        name_dict = {
            "family": patient.name.family,
            "given": patient.name.given,
            "full_name": patient.name.full_name
        }
    
    return PatientResponse(
        id=patient.id,
        source_id=patient.source_id,
        source_system=patient.source_system,
        name=name_dict,
        birth_date=patient.birth_date,
        gender=patient.gender.value,
        identifiers=[
            {"type": i.type.value, "value": i.value, "system": i.system}
            for i in patient.identifiers
        ],
        addresses=[
            {"full_address": a.full_address, "city": a.city, "state": a.state}
            for a in patient.addresses
        ],
        contact_points=[
            {"system": c.system.value, "value": c.value}
            for c in patient.contact_points
        ],
        empi_id=patient.empi_id,
        created_at=patient.created_at,
        updated_at=patient.updated_at
    )

def _match_to_response(match: MatchResult) -> MatchResultResponse:
    """Convert MatchResult to API response"""
    return MatchResultResponse(
        patient1_id=match.patient1_id,
        patient2_id=match.patient2_id,
        score=match.score,
        confidence=match.confidence.value,
        deterministic_score=match.deterministic_score,
        name_similarity=match.name_similarity,
        address_similarity=match.address_similarity,
        embedding_similarity=match.embedding_similarity,
        shared_identifiers=match.shared_identifiers,
        match_details=match.match_details
    )

# ========== Main Entry Point ==========

def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    return app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
