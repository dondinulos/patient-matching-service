"""
Patient Matching Service - Main Service Layer

Orchestrates the complete patient matching workflow:
1. Load patient data from FHIR files
2. Store in Graph Database
3. Find and score matches
4. Create EMPI Records
5. Handle human review workflow
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import uuid

from .models import (
    Patient, EmpiRecord, MatchResult, MatchConfidence,
    LinkageDecision, HumanName, Gender
)
from .graph_db import Neo4jConnection, PatientGraphDB
from .cosmos_graph_db import CosmosGraphDB
from .matching import PatientMatcher, MatchWeights
from .fhir_loader import FHIRPatientLoader, load_synthea_patients

logger = logging.getLogger(__name__)

class PatientMatchingService:
    """
    Main service for patient matching and MPI management
    
    Provides high-level API for:
    - Loading and indexing patients
    - Finding matches
    - Creating and managing EMPI Records
    - Human review workflow
    
    Supports both Neo4j and Cosmos DB Gremlin as the graph database backend.
    """
    
    def __init__(
        self,
        # Neo4j configuration
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
        # Cosmos DB configuration
        cosmos_endpoint: str = None,
        cosmos_database: str = None,
        cosmos_container: str = None,
        cosmos_key: str = None,
        # Database selection
        db_type: str = "neo4j",  # "neo4j" or "cosmos"
        # Matching configuration
        weights: MatchWeights = None,
        use_embeddings: bool = False,
        openai_api_key: str = None
    ):
        """
        Initialize the service
        
        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            cosmos_endpoint: Cosmos DB Gremlin endpoint
            cosmos_database: Cosmos DB database name
            cosmos_container: Cosmos DB container/graph name
            cosmos_key: Cosmos DB access key
            db_type: Database type ("neo4j" or "cosmos")
            weights: Custom match weights
            use_embeddings: Enable OpenAI embeddings
            openai_api_key: OpenAI API key
        """
        self.db_type = db_type
        
        # Initialize database connection based on type
        if db_type == "cosmos":
            self.db = CosmosGraphDB(
                endpoint=cosmos_endpoint,
                database=cosmos_database,
                container=cosmos_container,
                key=cosmos_key
            )
            self.connection = None  # Cosmos DB manages connection internally
            logger.info("Using Cosmos DB Gremlin backend")
        else:
            # Default to Neo4j
            self.connection = Neo4jConnection(
                uri=neo4j_uri,
                user=neo4j_user,
                password=neo4j_password
            )
            self.db = PatientGraphDB(self.connection)
            logger.info("Using Neo4j backend")
        
        # Initialize matcher
        self.matcher = PatientMatcher(
            weights=weights,
            use_embeddings=use_embeddings,
            openai_api_key=openai_api_key
        )
        
        self.weights = weights or MatchWeights()
        
        logger.info("PatientMatchingService initialized")
    
    def initialize(self):
        """Initialize database connection and schema"""
        if self.db_type == "cosmos":
            self.db.connect()
        else:
            self.connection.connect()
        self.db.initialize_schema()
        logger.info("Database schema initialized")
    
    def close(self):
        """Close database connection"""
        if self.db_type == "cosmos":
            self.db.close()
        elif self.connection:
            self.connection.close()
    
    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    # ========== Data Loading ==========
    
    def load_patient(self, patient: Patient) -> str:
        """
        Load a single patient into the database
        
        Args:
            patient: Patient object to load
        
        Returns:
            Patient ID
        """
        return self.db.create_patient(patient)
    
    def load_patients_from_fhir(
        self,
        directory: str,
        limit: int = None,
        source_system: str = "FHIR"
    ) -> int:
        """
        Load patients from FHIR Bundle files
        
        Args:
            directory: Directory containing FHIR JSON files
            limit: Maximum patients to load
            source_system: Source system name
        
        Returns:
            Number of patients loaded
        """
        loader = FHIRPatientLoader(source_system=source_system)
        count = 0
        
        for patient in loader.load_from_directory(directory, limit=limit):
            self.db.create_patient(patient)
            count += 1
            
            if count % 100 == 0:
                logger.info(f"Loaded {count} patients...")
        
        logger.info(f"Loaded {count} patients from {directory}")
        return count
    
    # ========== Matching ==========
    
    def find_matches_for_patient(
        self,
        patient_id: str,
        min_score: float = 0.3,
        use_graph_query: bool = True
    ) -> List[MatchResult]:
        """
        Find all potential matches for a patient
        
        Args:
            patient_id: ID of patient to match
            min_score: Minimum score threshold
            use_graph_query: Use graph-based candidate retrieval
        
        Returns:
            List of MatchResults sorted by score
        """
        patient = self.db.get_patient(patient_id)
        if not patient:
            logger.error(f"Patient not found: {patient_id}")
            return []
        
        # Get candidates using graph queries
        candidates = []
        candidate_ids = set()
        
        if use_graph_query:
            # Find by shared identifiers
            id_candidates = self.db.find_candidates_by_identifiers(patient)
            for cid, shared_ids in id_candidates:
                if cid not in candidate_ids:
                    candidate_ids.add(cid)
            
            # Find by demographics
            demo_candidates = self.db.find_candidates_by_demographics(patient)
            for cid in demo_candidates:
                if cid not in candidate_ids:
                    candidate_ids.add(cid)
            
            # Find by contacts
            contact_candidates = self.db.find_candidates_by_contact(patient)
            for cid, _ in contact_candidates:
                if cid not in candidate_ids:
                    candidate_ids.add(cid)
        
        # Load candidate patients
        for cid in candidate_ids:
            candidate = self.db.get_patient(cid)
            if candidate:
                candidates.append(candidate)
        
        logger.info(f"Found {len(candidates)} candidates for patient {patient_id}")
        
        # Compute match scores
        results = self.matcher.find_matches(
            patient,
            candidates,
            min_score=min_score
        )
        
        # Store potential matches in graph
        for result in results:
            self.db.store_potential_match(result)
        
        return results
    
    def run_batch_matching(
        self,
        patient_ids: List[str] = None,
        min_score: float = 0.3
    ) -> Dict[str, Any]:
        """
        Run matching for multiple patients
        
        Args:
            patient_ids: List of patient IDs to match (None = all)
            min_score: Minimum score threshold
        
        Returns:
            Summary statistics
        """
        stats = {
            "patients_processed": 0,
            "matches_found": 0,
            "auto_merge": 0,
            "human_review": 0,
            "no_match": 0
        }
        
        if patient_ids is None:
            # Would need to implement get_all_patient_ids
            logger.warning("Batch matching without patient_ids not implemented")
            return stats
        
        for pid in patient_ids:
            results = self.find_matches_for_patient(pid, min_score)
            stats["patients_processed"] += 1
            stats["matches_found"] += len(results)
            
            for r in results:
                if r.confidence == MatchConfidence.AUTO_MERGE:
                    stats["auto_merge"] += 1
                elif r.confidence == MatchConfidence.HUMAN_REVIEW:
                    stats["human_review"] += 1
                else:
                    stats["no_match"] += 1
        
        logger.info(f"Batch matching complete: {stats}")
        return stats
    
    # ========== EMPI Record Management ==========
    
    def create_empi_record_from_patient(
        self,
        patient: Patient,
        created_by: str = "AUTO"
    ) -> str:
        """
        Create a new EMPI Record from a patient
        
        Uses the patient's data as the initial survivorship values.
        
        Args:
            patient: Source patient
            created_by: Creator identifier
        
        Returns:
            EMPI Record ID
        """
        empi = EmpiRecord(
            id=str(uuid.uuid4()),
            name=patient.name,
            birth_date=patient.birth_date,
            gender=patient.gender,
            linked_patient_ids=[patient.id],
            created_by=created_by,
            auto_linked=created_by == "AUTO"
        )
        
        empi_id = self.db.create_empi_record(empi)
        self.db.link_patient_to_empi(
            patient.id,
            empi_id,
            score=1.0,
            auto_linked=created_by == "AUTO"
        )
        
        logger.info(f"Created EMPI Record {empi_id} from patient {patient.id}")
        return empi_id
    
    def merge_patients(
        self,
        patient_ids: List[str],
        match_results: List[MatchResult] = None,
        merged_by: str = "AUTO"
    ) -> str:
        """
        Merge multiple patients into a single EMPI Record
        
        Args:
            patient_ids: List of patient IDs to merge
            match_results: Optional match results for audit
            merged_by: User or system that initiated merge
        
        Returns:
            EMPI Record ID
        """
        if not patient_ids:
            raise ValueError("At least one patient ID required")
        
        patients = []
        for pid in patient_ids:
            patient = self.db.get_patient(pid)
            if patient:
                patients.append(patient)
        
        if not patients:
            raise ValueError("No valid patients found")
        
        # Apply survivorship rules
        empi_data = self._apply_survivorship(patients)
        
        empi = EmpiRecord(
            id=str(uuid.uuid4()),
            name=empi_data.get("name"),
            birth_date=empi_data.get("birth_date"),
            gender=empi_data.get("gender", Gender.UNKNOWN),
            linked_patient_ids=patient_ids,
            created_by=merged_by,
            auto_linked=merged_by == "AUTO"
        )
        
        empi_id = self.db.create_empi_record(empi)
        
        # Link all patients to EMPI record
        for pid in patient_ids:
            score = 1.0
            # Get match score if available
            if match_results:
                for mr in match_results:
                    if pid in [mr.patient1_id, mr.patient2_id]:
                        score = mr.score
                        break
            
            self.db.link_patient_to_empi(
                pid,
                empi_id,
                score=score,
                auto_linked=merged_by == "AUTO"
            )
        
        logger.info(f"Merged {len(patient_ids)} patients into EMPI {empi_id}")
        return empi_id
    
    def _apply_survivorship(
        self,
        patients: List[Patient]
    ) -> Dict[str, Any]:
        """
        Apply survivorship rules to determine best values
        
        Simple rules:
        - Most recent non-null value wins
        - Prefer more complete data
        """
        result = {
            "name": None,
            "birth_date": None,
            "gender": Gender.UNKNOWN
        }
        
        # Sort by updated_at descending
        sorted_patients = sorted(
            patients,
            key=lambda p: p.updated_at,
            reverse=True
        )
        
        for patient in sorted_patients:
            if result["name"] is None and patient.name:
                result["name"] = patient.name
            
            if result["birth_date"] is None and patient.birth_date:
                result["birth_date"] = patient.birth_date
            
            if result["gender"] == Gender.UNKNOWN and patient.gender != Gender.UNKNOWN:
                result["gender"] = patient.gender
        
        return result
    
    def process_auto_merges(self) -> int:
        """
        Process all matches that qualify for automatic merge
        
        Returns:
            Number of merges processed
        """
        # Get all high-confidence matches
        # This is a simplified implementation
        merge_count = 0
        
        # In a real implementation, we would:
        # 1. Query for all POTENTIAL_MATCH edges with score >= auto_merge_threshold
        # 2. Group connected components
        # 3. Create EMPI Records for each group
        
        logger.info(f"Processed {merge_count} auto-merges")
        return merge_count
    
    # ========== Human Review Workflow ==========
    
    def get_pending_reviews(
        self,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get matches pending human review
        
        Returns:
            List of match pairs with details
        """
        return self.db.get_matches_for_review(
            min_score=self.weights.human_review_threshold,
            max_score=self.weights.auto_merge_threshold,
            limit=limit
        )
    
    def approve_match(
        self,
        patient1_id: str,
        patient2_id: str,
        reviewed_by: str,
        notes: str = None
    ) -> str:
        """
        Approve a match and merge patients
        
        Args:
            patient1_id: First patient ID
            patient2_id: Second patient ID
            reviewed_by: Reviewer identifier
            notes: Optional review notes
        
        Returns:
            EMPI Record ID
        """
        # Check if either already has an EMPI record
        p1 = self.db.get_patient(patient1_id)
        p2 = self.db.get_patient(patient2_id)
        
        if p1.empi_id and p2.empi_id:
            if p1.empi_id == p2.empi_id:
                logger.info("Patients already linked to same EMPI")
                return p1.empi_id
            else:
                # Would need to merge EMPIs
                raise NotImplementedError("Merging EMPIs not yet implemented")
        
        if p1.empi_id:
            # Add p2 to p1's EMPI
            self.db.link_patient_to_empi(
                patient2_id,
                p1.empi_id,
                score=1.0,
                auto_linked=False
            )
            return p1.empi_id
        
        if p2.empi_id:
            # Add p1 to p2's EMPI
            self.db.link_patient_to_empi(
                patient1_id,
                p2.empi_id,
                score=1.0,
                auto_linked=False
            )
            return p2.empi_id
        
        # Neither has an EMPI - create new one
        empi_id = self.merge_patients(
            [patient1_id, patient2_id],
            merged_by=reviewed_by
        )
        
        logger.info(f"Match approved by {reviewed_by}: {patient1_id} + {patient2_id} -> EMPI {empi_id}")
        return empi_id
    
    def reject_match(
        self,
        patient1_id: str,
        patient2_id: str,
        reviewed_by: str,
        reason: str
    ):
        """
        Reject a potential match
        
        Args:
            patient1_id: First patient ID
            patient2_id: Second patient ID
            reviewed_by: Reviewer identifier
            reason: Rejection reason
        """
        # In a real implementation, we would:
        # 1. Remove or mark the POTENTIAL_MATCH edge as rejected
        # 2. Store the rejection decision for audit
        # 3. Possibly add a "DO_NOT_MERGE" edge for future reference
        
        logger.info(f"Match rejected by {reviewed_by}: {patient1_id} / {patient2_id} - {reason}")
    
    def unmerge_patient(
        self,
        patient_id: str,
        unmerged_by: str,
        reason: str
    ):
        """
        Remove a patient from its EMPI Record
        
        Args:
            patient_id: Patient to unmerge
            unmerged_by: User identifier
            reason: Reason for unmerge
        """
        self.db.unlink_patient(patient_id, reason, unmerged_by)
        logger.info(f"Patient {patient_id} unmerged by {unmerged_by}: {reason}")
    
    # ========== Statistics ==========
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            "total_patients": self.db.get_patient_count(),
            "total_empi_records": self.db.get_empi_record_count(),
            "pending_reviews": len(self.get_pending_reviews(limit=1000))
        }

class MatchingPipeline:
    """
    Complete matching pipeline for processing new patient data
    
    Steps:
    1. Load patient from FHIR
    2. Index in graph database
    3. Find matching candidates
    4. Score matches
    5. Auto-merge high confidence matches
    6. Queue low confidence for review
    """
    
    def __init__(self, service: PatientMatchingService):
        self.service = service
    
    def process_patient(
        self,
        patient: Patient
    ) -> Dict[str, Any]:
        """
        Process a single new patient through the matching pipeline
        
        Returns:
            Processing result with matches and decisions
        """
        result = {
            "patient_id": patient.id,
            "matches": [],
            "empi_id": None,
            "action": None
        }
        
        # Store patient
        self.service.load_patient(patient)
        
        # Find matches
        matches = self.service.find_matches_for_patient(patient.id)
        result["matches"] = [
            {
                "candidate_id": m.patient2_id,
                "score": m.score,
                "confidence": m.confidence.value
            }
            for m in matches
        ]
        
        if not matches:
            # No matches - create new EMPI
            empi_id = self.service.create_empi_record_from_patient(patient)
            result["empi_id"] = empi_id
            result["action"] = "new_empi"
        else:
            top_match = matches[0]
            
            if top_match.confidence == MatchConfidence.AUTO_MERGE:
                # Auto-merge with top match
                empi_id = self.service.approve_match(
                    patient.id,
                    top_match.patient2_id,
                    reviewed_by="AUTO"
                )
                result["empi_id"] = empi_id
                result["action"] = "auto_merged"
            
            elif top_match.confidence == MatchConfidence.HUMAN_REVIEW:
                result["action"] = "pending_review"
            
            else:
                # No good matches - create new EMPI
                empi_id = self.service.create_empi_record_from_patient(patient)
                result["empi_id"] = empi_id
                result["action"] = "new_empi"
        
        return result
    
    def process_batch(
        self,
        patients: List[Patient]
    ) -> Dict[str, Any]:
        """
        Process a batch of patients
        
        Returns:
            Batch processing statistics
        """
        stats = {
            "total": len(patients),
            "new_empi_records": 0,
            "auto_merged": 0,
            "pending_review": 0,
            "errors": 0
        }
        
        for patient in patients:
            try:
                result = self.process_patient(patient)
                
                if result["action"] == "new_empi":
                    stats["new_empi_records"] += 1
                elif result["action"] == "auto_merged":
                    stats["auto_merged"] += 1
                elif result["action"] == "pending_review":
                    stats["pending_review"] += 1
            
            except Exception as e:
                logger.error(f"Error processing patient {patient.id}: {e}")
                stats["errors"] += 1
        
        logger.info(f"Batch processing complete: {stats}")
        return stats
