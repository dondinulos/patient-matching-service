"""
Graph Database Layer for Patient Matching Service

Uses Neo4j to store patients and their relationships.
Provides CRUD operations and graph-based queries for matching.
"""

import os
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import uuid

from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, AuthError

from .models import (
    Patient, EmpiRecord, Identifier, Address, ContactPoint,
    HumanName, Gender, IdentifierType, ContactPointSystem,
    MatchResult, MatchConfidence, LinkageDecision
)

logger = logging.getLogger(__name__)


class Neo4jConnection:
    """Neo4j database connection manager"""
    
    def __init__(
        self,
        uri: str = None,
        user: str = None,
        password: str = None
    ):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self._driver: Optional[Driver] = None
    
    def connect(self) -> Driver:
        """Establish connection to Neo4j"""
        if self._driver is None:
            try:
                self._driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.user, self.password)
                )
                # Verify connectivity
                self._driver.verify_connectivity()
                logger.info(f"Connected to Neo4j at {self.uri}")
            except AuthError as e:
                logger.error(f"Authentication failed: {e}")
                raise
            except ServiceUnavailable as e:
                logger.error(f"Neo4j service unavailable: {e}")
                raise
        return self._driver
    
    def close(self):
        """Close the database connection"""
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed")
    
    def get_session(self) -> Session:
        """Get a new database session"""
        driver = self.connect()
        return driver.session()
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class PatientGraphDB:
    """
    Patient Graph Database operations
    
    Graph Schema:
    - (:Patient) - Core patient node
    - (:Identifier) - Patient identifiers (MRN, SSN, etc.)
    - (:Address) - Patient addresses
    - (:ContactPoint) - Phone, email, etc.
    - (:EmpiRecord) - Enterprise Master Patient Index record
    
    Relationships:
    - (Patient)-[:HAS_IDENTIFIER]->(Identifier)
    - (Patient)-[:HAS_ADDRESS]->(Address)
    - (Patient)-[:HAS_CONTACT]->(ContactPoint)
    - (Patient)-[:LINKED_TO]->(EmpiRecord)
    - (Patient)-[:POTENTIAL_MATCH {score}]->(Patient)
    """
    
    def __init__(self, connection: Neo4jConnection):
        self.connection = connection
    
    def initialize_schema(self):
        """Create indexes and constraints for optimal performance"""
        with self.connection.get_session() as session:
            # Patient constraints
            session.run("""
                CREATE CONSTRAINT patient_id IF NOT EXISTS
                FOR (p:Patient) REQUIRE p.id IS UNIQUE
            """)
            session.run("""
                CREATE INDEX patient_source IF NOT EXISTS
                FOR (p:Patient) ON (p.source_system, p.source_id)
            """)
            session.run("""
                CREATE INDEX patient_dob IF NOT EXISTS
                FOR (p:Patient) ON (p.birth_date)
            """)
            
            # Identifier constraints
            session.run("""
                CREATE INDEX identifier_value IF NOT EXISTS
                FOR (i:Identifier) ON (i.type, i.value)
            """)
            
            # EMPI Record constraints
            session.run("""
                CREATE CONSTRAINT empi_id IF NOT EXISTS
                FOR (g:EmpiRecord) REQUIRE g.id IS UNIQUE
            """)
            
            # ContactPoint index
            session.run("""
                CREATE INDEX contact_value IF NOT EXISTS
                FOR (c:ContactPoint) ON (c.system, c.normalized_value)
            """)
            
            logger.info("Graph schema initialized")
    
    def create_patient(self, patient: Patient) -> str:
        """
        Create a patient node with all related nodes and relationships
        
        Returns the patient ID
        """
        with self.connection.get_session() as session:
            result = session.execute_write(self._create_patient_tx, patient)
            logger.info(f"Created patient {result}")
            return result
    
    @staticmethod
    def _create_patient_tx(tx, patient: Patient) -> str:
        """Transaction function to create patient and related nodes"""
        patient_id = patient.id or str(uuid.uuid4())
        
        # Create patient node
        tx.run("""
            CREATE (p:Patient {
                id: $id,
                source_id: $source_id,
                source_system: $source_system,
                first_name: $first_name,
                last_name: $last_name,
                full_name: $full_name,
                birth_date: $birth_date,
                gender: $gender,
                created_at: $created_at,
                updated_at: $updated_at,
                empi_id: $empi_id
            })
        """, {
            "id": patient_id,
            "source_id": patient.source_id,
            "source_system": patient.source_system,
            "first_name": patient.name.first_name if patient.name else None,
            "last_name": patient.name.last_name if patient.name else None,
            "full_name": patient.name.full_name if patient.name else None,
            "birth_date": patient.birth_date.isoformat() if patient.birth_date else None,
            "gender": patient.gender.value,
            "created_at": patient.created_at.isoformat(),
            "updated_at": patient.updated_at.isoformat(),
            "empi_id": patient.empi_id
        })
        
        # Create identifiers
        for ident in patient.identifiers:
            tx.run("""
                MATCH (p:Patient {id: $patient_id})
                MERGE (i:Identifier {type: $type, value: $value})
                ON CREATE SET i.system = $system, i.assigner = $assigner
                MERGE (p)-[:HAS_IDENTIFIER]->(i)
            """, {
                "patient_id": patient_id,
                "type": ident.type.value,
                "value": ident.value,
                "system": ident.system,
                "assigner": ident.assigner
            })
        
        # Create addresses
        for idx, addr in enumerate(patient.addresses):
            tx.run("""
                MATCH (p:Patient {id: $patient_id})
                CREATE (a:Address {
                    line: $line,
                    city: $city,
                    state: $state,
                    postal_code: $postal_code,
                    country: $country,
                    full_address: $full_address,
                    normalized: $normalized
                })
                CREATE (p)-[:HAS_ADDRESS {order: $order}]->(a)
            """, {
                "patient_id": patient_id,
                "line": addr.line,
                "city": addr.city,
                "state": addr.state,
                "postal_code": addr.postal_code,
                "country": addr.country,
                "full_address": addr.full_address,
                "normalized": addr.normalize(),
                "order": idx
            })
        
        # Create contact points
        for cp in patient.contact_points:
            tx.run("""
                MATCH (p:Patient {id: $patient_id})
                MERGE (c:ContactPoint {system: $system, normalized_value: $normalized})
                ON CREATE SET c.value = $value, c.use = $use
                MERGE (p)-[:HAS_CONTACT]->(c)
            """, {
                "patient_id": patient_id,
                "system": cp.system.value,
                "value": cp.value,
                "normalized": cp.normalize(),
                "use": cp.use
            })
        
        # Store embedding if present
        if patient.embedding:
            tx.run("""
                MATCH (p:Patient {id: $patient_id})
                SET p.embedding = $embedding
            """, {
                "patient_id": patient_id,
                "embedding": patient.embedding
            })
        
        return patient_id
    
    def get_patient(self, patient_id: str) -> Optional[Patient]:
        """Retrieve a patient by ID with all related data"""
        with self.connection.get_session() as session:
            result = session.execute_read(self._get_patient_tx, patient_id)
            return result
    
    @staticmethod
    def _get_patient_tx(tx, patient_id: str) -> Optional[Patient]:
        """Transaction function to retrieve patient"""
        result = tx.run("""
            MATCH (p:Patient {id: $patient_id})
            OPTIONAL MATCH (p)-[:HAS_IDENTIFIER]->(i:Identifier)
            OPTIONAL MATCH (p)-[:HAS_ADDRESS]->(a:Address)
            OPTIONAL MATCH (p)-[:HAS_CONTACT]->(c:ContactPoint)
            RETURN p,
                   collect(DISTINCT i) as identifiers,
                   collect(DISTINCT a) as addresses,
                   collect(DISTINCT c) as contacts
        """, {"patient_id": patient_id})
        
        record = result.single()
        if not record:
            return None
        
        p = record["p"]
        
        # Build identifiers
        identifiers = []
        for i in record["identifiers"]:
            if i:
                identifiers.append(Identifier(
                    value=i["value"],
                    type=IdentifierType(i["type"]),
                    system=i.get("system"),
                    assigner=i.get("assigner")
                ))
        
        # Build addresses
        addresses = []
        for a in record["addresses"]:
            if a:
                addresses.append(Address(
                    line=a.get("line", []),
                    city=a.get("city"),
                    state=a.get("state"),
                    postal_code=a.get("postal_code"),
                    country=a.get("country")
                ))
        
        # Build contact points
        contact_points = []
        for c in record["contacts"]:
            if c:
                contact_points.append(ContactPoint(
                    system=ContactPointSystem(c["system"]),
                    value=c["value"],
                    use=c.get("use")
                ))
        
        # Build name
        name = None
        if p.get("first_name") or p.get("last_name"):
            name = HumanName(
                family=p.get("last_name"),
                given=[p["first_name"]] if p.get("first_name") else []
            )
        
        from datetime import date
        birth_date = None
        if p.get("birth_date"):
            birth_date = date.fromisoformat(p["birth_date"])
        
        return Patient(
            id=p["id"],
            source_id=p["source_id"],
            source_system=p["source_system"],
            name=name,
            birth_date=birth_date,
            gender=Gender(p.get("gender", "unknown")),
            identifiers=identifiers,
            addresses=addresses,
            contact_points=contact_points,
            empi_id=p.get("empi_id"),
            embedding=p.get("embedding")
        )
    
    def find_candidates_by_identifiers(
        self,
        patient: Patient,
        limit: int = 100
    ) -> List[Tuple[str, List[str]]]:
        """
        Find potential matching patients by shared identifiers
        
        Returns list of (patient_id, shared_identifier_values)
        """
        with self.connection.get_session() as session:
            result = session.execute_read(
                self._find_by_identifiers_tx,
                patient,
                limit
            )
            return result
    
    @staticmethod
    def _find_by_identifiers_tx(
        tx,
        patient: Patient,
        limit: int
    ) -> List[Tuple[str, List[str]]]:
        """Find candidates by shared identifiers"""
        identifier_values = [i.value for i in patient.identifiers]
        if not identifier_values:
            return []
        
        result = tx.run("""
            MATCH (p1:Patient)-[:HAS_IDENTIFIER]->(i:Identifier)<-[:HAS_IDENTIFIER]-(p2:Patient)
            WHERE p1.id = $patient_id AND p1 <> p2
            AND i.value IN $identifier_values
            WITH p2, collect(DISTINCT i.value) as shared_ids
            RETURN p2.id as patient_id, shared_ids
            ORDER BY size(shared_ids) DESC
            LIMIT $limit
        """, {
            "patient_id": patient.id,
            "identifier_values": identifier_values,
            "limit": limit
        })
        
        return [(r["patient_id"], r["shared_ids"]) for r in result]
    
    def find_candidates_by_demographics(
        self,
        patient: Patient,
        limit: int = 100
    ) -> List[str]:
        """
        Find potential matching patients by demographics (DOB, name)
        
        Returns list of patient IDs
        """
        with self.connection.get_session() as session:
            result = session.execute_read(
                self._find_by_demographics_tx,
                patient,
                limit
            )
            return result
    
    @staticmethod
    def _find_by_demographics_tx(
        tx,
        patient: Patient,
        limit: int
    ) -> List[str]:
        """Find candidates by demographics"""
        # Build conditions
        conditions = []
        params = {"patient_id": patient.id, "limit": limit}
        
        if patient.birth_date:
            conditions.append("p.birth_date = $birth_date")
            params["birth_date"] = patient.birth_date.isoformat()
        
        if patient.name and patient.name.last_name:
            # Use CONTAINS for partial matching
            conditions.append("toLower(p.last_name) = toLower($last_name)")
            params["last_name"] = patient.name.last_name
        
        if not conditions:
            return []
        
        query = f"""
            MATCH (p:Patient)
            WHERE p.id <> $patient_id
            AND ({' OR '.join(conditions)})
            RETURN p.id as patient_id
            LIMIT $limit
        """
        
        result = tx.run(query, params)
        return [r["patient_id"] for r in result]
    
    def find_candidates_by_contact(
        self,
        patient: Patient,
        limit: int = 100
    ) -> List[Tuple[str, str]]:
        """
        Find potential matching patients by shared contact points
        
        Returns list of (patient_id, shared_contact_value)
        """
        with self.connection.get_session() as session:
            result = session.execute_read(
                self._find_by_contact_tx,
                patient,
                limit
            )
            return result
    
    @staticmethod
    def _find_by_contact_tx(
        tx,
        patient: Patient,
        limit: int
    ) -> List[Tuple[str, str]]:
        """Find candidates by shared contacts"""
        normalized_contacts = [cp.normalize() for cp in patient.contact_points]
        if not normalized_contacts:
            return []
        
        result = tx.run("""
            MATCH (p1:Patient)-[:HAS_CONTACT]->(c:ContactPoint)<-[:HAS_CONTACT]-(p2:Patient)
            WHERE p1.id = $patient_id AND p1 <> p2
            AND c.normalized_value IN $contacts
            WITH p2, collect(DISTINCT c.normalized_value) as shared_contacts
            RETURN p2.id as patient_id, shared_contacts
            LIMIT $limit
        """, {
            "patient_id": patient.id,
            "contacts": normalized_contacts,
            "limit": limit
        })
        
        return [(r["patient_id"], r["shared_contacts"]) for r in result]
    
    def create_empi_record(self, empi: EmpiRecord) -> str:
        """Create an EMPI Record (master patient)"""
        with self.connection.get_session() as session:
            result = session.execute_write(self._create_empi_tx, empi)
            return result
    
    @staticmethod
    def _create_empi_tx(tx, empi: EmpiRecord) -> str:
        """Transaction to create EMPI record"""
        empi_id = empi.id or str(uuid.uuid4())
        
        tx.run("""
            CREATE (g:EmpiRecord {
                id: $id,
                first_name: $first_name,
                last_name: $last_name,
                birth_date: $birth_date,
                gender: $gender,
                created_at: $created_at,
                updated_at: $updated_at,
                created_by: $created_by,
                auto_linked: $auto_linked
            })
        """, {
            "id": empi_id,
            "first_name": empi.name.first_name if empi.name else None,
            "last_name": empi.name.last_name if empi.name else None,
            "birth_date": empi.birth_date.isoformat() if empi.birth_date else None,
            "gender": empi.gender.value,
            "created_at": empi.created_at.isoformat(),
            "updated_at": empi.updated_at.isoformat(),
            "created_by": empi.created_by,
            "auto_linked": empi.auto_linked
        })
        
        return empi_id
    
    def link_patient_to_empi(
        self,
        patient_id: str,
        empi_id: str,
        score: float,
        auto_linked: bool = True
    ):
        """Link a patient record to an EMPI Record"""
        with self.connection.get_session() as session:
            session.execute_write(
                self._link_to_empi_tx,
                patient_id,
                empi_id,
                score,
                auto_linked
            )
    
    @staticmethod
    def _link_to_empi_tx(
        tx,
        patient_id: str,
        empi_id: str,
        score: float,
        auto_linked: bool
    ):
        """Transaction to link patient to EMPI record"""
        tx.run("""
            MATCH (p:Patient {id: $patient_id})
            MATCH (g:EmpiRecord {id: $empi_id})
            MERGE (p)-[r:LINKED_TO]->(g)
            SET r.score = $score,
                r.auto_linked = $auto_linked,
                r.linked_at = $linked_at,
                p.empi_id = $empi_id
        """, {
            "patient_id": patient_id,
            "empi_id": empi_id,
            "score": score,
            "auto_linked": auto_linked,
            "linked_at": datetime.utcnow().isoformat()
        })
    
    def store_potential_match(self, match: MatchResult):
        """Store a potential match relationship between two patients"""
        with self.connection.get_session() as session:
            session.execute_write(self._store_match_tx, match)
    
    @staticmethod
    def _store_match_tx(tx, match: MatchResult):
        """Transaction to store potential match"""
        tx.run("""
            MATCH (p1:Patient {id: $p1_id})
            MATCH (p2:Patient {id: $p2_id})
            MERGE (p1)-[r:POTENTIAL_MATCH]-(p2)
            SET r.score = $score,
                r.confidence = $confidence,
                r.deterministic_score = $det_score,
                r.name_similarity = $name_sim,
                r.address_similarity = $addr_sim,
                r.embedding_similarity = $embed_sim,
                r.shared_identifiers = $shared_ids,
                r.updated_at = $updated_at
        """, {
            "p1_id": match.patient1_id,
            "p2_id": match.patient2_id,
            "score": match.score,
            "confidence": match.confidence.value,
            "det_score": match.deterministic_score,
            "name_sim": match.name_similarity,
            "addr_sim": match.address_similarity,
            "embed_sim": match.embedding_similarity,
            "shared_ids": match.shared_identifiers,
            "updated_at": datetime.utcnow().isoformat()
        })
    
    def get_matches_for_review(
        self,
        min_score: float = 0.65,
        max_score: float = 0.85,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get potential matches that need human review"""
        with self.connection.get_session() as session:
            result = session.execute_read(
                self._get_review_matches_tx,
                min_score,
                max_score,
                limit
            )
            return result
    
    @staticmethod
    def _get_review_matches_tx(
        tx,
        min_score: float,
        max_score: float,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Get matches needing review"""
        result = tx.run("""
            MATCH (p1:Patient)-[r:POTENTIAL_MATCH]-(p2:Patient)
            WHERE r.score >= $min_score AND r.score < $max_score
            AND r.confidence = 'human_review'
            RETURN p1, p2, r
            ORDER BY r.score DESC
            LIMIT $limit
        """, {
            "min_score": min_score,
            "max_score": max_score,
            "limit": limit
        })
        
        matches = []
        for record in result:
            matches.append({
                "patient1": dict(record["p1"]),
                "patient2": dict(record["p2"]),
                "match": dict(record["r"])
            })
        return matches
    
    def get_empi_record_patients(
        self,
        empi_id: str
    ) -> List[str]:
        """Get all patient IDs linked to an EMPI Record"""
        with self.connection.get_session() as session:
            result = session.execute_read(
                self._get_empi_patients_tx,
                empi_id
            )
            return result
    
    @staticmethod
    def _get_empi_patients_tx(tx, empi_id: str) -> List[str]:
        """Get patients in EMPI record"""
        result = tx.run("""
            MATCH (p:Patient)-[:LINKED_TO]->(g:EmpiRecord {id: $empi_id})
            RETURN p.id as patient_id
        """, {"empi_id": empi_id})
        
        return [r["patient_id"] for r in result]
    
    def unlink_patient(
        self,
        patient_id: str,
        reason: str,
        unlinked_by: str
    ):
        """Unlink a patient from its EMPI Record"""
        with self.connection.get_session() as session:
            session.execute_write(
                self._unlink_patient_tx,
                patient_id,
                reason,
                unlinked_by
            )
    
    @staticmethod
    def _unlink_patient_tx(
        tx,
        patient_id: str,
        reason: str,
        unlinked_by: str
    ):
        """Unlink patient transaction"""
        tx.run("""
            MATCH (p:Patient {id: $patient_id})-[r:LINKED_TO]->(g:EmpiRecord)
            DELETE r
            SET p.empi_id = null
            CREATE (u:UnlinkEvent {
                patient_id: $patient_id,
                empi_id: g.id,
                reason: $reason,
                unlinked_by: $unlinked_by,
                unlinked_at: $unlinked_at
            })
        """, {
            "patient_id": patient_id,
            "reason": reason,
            "unlinked_by": unlinked_by,
            "unlinked_at": datetime.utcnow().isoformat()
        })
    
    def run_match_query(self, patient_id: str) -> List[Dict[str, Any]]:
        """
        Run the comprehensive matching query from requirements
        
        This query finds candidates and computes similarity scores
        """
        with self.connection.get_session() as session:
            result = session.execute_read(
                self._run_match_query_tx,
                patient_id
            )
            return result
    
    @staticmethod
    def _run_match_query_tx(tx, patient_id: str) -> List[Dict[str, Any]]:
        """
        Comprehensive match query using Cypher
        
        Note: For Jaro-Winkler similarity, requires APOC plugin
        """
        result = tx.run("""
            // Find patients with shared identifiers
            MATCH (p1:Patient {id: $patient_id})-[:HAS_IDENTIFIER]->(id)<-[:HAS_IDENTIFIER]-(p2:Patient)
            WHERE p1 <> p2
            WITH p1, p2, collect(DISTINCT id.value) AS sharedIds
            
            // Get addresses for comparison
            OPTIONAL MATCH (p1)-[:HAS_ADDRESS]->(a1:Address)
            OPTIONAL MATCH (p2)-[:HAS_ADDRESS]->(a2:Address)
            
            WITH p1, p2, sharedIds,
                 collect(DISTINCT a1.normalized)[0] as addr1,
                 collect(DISTINCT a2.normalized)[0] as addr2
            
            // Compute scores
            WITH p1, p2, sharedIds, addr1, addr2,
                 CASE WHEN size(sharedIds) > 0 THEN 0.8 ELSE 0 END as deterministicScore,
                 CASE 
                     WHEN p1.full_name IS NOT NULL AND p2.full_name IS NOT NULL
                     THEN apoc.text.jaroWinklerDistance(
                         toLower(p1.full_name), 
                         toLower(p2.full_name)
                     )
                     ELSE 0 
                 END as nameSim,
                 CASE 
                     WHEN addr1 IS NOT NULL AND addr2 IS NOT NULL
                     THEN apoc.text.jaroWinklerDistance(addr1, addr2)
                     ELSE 0 
                 END as addrSim
            
            // Calculate final score
            WITH p1, p2, sharedIds, nameSim, addrSim, deterministicScore,
                 (0.4 * deterministicScore) + 
                 (0.35 * nameSim) + 
                 (0.15 * addrSim) as score
            
            WHERE score > 0.3  // Minimum threshold
            
            RETURN p2.id as candidate_id,
                   p2.full_name as candidate_name,
                   p2.birth_date as candidate_dob,
                   sharedIds,
                   nameSim,
                   addrSim,
                   deterministicScore,
                   score
            ORDER BY score DESC
            LIMIT 50
        """, {"patient_id": patient_id})
        
        return [dict(r) for r in result]
    
    def get_patient_count(self) -> int:
        """Get total number of patients in the database"""
        with self.connection.get_session() as session:
            result = session.run("MATCH (p:Patient) RETURN count(p) as count")
            return result.single()["count"]
    
    def get_empi_record_count(self) -> int:
        """Get total number of EMPI records"""
        with self.connection.get_session() as session:
            result = session.run("MATCH (g:EmpiRecord) RETURN count(g) as count")
            return result.single()["count"]
    
    def clear_all_data(self):
        """Clear all data from the database (use with caution!)"""
        with self.connection.get_session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.warning("All data cleared from database")
