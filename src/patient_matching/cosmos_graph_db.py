"""
Azure Cosmos DB Gremlin Graph Database Layer

Implements the graph database operations using Cosmos DB with Gremlin API.
"""

import os
import json
import logging
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import asdict

from gremlin_python.driver import client, serializer
from gremlin_python.driver.protocol import GremlinServerError

from .models import (
    Patient, EmpiRecord, MatchResult, HumanName, Address,
    ContactPoint, Identifier, MatchConfidence, Gender, IdentifierType,
    Encounter, Observation, Condition, Procedure, Immunization,
    MedicationRequest, DiagnosticReport, CodeableConcept, Period, Quantity,
    EncounterClass, EncounterStatus, ObservationCategory
)

logger = logging.getLogger(__name__)


class CosmosGraphDB:
    """
    Cosmos DB Gremlin API client for patient matching graph operations.
    
    Graph Schema:
    - Vertices: Patient, Identifier, Address, ContactPoint, EmpiRecord,
                Encounter, Observation, Condition, Procedure, Immunization,
                MedicationRequest, DiagnosticReport
    - Edges: HAS_IDENTIFIER, HAS_ADDRESS, HAS_CONTACT, LINKED_TO, POTENTIAL_MATCH,
             HAS_ENCOUNTER, HAS_OBSERVATION, HAS_CONDITION, HAS_PROCEDURE,
             HAS_IMMUNIZATION, HAS_MEDICATION, HAS_DIAGNOSTIC_REPORT,
             PART_OF_ENCOUNTER, HAS_RESULT
    
    Partition Strategy:
    - All vertices use source_system as partition key
    - Patient: source_system = patient source system
    - Clinical resources: source_system = same as patient for efficient traversals
    """
    
    def __init__(
        self,
        endpoint: str = None,
        database: str = None,
        container: str = None,
        key: str = None
    ):
        """
        Initialize Cosmos DB Gremlin connection.
        
        Args:
            endpoint: Cosmos DB Gremlin endpoint (e.g., 'your-account.gremlin.cosmos.azure.com')
            database: Database name
            container: Graph container name
            key: Cosmos DB primary key
        """
        self.endpoint = endpoint or os.getenv("COSMOS_GREMLIN_ENDPOINT", "localhost")
        self.database = database or os.getenv("COSMOS_DATABASE", "PatientMatching")
        self.container = container or os.getenv("COSMOS_CONTAINER", "PatientGraph")
        self.key = key or os.getenv("COSMOS_KEY", "")
        
        self._client: Optional[client.Client] = None
    
    def connect(self) -> None:
        """Establish connection to Cosmos DB Gremlin endpoint."""
        try:
            # Cosmos DB Gremlin endpoint format
            # Endpoint should be like: your-account.gremlin.cosmos.azure.com
            # Remove any protocol prefix if present
            endpoint = self.endpoint
            if endpoint.startswith("wss://"):
                endpoint = endpoint[6:]
            if endpoint.startswith("https://"):
                endpoint = endpoint[8:]
            if endpoint.endswith("/"):
                endpoint = endpoint[:-1]
            
            # Full URL must include the database and collection in the path
            gremlin_url = f"wss://{endpoint}:443/"
            
            logger.info(f"Connecting to Cosmos DB Gremlin...")
            logger.info(f"  URL: {gremlin_url}")
            logger.info(f"  Username: /dbs/{self.database}/colls/{self.container}")
            
            self._client = client.Client(
                gremlin_url,
                "g",
                username=f"/dbs/{self.database}/colls/{self.container}",
                password=self.key,
                message_serializer=serializer.GraphSONSerializersV2d0()
            )
            
            # Test connection
            self._execute_query("g.V().count()")
            logger.info(f"Connected to Cosmos DB Gremlin: {endpoint}/{self.database}/{self.container}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Cosmos DB: {e}")
            raise
    
    def close(self) -> None:
        """Close the database connection."""
        if self._client:
            self._client.close()
            self._client = None
    
    def _execute_query(self, query: str, bindings: Dict[str, Any] = None, max_retries: int = 5) -> List[Any]:
        """
        Execute a Gremlin query with retry logic for rate limiting.
        
        Args:
            query: Gremlin query string
            bindings: Parameter bindings (will be substituted into query for Cosmos DB)
            max_retries: Maximum number of retries for 429 errors
            
        Returns:
            Query results as a list
        """
        if not self._client:
            raise RuntimeError("Not connected to Cosmos DB. Call connect() first.")
        
        # Cosmos DB Gremlin has limited parameterized query support
        # Use placeholder markers for safe substitution
        final_query = query
        if bindings:
            # First, replace all binding references in query with unique placeholders
            # Then substitute the placeholders with properly escaped values
            # This prevents regex matching issues with substituted values
            
            # Build escaped values
            escaped_values = {}
            for key, value in bindings.items():
                if isinstance(value, str):
                    # Escape backslashes first, then single quotes with backslash (Cosmos DB Gremlin syntax)
                    # Per Microsoft docs: \' escapes apostrophe, \\ escapes backslash
                    escaped_value = value.replace("\\", "\\\\").replace("'", "\\'")
                    escaped_values[key] = f"'{escaped_value}'"
                elif isinstance(value, bool):
                    escaped_values[key] = str(value).lower()
                elif value is None:
                    escaped_values[key] = "''"
                else:
                    escaped_values[key] = str(value)
            
            # Sort by key length descending to prevent 'id' from matching before 'patientId'
            sorted_keys = sorted(escaped_values.keys(), key=len, reverse=True)
            
            # Replace each binding with a unique placeholder first
            placeholders = {}
            for i, key in enumerate(sorted_keys):
                placeholder = f"__PLACEHOLDER_{i:04d}__"
                placeholders[placeholder] = escaped_values[key]
                # Match binding name in proper context (after comma/paren/space, before comma/paren/space/newline)
                pattern = rf"(?<=[,\(\s]){re.escape(key)}(?=[,\)\s\n])"
                final_query = re.sub(pattern, placeholder, final_query)
            
            # Now replace placeholders with actual values (simple string replace, no regex)
            for placeholder, value in placeholders.items():
                final_query = final_query.replace(placeholder, value)
        
        # Debug: log the final query for troubleshooting
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Executing Gremlin query:\n{final_query}")
        
        # Retry loop with exponential backoff for rate limiting
        retry_count = 0
        base_delay = 1.0  # Start with 1 second delay
        
        while True:
            try:
                result_set = self._client.submit(final_query)
                results = []
                for result in result_set:
                    results.extend(result)
                return results
            except GremlinServerError as e:
                error_message = str(e)
                # Check for 429 TooManyRequests error
                if "429" in error_message or "TooManyRequests" in error_message or "RequestRateTooLarge" in error_message:
                    retry_count += 1
                    if retry_count > max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded for rate limiting")
                        raise
                    
                    # Extract RetryAfterInMs if available, otherwise use exponential backoff
                    delay = base_delay * (2 ** (retry_count - 1))  # Exponential backoff
                    
                    # Try to extract retry-after hint from error message
                    if "RetryAfterInMs" in error_message:
                        try:
                            import re as regex
                            match = regex.search(r'"RetryAfterInMs"\s*:\s*"?(\d+)"?', error_message)
                            if match:
                                delay = max(delay, int(match.group(1)) / 1000.0 + 0.5)  # Add buffer
                        except:
                            pass
                    
                    logger.warning(f"Rate limited (429). Retry {retry_count}/{max_retries} after {delay:.1f}s")
                    time.sleep(delay)
                # Check for 409 Conflict (already exists) - don't log, just re-raise for caller to handle
                elif self._is_conflict_error(e):
                    raise
                else:
                    logger.error(f"Gremlin query error: {e}")
                    raise
    
    def initialize_schema(self) -> None:
        """
        Initialize graph schema.
        
        Note: Cosmos DB Gremlin doesn't have explicit schema definition,
        but we can create indexes via Azure Portal or ARM templates.
        This method documents the expected structure.
        """
        logger.info("Cosmos DB Gremlin schema is schema-less. Ensure indexes are configured in Azure Portal.")
        logger.info("Expected vertex labels: Patient, Identifier, Address, ContactPoint, EmpiRecord")
        logger.info("Expected edge labels: HAS_IDENTIFIER, HAS_ADDRESS, HAS_CONTACT, LINKED_TO, POTENTIAL_MATCH")
    
    def clear_all_data(self) -> None:
        """
        Clear all vertices and edges from the graph.
        Use with caution - this deletes all data!
        Uses batched deletion to avoid RU throttling.
        """
        logger.warning("Clearing all data from the graph...")
        
        # Delete in batches to avoid RU exhaustion
        batch_size = 100
        total_deleted = 0
        
        while True:
            # Get count of remaining vertices
            count_result = self._execute_query("g.V().count()")
            remaining = count_result[0] if count_result else 0
            
            if remaining == 0:
                break
                
            # Delete a batch
            self._execute_query(f"g.V().limit({batch_size}).drop()")
            total_deleted += min(batch_size, remaining)
            
            if total_deleted % 500 == 0 or remaining <= batch_size:
                logger.info(f"  Deleted {total_deleted} vertices, {max(0, remaining - batch_size)} remaining...")
        
        logger.info(f"All data cleared from graph. Total deleted: {total_deleted}")

    # ==================== Patient Operations ====================
    
    def create_patient(self, patient: Patient) -> str:
        """
        Create a patient vertex with related vertices and edges.
        
        Args:
            patient: Patient model instance
            
        Returns:
            Patient ID
        """
        # Create patient vertex
        patient_props = self._build_patient_properties(patient)
        
        query = """
        g.addV('Patient')
            .property('id', id)
            .property('source_system', source_system)
            .property('sourceId', sourceId)
            .property('sourceSystem', sourceSystem)
            .property('firstName', firstName)
            .property('lastName', lastName)
            .property('fullName', fullName)
            .property('birthDate', birthDate)
            .property('gender', gender)
            .property('active', active)
            .property('createdAt', createdAt)
        """
        
        self._execute_query(query, patient_props)
        
        # Create identifiers and edges
        for identifier in patient.identifiers:
            self._create_identifier(patient.id, identifier, patient.source_system)
        
        # Create addresses and edges
        for address in patient.addresses:
            self._create_address(patient.id, address, patient.source_system)
        
        # Create contact points and edges
        for contact in patient.contact_points:
            self._create_contact_point(patient.id, contact, patient.source_system)
        
        logger.info(f"Created patient vertex: {patient.id}")
        return patient.id
    
    def _build_patient_properties(self, patient: Patient) -> Dict[str, Any]:
        """Build property bindings for patient vertex."""
        return {
            "id": patient.id,
            "source_system": patient.source_system,  # Partition key - matches /source_system in Cosmos DB
            "sourceId": patient.source_id,
            "sourceSystem": patient.source_system,
            "firstName": patient.name.given[0] if patient.name and patient.name.given else "",
            "lastName": patient.name.family if patient.name else "",
            "fullName": patient.name.full_name if patient.name else "",
            "birthDate": patient.birth_date.isoformat() if patient.birth_date else "",
            "gender": patient.gender.value if patient.gender else "",
            "active": getattr(patient, 'active', True),
            "createdAt": datetime.utcnow().isoformat()
        }
    
    def _create_identifier(self, patient_id: str, identifier: Identifier, source_system: str = None) -> None:
        """Create an identifier vertex and link to patient."""
        # Use type+value as unique id, but partition by source_system
        id_vertex_id = f"id_{patient_id}_{identifier.type.value}_{identifier.normalize()}"
        partition_key = source_system if source_system else getattr(identifier, 'source_system', None)
        if not partition_key:
            logger.warning("No source_system provided for identifier; using patient_id as fallback partition key.")
            partition_key = patient_id
        # Check if identifier already exists (shared identifier case)
        check_query = "g.V().has('Identifier', 'source_system', source_system).has('id', id).count()"
        result = self._execute_query(check_query, {"source_system": partition_key, "id": id_vertex_id})
        if result and result[0] == 0:
            # Create new identifier vertex
            create_query = """
            g.addV('Identifier')
                .property('id', id)
                .property('source_system', source_system)
                .property('type', type)
                .property('value', value)
                .property('normalizedValue', normalizedValue)
                .property('system', system)
            """
            self._execute_query(create_query, {
                "id": id_vertex_id,
                "source_system": partition_key,
                "type": identifier.type.value,
                "value": identifier.value,
                "normalizedValue": identifier.normalize(),
                "system": identifier.system or ""
            })
        # Create edge from patient to identifier
        edge_query = """
        g.V().has('Patient', 'id', patientId).has('source_system', source_system)
            .addE('HAS_IDENTIFIER')
            .to(g.V().has('Identifier', 'id', identifierId).has('source_system', source_system))
        """
        self._execute_query(edge_query, {
            "patientId": patient_id,
            "identifierId": id_vertex_id,
            "source_system": partition_key
        })

    def _create_address(self, patient_id: str, address: Address, source_system: str = None) -> None:
        """Create an address vertex and link to patient."""
        address_id = f"addr_{patient_id}_{hash(address.normalize())}"
        partition_key = source_system if source_system else getattr(address, 'source_system', None)
        if not partition_key:
            logger.warning("No source_system provided for address; using patient_id as fallback partition key.")
            partition_key = patient_id
        query = """
        g.addV('Address')
            .property('id', id)
            .property('source_system', source_system)
            .property('line', line)
            .property('city', city)
            .property('state', state)
            .property('postalCode', postalCode)
            .property('country', country)
            .property('fullAddress', fullAddress)
            .property('normalized', normalized)
        """
        self._execute_query(query, {
            "id": address_id,
            "source_system": partition_key,
            "line": ", ".join(address.line) if address.line else "",
            "city": address.city or "",
            "state": address.state or "",
            "postalCode": address.postal_code or "",
            "country": address.country or "",
            "fullAddress": address.full_address,
            "normalized": address.normalize()
        })
        # Create edge
        edge_query = """
        g.V().has('Patient', 'id', patientId).has('source_system', source_system)
            .addE('HAS_ADDRESS')
            .to(g.V().has('Address', 'id', addressId).has('source_system', source_system))
        """
        self._execute_query(edge_query, {
            "patientId": patient_id,
            "addressId": address_id,
            "source_system": partition_key
        })

    def _create_contact_point(self, patient_id: str, contact: ContactPoint, source_system: str = None) -> None:
        """Create a contact point vertex and link to patient."""
        contact_id = f"contact_{patient_id}_{contact.system.value}_{hash(contact.normalize())}"
        normalized = contact.normalize()
        partition_key = source_system if source_system else getattr(contact, 'source_system', None)
        if not partition_key:
            logger.warning("No source_system provided for contact; using patient_id as fallback partition key.")
            partition_key = patient_id
        # Check if contact already exists
        check_query = "g.V().has('ContactPoint', 'source_system', source_system).has('id', id).count()"
        result = self._execute_query(check_query, {"source_system": partition_key, "id": contact_id})
        if result and result[0] == 0:
            query = """
            g.addV('ContactPoint')
                .property('id', id)
                .property('source_system', source_system)
                .property('system', system)
                .property('value', value)
                .property('normalizedValue', normalizedValue)
            """
            self._execute_query(query, {
                "id": contact_id,
                "source_system": partition_key,
                "system": contact.system.value,
                "value": contact.value,
                "normalizedValue": normalized
            })
        # Create edge
        edge_query = """
        g.V().has('Patient', 'id', patientId).has('source_system', source_system)
            .addE('HAS_CONTACT')
            .to(g.V().has('ContactPoint', 'id', contactId).has('source_system', source_system))
        """
        self._execute_query(edge_query, {
            "patientId": patient_id,
            "contactId": contact_id,
            "source_system": partition_key
        })

    def get_patient(self, patient_id: str) -> Optional[Patient]:
        """
        Retrieve a patient by ID with all related data.
        
        Args:
            patient_id: Patient ID
            
        Returns:
            Patient object or None if not found
        """
        # Get patient vertex
        query = "g.V().has('Patient', 'id', patientId).valueMap(true)"
        results = self._execute_query(query, {"patientId": patient_id})
        
        if not results:
            return None
        
        patient_data = results[0]
        
        # Get identifiers
        id_query = """
        g.V().has('Patient', 'id', patientId)
            .out('HAS_IDENTIFIER')
            .valueMap(true)
        """
        id_results = self._execute_query(id_query, {"patientId": patient_id})
        
        # Get addresses
        addr_query = """
        g.V().has('Patient', 'id', patientId)
            .out('HAS_ADDRESS')
            .valueMap(true)
        """
        addr_results = self._execute_query(addr_query, {"patientId": patient_id})
        
        # Get contacts
        contact_query = """
        g.V().has('Patient', 'id', patientId)
            .out('HAS_CONTACT')
            .valueMap(true)
        """
        contact_results = self._execute_query(contact_query, {"patientId": patient_id})
        
        return self._build_patient_from_results(patient_data, id_results, addr_results, contact_results)
    
    def _build_patient_from_results(
        self,
        patient_data: Dict,
        identifiers: List[Dict],
        addresses: List[Dict],
        contacts: List[Dict]
    ) -> Patient:
        """Build Patient object from query results."""
        
        # Helper to extract value from Cosmos result (returns list)
        def get_val(data: Dict, key: str, default: Any = None) -> Any:
            val = data.get(key, [default])
            return val[0] if isinstance(val, list) and val else default
        
        # Note: 'id' is at top level, not in array (Gremlin valueMap(true))
        patient_id = patient_data.get("id", "")
        
        # Parse name
        first_name = get_val(patient_data, "firstName", "")
        last_name = get_val(patient_data, "lastName", "")
        name = HumanName(family=last_name, given=[first_name] if first_name else [])
        
        # Parse birth date
        birth_date_str = get_val(patient_data, "birthDate", "")
        birth_date = None
        if birth_date_str:
            from datetime import date
            try:
                birth_date = date.fromisoformat(birth_date_str)
            except ValueError:
                pass
        
        # Parse gender
        gender_str = get_val(patient_data, "gender", "")
        gender = Gender(gender_str) if gender_str else None
        
        # Parse identifiers
        parsed_identifiers = []
        for id_data in identifiers:
            id_type_str = get_val(id_data, "type", "other")
            try:
                id_type = IdentifierType(id_type_str)
            except ValueError:
                id_type = IdentifierType.OTHER
            
            parsed_identifiers.append(Identifier(
                type=id_type,
                value=get_val(id_data, "value", ""),
                system=get_val(id_data, "system", None)
            ))
        
        # Parse addresses
        parsed_addresses = []
        for addr_data in addresses:
            line_str = get_val(addr_data, "line", "")
            parsed_addresses.append(Address(
                line=line_str.split(", ") if line_str else [],
                city=get_val(addr_data, "city", None),
                state=get_val(addr_data, "state", None),
                postal_code=get_val(addr_data, "postalCode", None),
                country=get_val(addr_data, "country", None)
            ))
        
        # Parse contacts
        from .models import ContactPointSystem
        parsed_contacts = []
        for contact_data in contacts:
            system_str = get_val(contact_data, "system", "other")
            try:
                system = ContactPointSystem(system_str)
            except ValueError:
                system = ContactPointSystem.OTHER
            
            parsed_contacts.append(ContactPoint(
                system=system,
                value=get_val(contact_data, "value", "")
            ))
        
        return Patient(
            id=patient_id,
            source_id=get_val(patient_data, "sourceId", ""),
            source_system=get_val(patient_data, "sourceSystem", ""),
            name=name,
            birth_date=birth_date,
            gender=gender,
            identifiers=parsed_identifiers,
            addresses=parsed_addresses,
            contact_points=parsed_contacts
            # Note: 'active' not supported by Patient model
        )
    
    def search_patients(
        self,
        first_name: str = None,
        last_name: str = None,
        birth_date: str = None,
        identifier_value: str = None,
        limit: int = 100
    ) -> List[Patient]:
        """
        Search for patients by various criteria.
        
        Args:
            first_name: First name (partial match)
            last_name: Last name (partial match)
            birth_date: Birth date (ISO format)
            identifier_value: Identifier value
            limit: Maximum results
            
        Returns:
            List of matching patients
        """
        # Build query based on provided criteria
        query_parts = ["g.V().hasLabel('Patient')"]
        bindings = {"limit": limit}
        
        if last_name:
            query_parts.append(".has('lastName', lastName)")
            bindings["lastName"] = last_name
        
        if first_name:
            query_parts.append(".has('firstName', firstName)")
            bindings["firstName"] = first_name
        
        if birth_date:
            query_parts.append(".has('birthDate', birthDate)")
            bindings["birthDate"] = birth_date
        
        query_parts.append(".limit(limit).valueMap(true)")
        query = "".join(query_parts)
        
        results = self._execute_query(query, bindings)
        
        patients = []
        for patient_data in results:
            patient_id = patient_data.get("id", [None])[0]
            if patient_id:
                patient = self.get_patient(patient_id)
                if patient:
                    patients.append(patient)
        
        return patients
    
    # ==================== Graph-Based Matching ====================
    
    def find_candidates_by_identifier(self, patient_id: str) -> List[Tuple[str, List[str]]]:
        """
        Find candidate matches by shared identifiers using graph traversal.
        
        Args:
            patient_id: Source patient ID
            
        Returns:
            List of (candidate_id, shared_identifier_values) tuples
        """
        query = """
        g.V().has('Patient', 'id', patientId).as('p1')
            .out('HAS_IDENTIFIER').as('id')
            .in('HAS_IDENTIFIER').as('p2')
            .where('p1', neq('p2'))
            .select('p2', 'id')
            .by('id')
            .by('value')
            .dedup()
        """
        
        results = self._execute_query(query, {"patientId": patient_id})
        
        # Group by candidate patient
        candidates = {}
        for result in results:
            candidate_id = result.get("p2")
            id_value = result.get("id")
            if candidate_id:
                if candidate_id not in candidates:
                    candidates[candidate_id] = []
                candidates[candidate_id].append(id_value)
        
        return list(candidates.items())
    
    def find_candidates_by_contact(self, patient_id: str) -> List[Tuple[str, List[str]]]:
        """
        Find candidate matches by shared contact points (phone, email).
        
        Args:
            patient_id: Source patient ID
            
        Returns:
            List of (candidate_id, shared_contact_values) tuples
        """
        query = """
        g.V().has('Patient', 'id', patientId).as('p1')
            .out('HAS_CONTACT').as('contact')
            .in('HAS_CONTACT').as('p2')
            .where('p1', neq('p2'))
            .select('p2', 'contact')
            .by('id')
            .by('normalizedValue')
            .dedup()
        """
        
        results = self._execute_query(query, {"patientId": patient_id})
        
        candidates = {}
        for result in results:
            candidate_id = result.get("p2")
            contact_value = result.get("contact")
            if candidate_id:
                if candidate_id not in candidates:
                    candidates[candidate_id] = []
                candidates[candidate_id].append(contact_value)
        
        return list(candidates.items())
    
    def find_candidates_by_demographics(
        self,
        patient_id: str,
        limit: int = 50
    ) -> List[str]:
        """
        Find candidates with same DOB and last name first letter (blocking).
        
        Args:
            patient_id: Source patient ID
            limit: Maximum candidates
            
        Returns:
            List of candidate patient IDs
        """
        # Get patient's DOB and last name
        query = "g.V().has('Patient', 'id', patientId).valueMap('birthDate', 'lastName')"
        results = self._execute_query(query, {"patientId": patient_id})
        
        if not results:
            return []
        
        patient_data = results[0]
        birth_date = patient_data.get("birthDate", [None])[0]
        last_name = patient_data.get("lastName", [""])[0]
        
        if not birth_date:
            return []
        
        # Find patients with same DOB
        candidate_query = """
        g.V().hasLabel('Patient')
            .has('birthDate', birthDate)
            .has('id', neq(patientId))
            .limit(limit)
            .values('id')
        """
        
        candidates = self._execute_query(candidate_query, {
            "birthDate": birth_date,
            "patientId": patient_id,
            "limit": limit
        })
        
        return candidates
    
    # ==================== Potential Match Operations ====================
    
    def create_potential_match(
        self,
        patient1_id: str,
        patient2_id: str,
        score: float,
        confidence: MatchConfidence,
        details: Dict[str, Any] = None
    ) -> None:
        """
        Create a POTENTIAL_MATCH edge between two patients.
        
        Args:
            patient1_id: First patient ID
            patient2_id: Second patient ID
            score: Match score (0-1)
            confidence: Match confidence level
            details: Additional match details
        """
        # Check if edge already exists
        check_query = """
        g.V().has('Patient', 'id', p1Id)
            .outE('POTENTIAL_MATCH')
            .where(inV().has('id', p2Id))
            .count()
        """
        result = self._execute_query(check_query, {
            "p1Id": patient1_id,
            "p2Id": patient2_id
        })
        
        if result and result[0] > 0:
            # Update existing edge
            update_query = """
            g.V().has('Patient', 'id', p1Id)
                .outE('POTENTIAL_MATCH')
                .where(inV().has('id', p2Id))
                .property('score', score)
                .property('confidence', confidence)
                .property('updatedAt', updatedAt)
            """
        else:
            # Create new edge
            update_query = """
            g.V().has('Patient', 'id', p1Id)
                .addE('POTENTIAL_MATCH')
                .to(g.V().has('Patient', 'id', p2Id))
                .property('score', score)
                .property('confidence', confidence)
                .property('createdAt', updatedAt)
            """
        
        self._execute_query(update_query, {
            "p1Id": patient1_id,
            "p2Id": patient2_id,
            "score": score,
            "confidence": confidence.value,
            "updatedAt": datetime.utcnow().isoformat()
        })
    
    def store_potential_match(self, match: MatchResult) -> None:
        """
        Store a potential match relationship between two patients.
        
        Wrapper for create_potential_match that accepts a MatchResult object.
        
        Args:
            match: MatchResult object containing match details
        """
        self.create_potential_match(
            patient1_id=match.patient1_id,
            patient2_id=match.patient2_id,
            score=match.score,
            confidence=match.confidence,
            details={
                "deterministic_score": match.deterministic_score,
                "name_similarity": match.name_similarity,
                "address_similarity": match.address_similarity,
                "embedding_similarity": match.embedding_similarity,
                "shared_identifiers": match.shared_identifiers
            }
        )
    
    def get_potential_matches(
        self,
        patient_id: str,
        min_score: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Get all potential matches for a patient.
        
        Args:
            patient_id: Patient ID
            min_score: Minimum match score
            
        Returns:
            List of match results with scores
        """
        query = """
        g.V().has('Patient', 'id', patientId)
            .outE('POTENTIAL_MATCH')
            .has('score', gte(minScore))
            .as('e')
            .inV().as('p')
            .select('e', 'p')
            .by(valueMap())
            .by(valueMap('id', 'fullName', 'birthDate'))
        """
        
        results = self._execute_query(query, {
            "patientId": patient_id,
            "minScore": min_score
        })
        
        matches = []
        for result in results:
            edge = result.get("e", {})
            patient = result.get("p", {})
            matches.append({
                "patient_id": patient.get("id", [None])[0],
                "full_name": patient.get("fullName", [""])[0],
                "birth_date": patient.get("birthDate", [""])[0],
                "score": edge.get("score", [0])[0],
                "confidence": edge.get("confidence", [""])[0]
            })
        
        return matches
    
    def get_pending_reviews(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get potential matches requiring human review.
        
        Args:
            limit: Maximum results
            
        Returns:
            List of pending review match pairs
        """
        query = """
        g.E().hasLabel('POTENTIAL_MATCH')
            .has('confidence', 'human_review')
            .limit(limit)
            .as('e')
            .outV().as('p1')
            .select('e').inV().as('p2')
            .select('p1', 'p2', 'e')
            .by(valueMap('id', 'fullName', 'birthDate'))
            .by(valueMap('id', 'fullName', 'birthDate'))
            .by(valueMap())
        """
        
        results = self._execute_query(query, {"limit": limit})
        
        pending = []
        for result in results:
            p1 = result.get("p1", {})
            p2 = result.get("p2", {})
            edge = result.get("e", {})
            pending.append({
                "patient1": {
                    "id": p1.get("id", [None])[0],
                    "full_name": p1.get("fullName", [""])[0],
                    "birth_date": p1.get("birthDate", [""])[0]
                },
                "patient2": {
                    "id": p2.get("id", [None])[0],
                    "full_name": p2.get("fullName", [""])[0],
                    "birth_date": p2.get("birthDate", [""])[0]
                },
                "score": edge.get("score", [0])[0],
                "created_at": edge.get("createdAt", [""])[0]
            })
        
        return pending
    
    # ==================== EMPI Record Operations ====================
    
    def create_empi_record(
        self,
        patient_ids: List[str],
        empi_record: EmpiRecord
    ) -> str:
        """
        Create an EMPI Record and link patients to it.
        
        Args:
            patient_ids: List of patient IDs to link
            empi_record: EMPI Record data
            
        Returns:
            EMPI Record ID
        """
        # Create EMPI record vertex
        query = """
        g.addV('EmpiRecord')
            .property('id', id)
            .property('source_system', source_system)
            .property('firstName', firstName)
            .property('lastName', lastName)
            .property('fullName', fullName)
            .property('birthDate', birthDate)
            .property('gender', gender)
            .property('createdAt', createdAt)
            .property('sourceCount', sourceCount)
        """
        
        self._execute_query(query, {
            "id": empi_record.id,
            "source_system": getattr(empi_record, 'source_system', empi_record.id),
            "firstName": empi_record.name.given[0] if empi_record.name and empi_record.name.given else "",
            "lastName": empi_record.name.family if empi_record.name else "",
            "fullName": empi_record.name.full_name if empi_record.name else "",
            "birthDate": empi_record.birth_date.isoformat() if empi_record.birth_date else "",
            "gender": empi_record.gender.value if empi_record.gender else "",
            "createdAt": datetime.utcnow().isoformat(),
            "sourceCount": len(patient_ids)
        })
        
        # Link patients to EMPI record
        for patient_id in patient_ids:
            self.link_patient_to_empi_record(patient_id, empi_record.id, 1.0)
        
        logger.info(f"Created EMPI Record {empi_record.id} with {len(patient_ids)} patients")
        return empi_record.id
    
    def link_patient_to_empi_record(
        self,
        patient_id: str,
        empi_record_id: str,
        score: float
    ) -> None:
        """Link a patient to an EMPI Record."""
        query = """
        g.V().has('Patient', 'id', patientId)
            .addE('LINKED_TO')
            .to(g.V().has('EmpiRecord', 'id', empiId))
            .property('score', score)
            .property('linkedAt', linkedAt)
        """
        
        self._execute_query(query, {
            "patientId": patient_id,
            "empiId": empi_record_id,
            "score": score,
            "linkedAt": datetime.utcnow().isoformat()
        })
    
    def get_empi_record_patients(self, empi_record_id: str) -> List[Patient]:
        """Get all patients linked to an EMPI Record."""
        query = """
        g.V().has('EmpiRecord', 'id', empiId)
            .in('LINKED_TO')
            .values('id')
        """
        
        patient_ids = self._execute_query(query, {"empiId": empi_record_id})
        
        patients = []
        for patient_id in patient_ids:
            patient = self.get_patient(patient_id)
            if patient:
                patients.append(patient)
        
        return patients
    
    def unlink_patient_from_empi_record(
        self,
        patient_id: str,
        empi_record_id: str
    ) -> None:
        """Remove link between patient and EMPI Record."""
        query = """
        g.V().has('Patient', 'id', patientId)
            .outE('LINKED_TO')
            .where(inV().has('id', empiId))
            .drop()
        """
        
        self._execute_query(query, {
            "patientId": patient_id,
            "empiId": empi_record_id
        })
    
    # ==================== Clinical Data Operations ====================
    
    def _codeable_concept_to_props(self, cc: CodeableConcept) -> Dict[str, str]:
        """Convert CodeableConcept to property dict."""
        if not cc:
            return {"code": "", "display": "", "system": ""}
        return {
            "code": cc.code or "",
            "display": cc.display or cc.text or "",
            "system": cc.system or ""
        }
    
    def create_encounter(self, encounter: Encounter) -> str:
        """Create an Encounter vertex and link to patient."""
        # Build properties
        type_props = self._codeable_concept_to_props(encounter.type_code)
        reason_props = self._codeable_concept_to_props(encounter.reason_code)
        
        query = """
        g.addV('Encounter')
            .property('id', id)
            .property('source_system', source_system)
            .property('patientId', patientId)
            .property('status', status)
            .property('encounterClass', encounterClass)
            .property('typeCode', typeCode)
            .property('typeDisplay', typeDisplay)
            .property('periodStart', periodStart)
            .property('periodEnd', periodEnd)
            .property('serviceProvider', serviceProvider)
            .property('reasonCode', reasonCode)
            .property('reasonDisplay', reasonDisplay)
            .property('createdAt', createdAt)
        """
        
        period_start = ""
        period_end = ""
        if encounter.period:
            period_start = encounter.period.start.isoformat() if encounter.period.start else ""
            period_end = encounter.period.end.isoformat() if encounter.period.end else ""
        
        self._execute_query(query, {
            "id": encounter.id,
            "source_system": encounter.source_system,
            "patientId": encounter.patient_id,
            "status": encounter.status.value if encounter.status else "",
            "encounterClass": encounter.encounter_class.value if encounter.encounter_class else "",
            "typeCode": type_props["code"],
            "typeDisplay": type_props["display"],
            "periodStart": period_start,
            "periodEnd": period_end,
            "serviceProvider": encounter.service_provider or "",
            "reasonCode": reason_props["code"],
            "reasonDisplay": reason_props["display"],
            "createdAt": datetime.utcnow().isoformat()
        })
        
        # Create edge from patient to encounter
        edge_query = """
        g.V().has('Patient', 'id', patientId).has('source_system', source_system)
            .addE('HAS_ENCOUNTER')
            .to(g.V().has('Encounter', 'id', encounterId).has('source_system', source_system))
        """
        self._execute_query(edge_query, {
            "patientId": encounter.patient_id,
            "encounterId": encounter.id,
            "source_system": encounter.source_system
        })
        
        logger.debug(f"Created encounter vertex: {encounter.id}")
        return encounter.id
    
    def create_observation(self, observation: Observation) -> str:
        """Create an Observation vertex and link to patient (and optionally encounter)."""
        code_props = self._codeable_concept_to_props(observation.code)
        value_code_props = self._codeable_concept_to_props(observation.value_code)
        
        # Handle quantity value
        value_quantity_str = ""
        if observation.value_quantity:
            value_quantity_str = f"{observation.value_quantity.value} {observation.value_quantity.unit or ''}"
        
        query = """
        g.addV('Observation')
            .property('id', id)
            .property('source_system', source_system)
            .property('patientId', patientId)
            .property('status', status)
            .property('category', category)
            .property('code', code)
            .property('codeDisplay', codeDisplay)
            .property('codeSystem', codeSystem)
            .property('valueQuantity', valueQuantity)
            .property('valueString', valueString)
            .property('valueCode', valueCode)
            .property('effectiveDateTime', effectiveDateTime)
            .property('encounterId', encounterId)
            .property('createdAt', createdAt)
        """
        
        self._execute_query(query, {
            "id": observation.id,
            "source_system": observation.source_system,
            "patientId": observation.patient_id,
            "status": observation.status or "final",
            "category": observation.category.value if observation.category else "",
            "code": code_props["code"],
            "codeDisplay": code_props["display"],
            "codeSystem": code_props["system"],
            "valueQuantity": value_quantity_str,
            "valueString": observation.value_string or "",
            "valueCode": value_code_props.get("display", ""),
            "effectiveDateTime": observation.effective_datetime.isoformat() if observation.effective_datetime else "",
            "encounterId": observation.encounter_id or "",
            "createdAt": datetime.utcnow().isoformat()
        })
        
        # Create edge from patient to observation
        edge_query = """
        g.V().has('Patient', 'id', patientId).has('source_system', source_system)
            .addE('HAS_OBSERVATION')
            .to(g.V().has('Observation', 'id', observationId).has('source_system', source_system))
        """
        self._execute_query(edge_query, {
            "patientId": observation.patient_id,
            "observationId": observation.id,
            "source_system": observation.source_system
        })
        
        # If observation is part of an encounter, create PART_OF_ENCOUNTER edge
        if observation.encounter_id:
            enc_edge_query = """
            g.V().has('Observation', 'id', observationId).has('source_system', source_system)
                .addE('PART_OF_ENCOUNTER')
                .to(g.V().has('Encounter', 'id', encounterId).has('source_system', source_system))
            """
            try:
                self._execute_query(enc_edge_query, {
                    "observationId": observation.id,
                    "encounterId": observation.encounter_id,
                    "source_system": observation.source_system
                })
            except Exception as e:
                logger.debug(f"Could not link observation to encounter: {e}")
        
        logger.debug(f"Created observation vertex: {observation.id}")
        return observation.id
    
    def create_condition(self, condition: Condition) -> str:
        """Create a Condition vertex and link to patient."""
        code_props = self._codeable_concept_to_props(condition.code)
        
        query = """
        g.addV('Condition')
            .property('id', id)
            .property('source_system', source_system)
            .property('patientId', patientId)
            .property('code', code)
            .property('codeDisplay', codeDisplay)
            .property('codeSystem', codeSystem)
            .property('clinicalStatus', clinicalStatus)
            .property('verificationStatus', verificationStatus)
            .property('onsetDateTime', onsetDateTime)
            .property('abatementDateTime', abatementDateTime)
            .property('encounterId', encounterId)
            .property('createdAt', createdAt)
        """
        
        self._execute_query(query, {
            "id": condition.id,
            "source_system": condition.source_system,
            "patientId": condition.patient_id,
            "code": code_props["code"],
            "codeDisplay": code_props["display"],
            "codeSystem": code_props["system"],
            "clinicalStatus": condition.clinical_status or "",
            "verificationStatus": condition.verification_status or "",
            "onsetDateTime": condition.onset_datetime.isoformat() if condition.onset_datetime else "",
            "abatementDateTime": condition.abatement_datetime.isoformat() if condition.abatement_datetime else "",
            "encounterId": condition.encounter_id or "",
            "createdAt": datetime.utcnow().isoformat()
        })
        
        # Create edge from patient to condition
        edge_query = """
        g.V().has('Patient', 'id', patientId).has('source_system', source_system)
            .addE('HAS_CONDITION')
            .to(g.V().has('Condition', 'id', conditionId).has('source_system', source_system))
        """
        self._execute_query(edge_query, {
            "patientId": condition.patient_id,
            "conditionId": condition.id,
            "source_system": condition.source_system
        })
        
        # Link to encounter if present
        if condition.encounter_id:
            enc_edge_query = """
            g.V().has('Condition', 'id', conditionId).has('source_system', source_system)
                .addE('PART_OF_ENCOUNTER')
                .to(g.V().has('Encounter', 'id', encounterId).has('source_system', source_system))
            """
            try:
                self._execute_query(enc_edge_query, {
                    "conditionId": condition.id,
                    "encounterId": condition.encounter_id,
                    "source_system": condition.source_system
                })
            except Exception as e:
                logger.debug(f"Could not link condition to encounter: {e}")
        
        logger.debug(f"Created condition vertex: {condition.id}")
        return condition.id
    
    def create_procedure(self, procedure: Procedure) -> str:
        """Create a Procedure vertex and link to patient."""
        code_props = self._codeable_concept_to_props(procedure.code)
        
        performed_dt = ""
        if procedure.performed_datetime:
            performed_dt = procedure.performed_datetime.isoformat()
        elif procedure.performed_period and procedure.performed_period.start:
            performed_dt = procedure.performed_period.start.isoformat()
        
        query = """
        g.addV('Procedure')
            .property('id', id)
            .property('source_system', source_system)
            .property('patientId', patientId)
            .property('status', status)
            .property('code', code)
            .property('codeDisplay', codeDisplay)
            .property('codeSystem', codeSystem)
            .property('performedDateTime', performedDateTime)
            .property('encounterId', encounterId)
            .property('createdAt', createdAt)
        """
        
        self._execute_query(query, {
            "id": procedure.id,
            "source_system": procedure.source_system,
            "patientId": procedure.patient_id,
            "status": procedure.status or "completed",
            "code": code_props["code"],
            "codeDisplay": code_props["display"],
            "codeSystem": code_props["system"],
            "performedDateTime": performed_dt,
            "encounterId": procedure.encounter_id or "",
            "createdAt": datetime.utcnow().isoformat()
        })
        
        # Create edge from patient to procedure
        edge_query = """
        g.V().has('Patient', 'id', patientId).has('source_system', source_system)
            .addE('HAS_PROCEDURE')
            .to(g.V().has('Procedure', 'id', procedureId).has('source_system', source_system))
        """
        self._execute_query(edge_query, {
            "patientId": procedure.patient_id,
            "procedureId": procedure.id,
            "source_system": procedure.source_system
        })
        
        # Link to encounter if present
        if procedure.encounter_id:
            enc_edge_query = """
            g.V().has('Procedure', 'id', procedureId).has('source_system', source_system)
                .addE('PART_OF_ENCOUNTER')
                .to(g.V().has('Encounter', 'id', encounterId).has('source_system', source_system))
            """
            try:
                self._execute_query(enc_edge_query, {
                    "procedureId": procedure.id,
                    "encounterId": procedure.encounter_id,
                    "source_system": procedure.source_system
                })
            except Exception as e:
                logger.debug(f"Could not link procedure to encounter: {e}")
        
        logger.debug(f"Created procedure vertex: {procedure.id}")
        return procedure.id
    
    def create_immunization(self, immunization: Immunization) -> str:
        """Create an Immunization vertex and link to patient."""
        vaccine_props = self._codeable_concept_to_props(immunization.vaccine_code)
        
        query = """
        g.addV('Immunization')
            .property('id', id)
            .property('source_system', source_system)
            .property('patientId', patientId)
            .property('status', status)
            .property('vaccineCode', vaccineCode)
            .property('vaccineDisplay', vaccineDisplay)
            .property('vaccineSystem', vaccineSystem)
            .property('occurrenceDateTime', occurrenceDateTime)
            .property('lotNumber', lotNumber)
            .property('encounterId', encounterId)
            .property('createdAt', createdAt)
        """
        
        self._execute_query(query, {
            "id": immunization.id,
            "source_system": immunization.source_system,
            "patientId": immunization.patient_id,
            "status": immunization.status or "completed",
            "vaccineCode": vaccine_props["code"],
            "vaccineDisplay": vaccine_props["display"],
            "vaccineSystem": vaccine_props["system"],
            "occurrenceDateTime": immunization.occurrence_datetime.isoformat() if immunization.occurrence_datetime else "",
            "lotNumber": immunization.lot_number or "",
            "encounterId": immunization.encounter_id or "",
            "createdAt": datetime.utcnow().isoformat()
        })
        
        # Create edge from patient to immunization
        edge_query = """
        g.V().has('Patient', 'id', patientId).has('source_system', source_system)
            .addE('HAS_IMMUNIZATION')
            .to(g.V().has('Immunization', 'id', immunizationId).has('source_system', source_system))
        """
        self._execute_query(edge_query, {
            "patientId": immunization.patient_id,
            "immunizationId": immunization.id,
            "source_system": immunization.source_system
        })
        
        # Link to encounter if present
        if immunization.encounter_id:
            enc_edge_query = """
            g.V().has('Immunization', 'id', immunizationId).has('source_system', source_system)
                .addE('PART_OF_ENCOUNTER')
                .to(g.V().has('Encounter', 'id', encounterId).has('source_system', source_system))
            """
            try:
                self._execute_query(enc_edge_query, {
                    "immunizationId": immunization.id,
                    "encounterId": immunization.encounter_id,
                    "source_system": immunization.source_system
                })
            except Exception as e:
                logger.debug(f"Could not link immunization to encounter: {e}")
        
        logger.debug(f"Created immunization vertex: {immunization.id}")
        return immunization.id
    
    def create_medication_request(self, medication: MedicationRequest) -> str:
        """Create a MedicationRequest vertex and link to patient."""
        med_props = self._codeable_concept_to_props(medication.medication_code)
        
        query = """
        g.addV('MedicationRequest')
            .property('id', id)
            .property('source_system', source_system)
            .property('patientId', patientId)
            .property('status', status)
            .property('intent', intent)
            .property('medicationCode', medicationCode)
            .property('medicationDisplay', medicationDisplay)
            .property('medicationSystem', medicationSystem)
            .property('authoredOn', authoredOn)
            .property('dosageInstruction', dosageInstruction)
            .property('encounterId', encounterId)
            .property('createdAt', createdAt)
        """
        
        self._execute_query(query, {
            "id": medication.id,
            "source_system": medication.source_system,
            "patientId": medication.patient_id,
            "status": medication.status or "active",
            "intent": medication.intent or "order",
            "medicationCode": med_props["code"],
            "medicationDisplay": med_props["display"],
            "medicationSystem": med_props["system"],
            "authoredOn": medication.authored_on.isoformat() if medication.authored_on else "",
            "dosageInstruction": medication.dosage_instruction or "",
            "encounterId": medication.encounter_id or "",
            "createdAt": datetime.utcnow().isoformat()
        })
        
        # Create edge from patient to medication
        edge_query = """
        g.V().has('Patient', 'id', patientId).has('source_system', source_system)
            .addE('HAS_MEDICATION')
            .to(g.V().has('MedicationRequest', 'id', medicationId).has('source_system', source_system))
        """
        self._execute_query(edge_query, {
            "patientId": medication.patient_id,
            "medicationId": medication.id,
            "source_system": medication.source_system
        })
        
        # Link to encounter if present
        if medication.encounter_id:
            enc_edge_query = """
            g.V().has('MedicationRequest', 'id', medicationId).has('source_system', source_system)
                .addE('PART_OF_ENCOUNTER')
                .to(g.V().has('Encounter', 'id', encounterId).has('source_system', source_system))
            """
            try:
                self._execute_query(enc_edge_query, {
                    "medicationId": medication.id,
                    "encounterId": medication.encounter_id,
                    "source_system": medication.source_system
                })
            except Exception as e:
                logger.debug(f"Could not link medication to encounter: {e}")
        
        logger.debug(f"Created medication request vertex: {medication.id}")
        return medication.id
    
    def create_diagnostic_report(self, report: DiagnosticReport) -> str:
        """Create a DiagnosticReport vertex and link to patient and observations."""
        category_props = self._codeable_concept_to_props(report.category)
        code_props = self._codeable_concept_to_props(report.code)
        
        query = """
        g.addV('DiagnosticReport')
            .property('id', id)
            .property('source_system', source_system)
            .property('patientId', patientId)
            .property('status', status)
            .property('category', category)
            .property('categoryDisplay', categoryDisplay)
            .property('code', code)
            .property('codeDisplay', codeDisplay)
            .property('codeSystem', codeSystem)
            .property('effectiveDateTime', effectiveDateTime)
            .property('issued', issued)
            .property('conclusion', conclusion)
            .property('encounterId', encounterId)
            .property('createdAt', createdAt)
        """
        
        self._execute_query(query, {
            "id": report.id,
            "source_system": report.source_system,
            "patientId": report.patient_id,
            "status": report.status or "final",
            "category": category_props["code"],
            "categoryDisplay": category_props["display"],
            "code": code_props["code"],
            "codeDisplay": code_props["display"],
            "codeSystem": code_props["system"],
            "effectiveDateTime": report.effective_datetime.isoformat() if report.effective_datetime else "",
            "issued": report.issued.isoformat() if report.issued else "",
            "conclusion": report.conclusion or "",
            "encounterId": report.encounter_id or "",
            "createdAt": datetime.utcnow().isoformat()
        })
        
        # Create edge from patient to diagnostic report
        edge_query = """
        g.V().has('Patient', 'id', patientId).has('source_system', source_system)
            .addE('HAS_DIAGNOSTIC_REPORT')
            .to(g.V().has('DiagnosticReport', 'id', reportId).has('source_system', source_system))
        """
        self._execute_query(edge_query, {
            "patientId": report.patient_id,
            "reportId": report.id,
            "source_system": report.source_system
        })
        
        # Link to encounter if present
        if report.encounter_id:
            enc_edge_query = """
            g.V().has('DiagnosticReport', 'id', reportId).has('source_system', source_system)
                .addE('PART_OF_ENCOUNTER')
                .to(g.V().has('Encounter', 'id', encounterId).has('source_system', source_system))
            """
            try:
                self._execute_query(enc_edge_query, {
                    "reportId": report.id,
                    "encounterId": report.encounter_id,
                    "source_system": report.source_system
                })
            except Exception as e:
                logger.debug(f"Could not link report to encounter: {e}")
        
        # Link to observation results
        for obs_id in report.observation_ids or []:
            obs_edge_query = """
            g.V().has('DiagnosticReport', 'id', reportId).has('source_system', source_system)
                .addE('HAS_RESULT')
                .to(g.V().has('Observation', 'id', observationId).has('source_system', source_system))
            """
            try:
                self._execute_query(obs_edge_query, {
                    "reportId": report.id,
                    "observationId": obs_id,
                    "source_system": report.source_system
                })
            except Exception as e:
                logger.debug(f"Could not link report to observation {obs_id}: {e}")
        
        logger.debug(f"Created diagnostic report vertex: {report.id}")
        return report.id
    
    def _is_conflict_error(self, error: Exception) -> bool:
        """Check if an exception is a 409 Conflict error (resource already exists)."""
        error_str = str(error).lower()
        return "conflict" in error_str or "409" in error_str or "already exists" in error_str
    
    def create_patient_with_clinical_data(
        self, 
        bundle_data: Dict[str, Any],
        skip_existing: bool = True
    ) -> Tuple[str, Dict[str, int]]:
        """
        Create a patient with all clinical data from a parsed FHIR bundle.
        
        Args:
            bundle_data: Dictionary from FHIRBundleParser.parse_bundle_full()
            skip_existing: If True, silently skip resources that already exist (409 conflicts)
            
        Returns:
            Tuple of (Patient ID, stats dict with counts of created/skipped items)
        """
        stats = {
            "encounters_created": 0, "encounters_skipped": 0,
            "observations_created": 0, "observations_skipped": 0,
            "conditions_created": 0, "conditions_skipped": 0,
            "procedures_created": 0, "procedures_skipped": 0,
            "immunizations_created": 0, "immunizations_skipped": 0,
            "medication_requests_created": 0, "medication_requests_skipped": 0,
            "diagnostic_reports_created": 0, "diagnostic_reports_skipped": 0,
        }
        
        patient = bundle_data.get("patient")
        if not patient:
            raise ValueError("No patient in bundle data")
        
        # Create patient and related demographic data
        patient_id = self.create_patient(patient)
        
        # Create encounters first (other resources may reference them)
        for encounter in bundle_data.get("encounters", []):
            try:
                self.create_encounter(encounter)
                stats["encounters_created"] += 1
            except Exception as e:
                if skip_existing and self._is_conflict_error(e):
                    stats["encounters_skipped"] += 1
                else:
                    logger.warning(f"Failed to create encounter {encounter.id}: {e}")
        
        # Create observations
        for observation in bundle_data.get("observations", []):
            try:
                self.create_observation(observation)
                stats["observations_created"] += 1
            except Exception as e:
                if skip_existing and self._is_conflict_error(e):
                    stats["observations_skipped"] += 1
                else:
                    logger.warning(f"Failed to create observation {observation.id}: {e}")
        
        # Create conditions
        for condition in bundle_data.get("conditions", []):
            try:
                self.create_condition(condition)
                stats["conditions_created"] += 1
            except Exception as e:
                if skip_existing and self._is_conflict_error(e):
                    stats["conditions_skipped"] += 1
                else:
                    logger.warning(f"Failed to create condition {condition.id}: {e}")
        
        # Create procedures
        for procedure in bundle_data.get("procedures", []):
            try:
                self.create_procedure(procedure)
                stats["procedures_created"] += 1
            except Exception as e:
                if skip_existing and self._is_conflict_error(e):
                    stats["procedures_skipped"] += 1
                else:
                    logger.warning(f"Failed to create procedure {procedure.id}: {e}")
        
        # Create immunizations
        for immunization in bundle_data.get("immunizations", []):
            try:
                self.create_immunization(immunization)
                stats["immunizations_created"] += 1
            except Exception as e:
                if skip_existing and self._is_conflict_error(e):
                    stats["immunizations_skipped"] += 1
                else:
                    logger.warning(f"Failed to create immunization {immunization.id}: {e}")
        
        # Create medication requests
        for medication in bundle_data.get("medication_requests", []):
            try:
                self.create_medication_request(medication)
                stats["medication_requests_created"] += 1
            except Exception as e:
                if skip_existing and self._is_conflict_error(e):
                    stats["medication_requests_skipped"] += 1
                else:
                    logger.warning(f"Failed to create medication request {medication.id}: {e}")
        
        # Create diagnostic reports (after observations, since they reference them)
        for report in bundle_data.get("diagnostic_reports", []):
            try:
                self.create_diagnostic_report(report)
                stats["diagnostic_reports_created"] += 1
            except Exception as e:
                if skip_existing and self._is_conflict_error(e):
                    stats["diagnostic_reports_skipped"] += 1
                else:
                    logger.warning(f"Failed to create diagnostic report {report.id}: {e}")
        
        return patient_id, stats
    
    # ==================== Statistics ====================
    
    def get_stats(self) -> Dict[str, int]:
        """Get database statistics including clinical data counts."""
        stats = {}
        
        # Count vertices by label - demographic
        for label in ["Patient", "Identifier", "Address", "ContactPoint", "EmpiRecord"]:
            query = f"g.V().hasLabel('{label}').count()"
            result = self._execute_query(query)
            stats[f"total_{label.lower()}s"] = result[0] if result else 0
        
        # Count vertices by label - clinical
        for label in ["Encounter", "Observation", "Condition", "Procedure", 
                      "Immunization", "MedicationRequest", "DiagnosticReport"]:
            query = f"g.V().hasLabel('{label}').count()"
            result = self._execute_query(query)
            stats[f"total_{label.lower()}s"] = result[0] if result else 0
        
        # Count potential matches
        query = "g.E().hasLabel('POTENTIAL_MATCH').count()"
        result = self._execute_query(query)
        stats["potential_matches"] = result[0] if result else 0
        
        # Count by confidence
        for confidence in ["auto_merge", "human_review", "no_match"]:
            query = f"g.E().hasLabel('POTENTIAL_MATCH').has('confidence', '{confidence}').count()"
            result = self._execute_query(query)
            stats[f"matches_{confidence}"] = result[0] if result else 0
        
        return stats
    
    def get_patient_count(self) -> int:
        """Get total number of patients in the database."""
        query = "g.V().hasLabel('Patient').count()"
        result = self._execute_query(query)
        return result[0] if result else 0
    
    def get_empi_record_count(self) -> int:
        """Get total number of EMPI records in the database."""
        query = "g.V().hasLabel('EmpiRecord').count()"
        result = self._execute_query(query)
        return result[0] if result else 0
    
    def get_matches_for_review(
        self,
        min_score: float = 0.65,
        max_score: float = 0.85,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get potential matches that need human review.
        
        Args:
            min_score: Minimum match score
            max_score: Maximum match score
            limit: Maximum number of matches to return
        
        Returns:
            List of match pairs with patient and match details
        """
        # Query for POTENTIAL_MATCH edges in the human_review range
        query = f"""
            g.E().hasLabel('POTENTIAL_MATCH')
            .has('score', gte({min_score}))
            .has('score', lt({max_score}))
            .has('confidence', 'human_review')
            .limit({limit})
            .project('edge', 'outV', 'inV')
            .by(valueMap(true))
            .by(outV().valueMap(true))
            .by(inV().valueMap(true))
        """
        
        try:
            results = self._execute_query(query)
            
            matches = []
            if results:
                for record in results:
                    edge_props = self._flatten_value_map(record.get('edge', {}))
                    p1_props = self._flatten_value_map(record.get('outV', {}))
                    p2_props = self._flatten_value_map(record.get('inV', {}))
                    
                    matches.append({
                        "patient1": p1_props,
                        "patient2": p2_props,
                        "match": edge_props
                    })
            
            return matches
            
        except Exception as e:
            logger.error(f"Error getting matches for review: {e}")
            return []
    
    def _flatten_value_map(self, value_map: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten Gremlin valueMap results (single-item lists become values)."""
        result = {}
        for key, value in value_map.items():
            if isinstance(value, list) and len(value) == 1:
                result[key] = value[0]
            else:
                result[key] = value
        return result

    def get_all_patient_ids(self, limit: int = None, offset: int = 0) -> List[str]:
        """
        Get all patient IDs in the database.
        
        Args:
            limit: Maximum number of patient IDs to return (None for all)
            offset: Number of records to skip
        
        Returns:
            List of patient IDs
        """
        if limit:
            query = f"g.V().hasLabel('Patient').range({offset}, {offset + limit}).values('id')"
        else:
            query = f"g.V().hasLabel('Patient').range({offset}, -1).values('id')"
        
        results = self._execute_query(query)
        return results if results else []
    
    def get_all_patients(self, limit: int = None, offset: int = 0) -> List[Patient]:
        """
        Get all patients in the database with their full data.
        
        Args:
            limit: Maximum number of patients to return (None for all)
            offset: Number of records to skip
        
        Returns:
            List of Patient objects
        """
        patient_ids = self.get_all_patient_ids(limit=limit, offset=offset)
        patients = []
        for pid in patient_ids:
            patient = self.get_patient(pid)
            if patient:
                patients.append(patient)
        return patients
    
    def clear_all(self) -> None:
        """Clear all data from the graph. Use with caution!"""
        logger.warning("Clearing all data from Cosmos DB graph")
        self._execute_query("g.V().drop()")
