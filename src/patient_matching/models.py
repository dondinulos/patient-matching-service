"""
Data models for Patient Matching Service

Defines the core entities: Patient, Identifier, Address, ContactPoint, etc.
These models map to Neo4j graph nodes and relationships.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional, List
from enum import Enum


class Gender(Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"


class IdentifierType(Enum):
    MRN = "MRN"  # Medical Record Number
    SSN = "SSN"  # Social Security Number
    ENTERPRISE_ID = "ENTERPRISE_ID"
    DRIVERS_LICENSE = "DL"
    PASSPORT = "PASSPORT"
    FHIR_ID = "FHIR_ID"
    OTHER = "OTHER"


class ContactPointSystem(Enum):
    PHONE = "phone"
    EMAIL = "email"
    FAX = "fax"
    PAGER = "pager"
    OTHER = "other"


class MatchConfidence(Enum):
    """Match confidence levels for patient linkage"""
    AUTO_MERGE = "auto_merge"  # >= 0.85
    HUMAN_REVIEW = "human_review"  # 0.65 - 0.85
    NO_MATCH = "no_match"  # < 0.65


class EncounterClass(Enum):
    """Encounter class/type"""
    AMBULATORY = "AMB"
    EMERGENCY = "EMER"
    INPATIENT = "IMP"
    WELLNESS = "WELLNESS"
    VIRTUAL = "VR"
    OTHER = "OTHER"


class EncounterStatus(Enum):
    """Encounter status"""
    PLANNED = "planned"
    IN_PROGRESS = "in-progress"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class ObservationCategory(Enum):
    """Observation category"""
    VITAL_SIGNS = "vital-signs"
    LABORATORY = "laboratory"
    IMAGING = "imaging"
    PROCEDURE = "procedure"
    SURVEY = "survey"
    SOCIAL_HISTORY = "social-history"
    OTHER = "other"


@dataclass
class Identifier:
    """Patient identifier (MRN, SSN, Enterprise ID, etc.)"""
    value: str
    type: IdentifierType
    system: Optional[str] = None  # Source system
    assigner: Optional[str] = None  # Organization that assigned the ID
    
    def __hash__(self):
        return hash((self.value, self.type, self.system))
    
    def __eq__(self, other):
        if not isinstance(other, Identifier):
            return False
        return self.value == other.value and self.type == other.type

    def normalize(self) -> str:
        """Normalize identifier value for comparison and storage"""
        # Remove whitespace and convert to uppercase for consistent matching
        normalized = self.value.strip().upper()
        if self.type == IdentifierType.SSN:
            # Remove dashes from SSN
            normalized = normalized.replace("-", "")
        return normalized


@dataclass
class Address:
    """Patient address"""
    line: List[str] = field(default_factory=list)
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None
    country: Optional[str] = None
    
    @property
    def full_address(self) -> str:
        """Get full address as single string for comparison"""
        parts = []
        if self.line:
            parts.extend(self.line)
        if self.city:
            parts.append(self.city)
        if self.state:
            parts.append(self.state)
        if self.postal_code:
            parts.append(self.postal_code)
        if self.country:
            parts.append(self.country)
        return ", ".join(parts)
    
    def normalize(self) -> str:
        """Normalize address for comparison"""
        addr = self.full_address.lower()
        # Common abbreviation expansions
        replacements = {
            " st ": " street ",
            " st,": " street,",
            " ave ": " avenue ",
            " ave,": " avenue,",
            " blvd ": " boulevard ",
            " blvd,": " boulevard,",
            " dr ": " drive ",
            " dr,": " drive,",
            " ln ": " lane ",
            " ln,": " lane,",
            " rd ": " road ",
            " rd,": " road,",
            " apt ": " apartment ",
            " apt.": " apartment",
            " #": " apartment ",
        }
        for abbrev, full in replacements.items():
            addr = addr.replace(abbrev, full)
        # Remove punctuation
        addr = addr.replace(".", "").replace(",", " ").replace("-", " ")
        # Normalize whitespace
        addr = " ".join(addr.split())
        return addr


@dataclass
class ContactPoint:
    """Patient contact information (phone, email)"""
    system: ContactPointSystem
    value: str
    use: Optional[str] = None  # home, work, mobile
    
    def normalize(self) -> str:
        """Normalize contact for comparison"""
        if self.system == ContactPointSystem.PHONE:
            # Remove non-digits
            return "".join(c for c in self.value if c.isdigit())
        elif self.system == ContactPointSystem.EMAIL:
            return self.value.lower().strip()
        return self.value.lower().strip()


@dataclass
class HumanName:
    """Patient name"""
    family: Optional[str] = None
    given: List[str] = field(default_factory=list)
    prefix: List[str] = field(default_factory=list)
    suffix: List[str] = field(default_factory=list)
    
    @property
    def full_name(self) -> str:
        """Get full name as single string"""
        parts = []
        if self.prefix:
            parts.extend(self.prefix)
        if self.given:
            parts.extend(self.given)
        if self.family:
            parts.append(self.family)
        if self.suffix:
            parts.extend(self.suffix)
        return " ".join(parts)
    
    @property
    def first_name(self) -> Optional[str]:
        """Get first given name"""
        return self.given[0] if self.given else None
    
    @property
    def last_name(self) -> Optional[str]:
        """Get family name"""
        return self.family
    
    def normalize(self) -> str:
        """Normalize name for comparison"""
        return self.full_name.lower().strip()


@dataclass
class CodeableConcept:
    """FHIR CodeableConcept - a code with display text"""
    code: str
    display: Optional[str] = None
    system: Optional[str] = None
    text: Optional[str] = None
    
    def __str__(self) -> str:
        return self.text or self.display or self.code


@dataclass
class Period:
    """FHIR Period - a time range"""
    start: Optional[datetime] = None
    end: Optional[datetime] = None


@dataclass
class Quantity:
    """FHIR Quantity - a measured value with units"""
    value: float
    unit: Optional[str] = None
    system: Optional[str] = None
    code: Optional[str] = None
    
    def __str__(self) -> str:
        if self.unit:
            return f"{self.value} {self.unit}"
        return str(self.value)


@dataclass
class Encounter:
    """Healthcare encounter/visit"""
    id: str
    patient_id: str  # Reference to Patient
    source_system: str
    
    status: EncounterStatus = EncounterStatus.UNKNOWN
    encounter_class: EncounterClass = EncounterClass.OTHER
    type_code: Optional[CodeableConcept] = None
    period: Optional[Period] = None
    
    # Location/Provider
    service_provider: Optional[str] = None
    location: Optional[str] = None
    
    # Reason for visit
    reason_code: Optional[CodeableConcept] = None
    
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Observation:
    """Clinical observation (vital signs, lab results, etc.)"""
    id: str
    patient_id: str  # Reference to Patient
    source_system: str
    
    status: str = "final"
    category: ObservationCategory = ObservationCategory.OTHER
    code: Optional[CodeableConcept] = None
    
    # Value (one of these)
    value_quantity: Optional[Quantity] = None
    value_string: Optional[str] = None
    value_code: Optional[CodeableConcept] = None
    
    # When observed
    effective_datetime: Optional[datetime] = None
    
    # Related encounter
    encounter_id: Optional[str] = None
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def value_display(self) -> str:
        """Get displayable value"""
        if self.value_quantity:
            return str(self.value_quantity)
        if self.value_code:
            return str(self.value_code)
        if self.value_string:
            return self.value_string
        return ""


@dataclass
class Condition:
    """Medical condition/diagnosis"""
    id: str
    patient_id: str  # Reference to Patient
    source_system: str
    
    code: Optional[CodeableConcept] = None
    clinical_status: str = "active"  # active, resolved, inactive
    verification_status: str = "confirmed"
    
    # When diagnosed
    onset_datetime: Optional[datetime] = None
    abatement_datetime: Optional[datetime] = None  # When resolved
    
    # Related encounter
    encounter_id: Optional[str] = None
    
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Procedure:
    """Medical procedure"""
    id: str
    patient_id: str  # Reference to Patient
    source_system: str
    
    status: str = "completed"
    code: Optional[CodeableConcept] = None
    
    # When performed
    performed_datetime: Optional[datetime] = None
    performed_period: Optional[Period] = None
    
    # Related encounter
    encounter_id: Optional[str] = None
    
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Immunization:
    """Immunization/vaccination record"""
    id: str
    patient_id: str  # Reference to Patient
    source_system: str
    
    status: str = "completed"
    vaccine_code: Optional[CodeableConcept] = None
    
    # When given
    occurrence_datetime: Optional[datetime] = None
    
    # Related encounter
    encounter_id: Optional[str] = None
    
    # Lot/manufacturer info
    lot_number: Optional[str] = None
    
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MedicationRequest:
    """Medication prescription/order"""
    id: str
    patient_id: str  # Reference to Patient
    source_system: str
    
    status: str = "active"
    intent: str = "order"
    medication_code: Optional[CodeableConcept] = None
    
    # When prescribed
    authored_on: Optional[datetime] = None
    
    # Related encounter
    encounter_id: Optional[str] = None
    
    # Dosage instructions
    dosage_instruction: Optional[str] = None
    
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DiagnosticReport:
    """Diagnostic report (lab results, imaging, etc.)"""
    id: str
    patient_id: str  # Reference to Patient
    source_system: str
    
    status: str = "final"
    category: Optional[CodeableConcept] = None
    code: Optional[CodeableConcept] = None
    
    # When issued
    effective_datetime: Optional[datetime] = None
    issued: Optional[datetime] = None
    
    # Related encounter
    encounter_id: Optional[str] = None
    
    # Conclusion
    conclusion: Optional[str] = None
    
    # Related observations (IDs)
    observation_ids: List[str] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Patient:
    """Core patient entity"""
    id: str  # Internal ID
    source_id: str  # ID from source system
    source_system: str  # Source system name
    
    # Demographics
    name: Optional[HumanName] = None
    birth_date: Optional[date] = None
    gender: Gender = Gender.UNKNOWN
    
    # Identifiers
    identifiers: List[Identifier] = field(default_factory=list)
    
    # Contact information
    addresses: List[Address] = field(default_factory=list)
    contact_points: List[ContactPoint] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # EMPI (Enterprise Master Patient Index) linkage
    empi_id: Optional[str] = None  # Link to master record
    
    # Optional: Embedding for AI-enhanced matching
    embedding: Optional[List[float]] = None
    
    def get_mrn(self) -> Optional[str]:
        """Get MRN if available"""
        for ident in self.identifiers:
            if ident.type == IdentifierType.MRN:
                return ident.value
        return None
    
    def get_ssn(self) -> Optional[str]:
        """Get SSN if available"""
        for ident in self.identifiers:
            if ident.type == IdentifierType.SSN:
                return ident.value
        return None
    
    def get_enterprise_id(self) -> Optional[str]:
        """Get Enterprise ID if available"""
        for ident in self.identifiers:
            if ident.type == IdentifierType.ENTERPRISE_ID:
                return ident.value
        return None
    
    def get_primary_phone(self) -> Optional[str]:
        """Get primary phone number"""
        for cp in self.contact_points:
            if cp.system == ContactPointSystem.PHONE:
                return cp.normalize()
        return None
    
    def get_primary_email(self) -> Optional[str]:
        """Get primary email"""
        for cp in self.contact_points:
            if cp.system == ContactPointSystem.EMAIL:
                return cp.normalize()
        return None
    
    def get_primary_address(self) -> Optional[Address]:
        """Get primary address"""
        return self.addresses[0] if self.addresses else None
    
    def to_profile_text(self) -> str:
        """Generate text profile for embedding"""
        parts = []
        if self.name:
            parts.append(f"Name: {self.name.full_name}")
        if self.birth_date:
            parts.append(f"DOB: {self.birth_date.isoformat()}")
        if self.gender != Gender.UNKNOWN:
            parts.append(f"Gender: {self.gender.value}")
        for addr in self.addresses:
            parts.append(f"Address: {addr.full_address}")
        for cp in self.contact_points:
            parts.append(f"{cp.system.value}: {cp.value}")
        return "; ".join(parts)


@dataclass
class EmpiRecord:
    """
    Enterprise Master Patient Index (EMPI) Record
    
    Represents a unique individual across all source systems.
    Previously known as Golden Record in MDM terminology.
    """
    id: str
    
    # Survivorship - best values from linked records
    name: Optional[HumanName] = None
    birth_date: Optional[date] = None
    gender: Gender = Gender.UNKNOWN
    
    # Links to source records
    linked_patient_ids: List[str] = field(default_factory=list)
    
    # Audit trail
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    
    # Match metadata
    auto_linked: bool = True  # False if human-reviewed


@dataclass
class MatchResult:
    """Result of a patient matching operation"""
    patient1_id: str
    patient2_id: str
    score: float
    confidence: MatchConfidence
    
    # Component scores for explainability
    deterministic_score: float = 0.0
    name_similarity: float = 0.0
    address_similarity: float = 0.0
    embedding_similarity: float = 0.0
    
    # Matching details
    shared_identifiers: List[str] = field(default_factory=list)
    match_details: dict = field(default_factory=dict)
    
    @classmethod
    def from_scores(
        cls,
        patient1_id: str,
        patient2_id: str,
        deterministic_score: float,
        name_similarity: float,
        address_similarity: float,
        embedding_similarity: float = 0.0,
        shared_identifiers: List[str] = None,
        match_details: dict = None
    ) -> "MatchResult":
        """Create MatchResult with weighted score calculation"""
        # Weighted formula from requirements
        score = min(1.0,
            (0.4 * deterministic_score) +
            (0.35 * name_similarity) +
            (0.15 * address_similarity) +
            (0.1 * embedding_similarity)
        )
        
        # Determine confidence level
        if score >= 0.85:
            confidence = MatchConfidence.AUTO_MERGE
        elif score >= 0.65:
            confidence = MatchConfidence.HUMAN_REVIEW
        else:
            confidence = MatchConfidence.NO_MATCH
        
        return cls(
            patient1_id=patient1_id,
            patient2_id=patient2_id,
            score=score,
            confidence=confidence,
            deterministic_score=deterministic_score,
            name_similarity=name_similarity,
            address_similarity=address_similarity,
            embedding_similarity=embedding_similarity,
            shared_identifiers=shared_identifiers or [],
            match_details=match_details or {}
        )


@dataclass
class LinkageDecision:
    """Record of a linkage decision for audit trail"""
    id: str
    match_result: MatchResult
    decision: str  # "merge", "no_merge", "review"
    empi_id: Optional[str] = None  # Enterprise Master Patient Index ID
    
    # Audit
    decided_at: datetime = field(default_factory=datetime.utcnow)
    decided_by: Optional[str] = None  # User or "AUTO"
    reason: Optional[str] = None
    
    # For unmerge capability
    is_active: bool = True
    unmerged_at: Optional[datetime] = None
    unmerged_by: Optional[str] = None
