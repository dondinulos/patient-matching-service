"""
FHIR Data Loader for Patient Matching Service

Loads patient data from FHIR Bundle JSON files and converts
them to our internal Patient model for matching.
"""

import json
import logging
from pathlib import Path
from datetime import date, datetime
from typing import List, Optional, Dict, Any, Iterator
import uuid

from .models import (
    Patient, Identifier, Address, ContactPoint, HumanName,
    Gender, IdentifierType, ContactPointSystem,
    Encounter, Observation, Condition, Procedure, Immunization,
    MedicationRequest, DiagnosticReport,
    CodeableConcept, Period, Quantity,
    EncounterClass, EncounterStatus, ObservationCategory
)

logger = logging.getLogger(__name__)

class FHIRPatientLoader:
    """
    Load patients from FHIR Bundle JSON files
    
    Parses FHIR R4 Patient resources and related data.
    """
    
    # Map FHIR identifier systems to our types
    IDENTIFIER_SYSTEM_MAP = {
        "http://hl7.org/fhir/sid/us-ssn": IdentifierType.SSN,
        "urn:oid:2.16.840.1.113883.4.1": IdentifierType.SSN,
        "http://hospital.smarthealthit.org": IdentifierType.MRN,
        "urn:oid:1.2.36.146.595.217.0.1": IdentifierType.MRN,
    }
    
    # Map FHIR gender to our Gender enum
    GENDER_MAP = {
        "male": Gender.MALE,
        "female": Gender.FEMALE,
        "other": Gender.OTHER,
        "unknown": Gender.UNKNOWN
    }
    
    def __init__(self, source_system: str = "FHIR"):
        """
        Initialize the loader
        
        Args:
            source_system: Name of the source system for provenance
        """
        self.source_system = source_system
    
    def load_from_file(self, file_path: str) -> Optional[Patient]:
        """
        Load a patient from a FHIR Bundle JSON file
        
        Args:
            file_path: Path to the JSON file
        
        Returns:
            Patient object or None if loading fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                bundle = json.load(f)
            
            return self.parse_bundle(bundle, file_path)
        
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def load_from_directory(
        self,
        directory: str,
        pattern: str = "*.json",
        limit: int = None
    ) -> Iterator[Patient]:
        """
        Load all patients from a directory of FHIR files
        
        Args:
            directory: Path to directory containing JSON files
            pattern: Glob pattern for files
            limit: Maximum number of patients to load
        
        Yields:
            Patient objects
        """
        dir_path = Path(directory)
        files = list(dir_path.glob(pattern))
        
        logger.info(f"Found {len(files)} files in {directory}")
        
        count = 0
        for file_path in files:
            if limit and count >= limit:
                break
            
            patient = self.load_from_file(str(file_path))
            if patient:
                count += 1
                yield patient
        
        logger.info(f"Loaded {count} patients from {directory}")
    
    def parse_bundle(
        self,
        bundle: Dict[str, Any],
        source_file: str = None
    ) -> Optional[Patient]:
        """
        Parse a FHIR Bundle and extract the Patient resource
        
        Args:
            bundle: FHIR Bundle dictionary
            source_file: Source file path for provenance
        
        Returns:
            Patient object or None
        """
        if bundle.get("resourceType") != "Bundle":
            # Check if it's a direct Patient resource
            if bundle.get("resourceType") == "Patient":
                return self.parse_patient_resource(bundle, source_file)
            logger.warning(f"Not a FHIR Bundle: {bundle.get('resourceType')}")
            return None
        
        # Find Patient resource in bundle
        patient_resource = None
        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            if resource.get("resourceType") == "Patient":
                patient_resource = resource
                break
        
        if not patient_resource:
            logger.warning("No Patient resource found in bundle")
            return None
        
        return self.parse_patient_resource(patient_resource, source_file)
    
    def parse_patient_resource(
        self,
        resource: Dict[str, Any],
        source_file: str = None
    ) -> Patient:
        """
        Parse a FHIR Patient resource into our Patient model
        
        Args:
            resource: FHIR Patient resource dictionary
            source_file: Source file for provenance
        
        Returns:
            Patient object
        """
        # Generate internal ID
        fhir_id = resource.get("id", str(uuid.uuid4()))
        internal_id = str(uuid.uuid4())
        
        # Parse name
        name = self._parse_name(resource.get("name", []))
        
        # Parse birth date
        birth_date = self._parse_date(resource.get("birthDate"))
        
        # Parse gender
        gender = self.GENDER_MAP.get(
            resource.get("gender", "unknown"),
            Gender.UNKNOWN
        )
        
        # Parse identifiers
        identifiers = self._parse_identifiers(resource.get("identifier", []))
        
        # Add FHIR ID as an identifier
        identifiers.append(Identifier(
            value=fhir_id,
            type=IdentifierType.FHIR_ID,
            system=self.source_system
        ))
        
        # Parse addresses
        addresses = self._parse_addresses(resource.get("address", []))
        
        # Parse contact points (telecom)
        contact_points = self._parse_telecom(resource.get("telecom", []))
        
        # Create patient
        patient = Patient(
            id=internal_id,
            source_id=fhir_id,
            source_system=self.source_system,
            name=name,
            birth_date=birth_date,
            gender=gender,
            identifiers=identifiers,
            addresses=addresses,
            contact_points=contact_points
        )
        
        logger.debug(f"Parsed patient: {patient.id} ({name.full_name if name else 'No name'})")
        
        return patient
    
    def _parse_name(self, names: List[Dict]) -> Optional[HumanName]:
        """Parse FHIR HumanName array"""
        if not names:
            return None
        
        # Prefer official name, otherwise use first
        name_data = None
        for n in names:
            if n.get("use") == "official":
                name_data = n
                break
        
        if not name_data:
            name_data = names[0]
        
        return HumanName(
            family=name_data.get("family"),
            given=name_data.get("given", []),
            prefix=name_data.get("prefix", []),
            suffix=name_data.get("suffix", [])
        )
    
    def _parse_date(self, date_str: str) -> Optional[date]:
        """Parse FHIR date string"""
        if not date_str:
            return None
        
        try:
            # FHIR dates can be YYYY, YYYY-MM, or YYYY-MM-DD
            if len(date_str) == 4:
                return date(int(date_str), 1, 1)
            elif len(date_str) == 7:
                parts = date_str.split("-")
                return date(int(parts[0]), int(parts[1]), 1)
            else:
                return date.fromisoformat(date_str[:10])
        except ValueError as e:
            logger.warning(f"Invalid date format: {date_str}")
            return None
    
    def _parse_identifiers(
        self,
        identifiers: List[Dict]
    ) -> List[Identifier]:
        """Parse FHIR Identifier array"""
        result = []
        
        for ident in identifiers:
            value = ident.get("value")
            if not value:
                continue
            
            system = ident.get("system", "")
            
            # Determine identifier type
            ident_type = self.IDENTIFIER_SYSTEM_MAP.get(
                system,
                IdentifierType.OTHER
            )
            
            # Check type coding
            type_coding = ident.get("type", {}).get("coding", [])
            for coding in type_coding:
                code = coding.get("code", "").upper()
                if code == "MR":
                    ident_type = IdentifierType.MRN
                elif code == "SS":
                    ident_type = IdentifierType.SSN
                elif code == "DL":
                    ident_type = IdentifierType.DRIVERS_LICENSE
                elif code == "PPN":
                    ident_type = IdentifierType.PASSPORT
            
            # Get assigner
            assigner = ident.get("assigner", {}).get("display")
            
            result.append(Identifier(
                value=value,
                type=ident_type,
                system=system,
                assigner=assigner
            ))
        
        return result
    
    def _parse_addresses(self, addresses: List[Dict]) -> List[Address]:
        """Parse FHIR Address array"""
        result = []
        
        for addr in addresses:
            result.append(Address(
                line=addr.get("line", []),
                city=addr.get("city"),
                state=addr.get("state"),
                postal_code=addr.get("postalCode"),
                country=addr.get("country")
            ))
        
        return result
    
    def _parse_telecom(
        self,
        telecoms: List[Dict]
    ) -> List[ContactPoint]:
        """Parse FHIR ContactPoint (telecom) array"""
        result = []
        
        system_map = {
            "phone": ContactPointSystem.PHONE,
            "email": ContactPointSystem.EMAIL,
            "fax": ContactPointSystem.FAX,
            "pager": ContactPointSystem.PAGER,
        }
        
        for telecom in telecoms:
            value = telecom.get("value")
            if not value:
                continue
            
            system = system_map.get(
                telecom.get("system", ""),
                ContactPointSystem.OTHER
            )
            
            result.append(ContactPoint(
                system=system,
                value=value,
                use=telecom.get("use")
            ))
        
        return result

class FHIRBundleParser:
    """
    Extended parser for extracting all clinical resources from FHIR Bundles
    
    Extracts Encounters, Observations, Conditions, Procedures, Immunizations,
    MedicationRequests, and DiagnosticReports for building the patient graph.
    """
    
    ENCOUNTER_CLASS_MAP = {
        "AMB": EncounterClass.AMBULATORY,
        "EMER": EncounterClass.EMERGENCY,
        "IMP": EncounterClass.INPATIENT,
        "WELLNESS": EncounterClass.WELLNESS,
        "VR": EncounterClass.VIRTUAL,
    }
    
    ENCOUNTER_STATUS_MAP = {
        "planned": EncounterStatus.PLANNED,
        "in-progress": EncounterStatus.IN_PROGRESS,
        "finished": EncounterStatus.FINISHED,
        "cancelled": EncounterStatus.CANCELLED,
    }
    
    OBSERVATION_CATEGORY_MAP = {
        "vital-signs": ObservationCategory.VITAL_SIGNS,
        "laboratory": ObservationCategory.LABORATORY,
        "imaging": ObservationCategory.IMAGING,
        "procedure": ObservationCategory.PROCEDURE,
        "survey": ObservationCategory.SURVEY,
        "social-history": ObservationCategory.SOCIAL_HISTORY,
    }
    
    def __init__(self, source_system: str = "FHIR"):
        self.source_system = source_system
        self.patient_loader = FHIRPatientLoader(source_system=source_system)
        # Map urn:uuid references to internal IDs
        self._reference_map: Dict[str, str] = {}
    
    def parse_bundle_full(
        self,
        bundle: Dict[str, Any],
        source_file: str = None
    ) -> Dict[str, Any]:
        """
        Parse a complete FHIR Bundle extracting all clinical resources
        
        Returns:
            Dictionary with patient and related clinical resources
        """
        result = {
            "patient": None,
            "encounters": [],
            "observations": [],
            "conditions": [],
            "procedures": [],
            "immunizations": [],
            "medication_requests": [],
            "diagnostic_reports": [],
        }
        
        if bundle.get("resourceType") != "Bundle":
            logger.warning("Not a FHIR Bundle")
            return result
        
        # First pass: find Patient and build reference map
        patient_ref = None
        for entry in bundle.get("entry", []):
            full_url = entry.get("fullUrl", "")
            resource = entry.get("resource", {})
            resource_type = resource.get("resourceType")
            resource_id = resource.get("id")
            
            if resource_type == "Patient":
                result["patient"] = self.patient_loader.parse_patient_resource(resource, source_file)
                patient_ref = full_url
                if result["patient"]:
                    self._reference_map[full_url] = result["patient"].id
                    self._reference_map[f"Patient/{resource_id}"] = result["patient"].id
        
        if not result["patient"]:
            logger.warning("No Patient resource found in bundle")
            return result
        
        patient_id = result["patient"].id
        
        # Second pass: parse all clinical resources
        for entry in bundle.get("entry", []):
            full_url = entry.get("fullUrl", "")
            resource = entry.get("resource", {})
            resource_type = resource.get("resourceType")
            resource_id = resource.get("id")
            
            # Store reference mapping
            if resource_id:
                self._reference_map[full_url] = resource_id
                self._reference_map[f"{resource_type}/{resource_id}"] = resource_id
            
            if resource_type == "Encounter":
                enc = self._parse_encounter(resource, patient_id)
                if enc:
                    result["encounters"].append(enc)
                    
            elif resource_type == "Observation":
                obs = self._parse_observation(resource, patient_id)
                if obs:
                    result["observations"].append(obs)
                    
            elif resource_type == "Condition":
                cond = self._parse_condition(resource, patient_id)
                if cond:
                    result["conditions"].append(cond)
                    
            elif resource_type == "Procedure":
                proc = self._parse_procedure(resource, patient_id)
                if proc:
                    result["procedures"].append(proc)
                    
            elif resource_type == "Immunization":
                imm = self._parse_immunization(resource, patient_id)
                if imm:
                    result["immunizations"].append(imm)
                    
            elif resource_type == "MedicationRequest":
                med = self._parse_medication_request(resource, patient_id)
                if med:
                    result["medication_requests"].append(med)
                    
            elif resource_type == "DiagnosticReport":
                report = self._parse_diagnostic_report(resource, patient_id)
                if report:
                    result["diagnostic_reports"].append(report)
        
        return result
    
    def _parse_codeable_concept(self, cc: Dict) -> Optional[CodeableConcept]:
        """Parse FHIR CodeableConcept"""
        if not cc:
            return None
        
        codings = cc.get("coding", [])
        if codings:
            coding = codings[0]
            return CodeableConcept(
                code=coding.get("code", ""),
                display=coding.get("display"),
                system=coding.get("system"),
                text=cc.get("text")
            )
        elif cc.get("text"):
            return CodeableConcept(code="", text=cc.get("text"))
        return None
    
    def _parse_period(self, period: Dict) -> Optional[Period]:
        """Parse FHIR Period"""
        if not period:
            return None
        return Period(
            start=self._parse_datetime(period.get("start")),
            end=self._parse_datetime(period.get("end"))
        )
    
    def _parse_quantity(self, qty: Dict) -> Optional[Quantity]:
        """Parse FHIR Quantity"""
        if not qty or "value" not in qty:
            return None
        return Quantity(
            value=float(qty.get("value", 0)),
            unit=qty.get("unit"),
            system=qty.get("system"),
            code=qty.get("code")
        )
    
    def _parse_datetime(self, dt_str: str) -> Optional[datetime]:
        """Parse FHIR datetime string"""
        if not dt_str:
            return None
        try:
            # Handle various FHIR datetime formats
            if "T" in dt_str:
                # Remove timezone info for simplicity
                dt_str = dt_str.replace("Z", "+00:00")
                if "+" in dt_str or dt_str.count("-") > 2:
                    # Has timezone
                    return datetime.fromisoformat(dt_str)
                return datetime.fromisoformat(dt_str)
            else:
                # Date only
                return datetime.fromisoformat(dt_str + "T00:00:00")
        except (ValueError, TypeError):
            return None
    
    def _resolve_reference(self, reference: str) -> Optional[str]:
        """Resolve a FHIR reference to an internal ID"""
        if not reference:
            return None
        return self._reference_map.get(reference, reference.split("/")[-1] if "/" in reference else reference)
    
    def _parse_encounter(self, resource: Dict, patient_id: str) -> Optional[Encounter]:
        """Parse FHIR Encounter resource"""
        enc_id = resource.get("id")
        if not enc_id:
            return None
        
        # Parse encounter class
        enc_class_data = resource.get("class", {})
        enc_class = self.ENCOUNTER_CLASS_MAP.get(
            enc_class_data.get("code", ""),
            EncounterClass.OTHER
        )
        
        # Parse status
        status = self.ENCOUNTER_STATUS_MAP.get(
            resource.get("status", ""),
            EncounterStatus.UNKNOWN
        )
        
        # Parse type
        type_code = None
        types = resource.get("type", [])
        if types:
            type_code = self._parse_codeable_concept(types[0])
        
        # Parse period
        period = self._parse_period(resource.get("period"))
        
        # Parse service provider
        service_provider = resource.get("serviceProvider", {}).get("display")
        if not service_provider:
            service_provider = resource.get("serviceProvider", {}).get("reference")
        
        # Parse reason
        reason_code = None
        reasons = resource.get("reasonCode", [])
        if reasons:
            reason_code = self._parse_codeable_concept(reasons[0])
        
        return Encounter(
            id=enc_id,
            patient_id=patient_id,
            source_system=self.source_system,
            status=status,
            encounter_class=enc_class,
            type_code=type_code,
            period=period,
            service_provider=service_provider,
            reason_code=reason_code
        )
    
    def _parse_observation(self, resource: Dict, patient_id: str) -> Optional[Observation]:
        """Parse FHIR Observation resource"""
        obs_id = resource.get("id")
        if not obs_id:
            return None
        
        # Parse category
        category = ObservationCategory.OTHER
        categories = resource.get("category", [])
        if categories:
            cat_codings = categories[0].get("coding", [])
            if cat_codings:
                cat_code = cat_codings[0].get("code", "")
                category = self.OBSERVATION_CATEGORY_MAP.get(cat_code, ObservationCategory.OTHER)
        
        # Parse code
        code = self._parse_codeable_concept(resource.get("code"))
        
        # Parse value
        value_quantity = self._parse_quantity(resource.get("valueQuantity"))
        value_string = resource.get("valueString")
        value_code = self._parse_codeable_concept(resource.get("valueCodeableConcept"))
        
        # Parse effective datetime
        effective = self._parse_datetime(resource.get("effectiveDateTime"))
        
        # Parse encounter reference
        encounter_ref = resource.get("encounter", {}).get("reference")
        encounter_id = self._resolve_reference(encounter_ref)
        
        return Observation(
            id=obs_id,
            patient_id=patient_id,
            source_system=self.source_system,
            status=resource.get("status", "final"),
            category=category,
            code=code,
            value_quantity=value_quantity,
            value_string=value_string,
            value_code=value_code,
            effective_datetime=effective,
            encounter_id=encounter_id
        )
    
    def _parse_condition(self, resource: Dict, patient_id: str) -> Optional[Condition]:
        """Parse FHIR Condition resource"""
        cond_id = resource.get("id")
        if not cond_id:
            return None
        
        # Parse code
        code = self._parse_codeable_concept(resource.get("code"))
        
        # Parse clinical status
        clinical_status = "unknown"
        status_cc = resource.get("clinicalStatus", {})
        status_codings = status_cc.get("coding", [])
        if status_codings:
            clinical_status = status_codings[0].get("code", "unknown")
        
        # Parse verification status
        verification_status = "confirmed"
        verif_cc = resource.get("verificationStatus", {})
        verif_codings = verif_cc.get("coding", [])
        if verif_codings:
            verification_status = verif_codings[0].get("code", "confirmed")
        
        # Parse onset
        onset = self._parse_datetime(resource.get("onsetDateTime"))
        
        # Parse abatement
        abatement = self._parse_datetime(resource.get("abatementDateTime"))
        
        # Parse encounter reference
        encounter_ref = resource.get("encounter", {}).get("reference")
        encounter_id = self._resolve_reference(encounter_ref)
        
        return Condition(
            id=cond_id,
            patient_id=patient_id,
            source_system=self.source_system,
            code=code,
            clinical_status=clinical_status,
            verification_status=verification_status,
            onset_datetime=onset,
            abatement_datetime=abatement,
            encounter_id=encounter_id
        )
    
    def _parse_procedure(self, resource: Dict, patient_id: str) -> Optional[Procedure]:
        """Parse FHIR Procedure resource"""
        proc_id = resource.get("id")
        if not proc_id:
            return None
        
        # Parse code
        code = self._parse_codeable_concept(resource.get("code"))
        
        # Parse performed datetime/period
        performed_dt = self._parse_datetime(resource.get("performedDateTime"))
        performed_period = self._parse_period(resource.get("performedPeriod"))
        
        # Parse encounter reference
        encounter_ref = resource.get("encounter", {}).get("reference")
        encounter_id = self._resolve_reference(encounter_ref)
        
        return Procedure(
            id=proc_id,
            patient_id=patient_id,
            source_system=self.source_system,
            status=resource.get("status", "completed"),
            code=code,
            performed_datetime=performed_dt,
            performed_period=performed_period,
            encounter_id=encounter_id
        )
    
    def _parse_immunization(self, resource: Dict, patient_id: str) -> Optional[Immunization]:
        """Parse FHIR Immunization resource"""
        imm_id = resource.get("id")
        if not imm_id:
            return None
        
        # Parse vaccine code
        vaccine_code = self._parse_codeable_concept(resource.get("vaccineCode"))
        
        # Parse occurrence
        occurrence = self._parse_datetime(resource.get("occurrenceDateTime"))
        
        # Parse encounter reference
        encounter_ref = resource.get("encounter", {}).get("reference")
        encounter_id = self._resolve_reference(encounter_ref)
        
        return Immunization(
            id=imm_id,
            patient_id=patient_id,
            source_system=self.source_system,
            status=resource.get("status", "completed"),
            vaccine_code=vaccine_code,
            occurrence_datetime=occurrence,
            encounter_id=encounter_id,
            lot_number=resource.get("lotNumber")
        )
    
    def _parse_medication_request(self, resource: Dict, patient_id: str) -> Optional[MedicationRequest]:
        """Parse FHIR MedicationRequest resource"""
        med_id = resource.get("id")
        if not med_id:
            return None
        
        # Parse medication code
        medication_code = self._parse_codeable_concept(resource.get("medicationCodeableConcept"))
        
        # Parse authored on
        authored = self._parse_datetime(resource.get("authoredOn"))
        
        # Parse encounter reference
        encounter_ref = resource.get("encounter", {}).get("reference")
        encounter_id = self._resolve_reference(encounter_ref)
        
        # Parse dosage instructions
        dosage_instructions = resource.get("dosageInstruction", [])
        dosage_text = None
        if dosage_instructions:
            dosage_text = dosage_instructions[0].get("text")
        
        return MedicationRequest(
            id=med_id,
            patient_id=patient_id,
            source_system=self.source_system,
            status=resource.get("status", "active"),
            intent=resource.get("intent", "order"),
            medication_code=medication_code,
            authored_on=authored,
            encounter_id=encounter_id,
            dosage_instruction=dosage_text
        )
    
    def _parse_diagnostic_report(self, resource: Dict, patient_id: str) -> Optional[DiagnosticReport]:
        """Parse FHIR DiagnosticReport resource"""
        report_id = resource.get("id")
        if not report_id:
            return None
        
        # Parse category
        category = None
        categories = resource.get("category", [])
        if categories:
            category = self._parse_codeable_concept(categories[0])
        
        # Parse code
        code = self._parse_codeable_concept(resource.get("code"))
        
        # Parse effective datetime
        effective = self._parse_datetime(resource.get("effectiveDateTime"))
        
        # Parse issued
        issued = self._parse_datetime(resource.get("issued"))
        
        # Parse encounter reference
        encounter_ref = resource.get("encounter", {}).get("reference")
        encounter_id = self._resolve_reference(encounter_ref)
        
        # Parse observation references
        observation_ids = []
        for result in resource.get("result", []):
            obs_ref = result.get("reference")
            obs_id = self._resolve_reference(obs_ref)
            if obs_id:
                observation_ids.append(obs_id)
        
        return DiagnosticReport(
            id=report_id,
            patient_id=patient_id,
            source_system=self.source_system,
            status=resource.get("status", "final"),
            category=category,
            code=code,
            effective_datetime=effective,
            issued=issued,
            encounter_id=encounter_id,
            conclusion=resource.get("conclusion"),
            observation_ids=observation_ids
        )

def load_synthea_patients(
    directory: str,
    limit: int = None,
    source_system: str = "Synthea"
) -> List[Patient]:
    """
    Convenience function to load Synthea-generated FHIR data
    
    Args:
        directory: Path to directory with Synthea JSON files
        limit: Maximum patients to load
        source_system: Source system name
    
    Returns:
        List of Patient objects
    """
    loader = FHIRPatientLoader(source_system=source_system)
    patients = list(loader.load_from_directory(directory, limit=limit))
    
    logger.info(f"Loaded {len(patients)} patients from Synthea data")
    return patients
