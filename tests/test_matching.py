"""
Tests for Patient Matching Algorithms

Tests deterministic, probabilistic, and combined matching.
"""

import pytest
from datetime import date

from src.patient_matching.models import (
    Patient, HumanName, Address, ContactPoint, Identifier,
    Gender, IdentifierType, ContactPointSystem, MatchConfidence
)
from src.patient_matching.matching import (
    DeterministicMatcher, ProbabilisticMatcher, PatientMatcher,
    MatchWeights, EmbeddingMatcher
)


# ========== Test Fixtures ==========

@pytest.fixture
def patient_john():
    """Create a test patient: John Smith"""
    return Patient(
        id="p1",
        source_id="src1",
        source_system="TestSystem",
        name=HumanName(family="Smith", given=["John", "Robert"]),
        birth_date=date(1985, 3, 15),
        gender=Gender.MALE,
        identifiers=[
            Identifier(value="MRN12345", type=IdentifierType.MRN, system="Hospital A"),
            Identifier(value="123-45-6789", type=IdentifierType.SSN),
        ],
        addresses=[
            Address(
                line=["123 Main St", "Apt 4"],
                city="Boston",
                state="MA",
                postal_code="02101",
                country="USA"
            )
        ],
        contact_points=[
            ContactPoint(system=ContactPointSystem.PHONE, value="555-123-4567"),
            ContactPoint(system=ContactPointSystem.EMAIL, value="john.smith@email.com"),
        ]
    )

@pytest.fixture
def patient_john_duplicate():
    """Create a duplicate of John Smith (same person, different source)"""
    return Patient(
        id="p2",
        source_id="src2",
        source_system="AnotherSystem",
        name=HumanName(family="Smith", given=["John"]),
        birth_date=date(1985, 3, 15),
        gender=Gender.MALE,
        identifiers=[
            Identifier(value="MRN12345", type=IdentifierType.MRN, system="Hospital A"),
        ],
        addresses=[
            Address(
                line=["123 Main Street"],
                city="Boston",
                state="MA",
                postal_code="02101",
            )
        ],
        contact_points=[
            ContactPoint(system=ContactPointSystem.PHONE, value="(555) 123-4567"),
        ]
    )

@pytest.fixture
def patient_john_similar():
    """Create a similar but different person: Jon Smith"""
    return Patient(
        id="p3",
        source_id="src3",
        source_system="TestSystem",
        name=HumanName(family="Smith", given=["Jon"]),
        birth_date=date(1985, 4, 15),  # Different month
        gender=Gender.MALE,
        identifiers=[
            Identifier(value="MRN99999", type=IdentifierType.MRN, system="Hospital B"),
        ],
        addresses=[
            Address(
                line=["456 Oak Ave"],
                city="Boston",
                state="MA",
                postal_code="02102",
            )
        ],
        contact_points=[
            ContactPoint(system=ContactPointSystem.PHONE, value="555-999-8888"),
        ]
    )

@pytest.fixture
def patient_jane():
    """Create a different patient: Jane Doe"""
    return Patient(
        id="p4",
        source_id="src4",
        source_system="TestSystem",
        name=HumanName(family="Doe", given=["Jane"]),
        birth_date=date(1990, 7, 22),
        gender=Gender.FEMALE,
        identifiers=[
            Identifier(value="MRN55555", type=IdentifierType.MRN, system="Hospital A"),
        ],
        addresses=[
            Address(
                line=["789 Pine St"],
                city="Cambridge",
                state="MA",
                postal_code="02139",
            )
        ],
        contact_points=[
            ContactPoint(system=ContactPointSystem.EMAIL, value="jane.doe@email.com"),
        ]
    )


# ========== Deterministic Matcher Tests ==========

class TestDeterministicMatcher:
    """Tests for DeterministicMatcher"""
    
    def test_exact_mrn_match(self, patient_john, patient_john_duplicate):
        """Test exact MRN match between patients"""
        matcher = DeterministicMatcher()
        score, details = matcher.compute_score(patient_john, patient_john_duplicate)
        
        assert score >= 0.8  # MRN match weight
        assert len(details["matched_identifiers"]) > 0
        assert details["dob_match"] is True
    
    def test_exact_ssn_match(self, patient_john):
        """Test SSN match"""
        # Create patient with same SSN
        patient2 = Patient(
            id="p5",
            source_id="src5",
            source_system="Test",
            name=HumanName(family="Smith", given=["Johnny"]),
            identifiers=[
                Identifier(value="123-45-6789", type=IdentifierType.SSN),
            ]
        )
        
        matcher = DeterministicMatcher()
        score, details = matcher.compute_score(patient_john, patient2)
        
        assert score >= 0.9  # SSN match weight
        assert any("SSN" in str(i) for i in details["matched_identifiers"])
    
    def test_dob_match(self, patient_john, patient_john_duplicate):
        """Test DOB exact match contributes to score"""
        matcher = DeterministicMatcher()
        score, details = matcher.compute_score(patient_john, patient_john_duplicate)
        
        assert details["dob_match"] is True
    
    def test_phone_match(self, patient_john, patient_john_duplicate):
        """Test phone number match (normalized)"""
        matcher = DeterministicMatcher()
        score, details = matcher.compute_score(patient_john, patient_john_duplicate)
        
        # Phone numbers should match after normalization
        # "555-123-4567" == "(555) 123-4567"
        assert len(details["matched_contacts"]) > 0
    
    def test_no_match(self, patient_john, patient_jane):
        """Test no match between different patients"""
        matcher = DeterministicMatcher()
        score, details = matcher.compute_score(patient_john, patient_jane)
        
        assert score < 0.3
        assert len(details["matched_identifiers"]) == 0
        assert details["dob_match"] is False


# ========== Probabilistic Matcher Tests ==========

class TestProbabilisticMatcher:
    """Tests for ProbabilisticMatcher"""
    
    def test_exact_name_match(self, patient_john, patient_john_duplicate):
        """Test high similarity for same name"""
        matcher = ProbabilisticMatcher()
        score, details = matcher.compute_name_similarity(patient_john, patient_john_duplicate)
        
        assert score > 0.8
        assert "full_name_jaro_winkler" in details
    
    def test_similar_name_match(self, patient_john, patient_john_similar):
        """Test moderate similarity for similar names (John vs Jon)"""
        matcher = ProbabilisticMatcher()
        score, details = matcher.compute_name_similarity(patient_john, patient_john_similar)
        
        # Jon vs John should have high but not perfect similarity
        assert 0.7 < score < 1.0
    
    def test_different_name_match(self, patient_john, patient_jane):
        """Test low similarity for different names"""
        matcher = ProbabilisticMatcher()
        score, details = matcher.compute_name_similarity(patient_john, patient_jane)
        
        assert score < 0.5
    
    def test_address_similarity_exact(self, patient_john, patient_john_duplicate):
        """Test address similarity for same address"""
        matcher = ProbabilisticMatcher()
        score, details = matcher.compute_address_similarity(patient_john, patient_john_duplicate)
        
        # "123 Main St" vs "123 Main Street" should be very similar
        assert score > 0.8
    
    def test_address_similarity_different(self, patient_john, patient_jane):
        """Test address similarity for different addresses"""
        matcher = ProbabilisticMatcher()
        score, details = matcher.compute_address_similarity(patient_john, patient_jane)
        
        # Different addresses but same state
        assert score < 0.7
    
    def test_dob_similarity_exact(self, patient_john, patient_john_duplicate):
        """Test DOB similarity for exact match"""
        matcher = ProbabilisticMatcher()
        score, details = matcher.compute_dob_similarity(patient_john, patient_john_duplicate)
        
        assert score == 1.0
        assert details.get("exact_match") is True
    
    def test_dob_similarity_different_month(self, patient_john, patient_john_similar):
        """Test DOB similarity for different month"""
        matcher = ProbabilisticMatcher()
        score, details = matcher.compute_dob_similarity(patient_john, patient_john_similar)
        
        # Same year, different month/day
        assert 0.0 < score < 0.5
    
    def test_dob_transposition_detection(self):
        """Test detection of transposed month/day"""
        p1 = Patient(
            id="t1", source_id="s1", source_system="T",
            birth_date=date(1985, 3, 12)  # March 12
        )
        p2 = Patient(
            id="t2", source_id="s2", source_system="T",
            birth_date=date(1985, 12, 3)  # December 3 (transposed)
        )
        
        matcher = ProbabilisticMatcher()
        score, details = matcher.compute_dob_similarity(p1, p2)
        
        assert details.get("transposition_detected") is True
        assert score >= 0.8


# ========== Combined Matcher Tests ==========

class TestPatientMatcher:
    """Tests for the combined PatientMatcher"""
    
    def test_high_confidence_match(self, patient_john, patient_john_duplicate):
        """Test high confidence (auto-merge) for duplicate patients"""
        matcher = PatientMatcher(use_embeddings=False)
        result = matcher.match(patient_john, patient_john_duplicate)
        
        assert result.score >= 0.85
        assert result.confidence == MatchConfidence.AUTO_MERGE
        assert len(result.shared_identifiers) > 0
    
    def test_medium_confidence_match(self, patient_john, patient_john_similar):
        """Test medium confidence (human review) for similar patients"""
        matcher = PatientMatcher(use_embeddings=False)
        result = matcher.match(patient_john, patient_john_similar)
        
        # Similar name, same state, different identifiers
        assert 0.3 < result.score < 0.85
    
    def test_no_match(self, patient_john, patient_jane):
        """Test no match for different patients"""
        matcher = PatientMatcher(use_embeddings=False)
        result = matcher.match(patient_john, patient_jane)
        
        assert result.score < 0.65
        assert result.confidence == MatchConfidence.NO_MATCH
    
    def test_find_matches(self, patient_john, patient_john_duplicate, patient_jane):
        """Test finding matches from a candidate list"""
        matcher = PatientMatcher(use_embeddings=False)
        candidates = [patient_john_duplicate, patient_jane]
        
        results = matcher.find_matches(patient_john, candidates, min_score=0.3)
        
        assert len(results) >= 1
        # Best match should be the duplicate
        assert results[0].patient2_id == patient_john_duplicate.id
    
    def test_batch_match(self, patient_john, patient_john_duplicate, patient_jane):
        """Test batch matching"""
        matcher = PatientMatcher(use_embeddings=False)
        patients = [patient_john, patient_john_duplicate, patient_jane]
        
        results = matcher.batch_match(patients, min_score=0.3)
        
        # Should find at least the John/John duplicate match
        assert len(results) >= 1
    
    def test_custom_weights(self, patient_john, patient_john_duplicate):
        """Test with custom match weights"""
        custom_weights = MatchWeights(
            deterministic_weight=0.6,  # Higher weight for deterministic
            name_weight=0.2,
            address_weight=0.1,
            embedding_weight=0.1,
            auto_merge_threshold=0.9,  # Stricter threshold
        )
        
        matcher = PatientMatcher(weights=custom_weights, use_embeddings=False)
        result = matcher.match(patient_john, patient_john_duplicate)
        
        # Score should be calculated with new weights
        assert result.score > 0


# ========== Address Normalization Tests ==========

class TestAddressNormalization:
    """Tests for address normalization"""
    
    def test_abbreviation_expansion(self):
        """Test that address abbreviations are expanded"""
        addr = Address(line=["123 Main St"], city="Boston", state="MA")
        normalized = addr.normalize()
        
        assert "street" in normalized
        assert "st" not in normalized.split()
    
    def test_case_normalization(self):
        """Test case is normalized to lowercase"""
        addr = Address(line=["123 MAIN STREET"], city="BOSTON", state="MA")
        normalized = addr.normalize()
        
        assert normalized == normalized.lower()
    
    def test_punctuation_removal(self):
        """Test punctuation is removed"""
        addr = Address(line=["123 Main St."], city="Boston", state="MA")
        normalized = addr.normalize()
        
        assert "." not in normalized


# ========== Contact Point Normalization Tests ==========

class TestContactPointNormalization:
    """Tests for contact point normalization"""
    
    def test_phone_normalization(self):
        """Test phone numbers are normalized to digits only"""
        phones = [
            "555-123-4567",
            "(555) 123-4567",
            "555.123.4567",
            "+1 555 123 4567",
        ]
        
        expected = "5551234567"
        
        for phone in phones:
            cp = ContactPoint(system=ContactPointSystem.PHONE, value=phone)
            # Note: our normalize removes all non-digits
            normalized = cp.normalize()
            assert normalized.endswith("5551234567")
    
    def test_email_normalization(self):
        """Test email is lowercased"""
        cp = ContactPoint(
            system=ContactPointSystem.EMAIL,
            value="John.Smith@Email.COM"
        )
        
        assert cp.normalize() == "john.smith@email.com"


# ========== Edge Cases ==========

class TestEdgeCases:
    """Tests for edge cases"""
    
    def test_missing_name(self):
        """Test matching with missing name"""
        p1 = Patient(id="e1", source_id="s1", source_system="T")
        p2 = Patient(id="e2", source_id="s2", source_system="T")
        
        matcher = ProbabilisticMatcher()
        score, details = matcher.compute_name_similarity(p1, p2)
        
        assert score == 0.0
        assert details.get("reason") == "missing_name"
    
    def test_missing_address(self):
        """Test matching with missing address"""
        p1 = Patient(id="e1", source_id="s1", source_system="T")
        p2 = Patient(id="e2", source_id="s2", source_system="T")
        
        matcher = ProbabilisticMatcher()
        score, details = matcher.compute_address_similarity(p1, p2)
        
        assert score == 0.0
        assert details.get("reason") == "missing_address"
    
    def test_missing_dob(self):
        """Test matching with missing DOB"""
        p1 = Patient(id="e1", source_id="s1", source_system="T")
        p2 = Patient(id="e2", source_id="s2", source_system="T")
        
        matcher = ProbabilisticMatcher()
        score, details = matcher.compute_dob_similarity(p1, p2)
        
        assert score == 0.0
        assert details.get("reason") == "missing_dob"
    
    def test_empty_patient(self):
        """Test matching completely empty patients"""
        p1 = Patient(id="e1", source_id="s1", source_system="T")
        p2 = Patient(id="e2", source_id="s2", source_system="T")
        
        matcher = PatientMatcher(use_embeddings=False)
        result = matcher.match(p1, p2)
        
        # Should not crash, just return low score
        assert result.score >= 0
        assert result.confidence == MatchConfidence.NO_MATCH


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
