"""
Tests for Patient Matching MCP Server

Unit tests that mock PatientMatchingService to verify MCP tool functions,
resources, and prompts without requiring a database connection.
"""

import json
import pytest
from datetime import date, datetime
from unittest.mock import MagicMock, patch

from src.patient_matching.models import (
    Patient, HumanName, Address, ContactPoint, Identifier,
    Gender, IdentifierType, ContactPointSystem, MatchConfidence, MatchResult,
)

# Import the MCP server module (tools are module-level functions)
import src.patient_matching.mcp_server as mcp_mod


# ========== Test Fixtures ==========


@pytest.fixture
def patient_john():
    """Create a test patient: John Smith."""
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
                country="USA",
            )
        ],
        contact_points=[
            ContactPoint(system=ContactPointSystem.PHONE, value="555-123-4567"),
            ContactPoint(system=ContactPointSystem.EMAIL, value="john.smith@email.com"),
        ],
    )


@pytest.fixture
def patient_jane():
    """Create a different patient: Jane Doe."""
    return Patient(
        id="p2",
        source_id="src2",
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
        ],
    )


@pytest.fixture
def mock_service(patient_john, patient_jane):
    """Create a fully mocked PatientMatchingService."""
    service = MagicMock()

    # get_patient routing
    def _get_patient(pid):
        return {"p1": patient_john, "p2": patient_jane}.get(pid)

    service.db.get_patient.side_effect = _get_patient

    # get_all_patients returns both
    service.db.get_all_patients.return_value = [patient_john, patient_jane]

    # Default match results
    service.find_matches_for_patient.return_value = [
        MatchResult(
            patient1_id="p1",
            patient2_id="p2",
            score=0.72,
            confidence=MatchConfidence.HUMAN_REVIEW,
            deterministic_score=0.35,
            name_similarity=0.45,
            address_similarity=0.30,
            embedding_similarity=0.0,
            shared_identifiers=["DOB"],
        )
    ]
    service.find_all_matches_for_patient.return_value = (
        service.find_matches_for_patient.return_value
    )

    # Matcher for compare
    service.matcher.match.return_value = MatchResult(
        patient1_id="p1",
        patient2_id="p2",
        score=0.72,
        confidence=MatchConfidence.HUMAN_REVIEW,
        deterministic_score=0.35,
        name_similarity=0.45,
        address_similarity=0.30,
        embedding_similarity=0.0,
        shared_identifiers=["DOB"],
    )

    # Batch matching
    service.run_global_matching.return_value = {
        "total_patients": 2,
        "patients_processed": 2,
        "matches_found": 1,
        "auto_merge": 0,
        "human_review": 1,
        "unique_pairs": 1,
    }

    # Approve / reject
    service.approve_match.return_value = "empi-001"
    service.reject_match.return_value = None

    # Pending reviews
    service.get_pending_reviews.return_value = [
        {
            "patient1": {"id": "p1", "name": "John Robert Smith"},
            "patient2": {"id": "p2", "name": "Jane Doe"},
            "match": {"score": 0.72, "confidence": "human_review"},
        }
    ]

    # Stats
    service.get_stats.return_value = {
        "total_patients": 2,
        "total_empi_records": 0,
        "pending_reviews": 1,
    }

    return service


@pytest.fixture(autouse=True)
def _patch_service(mock_service):
    """Patch the module-level _get_service to return our mock."""
    with patch.object(mcp_mod, "_get_service", return_value=mock_service):
        yield


# ========== Tool Tests ==========


class TestFindPatientMatches:
    """Tests for find_patient_matches tool."""

    def test_returns_matches_json(self, mock_service):
        result = mcp_mod.find_patient_matches(patient_id="p1")
        data = json.loads(result)

        assert data["patient_id"] == "p1"
        assert data["matches_found"] == 1
        assert data["matches"][0]["matched_patient_id"] == "p2"
        assert data["matches"][0]["score"] == 0.72
        assert data["matches"][0]["confidence"] == "human_review"

    def test_no_matches_message(self, mock_service):
        mock_service.find_matches_for_patient.return_value = []
        result = mcp_mod.find_patient_matches(patient_id="p1")

        assert "No matches found" in result

    def test_search_entire_database(self, mock_service):
        mcp_mod.find_patient_matches(patient_id="p1", search_entire_database=True)

        mock_service.find_all_matches_for_patient.assert_called_once_with(
            patient_id="p1", min_score=0.3, limit=20
        )

    def test_custom_thresholds(self, mock_service):
        mcp_mod.find_patient_matches(patient_id="p1", min_score=0.5, max_results=5)

        mock_service.find_matches_for_patient.assert_called_once_with(
            patient_id="p1", min_score=0.5
        )

    def test_max_results_slicing(self, mock_service):
        # Return 3 matches but max_results=2
        mock_service.find_matches_for_patient.return_value = [
            MatchResult(
                patient1_id="p1", patient2_id=f"px{i}",
                score=0.7 - i * 0.1, confidence=MatchConfidence.HUMAN_REVIEW,
            )
            for i in range(3)
        ]
        result = mcp_mod.find_patient_matches(patient_id="p1", max_results=2)
        data = json.loads(result)

        assert data["matches_found"] == 2


class TestGetPatientDetails:
    """Tests for get_patient_details tool."""

    def test_returns_patient_json(self):
        result = mcp_mod.get_patient_details(patient_id="p1")
        data = json.loads(result)

        assert data["id"] == "p1"
        assert "John" in data["name"]
        assert "Smith" in data["name"]
        assert data["birth_date"] == "1985-03-15"
        assert data["gender"] == "male"
        assert len(data["identifiers"]) == 2
        assert len(data["addresses"]) == 1
        assert len(data["contact_points"]) == 2

    def test_patient_not_found(self, mock_service):
        mock_service.db.get_patient.side_effect = lambda pid: None
        result = mcp_mod.get_patient_details(patient_id="nonexistent")

        assert "not found" in result


class TestCompareTwoPatients:
    """Tests for compare_two_patients tool."""

    def test_returns_comparison_json(self, mock_service):
        result = mcp_mod.compare_two_patients(patient1_id="p1", patient2_id="p2")
        data = json.loads(result)

        assert data["patient1"]["id"] == "p1"
        assert data["patient2"]["id"] == "p2"
        assert data["match_result"]["overall_score"] == 0.72
        assert data["match_result"]["confidence"] == "human_review"
        assert "recommendation" in data["match_result"]

    def test_patient1_not_found(self, mock_service):
        mock_service.db.get_patient.side_effect = lambda pid: None
        result = mcp_mod.compare_two_patients(patient1_id="bad", patient2_id="p2")

        assert "not found" in result

    def test_patient2_not_found(self, mock_service):
        def _side(pid):
            if pid == "p1":
                return Patient(id="p1", source_id="s", source_system="T",
                               name=HumanName(family="X"))
            return None

        mock_service.db.get_patient.side_effect = _side
        result = mcp_mod.compare_two_patients(patient1_id="p1", patient2_id="bad")

        assert "not found" in result


class TestRunBatchMatching:
    """Tests for run_batch_matching tool."""

    def test_returns_statistics(self, mock_service):
        result = mcp_mod.run_batch_matching()
        data = json.loads(result)

        assert data["operation"] == "global_matching"
        assert data["status"] == "completed"
        assert data["statistics"]["total_patients"] == 2
        assert data["statistics"]["matches_found"] == 1

    def test_custom_min_score(self, mock_service):
        mcp_mod.run_batch_matching(min_score=0.7)

        mock_service.run_global_matching.assert_called_once_with(min_score=0.7)


class TestApprovePatientMatch:
    """Tests for approve_patient_match tool."""

    def test_approve_returns_empi(self, mock_service):
        result = mcp_mod.approve_patient_match(
            patient1_id="p1", patient2_id="p2", reviewed_by="admin"
        )
        data = json.loads(result)

        assert data["action"] == "match_approved"
        assert data["empi_id"] == "empi-001"
        assert data["reviewed_by"] == "admin"

    def test_approve_with_notes(self, mock_service):
        mcp_mod.approve_patient_match(
            patient1_id="p1", patient2_id="p2",
            reviewed_by="admin", notes="Verified by phone"
        )

        mock_service.approve_match.assert_called_once_with(
            patient1_id="p1", patient2_id="p2",
            reviewed_by="admin", notes="Verified by phone"
        )


class TestRejectPatientMatch:
    """Tests for reject_patient_match tool."""

    def test_reject_returns_confirmation(self, mock_service):
        result = mcp_mod.reject_patient_match(
            patient1_id="p1", patient2_id="p2",
            reviewed_by="admin", reason="Different people"
        )
        data = json.loads(result)

        assert data["action"] == "match_rejected"
        assert data["reason"] == "Different people"

    def test_reject_calls_service(self, mock_service):
        mcp_mod.reject_patient_match(
            patient1_id="p1", patient2_id="p2",
            reviewed_by="admin", reason="Different DOB"
        )

        mock_service.reject_match.assert_called_once_with(
            patient1_id="p1", patient2_id="p2",
            reviewed_by="admin", reason="Different DOB"
        )


class TestGetPendingReviews:
    """Tests for get_pending_reviews tool."""

    def test_returns_reviews_json(self, mock_service):
        result = mcp_mod.get_pending_reviews()
        data = json.loads(result)

        assert data["pending_reviews"] == 1
        assert data["reviews"][0]["match_score"] == 0.72

    def test_no_pending_reviews(self, mock_service):
        mock_service.get_pending_reviews.return_value = []
        result = mcp_mod.get_pending_reviews()

        assert "No pending reviews" in result

    def test_custom_limit(self, mock_service):
        mcp_mod.get_pending_reviews(limit=5)

        mock_service.get_pending_reviews.assert_called_once_with(limit=5)


class TestGetServiceStatistics:
    """Tests for get_service_statistics tool."""

    def test_returns_stats_json(self, mock_service):
        result = mcp_mod.get_service_statistics()
        data = json.loads(result)

        assert data["service"] == "Patient Matching Service"
        assert data["statistics"]["total_patients"] == 2
        assert data["statistics"]["total_empi_records"] == 0
        assert data["statistics"]["pending_reviews"] == 1


class TestSearchPatients:
    """Tests for search_patients tool."""

    def test_search_by_name(self):
        result = mcp_mod.search_patients(name="John")
        data = json.loads(result)

        assert data["patients_found"] == 1
        assert "John" in data["patients"][0]["name"]

    def test_search_by_birth_date(self):
        result = mcp_mod.search_patients(birth_date="1990-07-22")
        data = json.loads(result)

        assert data["patients_found"] == 1
        assert data["patients"][0]["birth_date"] == "1990-07-22"

    def test_search_by_identifier(self):
        result = mcp_mod.search_patients(identifier_value="MRN12345")
        data = json.loads(result)

        assert data["patients_found"] == 1
        assert data["patients"][0]["id"] == "p1"

    def test_search_no_results(self):
        result = mcp_mod.search_patients(name="Nobody")

        assert "No patients found" in result

    def test_search_limit(self, mock_service):
        # Both patients match when no filter is specified;
        # limit=1 should truncate
        result = mcp_mod.search_patients(limit=1)
        data = json.loads(result)

        assert data["patients_found"] == 1


# ========== Resource Tests ==========


class TestResources:
    """Tests for MCP resource functions."""

    def test_patient_resource(self):
        result = mcp_mod.patient_resource("p1")
        data = json.loads(result)

        assert data["id"] == "p1"
        assert "John" in data["name"]

    def test_service_stats_resource(self):
        result = mcp_mod.service_stats_resource()
        data = json.loads(result)

        assert "statistics" in data
        assert data["statistics"]["total_patients"] == 2

    def test_pending_reviews_resource(self):
        result = mcp_mod.pending_reviews_resource()
        data = json.loads(result)

        assert data["pending_reviews"] == 1


# ========== Prompt Tests ==========


class TestPrompts:
    """Tests for MCP prompt templates."""

    def test_analyze_patient_prompt(self):
        result = mcp_mod.analyze_patient("p1")

        assert "p1" in result
        assert "get_patient_details" in result
        assert "find_patient_matches" in result
        assert "compare_two_patients" in result

    def test_review_pending_matches_prompt(self):
        result = mcp_mod.review_pending_matches()

        assert "get_pending_reviews" in result
        assert "compare_two_patients" in result
        assert "approve" in result.lower()
        assert "reject" in result.lower()

    def test_batch_matching_report_prompt(self):
        result = mcp_mod.batch_matching_report(min_score="0.5")

        assert "0.5" in result
        assert "run_batch_matching" in result
        assert "get_service_statistics" in result


# ========== Helper Tests ==========


class TestHelpers:
    """Tests for internal helper functions."""

    def test_get_recommendation_auto_merge(self):
        result = mcp_mod._get_recommendation(MatchConfidence.AUTO_MERGE)
        assert "automatic merge" in result.lower()

    def test_get_recommendation_human_review(self):
        result = mcp_mod._get_recommendation(MatchConfidence.HUMAN_REVIEW)
        assert "human review" in result.lower()

    def test_get_recommendation_no_match(self):
        result = mcp_mod._get_recommendation(MatchConfidence.NO_MATCH)
        assert "different patients" in result.lower()


# ========== MCP Server Registration Tests ==========


class TestMCPRegistration:
    """Tests that the FastMCP server has the expected tools/resources/prompts registered."""

    def test_server_name(self):
        assert mcp_mod.mcp.name == "Patient Matching Service"

    def test_server_has_instructions(self):
        assert mcp_mod.mcp.instructions is not None
        assert "AUTO_MERGE" in mcp_mod.mcp.instructions
