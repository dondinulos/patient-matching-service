"""
Test Patient Matching with Real FHIR Data

This test loads actual FHIR bundle files from data/fhir and tests
the matching algorithms with real patient data.
"""

import pytest
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.patient_matching.fhir_loader import FHIRPatientLoader
from src.patient_matching.matching import (
    PatientMatcher, DeterministicMatcher, ProbabilisticMatcher, MatchWeights
)
from src.patient_matching.models import Patient, MatchConfidence

# Path to FHIR data
FHIR_DATA_DIR = Path(__file__).parent.parent / "data" / "fhir"


class TestFHIRDataLoading:
    """Test loading FHIR bundle files"""
    
    def test_fhir_directory_exists(self):
        """Verify FHIR data directory exists"""
        assert FHIR_DATA_DIR.exists(), f"FHIR data directory not found: {FHIR_DATA_DIR}"
    
    def test_load_single_fhir_file(self):
        """Test loading a single FHIR bundle file"""
        loader = FHIRPatientLoader(source_system="Synthea")
        
        # Get first file
        fhir_files = list(FHIR_DATA_DIR.glob("*.json"))
        assert len(fhir_files) > 0, "No FHIR JSON files found"
        
        patient = loader.load_from_file(str(fhir_files[0]))
        
        assert patient is not None
        assert patient.id is not None
        print(f"\nLoaded patient: {patient.name.full_name if patient.name else 'No name'}")
        print(f"  DOB: {patient.birth_date}")
        print(f"  Gender: {patient.gender}")
        print(f"  Identifiers: {len(patient.identifiers)}")
        print(f"  Addresses: {len(patient.addresses)}")
        print(f"  Contact Points: {len(patient.contact_points)}")
    
    def test_load_multiple_fhir_files(self):
        """Test loading multiple FHIR files"""
        loader = FHIRPatientLoader(source_system="Synthea")
        
        patients = list(loader.load_from_directory(str(FHIR_DATA_DIR), limit=10))
        
        assert len(patients) > 0
        print(f"\nLoaded {len(patients)} patients:")
        for p in patients[:5]:
            print(f"  - {p.name.full_name if p.name else 'Unknown'} ({p.birth_date})")
    
    def test_fhir_patient_has_required_fields(self):
        """Verify loaded patients have necessary fields for matching"""
        loader = FHIRPatientLoader(source_system="Synthea")
        
        patients = list(loader.load_from_directory(str(FHIR_DATA_DIR), limit=20))
        
        patients_with_name = sum(1 for p in patients if p.name and p.name.full_name)
        patients_with_dob = sum(1 for p in patients if p.birth_date)
        patients_with_ids = sum(1 for p in patients if len(p.identifiers) > 0)
        patients_with_address = sum(1 for p in patients if len(p.addresses) > 0)
        
        print(f"\nPatient field coverage ({len(patients)} patients):")
        print(f"  With name: {patients_with_name} ({100*patients_with_name/len(patients):.1f}%)")
        print(f"  With DOB: {patients_with_dob} ({100*patients_with_dob/len(patients):.1f}%)")
        print(f"  With identifiers: {patients_with_ids} ({100*patients_with_ids/len(patients):.1f}%)")
        print(f"  With address: {patients_with_address} ({100*patients_with_address/len(patients):.1f}%)")
        
        # Most patients should have these fields
        assert patients_with_name >= len(patients) * 0.9, "Most patients should have names"
        assert patients_with_dob >= len(patients) * 0.9, "Most patients should have DOB"


class TestMatchingWithFHIRData:
    """Test matching algorithms with real FHIR data"""
    
    @pytest.fixture
    def patients(self):
        """Load patients from FHIR files"""
        loader = FHIRPatientLoader(source_system="Synthea")
        return list(loader.load_from_directory(str(FHIR_DATA_DIR), limit=50))
    
    @pytest.fixture
    def matcher(self):
        """Create a matcher without embeddings for testing"""
        return PatientMatcher(use_embeddings=False)
    
    def test_deterministic_matching_same_patient(self, patients):
        """Test deterministic matching - same patient should match perfectly"""
        if not patients:
            pytest.skip("No patients loaded")
        
        matcher = DeterministicMatcher()
        patient = patients[0]
        
        # Match patient with itself
        score, details = matcher.compute_score(patient, patient)
        
        print(f"\nDeterministic self-match for: {patient.name.full_name if patient.name else 'Unknown'}")
        print(f"  Score: {score:.3f}")
        print(f"  DOB Match: {details.get('dob_match')}")
        print(f"  Matched Identifiers: {len(details.get('matched_identifiers', []))}")
        
        # Should have perfect or near-perfect match
        assert score > 0.5, "Self-match should have high score"
    
    def test_probabilistic_name_similarity(self, patients):
        """Test probabilistic name matching between different patients"""
        if len(patients) < 2:
            pytest.skip("Need at least 2 patients")
        
        matcher = ProbabilisticMatcher()
        
        print("\nName similarity between patients:")
        # Compare first few patients
        for i in range(min(5, len(patients))):
            for j in range(i+1, min(5, len(patients))):
                p1, p2 = patients[i], patients[j]
                if p1.name and p2.name:
                    score, details = matcher.compute_name_similarity(p1, p2)
                    print(f"  {p1.name.full_name} vs {p2.name.full_name}: {score:.3f}")
    
    def test_full_matching_pipeline(self, patients, matcher):
        """Test the full matching pipeline with real data"""
        if len(patients) < 2:
            pytest.skip("Need at least 2 patients")
        
        print(f"\nRunning full matching on {len(patients)} patients...")
        
        # Match first patient against all others
        source_patient = patients[0]
        candidates = patients[1:]
        
        results = matcher.find_matches(source_patient, candidates, min_score=0.2)
        
        print(f"\nMatches for: {source_patient.name.full_name if source_patient.name else 'Unknown'}")
        print(f"  DOB: {source_patient.birth_date}")
        print(f"  Found {len(results)} potential matches")
        
        if results:
            print("\nTop matches:")
            for i, result in enumerate(results[:5]):
                candidate = next((p for p in candidates if p.id == result.patient2_id), None)
                if candidate:
                    print(f"  {i+1}. {candidate.name.full_name if candidate.name else 'Unknown'}")
                    print(f"     Score: {result.score:.3f}")
                    print(f"     Confidence: {result.confidence.value}")
                    print(f"     Shared IDs: {result.shared_identifiers}")
    
    def test_batch_matching(self, patients, matcher):
        """Test batch matching - all patients against each other"""
        if len(patients) < 5:
            pytest.skip("Need at least 5 patients")
        
        # Use subset for performance
        test_patients = patients[:20]
        
        print(f"\nBatch matching {len(test_patients)} patients...")
        
        results = matcher.batch_match(test_patients, min_score=0.3)
        
        print(f"Found {len(results)} potential match pairs")
        
        # Categorize by confidence
        auto_merge = sum(1 for r in results if r.confidence == MatchConfidence.AUTO_MERGE)
        human_review = sum(1 for r in results if r.confidence == MatchConfidence.HUMAN_REVIEW)
        no_match = sum(1 for r in results if r.confidence == MatchConfidence.NO_MATCH)
        
        print(f"\nConfidence distribution:")
        print(f"  Auto-merge: {auto_merge}")
        print(f"  Human review: {human_review}")
        print(f"  No match: {no_match}")
        
        if results:
            print("\nHighest scoring pairs:")
            sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
            for result in sorted_results[:5]:
                p1 = next((p for p in test_patients if p.id == result.patient1_id), None)
                p2 = next((p for p in test_patients if p.id == result.patient2_id), None)
                if p1 and p2:
                    print(f"  {p1.name.full_name if p1.name else 'Unknown'} <-> "
                          f"{p2.name.full_name if p2.name else 'Unknown'}: {result.score:.3f}")
    
    def test_find_similar_names(self, patients):
        """Find patients with similar names (potential duplicates or relatives)"""
        if len(patients) < 10:
            pytest.skip("Need at least 10 patients")
        
        matcher = ProbabilisticMatcher()
        
        similar_pairs = []
        for i in range(len(patients)):
            for j in range(i+1, len(patients)):
                p1, p2 = patients[i], patients[j]
                if p1.name and p2.name:
                    score, _ = matcher.compute_name_similarity(p1, p2)
                    if score > 0.7:  # High name similarity
                        similar_pairs.append((p1, p2, score))
        
        if similar_pairs:
            print(f"\nFound {len(similar_pairs)} pairs with similar names (>0.7):")
            for p1, p2, score in sorted(similar_pairs, key=lambda x: x[2], reverse=True)[:10]:
                print(f"  {p1.name.full_name} vs {p2.name.full_name}: {score:.3f}")
                if p1.birth_date and p2.birth_date:
                    print(f"    DOB: {p1.birth_date} vs {p2.birth_date}")
        else:
            print("\nNo pairs with highly similar names found")
    
    def test_same_dob_patients(self, patients):
        """Find patients with the same date of birth"""
        from collections import defaultdict
        
        dob_groups = defaultdict(list)
        for p in patients:
            if p.birth_date:
                dob_groups[p.birth_date].append(p)
        
        same_dob = {dob: pts for dob, pts in dob_groups.items() if len(pts) > 1}
        
        print(f"\nPatients with same DOB: {len(same_dob)} groups")
        for dob, pts in list(same_dob.items())[:5]:
            print(f"\n  {dob}:")
            for p in pts:
                print(f"    - {p.name.full_name if p.name else 'Unknown'}")


class TestMatchingScoreDistribution:
    """Analyze the distribution of matching scores"""
    
    def test_score_distribution(self):
        """Generate statistics on match scores across the dataset"""
        loader = FHIRPatientLoader(source_system="Synthea")
        patients = list(loader.load_from_directory(str(FHIR_DATA_DIR), limit=30))
        
        if len(patients) < 10:
            pytest.skip("Need at least 10 patients")
        
        matcher = PatientMatcher(use_embeddings=False)
        
        # Collect all pairwise scores
        scores = []
        for i in range(len(patients)):
            for j in range(i+1, len(patients)):
                result = matcher.match(patients[i], patients[j])
                scores.append(result.score)
        
        if not scores:
            pytest.skip("No scores generated")
        
        # Calculate statistics
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        
        # Distribution buckets
        buckets = {
            "0.0-0.2": 0,
            "0.2-0.4": 0,
            "0.4-0.6": 0,
            "0.6-0.8": 0,
            "0.8-1.0": 0,
        }
        
        for score in scores:
            if score < 0.2:
                buckets["0.0-0.2"] += 1
            elif score < 0.4:
                buckets["0.2-0.4"] += 1
            elif score < 0.6:
                buckets["0.4-0.6"] += 1
            elif score < 0.8:
                buckets["0.6-0.8"] += 1
            else:
                buckets["0.8-1.0"] += 1
        
        print(f"\nScore Distribution ({len(scores)} pairs):")
        print(f"  Min: {min_score:.3f}")
        print(f"  Max: {max_score:.3f}")
        print(f"  Avg: {avg_score:.3f}")
        print("\nDistribution:")
        for bucket, count in buckets.items():
            pct = 100 * count / len(scores)
            bar = "█" * int(pct / 5)
            print(f"  {bucket}: {count:4d} ({pct:5.1f}%) {bar}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
