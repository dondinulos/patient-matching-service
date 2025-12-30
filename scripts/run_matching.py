"""
Patient Matching Service - Run Matching on Cosmos DB Graph

This script:
1. Queries all patients from Cosmos DB Gremlin graph
2. Runs pairwise matching using deterministic and probabilistic algorithms
3. Optionally uses AI-enhanced matching (embeddings and/or LLM)
4. Creates POTENTIAL_MATCH edges in the graph for high-scoring pairs
5. Stores detailed match results in a Cosmos DB NoSQL collection for review

Usage:
    python scripts/run_matching.py [--min-score 0.5] [--limit 100] [--verbose]
    
    # With AI-enhanced matching (requires Azure OpenAI or OpenAI API key):
    python scripts/run_matching.py --use-embeddings --use-llm --verbose
"""

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.patient_matching.cosmos_graph_db import CosmosGraphDB
from src.patient_matching.matching import (
    PatientMatcher, MatchWeights, DeterministicMatcher, ProbabilisticMatcher,
    EmbeddingMatcher, LLMMatcher
)
from src.patient_matching.models import Patient, MatchConfidence

# Azure SDK for Cosmos DB NoSQL
from azure.cosmos import CosmosClient, PartitionKey
from azure.identity import DefaultAzureCredential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_cosmos_credentials():
    """Get Cosmos DB credentials from Azure CLI."""
    resource_group = "rg-patient-matching"
    use_shell = platform.system() == "Windows"
    
    # Get Cosmos DB account name
    print("Fetching Cosmos DB credentials from Azure...")
    cmd = f'az cosmosdb list --resource-group {resource_group} --query "[0].name" -o tsv'
    result = subprocess.run(cmd, capture_output=True, text=True, shell=use_shell)
    if result.returncode == 0:
        account_name = result.stdout.strip()
        print(f"Found Cosmos DB account: {account_name}")
    else:
        raise Exception(f"Failed to get Cosmos DB account: {result.stderr}")
    
    # Get endpoint for NoSQL
    nosql_endpoint = f"https://{account_name}.documents.azure.com:443/"
    
    # Get Gremlin endpoint
    gremlin_endpoint = f"{account_name}.gremlin.cosmos.azure.com"
    
    # Get primary key
    cmd = f'az cosmosdb keys list --name {account_name} --resource-group {resource_group} --query "primaryMasterKey" -o tsv'
    result = subprocess.run(cmd, capture_output=True, text=True, shell=use_shell)
    if result.returncode == 0:
        key = result.stdout.strip()
    else:
        raise Exception(f"Failed to get Cosmos DB key: {result.stderr}")
    
    return {
        "gremlin_endpoint": gremlin_endpoint,
        "nosql_endpoint": nosql_endpoint,
        "database": "patient-matching-db",
        "graph_container": "patients",
        "key": key
    }


def ensure_match_results_container(nosql_client: CosmosClient, database_name: str) -> Any:
    """Ensure the match_results container exists for storing match results."""
    try:
        database = nosql_client.create_database_if_not_exists(database_name)
    except Exception as e:
        # Database might exist, try to get it
        database = nosql_client.get_database_client(database_name)
    
    # Create match_results container with patient_id as partition key
    try:
        container = database.create_container_if_not_exists(
            id="match_results",
            partition_key=PartitionKey(path="/patient1_id"),
            offer_throughput=400
        )
        logger.info("Created/connected to match_results container")
        return container
    except Exception as e:
        # Container might exist, try to get it
        container = database.get_container_client("match_results")
        logger.info("Connected to existing match_results container")
        return container


def get_all_patient_ids(graph_db: CosmosGraphDB) -> List[Dict[str, str]]:
    """Get all patient IDs and basic info from the graph."""
    query = """
    g.V().hasLabel('Patient')
        .valueMap(true)
    """
    results = graph_db._execute_query(query, {})
    
    patients = []
    for r in results:
        # 'id' is at top level, other props are in arrays
        patient_id = r.get("id")  # Not in array
        full_name = r.get("fullName", [""])[0] if r.get("fullName") else ""
        birth_date = r.get("birthDate", [""])[0] if r.get("birthDate") else ""
        source_system = r.get("source_system", [""])[0] if r.get("source_system") else ""
        
        if patient_id:
            patients.append({
                "id": patient_id,
                "full_name": full_name,
                "birth_date": birth_date,
                "source_system": source_system
            })
    
    return patients


def compute_match_score(
    patient1: Patient,
    patient2: Patient,
    deterministic_matcher: DeterministicMatcher,
    probabilistic_matcher: ProbabilisticMatcher,
    weights: MatchWeights,
    embedding_matcher: EmbeddingMatcher = None,
    llm_matcher: LLMMatcher = None
) -> Tuple[float, MatchConfidence, Dict[str, Any]]:
    """
    Compute comprehensive match score between two patients.
    
    Returns:
        (overall_score, confidence, details)
    """
    # Deterministic matching
    det_score, det_details = deterministic_matcher.compute_score(patient1, patient2)
    
    # Probabilistic name matching
    name_score, name_details = probabilistic_matcher.compute_name_similarity(patient1, patient2)
    
    # Probabilistic address matching
    addr_score, addr_details = probabilistic_matcher.compute_address_similarity(patient1, patient2)
    
    # AI-enhanced matching (optional)
    embed_score = 0.0
    embed_details = {}
    llm_score = 0.0
    llm_details = {}
    
    if embedding_matcher:
        try:
            embed_score, embed_details = embedding_matcher.compute_similarity(patient1, patient2)
            logger.debug(f"Embedding score: {embed_score:.3f}")
        except Exception as e:
            logger.warning(f"Embedding matching failed: {e}")
            embed_details = {"error": str(e)}
    
    if llm_matcher:
        try:
            llm_score, llm_details = llm_matcher.analyze_match(patient1, patient2)
            logger.debug(f"LLM score: {llm_score:.3f}")
        except Exception as e:
            logger.warning(f"LLM matching failed: {e}")
            llm_details = {"error": str(e)}
    
    # Combined score - adjust weights if AI matchers are enabled
    if embedding_matcher or llm_matcher:
        # When AI is enabled, give it significant weight
        embed_weight = 0.15 if embedding_matcher else 0.0
        llm_weight = 0.20 if llm_matcher else 0.0
        
        # Reduce other weights proportionally
        remaining_weight = 1.0 - embed_weight - llm_weight
        scale = remaining_weight / (weights.deterministic_weight + weights.name_weight + weights.address_weight)
        
        overall_score = (
            det_score * weights.deterministic_weight * scale +
            name_score * weights.name_weight * scale +
            addr_score * weights.address_weight * scale +
            embed_score * embed_weight +
            llm_score * llm_weight
        )
    else:
        # Standard scoring without AI
        overall_score = (
            det_score * weights.deterministic_weight +
            name_score * weights.name_weight +
            addr_score * weights.address_weight
        )
    
    # Gender match bonus
    if patient1.gender == patient2.gender and patient1.gender is not None:
        overall_score += weights.gender_match_bonus
    
    # Cap at 1.0
    overall_score = min(1.0, overall_score)
    
    # Determine confidence level
    if overall_score >= weights.auto_merge_threshold:
        confidence = MatchConfidence.AUTO_MERGE
    elif overall_score >= weights.human_review_threshold:
        confidence = MatchConfidence.HUMAN_REVIEW
    else:
        confidence = MatchConfidence.NO_MATCH
    
    # Compile details
    details = {
        "deterministic_score": det_score,
        "deterministic_details": det_details,
        "name_score": name_score,
        "name_details": name_details,
        "address_score": addr_score,
        "address_details": addr_details,
        "gender_match": patient1.gender == patient2.gender,
        "embedding_score": embed_score,
        "embedding_details": embed_details,
        "llm_score": llm_score,
        "llm_details": llm_details
    }
    
    return overall_score, confidence, details


def run_matching(
    min_score: float = 0.5,
    limit: int = None,
    verbose: bool = False,
    use_embeddings: bool = False,
    use_llm: bool = False
) -> Dict[str, Any]:
    """
    Run matching on all patients in the graph.
    
    Args:
        min_score: Minimum score to record a match
        limit: Limit number of patients to process
        verbose: Enable verbose logging
        use_embeddings: Use AI embeddings for semantic matching
        use_llm: Use LLM (GPT-4) for intelligent matching
        
    Returns:
        Statistics about the matching run
    """
    stats = {
        "patients_processed": 0,
        "pairs_compared": 0,
        "matches_found": 0,
        "auto_merge": 0,
        "human_review": 0,
        "no_match": 0,
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": None
    }
    
    # Get credentials
    creds = get_cosmos_credentials()
    
    # Connect to Gremlin graph
    logger.info("Connecting to Cosmos DB Gremlin...")
    graph_db = CosmosGraphDB(
        endpoint=creds["gremlin_endpoint"],
        database=creds["database"],
        container=creds["graph_container"],
        key=creds["key"]
    )
    graph_db.connect()
    logger.info("[OK] Connected to Gremlin graph")
    
    # Connect to NoSQL for match results
    logger.info("Connecting to Cosmos DB NoSQL for match results...")
    nosql_client = CosmosClient(creds["nosql_endpoint"], credential=creds["key"])
    results_container = ensure_match_results_container(nosql_client, creds["database"])
    logger.info("[OK] Connected to match_results container")
    
    # Get all patients
    logger.info("Fetching all patients from graph...")
    patient_infos = get_all_patient_ids(graph_db)
    logger.info(f"Found {len(patient_infos)} patients")
    
    if limit:
        patient_infos = patient_infos[:limit]
        logger.info(f"Limited to {limit} patients")
    
    stats["patients_processed"] = len(patient_infos)
    
    # Initialize matchers
    weights = MatchWeights()
    deterministic_matcher = DeterministicMatcher(weights)
    probabilistic_matcher = ProbabilisticMatcher(weights)
    
    # Initialize AI matchers if enabled
    embedding_matcher = None
    llm_matcher = None
    
    if use_embeddings:
        logger.info("Initializing EmbeddingMatcher (Azure OpenAI)...")
        try:
            embedding_matcher = EmbeddingMatcher(use_azure=True)
            logger.info("[OK] EmbeddingMatcher initialized")
        except Exception as e:
            logger.error(f"[FAIL] Failed to initialize EmbeddingMatcher: {e}")
            logger.info("Continuing without embeddings...")
    
    if use_llm:
        logger.info("Initializing LLMMatcher (Azure OpenAI GPT-4)...")
        try:
            llm_matcher = LLMMatcher(use_azure=True, model="gpt-4o")
            logger.info("[OK] LLMMatcher initialized")
        except Exception as e:
            logger.error(f"[FAIL] Failed to initialize LLMMatcher: {e}")
            logger.info("Continuing without LLM...")
    
    # Load full patient objects
    logger.info("Loading patient details...")
    patients = []
    for info in patient_infos:
        patient = graph_db.get_patient(info["id"])
        if patient:
            patients.append(patient)
    
    logger.info(f"Loaded {len(patients)} patients with full details")
    
    # Pairwise comparison
    matches = []
    total_pairs = len(patients) * (len(patients) - 1) // 2
    logger.info(f"Comparing {total_pairs} patient pairs...")
    
    pair_count = 0
    for i in range(len(patients)):
        for j in range(i + 1, len(patients)):
            pair_count += 1
            patient1 = patients[i]
            patient2 = patients[j]
            
            stats["pairs_compared"] += 1
            
            # Compute match score
            score, confidence, details = compute_match_score(
                patient1, patient2,
                deterministic_matcher,
                probabilistic_matcher,
                weights,
                embedding_matcher=embedding_matcher,
                llm_matcher=llm_matcher
            )
            
            if score >= min_score:
                match_result = {
                    "id": str(uuid.uuid4()),
                    "patient1_id": patient1.id,
                    "patient1_name": patient1.name.normalize() if patient1.name else "",
                    "patient1_dob": patient1.birth_date.isoformat() if patient1.birth_date else None,
                    "patient2_id": patient2.id,
                    "patient2_name": patient2.name.normalize() if patient2.name else "",
                    "patient2_dob": patient2.birth_date.isoformat() if patient2.birth_date else None,
                    "score": score,
                    "confidence": confidence.value,
                    "status": "pending_review",
                    "details": details,
                    "created_at": datetime.utcnow().isoformat(),
                    "reviewed_at": None,
                    "reviewed_by": None,
                    "decision": None
                }
                matches.append(match_result)
                stats["matches_found"] += 1
                
                if confidence == MatchConfidence.AUTO_MERGE:
                    stats["auto_merge"] += 1
                elif confidence == MatchConfidence.HUMAN_REVIEW:
                    stats["human_review"] += 1
                else:
                    stats["no_match"] += 1
                
                if verbose:
                    logger.info(
                        f"  Match: {patient1.name.normalize()} <-> {patient2.name.normalize()} "
                        f"score={score:.3f} confidence={confidence.value}"
                    )
                
                # Create POTENTIAL_MATCH edge in graph
                try:
                    graph_db.create_potential_match(
                        patient1.id,
                        patient2.id,
                        score,
                        confidence,
                        details
                    )
                except Exception as e:
                    logger.warning(f"Failed to create POTENTIAL_MATCH edge: {e}")
            
            if pair_count % 10 == 0:
                logger.info(f"  Processed {pair_count}/{total_pairs} pairs, found {stats['matches_found']} matches")
    
    # Store match results in NoSQL container
    logger.info(f"Storing {len(matches)} match results in Cosmos DB...")
    for match in matches:
        try:
            results_container.upsert_item(match)
        except Exception as e:
            logger.error(f"Failed to store match result: {e}")
    
    logger.info("[OK] Match results stored")
    
    stats["completed_at"] = datetime.utcnow().isoformat()
    
    # Close connections
    graph_db.close()
    
    return stats, matches


def main():
    parser = argparse.ArgumentParser(description="Run patient matching on Cosmos DB graph")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.3,
        help="Minimum match score to record (default: 0.3)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of patients to process (default: all)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--use-embeddings",
        action="store_true",
        help="Use AI embeddings for semantic matching (requires Azure OpenAI)"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM (GPT-4) for intelligent matching (requires Azure OpenAI)"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("=" * 60)
    print("Patient Matching Service")
    print("=" * 60)
    print(f"Minimum Score: {args.min_score}")
    print(f"Patient Limit: {args.limit or 'All'}")
    print(f"AI Embeddings: {'Enabled' if args.use_embeddings else 'Disabled'}")
    print(f"AI LLM (GPT-4): {'Enabled' if args.use_llm else 'Disabled'}")
    print("=" * 60)
    
    try:
        stats, matches = run_matching(
            min_score=args.min_score,
            limit=args.limit,
            verbose=args.verbose,
            use_embeddings=args.use_embeddings,
            use_llm=args.use_llm
        )
        
        print("\n" + "=" * 60)
        print("Matching Complete - Summary")
        print("=" * 60)
        print(f"Patients processed:    {stats['patients_processed']}")
        print(f"Pairs compared:        {stats['pairs_compared']}")
        print(f"Matches found:         {stats['matches_found']}")
        print("-" * 60)
        print(f"Auto merge:            {stats['auto_merge']}")
        print(f"Human review:          {stats['human_review']}")
        print(f"No match:              {stats['no_match']}")
        print("-" * 60)
        print(f"Started:              {stats['started_at']}")
        print(f"Completed:            {stats['completed_at']}")
        print("=" * 60)
        
        if matches:
            print("\nTop Matches:")
            sorted_matches = sorted(matches, key=lambda m: m["score"], reverse=True)[:10]
            for m in sorted_matches:
                print(f"  {m['patient1_name']} <-> {m['patient2_name']}: {m['score']:.3f} ({m['confidence']})")
        
        print("\n[OK] Matching completed successfully!")
        print(f"Results stored in Cosmos DB 'match_results' container")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"[FAIL] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
