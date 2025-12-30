"""
Load FHIR Patient Data into Azure Cosmos DB Gremlin

This script loads all FHIR bundle files from data/fhir into Azure Cosmos DB
Graph database, including all clinical data (encounters, observations, 
conditions, procedures, immunizations, medication requests, diagnostic reports).

Clinical data loaded:
- Patients (with identifiers, addresses, contact points)
- Encounters
- Observations  
- Conditions
- Procedures
- Immunizations
- MedicationRequests
- DiagnosticReports
"""

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.patient_matching.cosmos_graph_db import CosmosGraphDB
from src.patient_matching.fhir_loader import FHIRBundleParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress verbose Gremlin driver error logs (conflicts are handled in code)
# The gremlin_python driver uses logger name "gremlinpython" (no underscore)
logging.getLogger("gremlinpython").setLevel(logging.CRITICAL)


def get_cosmos_credentials():
    """Get Cosmos DB credentials from Azure CLI"""
    
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
    
    # Get endpoint
    endpoint = f"{account_name}.gremlin.cosmos.azure.com"
    
    # Get primary key
    cmd = f'az cosmosdb keys list --name {account_name} --resource-group {resource_group} --query "primaryMasterKey" -o tsv'
    result = subprocess.run(cmd, capture_output=True, text=True, shell=use_shell)
    if result.returncode == 0:
        key = result.stdout.strip()
    else:
        raise Exception(f"Failed to get Cosmos DB key: {result.stderr}")
    
    return endpoint, "patient-matching-db", "patients", key


def load_fhir_to_cosmos(
    fhir_dir: str,
    clear_existing: bool = False,
    limit: int = None,
    skip_existing: bool = True,
    batch_reconnect: int = 5
) -> dict:
    """
    Load FHIR bundles into Cosmos DB.
    
    Args:
        fhir_dir: Directory containing FHIR JSON bundle files
        clear_existing: Whether to clear existing data before loading
        limit: Maximum number of bundles to load (None for all)
        skip_existing: Skip patients that already exist (handle 409 conflicts)
        batch_reconnect: Reconnect to DB every N patients to avoid timeouts
        
    Returns:
        Dictionary with load statistics
    """
    stats = {
        "bundles_found": 0,
        "bundles_loaded": 0,
        "bundles_failed": 0,
        "bundles_skipped": 0,
        "patients": 0,
        "encounters": 0,
        "observations": 0,
        "conditions": 0,
        "procedures": 0,
        "immunizations": 0,
        "medication_requests": 0,
        "diagnostic_reports": 0,
        "clinical_skipped": 0,
    }
    
    # Get credentials from Azure CLI
    endpoint, database, container, key = get_cosmos_credentials()
    
    def connect_db():
        """Create and connect to database."""
        logger.info("Connecting to Azure Cosmos DB Gremlin...")
        db = CosmosGraphDB(
            endpoint=endpoint,
            database=database,
            container=container,
            key=key
        )
        db.connect()
        logger.info("[OK] Connected to Cosmos DB")
        return db
    
    db = connect_db()
    
    # Clear existing data if requested
    if clear_existing:
        logger.info("[WARNING] Clearing all existing data...")
        try:
            db.clear_all_data()
            logger.info("[OK] Data cleared")
        except Exception as e:
            logger.error(f"[FAIL] Failed to clear data: {e}")
            raise
    
    # Find all FHIR bundle files
    fhir_path = Path(fhir_dir)
    bundle_files = list(fhir_path.glob("*.json"))
    stats["bundles_found"] = len(bundle_files)
    
    logger.info(f"Found {len(bundle_files)} FHIR bundle files in {fhir_dir}")
    
    if limit:
        bundle_files = bundle_files[:limit]
        logger.info(f"Limiting to {limit} bundles")
    
    # Initialize FHIR parser
    parser = FHIRBundleParser()
    
    # Track loaded patients to reconnect periodically
    patients_since_reconnect = 0
    
    # Process each bundle
    for i, bundle_file in enumerate(bundle_files):
        try:
            # Reconnect periodically to avoid connection timeouts
            if batch_reconnect and patients_since_reconnect >= batch_reconnect:
                logger.info(f"[INFO] Reconnecting after {patients_since_reconnect} patients...")
                try:
                    db.close()
                except Exception:
                    pass
                import time
                time.sleep(1)  # Brief pause before reconnect
                db = connect_db()
                patients_since_reconnect = 0
            
            logger.info(f"[{i+1}/{len(bundle_files)}] Loading {bundle_file.name}...")
            
            # Parse the bundle
            with open(bundle_file, "r", encoding="utf-8") as f:
                bundle = json.load(f)
            
            # Parse all clinical data from bundle
            bundle_data = parser.parse_bundle_full(bundle, str(bundle_file))
            
            if not bundle_data.get("patient"):
                logger.warning(f"  [WARNING] No patient found in {bundle_file.name}, skipping")
                stats["bundles_failed"] += 1
                continue
            
            # Load patient with all clinical data
            try:
                patient_id, clinical_stats = db.create_patient_with_clinical_data(bundle_data, skip_existing=skip_existing)
                patients_since_reconnect += 1
            except Exception as e:
                error_msg = str(e).lower()
                # Check for conflict (already exists) or connection errors
                if "conflict" in error_msg or "409" in error_msg or "already exists" in error_msg:
                    if skip_existing:
                        logger.info(f"  [SKIP] Patient already exists: {bundle_file.name}")
                        stats["bundles_skipped"] += 1
                        continue
                    else:
                        raise
                elif "closing" in error_msg or "connection" in error_msg or "transport" in error_msg:
                    # Connection issue - try to reconnect
                    logger.warning(f"  [RECONNECT] Connection lost, reconnecting...")
                    try:
                        db.close()
                    except Exception:
                        pass
                    import time
                    time.sleep(2)
                    db = connect_db()
                    patients_since_reconnect = 0
                    # Retry loading this patient
                    patient_id, clinical_stats = db.create_patient_with_clinical_data(bundle_data, skip_existing=skip_existing)
                    patients_since_reconnect += 1
                else:
                    raise
            
            # Update statistics from clinical data
            stats["bundles_loaded"] += 1
            stats["patients"] += 1
            stats["encounters"] += clinical_stats.get("encounters_created", 0)
            stats["observations"] += clinical_stats.get("observations_created", 0)
            stats["conditions"] += clinical_stats.get("conditions_created", 0)
            stats["procedures"] += clinical_stats.get("procedures_created", 0)
            stats["immunizations"] += clinical_stats.get("immunizations_created", 0)
            stats["medication_requests"] += clinical_stats.get("medication_requests_created", 0)
            stats["diagnostic_reports"] += clinical_stats.get("diagnostic_reports_created", 0)
            
            # Track skipped clinical items
            total_skipped = sum(v for k, v in clinical_stats.items() if k.endswith("_skipped"))
            stats["clinical_skipped"] += total_skipped
            
            patient_name = bundle_data["patient"].name if bundle_data["patient"].name else "Unknown"
            logger.info(f"  [OK] Loaded patient: {patient_name}")
            
            # Show created counts (compact logging)
            created_summary = []
            if clinical_stats.get("encounters_created", 0) > 0:
                created_summary.append(f"Enc:{clinical_stats['encounters_created']}")
            if clinical_stats.get("observations_created", 0) > 0:
                created_summary.append(f"Obs:{clinical_stats['observations_created']}")
            if clinical_stats.get("conditions_created", 0) > 0:
                created_summary.append(f"Cond:{clinical_stats['conditions_created']}")
            if clinical_stats.get("procedures_created", 0) > 0:
                created_summary.append(f"Proc:{clinical_stats['procedures_created']}")
            if created_summary:
                logger.info(f"       Created: {', '.join(created_summary)}")
            if total_skipped > 0:
                logger.info(f"       Skipped (already exist): {total_skipped} clinical items")
            
        except Exception as e:
            logger.error(f"  [FAIL] Error loading {bundle_file.name}: {e}")
            stats["bundles_failed"] += 1
            continue
    
    # Close connection
    db.close()
    
    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Load FHIR bundles into Azure Cosmos DB Gremlin"
    )
    parser.add_argument(
        "--fhir-dir",
        default="data/fhir",
        help="Directory containing FHIR bundle JSON files (default: data/fhir)"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all existing data before loading"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of bundles to load (default: all)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("=" * 60)
    print("FHIR to Cosmos DB Loader")
    print("=" * 60)
    print(f"FHIR Directory: {args.fhir_dir}")
    print(f"Clear existing: {args.clear}")
    print(f"Limit: {args.limit or 'All'}")
    print("=" * 60)
    
    try:
        stats = load_fhir_to_cosmos(
            fhir_dir=args.fhir_dir,
            clear_existing=args.clear,
            limit=args.limit
        )
        
        print("\n" + "=" * 60)
        print("Load Complete - Summary")
        print("=" * 60)
        print(f"Bundles found:        {stats['bundles_found']}")
        print(f"Bundles loaded:       {stats['bundles_loaded']}")
        print(f"Bundles skipped:      {stats['bundles_skipped']}")
        print(f"Bundles failed:       {stats['bundles_failed']}")
        print("-" * 60)
        print(f"Patients:             {stats['patients']}")
        print(f"Encounters:           {stats['encounters']}")
        print(f"Observations:         {stats['observations']}")
        print(f"Conditions:           {stats['conditions']}")
        print(f"Procedures:           {stats['procedures']}")
        print(f"Immunizations:        {stats['immunizations']}")
        print(f"Medication Requests:  {stats['medication_requests']}")
        print(f"Diagnostic Reports:   {stats['diagnostic_reports']}")
        print("-" * 60)
        print(f"Clinical items skipped (existing): {stats['clinical_skipped']}")
        print("=" * 60)
        
        if stats["bundles_failed"] > 0:
            print(f"\n[WARNING] {stats['bundles_failed']} bundles failed to load")
            sys.exit(1)
        else:
            print("\n[OK] All bundles loaded successfully!")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"[FAIL] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
