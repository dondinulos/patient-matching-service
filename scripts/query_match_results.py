"""
Query Match Results from Cosmos DB NoSQL Container

This script queries the match_results container using the SQL API (not Gremlin).
The match_results container stores patient matching results for human review.

Usage:
    python scripts/query_match_results.py [--status pending_review|approved|rejected] [--min-score 0.5]
"""

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime


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
    
    # Get primary key
    cmd = f'az cosmosdb keys list --name {account_name} --resource-group {resource_group} --query "primaryMasterKey" -o tsv'
    result = subprocess.run(cmd, capture_output=True, text=True, shell=use_shell)
    if result.returncode == 0:
        key = result.stdout.strip()
    else:
        raise Exception(f"Failed to get Cosmos DB key: {result.stderr}")
    
    return {
        "endpoint": nosql_endpoint,
        "database": "patient-matching-db",
        "container": "match_results",
        "key": key
    }


def query_match_results(status: str = None, min_score: float = None, limit: int = 100):
    """
    Query match results from the NoSQL container.
    
    Args:
        status: Filter by status (pending_review, approved, rejected)
        min_score: Minimum score filter
        limit: Maximum results to return
    """
    from azure.cosmos import CosmosClient
    
    creds = get_cosmos_credentials()
    
    # Connect to Cosmos DB NoSQL
    print(f"Connecting to {creds['endpoint']}...")
    client = CosmosClient(creds["endpoint"], credential=creds["key"])
    database = client.get_database_client(creds["database"])
    container = database.get_container_client(creds["container"])
    
    # Build query
    query = "SELECT * FROM c"
    conditions = []
    parameters = []
    
    if status:
        conditions.append("c.status = @status")
        parameters.append({"name": "@status", "value": status})
    
    if min_score is not None:
        conditions.append("c.score >= @min_score")
        parameters.append({"name": "@min_score", "value": min_score})
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    query += " ORDER BY c.score DESC"
    
    print(f"Executing query: {query}")
    print(f"Parameters: {parameters}")
    print("-" * 60)
    
    # Execute query with cross-partition enabled
    results = list(container.query_items(
        query=query,
        parameters=parameters if parameters else None,
        enable_cross_partition_query=True,
        max_item_count=limit
    ))
    
    print(f"Found {len(results)} match results\n")
    
    # Display results
    for i, result in enumerate(results, 1):
        print(f"Match #{i}")
        print(f"  ID: {result.get('id')}")
        print(f"  Patient 1: {result.get('patient1_name')} ({result.get('patient1_id')})")
        print(f"  Patient 2: {result.get('patient2_name')} ({result.get('patient2_id')})")
        print(f"  Score: {result.get('score', 0):.3f}")
        print(f"  Confidence: {result.get('confidence')}")
        print(f"  Status: {result.get('status')}")
        print(f"  Created: {result.get('created_at')}")
        
        if result.get('reviewed_at'):
            print(f"  Reviewed: {result.get('reviewed_at')} by {result.get('reviewed_by')}")
            print(f"  Decision: {result.get('decision')}")
        
        # Show matching details
        details = result.get('details', {})
        if details:
            print(f"  Details:")
            print(f"    Deterministic Score: {details.get('deterministic_score', 0):.3f}")
            print(f"    Name Score: {details.get('name_score', 0):.3f}")
            print(f"    Address Score: {details.get('address_score', 0):.3f}")
        
        print()
    
    return results


def update_match_status(match_id: str, patient1_id: str, decision: str, reviewed_by: str = "system"):
    """
    Update the status of a match result.
    
    Args:
        match_id: The match result ID
        patient1_id: The patient1_id (partition key)
        decision: approved, rejected, or pending_review
        reviewed_by: Who made the decision
    """
    from azure.cosmos import CosmosClient
    
    creds = get_cosmos_credentials()
    
    client = CosmosClient(creds["endpoint"], credential=creds["key"])
    database = client.get_database_client(creds["database"])
    container = database.get_container_client(creds["container"])
    
    # Read existing item
    item = container.read_item(item=match_id, partition_key=patient1_id)
    
    # Update fields
    item["status"] = decision
    item["decision"] = decision
    item["reviewed_at"] = datetime.utcnow().isoformat()
    item["reviewed_by"] = reviewed_by
    
    # Upsert
    container.upsert_item(item)
    
    print(f"Updated match {match_id} to status: {decision}")
    return item


def main():
    parser = argparse.ArgumentParser(description="Query match results from Cosmos DB")
    parser.add_argument(
        "--status",
        choices=["pending_review", "approved", "rejected"],
        help="Filter by status"
    )
    parser.add_argument(
        "--min-score",
        type=float,
        help="Minimum score filter"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum results to return"
    )
    parser.add_argument(
        "--update",
        metavar="MATCH_ID",
        help="Update a match result status"
    )
    parser.add_argument(
        "--patient1-id",
        help="Patient1 ID (partition key) for update"
    )
    parser.add_argument(
        "--decision",
        choices=["approved", "rejected", "pending_review"],
        help="Decision for update"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Match Results Query Tool")
    print("=" * 60)
    
    try:
        if args.update:
            if not args.patient1_id or not args.decision:
                print("Error: --patient1-id and --decision required for update")
                sys.exit(1)
            update_match_status(args.update, args.patient1_id, args.decision)
        else:
            query_match_results(
                status=args.status,
                min_score=args.min_score,
                limit=args.limit
            )
        
        print("[OK] Query completed successfully")
        
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
