"""
Query Cosmos DB to verify patient data was loaded successfully
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.patient_matching.cosmos_graph_db import CosmosGraphDB
import subprocess
import platform

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

def main():
    print("=" * 60)
    print("Cosmos DB Patient Data Query")
    print("=" * 60)
    
    # Get credentials
    endpoint, database, container, key = get_cosmos_credentials()
    
    print(f"\nCosmos DB Configuration:")
    print(f"  Endpoint: {endpoint}")
    print(f"  Database: {database}")
    print(f"  Container: {container}")
    
    # Initialize Cosmos DB connection
    print("\nConnecting to Cosmos DB...")
    cosmos_db = CosmosGraphDB(
        endpoint=endpoint,
        database=database,
        container=container,
        key=key
    )
    cosmos_db.connect()
    print("Connected successfully!\n")
    
    # Get statistics
    print("=" * 60)
    print("Database Statistics")
    print("=" * 60)
    
    try:
        stats = cosmos_db.get_stats()
        for key, value in stats.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
    except Exception as e:
        print(f"Error getting stats: {e}")
    
    # Query some sample patients
    print("\n" + "=" * 60)
    print("Sample Patients (First 5)")
    print("=" * 60)
    
    try:
        query = "g.V().hasLabel('Patient').limit(5).valueMap('id', 'fullName', 'birthDate', 'gender')"
        results = cosmos_db._execute_query(query)
        
        for i, patient in enumerate(results, 1):
            print(f"\nPatient {i}:")
            print(f"  ID: {patient.get('id', ['N/A'])[0]}")
            print(f"  Name: {patient.get('fullName', ['N/A'])[0]}")
            print(f"  Birth Date: {patient.get('birthDate', ['N/A'])[0]}")
            print(f"  Gender: {patient.get('gender', ['N/A'])[0]}")
    except Exception as e:
        print(f"Error querying patients: {e}")
    
    # Query identifiers
    print("\n" + "=" * 60)
    print("Sample Identifiers (First 5)")
    print("=" * 60)
    
    try:
        query = "g.V().hasLabel('Identifier').limit(5).valueMap('type', 'value')"
        results = cosmos_db._execute_query(query)
        
        for i, identifier in enumerate(results, 1):
            print(f"\nIdentifier {i}:")
            print(f"  Type: {identifier.get('type', ['N/A'])[0]}")
            print(f"  Value: {identifier.get('value', ['N/A'])[0]}")
    except Exception as e:
        print(f"Error querying identifiers: {e}")
    
    # Query addresses
    print("\n" + "=" * 60)
    print("Sample Addresses (First 5)")
    print("=" * 60)
    
    try:
        query = "g.V().hasLabel('Address').limit(5).valueMap('city', 'state', 'postalCode')"
        results = cosmos_db._execute_query(query)
        
        for i, address in enumerate(results, 1):
            print(f"\nAddress {i}:")
            print(f"  City: {address.get('city', ['N/A'])[0]}")
            print(f"  State: {address.get('state', ['N/A'])[0]}")
            print(f"  Postal Code: {address.get('postalCode', ['N/A'])[0]}")
    except Exception as e:
        print(f"Error querying addresses: {e}")
    
    # Query contact points
    print("\n" + "=" * 60)
    print("Sample Contact Points (First 5)")
    print("=" * 60)
    
    try:
        query = "g.V().hasLabel('ContactPoint').limit(5).valueMap('system', 'value')"
        results = cosmos_db._execute_query(query)
        
        for i, contact in enumerate(results, 1):
            print(f"\nContact {i}:")
            print(f"  System: {contact.get('system', ['N/A'])[0]}")
            print(f"  Value: {contact.get('value', ['N/A'])[0]}")
    except Exception as e:
        print(f"Error querying contacts: {e}")
    
    # Query relationships
    print("\n" + "=" * 60)
    print("Sample Relationships")
    print("=" * 60)
    
    try:
        query = """
        g.V().hasLabel('Patient').limit(1).as('p')
         .select('p').by(valueMap('id', 'fullName'))
        """
        patient_results = cosmos_db._execute_query(query)
        
        if patient_results:
            patient = patient_results[0]
            patient_id = patient.get('id', ['N/A'])[0]
            patient_name = patient.get('fullName', ['N/A'])[0]
            
            print(f"\nPatient: {patient_name} (ID: {patient_id})")
            
            # Get identifiers for this patient
            query = f"""
            g.V().has('Patient', 'id', '{patient_id}')
             .out('HAS_IDENTIFIER')
             .valueMap('type', 'value')
             .limit(3)
            """
            id_results = cosmos_db._execute_query(query)
            
            if id_results:
                print(f"  Identifiers ({len(id_results)}):")
                for id_data in id_results:
                    print(f"    - {id_data.get('type', ['N/A'])[0]}: {id_data.get('value', ['N/A'])[0]}")
            
            # Get addresses for this patient
            query = f"""
            g.V().has('Patient', 'id', '{patient_id}')
             .out('HAS_ADDRESS')
             .valueMap('city', 'state')
             .limit(2)
            """
            addr_results = cosmos_db._execute_query(query)
            
            if addr_results:
                print(f"  Addresses ({len(addr_results)}):")
                for addr_data in addr_results:
                    print(f"    - {addr_data.get('city', ['N/A'])[0]}, {addr_data.get('state', ['N/A'])[0]}")
    except Exception as e:
        print(f"Error querying relationships: {e}")
    
    # Close connection
    cosmos_db.close()
    
    print("\n" + "=" * 60)
    print("Query Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
