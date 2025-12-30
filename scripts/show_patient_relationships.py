"""
Show Patient Relationships - Visualize graph connections

Displays how a patient is connected to:
- Identifiers
- Addresses  
- Contact Points
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
    cmd = f'az cosmosdb list --resource-group {resource_group} --query "[0].name" -o tsv'
    result = subprocess.run(cmd, capture_output=True, text=True, shell=use_shell)
    if result.returncode == 0:
        account_name = result.stdout.strip()
    else:
        raise Exception(f"Failed to get Cosmos DB account: {result.stderr}")
    
    endpoint = f"{account_name}.gremlin.cosmos.azure.com"
    
    # Get primary key
    cmd = f'az cosmosdb keys list --name {account_name} --resource-group {resource_group} --query "primaryMasterKey" -o tsv'
    result = subprocess.run(cmd, capture_output=True, text=True, shell=use_shell)
    if result.returncode == 0:
        key = result.stdout.strip()
    else:
        raise Exception(f"Failed to get Cosmos DB key: {result.stderr}")
    
    return endpoint, "patient-matching-db", "patients", key

def print_box(title, content, width=70):
    """Print a nice box around content"""
    print("┌" + "─" * (width - 2) + "┐")
    print("│ " + title.ljust(width - 4) + " │")
    print("├" + "─" * (width - 2) + "┤")
    for line in content:
        if isinstance(line, str):
            print("│ " + line.ljust(width - 4) + " │")
        else:
            print("│ " + " " * (width - 4) + " │")
    print("└" + "─" * (width - 2) + "┘")

def main():
    print("\n" + "=" * 70)
    print("Patient Relationship Visualizer")
    print("=" * 70 + "\n")
    
    # Get credentials
    endpoint, database, container, key = get_cosmos_credentials()
    
    # Initialize Cosmos DB connection
    print("Connecting to Cosmos DB...")
    cosmos_db = CosmosGraphDB(
        endpoint=endpoint,
        database=database,
        container=container,
        key=key
    )
    cosmos_db.connect()
    print("✓ Connected\n")
    
    # Get a sample patient with relationships
    query = """
    g.V().hasLabel('Patient').limit(1).valueMap('id', 'fullName', 'birthDate', 'gender')
    """
    
    patient_results = cosmos_db._execute_query(query)
    
    if not patient_results:
        print("No patients found in database")
        cosmos_db.close()
        return
    
    patient = patient_results[0]
    patient_id = patient.get('id', ['N/A'])[0]
    patient_name = patient.get('fullName', ['N/A'])[0]
    birth_date = patient.get('birthDate', ['N/A'])[0]
    gender = patient.get('gender', ['N/A'])[0]
    
    # Display patient info
    print_box("PATIENT", [
        f"ID: {patient_id}",
        f"Name: {patient_name}",
        f"Birth Date: {birth_date}",
        f"Gender: {gender}",
    ])
    print()
    
    # Get identifiers
    id_query = f"""
    g.V().has('Patient', 'id', '{patient_id}')
     .outE('HAS_IDENTIFIER').as('edge')
     .inV().as('identifier')
     .select('identifier')
     .by(valueMap('type', 'value', 'system'))
    """
    
    id_results = cosmos_db._execute_query(id_query)
    
    if id_results:
        print("    │")
        print("    ├─── HAS_IDENTIFIER ───> [Identifiers]")
        print("    │")
        
        identifier_lines = []
        for i, id_data in enumerate(id_results, 1):
            id_type = id_data.get('type', ['N/A'])[0]
            id_value = id_data.get('value', ['N/A'])[0]
            id_system = id_data.get('system', [''])[0]
            
            identifier_lines.append(f"{i}. {id_type}: {id_value}")
            if id_system:
                identifier_lines.append(f"   System: {id_system}")
        
        print_box("IDENTIFIERS", identifier_lines)
        print()
    
    # Get addresses
    addr_query = f"""
    g.V().has('Patient', 'id', '{patient_id}')
     .outE('HAS_ADDRESS').as('edge')
     .inV().as('address')
     .select('address')
     .by(valueMap('line', 'city', 'state', 'postalCode', 'country'))
    """
    
    addr_results = cosmos_db._execute_query(addr_query)
    
    if addr_results:
        print("    │")
        print("    ├─── HAS_ADDRESS ───> [Addresses]")
        print("    │")
        
        address_lines = []
        for i, addr_data in enumerate(addr_results, 1):
            line = addr_data.get('line', [''])[0]
            city = addr_data.get('city', [''])[0]
            state = addr_data.get('state', [''])[0]
            postal = addr_data.get('postalCode', [''])[0]
            country = addr_data.get('country', [''])[0]
            
            address_lines.append(f"{i}. {line}")
            if city or state:
                address_lines.append(f"   {city}, {state} {postal}")
            if country:
                address_lines.append(f"   {country}")
        
        print_box("ADDRESSES", address_lines)
        print()
    
    # Get contact points
    contact_query = f"""
    g.V().has('Patient', 'id', '{patient_id}')
     .outE('HAS_CONTACT').as('edge')
     .inV().as('contact')
     .select('contact')
     .by(valueMap('system', 'value'))
    """
    
    contact_results = cosmos_db._execute_query(contact_query)
    
    if contact_results:
        print("    │")
        print("    └─── HAS_CONTACT ───> [Contact Points]")
        print()
        
        contact_lines = []
        for i, contact_data in enumerate(contact_results, 1):
            system = contact_data.get('system', ['N/A'])[0]
            value = contact_data.get('value', ['N/A'])[0]
            
            contact_lines.append(f"{i}. {system.upper()}: {value}")
        
        print_box("CONTACT POINTS", contact_lines)
        print()
    
    # Show graph structure
    print("\n" + "=" * 70)
    print("Graph Structure Summary")
    print("=" * 70)
    print("""
The graph database uses vertices (nodes) and edges (relationships):

    [Patient] ──HAS_IDENTIFIER──> [Identifier]
              |
              ├──HAS_ADDRESS──> [Address]
              |
              └──HAS_CONTACT──> [ContactPoint]

Key Benefits:
• Efficient traversal to find related data
• Easy to query "find patients who share an identifier"
• Flexible schema for adding new relationship types
• Supports complex pattern matching for duplicate detection
    """)
    
    # Show how to find shared relationships
    print("=" * 70)
    print("Finding Patients with Shared Addresses")
    print("=" * 70)
    
    shared_query = f"""
    g.V().has('Patient', 'id', '{patient_id}')
     .out('HAS_ADDRESS')
     .in('HAS_ADDRESS')
     .where(neq('{patient_id}'))
     .dedup()
     .limit(3)
     .valueMap('id', 'fullName')
    """
    
    try:
        shared_results = cosmos_db._execute_query(shared_query)
        
        if shared_results:
            print(f"\nFound {len(shared_results)} other patient(s) sharing an address:\n")
            for i, shared_patient in enumerate(shared_results, 1):
                shared_id = shared_patient.get('id', ['N/A'])[0]
                shared_name = shared_patient.get('fullName', ['N/A'])[0]
                print(f"  {i}. {shared_name} (ID: {shared_id})")
        else:
            print("\nNo other patients share an address with this patient.")
    except Exception as e:
        print(f"\nCould not query shared addresses: {e}")
    
    # Close connection
    cosmos_db.close()
    
    print("\n" + "=" * 70)
    print("Visualization Complete!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
