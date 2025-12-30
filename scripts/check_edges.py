"""Check edge counts in Cosmos DB"""
import subprocess
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.patient_matching.cosmos_graph_db import CosmosGraphDB

def main():
    use_shell = platform.system() == 'Windows'
    rg = 'rg-patient-matching'
    
    r = subprocess.run(f'az cosmosdb list --resource-group {rg} --query "[0].name" -o tsv', 
                       capture_output=True, text=True, shell=use_shell)
    acct = r.stdout.strip()
    
    r2 = subprocess.run(f'az cosmosdb keys list --name {acct} --resource-group {rg} --query "primaryMasterKey" -o tsv', 
                        capture_output=True, text=True, shell=use_shell)
    key = r2.stdout.strip()
    
    db = CosmosGraphDB(
        endpoint=f'{acct}.gremlin.cosmos.azure.com', 
        database='patient-matching-db', 
        container='patients', 
        key=key
    )
    db.connect()
    
    print("Edge counts:")
    addr_count = db._execute_query("g.E().hasLabel('HAS_ADDRESS').count()")
    id_count = db._execute_query("g.E().hasLabel('HAS_IDENTIFIER').count()")
    contact_count = db._execute_query("g.E().hasLabel('HAS_CONTACT').count()")
    print(f"  HAS_ADDRESS: {addr_count}")
    print(f"  HAS_IDENTIFIER: {id_count}")
    print(f"  HAS_CONTACT: {contact_count}")
    
    print("\nSample address with patient:")
    sample = db._execute_query("g.V().hasLabel('Address').limit(1).as('a').in('HAS_ADDRESS').as('p').select('p','a').by(valueMap('fullName')).by(valueMap('city','state'))")
    print(f"  {sample}")
    
    print("\nPatient with addresses (correct query - limit BEFORE project):")
    import json
    query = "g.V().hasLabel('Patient').limit(3).project('patient', 'addresses').by(valueMap(true)).by(out('HAS_ADDRESS').valueMap(true).fold())"
    result = db._execute_query(query)
    print(json.dumps(result, indent=2, default=str))
    
    # Debug: Check edges directly
    print("\nDirect edge check for first patient:")
    first_patient = db._execute_query("g.V().hasLabel('Patient').limit(1).values('id')")
    print(f"  First patient ID: {first_patient}")
    
    edges = db._execute_query("g.V().hasLabel('Patient').limit(1).outE('HAS_ADDRESS').valueMap(true)")
    print(f"  Outgoing HAS_ADDRESS edges: {edges}")
    
    # Try with partition key
    print("\nTry traversal with inV():")
    result2 = db._execute_query("g.V().hasLabel('Patient').limit(1).outE('HAS_ADDRESS').inV().valueMap(true)")
    print(f"  Result: {result2}")
    
    # Check which patients have HAS_ADDRESS edges
    print("\nPatients with HAS_ADDRESS edges (via inV from Address):")
    linked_patients = db._execute_query("g.V().hasLabel('Address').limit(3).in('HAS_ADDRESS').values('id', 'fullName')")
    print(f"  {linked_patients}")
    
    # Check edge structure
    print("\nSample HAS_ADDRESS edge details:")
    edge_details = db._execute_query("g.E().hasLabel('HAS_ADDRESS').limit(1).project('outV','inV','label').by(outV().values('id')).by(inV().values('id')).by(label())")
    print(f"  {edge_details}")
    
    # Check for duplicate patients
    print("\nAaron697 Brekke496 patients (check for duplicates):")
    result = db._execute_query("g.V().hasLabel('Patient').has('fullName', 'Mr. Aaron697 Brekke496').valueMap('id', 'fullName', 'sourceId')")
    for r in result:
        print(f"  {r}")
    
    print("\nTotal patient count:", db._execute_query("g.V().hasLabel('Patient').count()"))
    
    db.close()

if __name__ == "__main__":
    main()
