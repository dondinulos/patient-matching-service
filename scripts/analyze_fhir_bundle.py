"""
Analyze FHIR bundle structure to understand resource types and relationships
"""
import json
from pathlib import Path
from collections import Counter

def analyze_bundle(fhir_file: Path):
    """Analyze a single FHIR bundle file."""
    with open(fhir_file, encoding='utf-8') as f:
        bundle = json.load(f)
    
    # Count resource types
    resource_types = Counter()
    resources_by_type = {}
    
    for entry in bundle.get('entry', []):
        resource = entry.get('resource', {})
        rt = resource.get('resourceType', 'Unknown')
        resource_types[rt] += 1
        
        if rt not in resources_by_type:
            resources_by_type[rt] = []
        resources_by_type[rt].append(resource)
    
    return resource_types, resources_by_type

def main():
    # Analyze a sample FHIR bundle
    fhir_file = Path(r'c:\Git\PatientMatching\data\fhir\Zula72_Ondricka197_907bd608-e768-4ce4-a5d8-4c7ba87bffa5.json')
    
    print(f"Analyzing: {fhir_file.name}")
    print("=" * 60)
    
    resource_types, resources_by_type = analyze_bundle(fhir_file)
    
    print("\nResource types in this FHIR bundle:")
    print("-" * 40)
    for rt, count in resource_types.most_common():
        print(f"  {rt}: {count}")
    
    total = sum(resource_types.values())
    print(f"\nTotal entries: {total}")
    
    # Show sample Encounter
    if 'Encounter' in resources_by_type:
        print("\n" + "=" * 60)
        print("Sample Encounter:")
        print("-" * 40)
        enc = resources_by_type['Encounter'][0]
        print(f"  ID: {enc.get('id')}")
        print(f"  Status: {enc.get('status')}")
        print(f"  Class: {enc.get('class', {}).get('code')}")
        print(f"  Type: {enc.get('type', [{}])[0].get('text', 'N/A')}")
        print(f"  Period: {enc.get('period', {})}")
        
        # Subject reference (patient)
        subject = enc.get('subject', {})
        print(f"  Subject (Patient): {subject.get('reference')}")
        
        # Service provider
        provider = enc.get('serviceProvider', {})
        print(f"  Service Provider: {provider.get('display', provider.get('reference', 'N/A'))}")
    
    # Show sample Observation
    if 'Observation' in resources_by_type:
        print("\n" + "=" * 60)
        print("Sample Observations (first 3):")
        print("-" * 40)
        for i, obs in enumerate(resources_by_type['Observation'][:3], 1):
            print(f"\n  Observation {i}:")
            print(f"    ID: {obs.get('id')}")
            print(f"    Status: {obs.get('status')}")
            
            # Category
            categories = obs.get('category', [])
            if categories:
                cat_text = categories[0].get('coding', [{}])[0].get('display', 'N/A')
                print(f"    Category: {cat_text}")
            
            # Code (what was observed)
            code = obs.get('code', {})
            code_text = code.get('text') or code.get('coding', [{}])[0].get('display', 'N/A')
            print(f"    Code: {code_text}")
            
            # Value
            if 'valueQuantity' in obs:
                vq = obs['valueQuantity']
                print(f"    Value: {vq.get('value')} {vq.get('unit')}")
            elif 'valueCodeableConcept' in obs:
                vc = obs['valueCodeableConcept']
                print(f"    Value: {vc.get('text', vc.get('coding', [{}])[0].get('display', 'N/A'))}")
            
            # Subject (patient reference)
            subject = obs.get('subject', {})
            print(f"    Subject: {subject.get('reference')}")
            
            # Encounter reference
            encounter = obs.get('encounter', {})
            print(f"    Encounter: {encounter.get('reference', 'N/A')}")
    
    # Show sample Condition
    if 'Condition' in resources_by_type:
        print("\n" + "=" * 60)
        print("Sample Conditions (first 3):")
        print("-" * 40)
        for i, cond in enumerate(resources_by_type['Condition'][:3], 1):
            print(f"\n  Condition {i}:")
            print(f"    ID: {cond.get('id')}")
            
            code = cond.get('code', {})
            code_text = code.get('text') or code.get('coding', [{}])[0].get('display', 'N/A')
            print(f"    Code: {code_text}")
            
            clinical_status = cond.get('clinicalStatus', {}).get('coding', [{}])[0].get('code', 'N/A')
            print(f"    Clinical Status: {clinical_status}")
            
            subject = cond.get('subject', {})
            print(f"    Subject: {subject.get('reference')}")

if __name__ == "__main__":
    main()
