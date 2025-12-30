"""Test query binding replacement logic"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import re

# Test query similar to create_encounter
query = """
g.addV('Encounter')
    .property('id', id)
    .property('source_system', source_system)
    .property('patientId', patientId)
    .property('status', status)
    .property('encounterClass', encounterClass)
    .property('typeCode', typeCode)
    .property('typeDisplay', typeDisplay)
    .property('periodStart', periodStart)
    .property('periodEnd', periodEnd)
    .property('serviceProvider', serviceProvider)
    .property('reasonCode', reasonCode)
    .property('reasonDisplay', reasonDisplay)
    .property('createdAt', createdAt)
"""

bindings = {
    'id': 'test-enc-1',
    'source_system': 'FHIR',
    'patientId': 'patient-123',
    'status': 'finished',
    'encounterClass': 'ambulatory',
    'typeCode': '185349003',
    'typeDisplay': "Encounter for 'check-up' routine",  # Note single quote!
    'periodStart': '2024-01-01T10:00:00',
    'periodEnd': '2024-01-01T11:00:00',
    'serviceProvider': 'General Hospital',
    'reasonCode': '123456',
    'reasonDisplay': 'Routine visit',
    'createdAt': '2024-01-01T10:00:00'
}

# Simulate the binding replacement
final_query = query
escaped_values = {}
for key, value in bindings.items():
    if isinstance(value, str):
        escaped_value = value.replace("'", "''")
        escaped_values[key] = f"'{escaped_value}'"
    elif isinstance(value, bool):
        escaped_values[key] = str(value).lower()
    elif value is None:
        escaped_values[key] = "''"
    else:
        escaped_values[key] = str(value)

sorted_keys = sorted(escaped_values.keys(), key=len, reverse=True)

print("Sorted keys:", sorted_keys[:5])
print()

placeholders = {}
for i, key in enumerate(sorted_keys):
    placeholder = f'__PLACEHOLDER_{i:04d}__'
    placeholders[placeholder] = escaped_values[key]
    pattern = rf"(?<=[,\(\s]){re.escape(key)}(?=[,\)\s\n])"
    matches = list(re.finditer(pattern, final_query))
    if matches:
        print(f"Key '{key}' -> matches at positions: {[m.start() for m in matches]}")
    final_query = re.sub(pattern, placeholder, final_query)

for placeholder, value in placeholders.items():
    final_query = final_query.replace(placeholder, value)

print()
print("=" * 60)
print("Final Query:")
print("=" * 60)
print(final_query)
