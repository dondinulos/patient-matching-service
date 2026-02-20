# Patient Matching Service — Coding Instructions

> Domain-specific guidance for AI coding assistants and developers working on this codebase.

## Project Overview

This is a **Master Patient Index (MPI)** service for healthcare data interoperability. It finds and resolves duplicate patient records across healthcare systems using a combination of deterministic, probabilistic, and AI-enhanced matching algorithms.

### Key Interfaces

| Interface | Entry Point | Purpose |
|-----------|-------------|---------|
| **FastAPI REST** | `src/patient_matching/api.py` | HTTP endpoints for integrations |
| **Streamlit Dashboard** | `app/dashboard.py` | Web UI for match review |
| **Agent Framework** | `src/patient_matching/agent.py` | Conversational AI agent |
| **MCP Server** | `src/patient_matching/mcp_server.py` | Model Context Protocol server |
| **CLI Scripts** | `scripts/` | Batch operations and data loading |

All interfaces share the same core service layer — never duplicate business logic.

---

## Architecture & Layering

```
Interfaces (API, Agent, MCP, Dashboard, CLI)
        │
        ▼
  PatientMatchingService  (src/patient_matching/service.py)
        │
   ┌────┴────┐
   ▼         ▼
Matcher    GraphDB
(matching.py)  (graph_db.py / cosmos_graph_db.py)
```

### Rules

1. **Service layer is the single orchestration point.** All interfaces call `PatientMatchingService` methods — never reach into the DB or matcher directly from an interface.
2. **Database layer is swappable.** `graph_db.py` (Neo4j) and `cosmos_graph_db.py` (Cosmos DB Gremlin) implement the same interface. Code in the service layer should work with either backend.
3. **Models are shared.** All data classes live in `models.py`. Do NOT create duplicate Pydantic/dataclass models in interface modules; convert at the boundary.

---

## Matching Algorithms

### Three-Layer Scoring

| Layer | Weight | Components |
|-------|--------|------------|
| **Deterministic** | 0.4 | Enterprise ID (1.0), SSN (0.9), MRN (0.8), DOB (0.35), Phone (0.3), Email (0.3) |
| **Probabilistic** | 0.5 | Name similarity via Jaro-Winkler (0.35), Address token matching (0.15), Phonetic boost (Soundex/Metaphone) |
| **AI-Enhanced** | 0.1 | OpenAI text embeddings cosine similarity |

With LLM analysis enabled: `Final = Traditional × 0.8 + LLM × 0.2`

### Confidence Thresholds

| Score | Confidence | Action |
|-------|-----------|--------|
| >= 0.85 | `AUTO_MERGE` | Link to EMPI automatically |
| 0.65–0.85 | `HUMAN_REVIEW` | Queue for manual review |
| < 0.65 | `NO_MATCH` | Treat as separate patients |

These thresholds are configurable via `MatchWeights` dataclass.

### When Modifying Matching Logic

- All scoring lives in `matching.py` (classes: `DeterministicMatcher`, `ProbabilisticMatcher`, `PatientMatcher`).
- Weights are in the `MatchWeights` dataclass — add new fields there, not as magic constants.
- Always keep the composite score normalized to `[0.0, 1.0]`.
- Run `pytest tests/test_matching.py` after any changes to scoring logic.

---

## FHIR & Healthcare Domain

### Data Format

Patient data comes from **FHIR R4 Bundles** (Synthea-generated). Key mappings:

| FHIR Field | Internal Model |
|------------|---------------|
| `Patient.name` | `HumanName(family, given, prefix, suffix)` |
| `Patient.birthDate` | `Patient.birth_date: date` |
| `Patient.gender` | `Gender` enum (male/female/other/unknown) |
| `Patient.identifier` | `Identifier(value, type, system, assigner)` |
| `Patient.address` | `Address(line, city, state, postal_code, country)` |
| `Patient.telecom` | `ContactPoint(system, value, use)` |

FHIR parsing lives in `fhir_loader.py`. When adding new FHIR resource support, follow the existing `_parse_patient()` pattern.

### Identifier Normalization

- SSNs: strip dashes, uppercase → `123456789`
- MRNs: preserve system URI for cross-system matching
- Enterprise IDs: exact match only

### Privacy & Compliance (HIPAA)

- **Never log raw PHI** (SSN, full name + DOB) at INFO level. Use DEBUG only.
- Patient search returns IDs by default; full demographics require explicit tool call.
- The `reject_match` action must record `reviewed_by` and `reason` for audit trail.

---

## Graph Database Schema

### Vertices

| Label | Key Properties |
|-------|---------------|
| `Patient` | id, source_id, source_system, name, birth_date, gender |
| `Identifier` | value, type, system |
| `Address` | line, city, state, postal_code |
| `ContactPoint` | system, value |
| `EmpiRecord` | id, name, birth_date, gender, created_by |

### Edges

| Edge | From → To | Properties |
|------|-----------|-----------|
| `HAS_IDENTIFIER` | Patient → Identifier | — |
| `HAS_ADDRESS` | Patient → Address | — |
| `HAS_CONTACT` | Patient → ContactPoint | — |
| `POTENTIAL_MATCH` | Patient → Patient | score, confidence, timestamp |
| `LINKED_TO` | Patient → EmpiRecord | score, auto_linked, linked_at |

### Partition Strategy (Cosmos DB)

All vertices use `source_system` as the partition key. Keep clinical resources in the same partition as their patient for efficient traversals.

---

## Adding a New Tool / Capability

When adding a new operation to the service:

1. **Service layer** — Add the method to `PatientMatchingService` in `service.py`.
2. **Agent tool** — Add a function in `agent.py` and register it in `get_patient_matching_tools()`.
3. **MCP tool** — Add a `@mcp.tool()` function in `mcp_server.py`.
4. **REST endpoint** — Add a FastAPI route in `api.py`.
5. **Tests** — Add unit tests in `tests/test_matching.py` or integration tests in `tests/test_with_fhir_data.py`.

Keep the function signatures aligned across agent.py and mcp_server.py — both wrap the same service call and should accept the same parameters.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PM_DB_TYPE` | `cosmos` | Database backend: `cosmos` or `neo4j` |
| `COSMOS_GREMLIN_ENDPOINT` | — | Cosmos Gremlin endpoint |
| `COSMOS_DATABASE` | `PatientMatching` | Cosmos database name |
| `COSMOS_CONTAINER` | `PatientGraph` | Cosmos graph container |
| `COSMOS_KEY` | — | Cosmos primary key |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `password` | Neo4j password |
| `USE_EMBEDDINGS` | `false` | Enable OpenAI embedding matching |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | — | Azure OpenAI endpoint (for Agent) |
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-4o` | Azure OpenAI model deployment |

---

## Testing

```bash
# Unit tests (matching algorithms, models)
pytest tests/test_matching.py -v

# Integration tests (requires database connection)
pytest tests/test_with_fhir_data.py -v

# Full suite with coverage
pytest tests/ --cov=src/patient_matching --cov-report=html
```

When writing tests for matching logic, use the existing `Patient` factory helpers and assert on both score ranges and confidence classifications — not exact floating-point values.

---

## Deployment

Infrastructure is defined in `deploy/main.bicep`. Key resources:

- Azure Cosmos DB (Gremlin + NoSQL APIs)
- Azure OpenAI (GPT-4o + embeddings)
- Azure Container Apps
- Azure Container Registry
- Log Analytics Workspace

Deploy via `deploy/deploy.ps1` (Windows) or `deploy/deploy.sh` (Linux/macOS).
