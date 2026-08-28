"""Evidence ingestion service.

The service used to return the literal string "Sample data from PubMed",
normalise it with empty entity lists and a hardcoded trial id of
"NCT00000000", and emit an `evidence.ingested` event describing nothing.
It is now an orchestration layer over `src/data_ingestion.py`, and the
retrieval function is injected so these tests need no network.
"""
from __future__ import annotations

import pytest

from src.data_ingestion import MedicalEvidence
from src.evidence_ingestion_service.main import EvidenceIngestionService


ARTICLE = MedicalEvidence(
    id="pubmed_31234567",
    title="Dapagliflozin in Patients with Heart Failure",
    abstract="CONCLUSIONS: mortality and cardiac outcomes improved with this therapy.",
    pub_date="2019",
    authors=["John J V McMurray"],
    journal="NEJM",
    source="PubMed",
    pmid="31234567",
    mesh_terms=["Heart Failure", "Sodium-Glucose Transporter 2 Inhibitors"],
)

TRIAL = MedicalEvidence(
    id="trial_NCT03036124",
    title="Metformin in Pre-Diabetes on Cardiovascular Outcomes",
    abstract="A trial of metformin therapy for diabetes and cardiac outcomes.",
    pub_date="2017",
    authors=[],
    journal="ClinicalTrials.gov",
    source="ClinicalTrials.gov",
    nct_id="NCT03036124",
    mesh_terms=["Diabetes Mellitus"],
)


def service_returning(*records) -> EvidenceIngestionService:
    async def fake_ingest(search_terms, max_per_source):
        return list(records)
    return EvidenceIngestionService(ingest_fn=fake_ingest)


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

def test_define_sources_lists_what_is_actually_retrieved():
    sources = EvidenceIngestionService().define_sources()

    assert {s.source_type for s in sources} == {"pubmed", "clinical_trial"}


async def test_fetch_sources_returns_real_records():
    service = service_returning(ARTICLE, TRIAL)

    evidence = await service.fetch_sources(["heart failure"], max_per_source=5)

    assert [e.id for e in evidence] == ["pubmed_31234567", "trial_NCT03036124"]


async def test_fetch_sources_requires_search_terms():
    with pytest.raises(ValueError, match="No search terms"):
        await service_returning().fetch_sources([])


async def test_a_retrieval_failure_is_not_swallowed():
    """data_ingestion raises rather than returning [] so a broken ingest
    cannot be mistaken for a search that matched nothing. Catching it here
    would undo that."""
    async def failing_ingest(search_terms, max_per_source):
        raise RuntimeError("PubMed search returned HTTP 429")

    service = EvidenceIngestionService(ingest_fn=failing_ingest)

    with pytest.raises(RuntimeError, match="429"):
        await service.fetch_sources(["heart failure"])


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------

def test_normalisation_carries_the_real_identifiers():
    """Regression guard: trials were normalised with a hardcoded
    "NCT00000000"."""
    records = service_returning().parse_normalize([ARTICLE, TRIAL])

    assert records[0]["metadata"]["pmid"] == "31234567"
    assert records[1]["metadata"]["nct_id"] == "NCT03036124"
    assert records[1]["id"] == "trial_NCT03036124"


def test_records_are_typed_by_what_they_are():
    records = service_returning().parse_normalize([ARTICLE, TRIAL])

    assert records[0]["type"] == "publication"
    assert records[1]["type"] == "clinical_trial"


def test_entities_are_extracted_from_the_record_text():
    """Previously every entity list was empty regardless of content."""
    records = service_returning().parse_normalize([ARTICLE])

    entities = records[0]["entities"]
    assert "Heart Failure" in entities["conditions"]  # from MeSH
    assert "therapy" in entities["interventions"]     # from the abstract
    assert "mortality" in entities["outcomes"]


def test_mesh_headings_are_preferred_and_deduplicated():
    records = service_returning().parse_normalize([TRIAL])

    conditions = records[0]["entities"]["conditions"]
    assert conditions[0] == "Diabetes Mellitus"  # curated heading comes first
    assert len(conditions) == len(set(conditions))


def test_a_record_with_no_text_yields_empty_entities():
    """Empty because nothing was found, not because nothing was looked for."""
    bare = MedicalEvidence(
        id="pubmed_1", title="", abstract="", pub_date="2020",
        authors=[], journal="J", source="PubMed", pmid="1",
    )

    records = service_returning().parse_normalize([bare])

    assert records[0]["entities"]["conditions"] == []
    assert records[0]["metadata"]["has_abstract"] is False


def test_normalised_records_match_the_graph_service_input_shape():
    """The two services compose: ingestion output feeds the graph builder."""
    from src.evidence_graph_service.main import EvidenceGraphService

    records = service_returning().parse_normalize([ARTICLE, TRIAL])
    elements = EvidenceGraphService().extract_entities_relations(records)

    assert elements["nodes"]
    assert elements["edges"]


# --------------------------------------------------------------------------
# Ontology mapping
# --------------------------------------------------------------------------

def test_available_ontologies_are_mapped_and_unavailable_ones_are_named():
    """"We cannot look" is a different claim from "we looked and found
    nothing", so unavailable vocabularies are absent rather than empty."""
    service = service_returning()
    records = service.map_ontologies(service.parse_normalize([ARTICLE]))

    mappings = records[0]["ontology_mappings"]
    assert "Heart Failure" in mappings["mesh_terms"]
    assert mappings["unavailable"] == ["snomed_ct", "icd"]
    assert "snomed_ct" not in mappings
    assert "UMLS" in mappings["unavailable_reason"]


# --------------------------------------------------------------------------
# Summary and event
# --------------------------------------------------------------------------

def test_the_summary_says_it_was_not_persisted():
    service = service_returning()
    records = service.parse_normalize([ARTICLE, TRIAL])

    summary = service.store_metadata(records)

    assert summary["total"] == 2
    assert summary["by_source"] == {"PubMed": 1, "ClinicalTrials.gov": 1}
    assert summary["with_abstract"] == 2
    assert summary["persisted"] is False


def test_the_event_says_it_was_not_delivered():
    """It used to report status "completed" for an event that was never
    sent, about data that was never fetched."""
    service = service_returning()
    summary = service.store_metadata(service.parse_normalize([ARTICLE]))

    event = service.emit_evidence_ingested(summary)

    assert event["event"] == "evidence.ingested"
    assert event["delivered"] is False
    assert event["payload"]["total"] == 1


async def test_the_full_pipeline_runs_end_to_end():
    async with service_returning(ARTICLE, TRIAL) as service:
        result = await service.run(["heart failure"], max_per_source=5)

    assert result["summary"]["total"] == 2
    assert len(result["records"]) == 2
    assert result["records"][0]["ontology_mappings"]["mesh_terms"]
    assert result["event"]["delivered"] is False
