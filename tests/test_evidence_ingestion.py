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
    async def fake_ingest(search_terms, max_per_source, since=None):
        fake_ingest.calls.append((list(search_terms), max_per_source, since))
        return list(records)
    fake_ingest.calls = []
    service = EvidenceIngestionService(ingest_fn=fake_ingest)
    service.ingest_calls = fake_ingest.calls
    return service


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
    async def failing_ingest(search_terms, max_per_source, since=None):
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


# --------------------------------------------------------------------------
# Curated entities take precedence over keyword matching
# --------------------------------------------------------------------------

CURATED = MedicalEvidence(
    id="pubmed_1", title="Dapagliflozin in heart failure",
    abstract="A trial of therapy and treatment for diabetes outcomes.",
    pub_date="2019", authors=[], journal="NEJM", source="PubMed", pmid="1",
    mesh_terms=["Heart Failure"],
    entities={
        "conditions": ["Heart Failure"],
        "interventions": ["dapagliflozin"],
        "outcomes": ["mortality"],
        "populations": ["Humans", "Aged"],
    },
    publication_types=["Randomized Controlled Trial"],
)


def test_curated_entities_are_used_verbatim():
    """The abstract contains "therapy", "treatment", "diabetes" and
    "outcomes" -- exactly the words the regex matcher would file as
    entities. With curated axes present it must not run at all."""
    entities = service_returning()._entities_for(CURATED)

    assert entities["interventions"] == ["dapagliflozin"]
    assert entities["outcomes"] == ["mortality"]
    assert entities["populations"] == ["Humans", "Aged"]
    assert "treatment" not in entities["interventions"]
    assert "therapy" not in entities["interventions"]
    assert "outcome" not in entities["outcomes"]


def test_keyword_matching_is_the_fallback_not_the_default():
    """A record with no indexed entities still yields something, coarse."""
    uncurated = MedicalEvidence(
        id="pubmed_2", title="A study of metformin",
        abstract="Metformin therapy in diabetes reduced mortality.",
        pub_date="2020", authors=[], journal="J", source="PubMed", pmid="2",
        entities={},
    )

    entities = service_returning()._entities_for(uncurated)

    assert "metformin" in entities["interventions"]
    assert "diabetes" in entities["conditions"]


def test_a_record_with_neither_curated_entities_nor_text_is_empty():
    bare = MedicalEvidence(
        id="pubmed_3", title="", abstract="", pub_date="2020",
        authors=[], journal="J", source="PubMed", pmid="3", entities={})

    entities = service_returning()._entities_for(bare)

    assert all(entities[axis] == [] for axis in
               ("conditions", "interventions", "outcomes", "populations"))


def test_study_design_reaches_the_normalised_record():
    records = service_returning().parse_normalize([CURATED])

    assert records[0]["metadata"]["publication_types"] == [
        "Randomized Controlled Trial"]


# --------------------------------------------------------------------------
# Incremental windows
# --------------------------------------------------------------------------

async def test_a_date_window_reaches_the_fetchers():
    """Without this, the only refresh available is a full re-fetch of every
    term ever searched -- which is slow enough that it does not get run."""
    from datetime import date

    service = service_returning(ARTICLE)
    await service.fetch_sources(["heart failure"], 5, since=date(2026, 8, 1))

    assert service.ingest_calls == [(["heart failure"], 5, date(2026, 8, 1))]


async def test_no_window_means_fetch_everything():
    service = service_returning(ARTICLE)
    await service.fetch_sources(["heart failure"], 5)

    assert service.ingest_calls[0][2] is None


# --------------------------------------------------------------------------
# MeSH descriptors reach the axis their tree names
# --------------------------------------------------------------------------

def mesh_service(*records):
    """A service whose classifier is seeded and offline."""
    import json
    import tempfile
    from pathlib import Path

    from src.mesh import MeshClassifier

    cache = Path(tempfile.mkdtemp()) / "mesh.json"
    cache.write_text(json.dumps({
        "heart failure": ["C14.280.434"],
        "treatment outcome": ["E01.789.800"],
        "kidney": ["A05.810.453"],
        "stroke volume": ["G09.330.553.700"],
        "aged": ["M01.060.116"],
        "metformin": ["D02.078.370.141.450"],
    }))

    async def fake_ingest(search_terms, max_per_source, since=None):
        return list(records)

    return EvidenceIngestionService(
        ingest_fn=fake_ingest,
        mesh_classifier=MeshClassifier(cache, offline=True))


PUBMED_RECORD = MedicalEvidence(
    id="pubmed_1", title="Dapagliflozin in heart failure",
    abstract="A trial.", pub_date="2019", authors=[], journal="NEJM",
    source="PubMed", pmid="1",
    mesh_terms=["Heart Failure", "Treatment Outcome", "Kidney",
                "Stroke Volume", "Aged"],
    entities={"conditions": ["Heart Failure", "Treatment Outcome", "Kidney",
                             "Stroke Volume"],
              "interventions": ["dapagliflozin"], "outcomes": ["mortality"],
              "populations": ["Aged"]},
)


def test_misfiled_descriptors_leave_the_condition_axis():
    """Treatment Outcome, Kidney and Stroke Volume were among the
    highest-degree Condition nodes in the corpus."""
    entities = mesh_service()._entities_for(PUBMED_RECORD)

    assert entities["conditions"] == ["Heart Failure"]
    for misfiled in ("Treatment Outcome", "Kidney", "Stroke Volume"):
        assert misfiled not in entities["conditions"]


def test_the_chemical_list_still_supplies_interventions():
    entities = mesh_service()._entities_for(PUBMED_RECORD)

    assert "dapagliflozin" in entities["interventions"]


def test_check_tags_still_supply_populations():
    entities = mesh_service()._entities_for(PUBMED_RECORD)

    assert "Aged" in entities["populations"]


def test_qualifier_outcomes_are_untouched():
    entities = mesh_service()._entities_for(PUBMED_RECORD)

    assert entities["outcomes"] == ["mortality"]


def test_registry_entities_are_not_run_through_a_mesh_lookup():
    """ClinicalTrials.gov supplies its own conditions and interventions,
    already typed by the registry. A MeSH lookup would fail to find
    "COVID-19" as indexed there and strip a correct entity out.
    """
    trial = MedicalEvidence(
        id="trial_NCT1", title="A trial", abstract="", pub_date="2021",
        authors=[], journal="ClinicalTrials.gov", source="ClinicalTrials.gov",
        nct_id="NCT1", mesh_terms=["COVID-19"],
        entities={"conditions": ["COVID-19"],
                  "interventions": ["Dapagliflozin 10 milligram (mg)"],
                  "outcomes": ["Time to recovery"], "populations": ["ADULT"]},
    )

    entities = mesh_service()._entities_for(trial)

    assert entities["conditions"] == ["COVID-19"]
    assert entities["interventions"] == ["Dapagliflozin 10 milligram (mg)"]


def test_an_unclassifiable_descriptor_is_not_forced_onto_an_axis():
    record = MedicalEvidence(
        id="pubmed_2", title="T", abstract="", pub_date="2020", authors=[],
        journal="J", source="PubMed", pmid="2",
        mesh_terms=["Some Descriptor Not In The Cache"],
        entities={"conditions": ["Some Descriptor Not In The Cache"],
                  "interventions": [], "outcomes": [], "populations": []},
    )

    entities = mesh_service()._entities_for(record)

    assert entities["conditions"] == []


def test_unclassified_descriptors_stay_searchable():
    """They remain in mesh_terms, so OpenSearch still finds them; they
    simply stop being entity nodes.

    Dropping them entirely would trade one loss for another: the point is
    that they are not conditions, not that they are worthless.
    """
    from src.integration import evidence_to_document

    service = mesh_service()
    record = MedicalEvidence(**{**vars(PUBMED_RECORD)})
    record.entities = service._entities_for(record)

    document = evidence_to_document(record)

    # Off the entity axis...
    assert "Treatment Outcome" not in document["conditions"]
    # ...but still indexed, and still findable.
    assert "Treatment Outcome" in document["mesh_terms"]
    assert "Kidney" in document["mesh_terms"]
