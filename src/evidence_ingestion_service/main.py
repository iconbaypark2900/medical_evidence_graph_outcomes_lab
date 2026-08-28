"""
Main entry point for the Evidence Ingestion Service.

This is an orchestration layer over `src/data_ingestion.py`, which does
the actual retrieval from PubMed and ClinicalTrials.gov. It previously
did neither: `fetch_sources` returned the literal string
"Sample data from PubMed", `parse_normalize` attached empty entity lists
to it, and clinical trials were normalised with a hardcoded trial id of
"NCT00000000". The pipeline reported success and emitted an
`evidence.ingested` event describing nothing.

Output is shaped for `evidence_graph_service.extract_entities_relations`,
so ingestion feeds the graph directly.
"""
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

from src.data_ingestion import (
    MedicalEvidence,
    extract_entities_from_text,
    ingest_medical_evidence,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvidenceSource:
    """Represents an evidence source to be ingested"""
    name: str
    url: str
    source_type: str  # 'pubmed', 'clinical_trial', 'guideline', etc.
    last_updated: str


IngestFn = Callable[[List[str], int], Awaitable[List[MedicalEvidence]]]


class EvidenceIngestionService:
    def __init__(self, ingest_fn: Optional[IngestFn] = None):
        # Injectable so the pipeline can be exercised without network access.
        self._ingest = ingest_fn or ingest_medical_evidence

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

    def define_sources(self) -> List[EvidenceSource]:
        """The sources `data_ingestion` actually retrieves from."""
        return [
            EvidenceSource(
                name="PubMed",
                url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
                source_type="pubmed",
                last_updated=datetime.now(timezone.utc).isoformat(),
            ),
            EvidenceSource(
                name="ClinicalTrials.gov",
                url="https://clinicaltrials.gov/api/v2/",
                source_type="clinical_trial",
                last_updated=datetime.now(timezone.utc).isoformat(),
            ),
        ]

    async def fetch_sources(
        self, search_terms: List[str], max_per_source: int = 5
    ) -> List[MedicalEvidence]:
        """Retrieve real records for the given search terms.

        Failures propagate. `data_ingestion` raises rather than returning
        [] on a transport error precisely so a broken ingest cannot be
        mistaken for a search that matched nothing, and swallowing that
        here would undo it.
        """
        if not search_terms:
            raise ValueError("No search terms supplied; nothing to ingest")

        logger.info(f"Ingesting for {search_terms} (max {max_per_source} per source)")
        evidence = await self._ingest(search_terms, max_per_source)
        logger.info(f"Retrieved {len(evidence)} records")
        return evidence

    def enrich_entities(self, evidence: List[MedicalEvidence]) -> List[MedicalEvidence]:
        """Populate `.entities` on each record, in place.

        `ingest_medical_evidence` leaves entities empty; this fills them
        from the record's own title, abstract and MeSH headings. The
        storage layer reads `.entities` to build the graph, so without
        this step every indexed document arrives with no edges.
        """
        for item in evidence:
            item.entities = self._entities_for(item)
        return evidence

    AXES = ("conditions", "interventions", "outcomes", "populations")

    def _entities_for(self, item: MedicalEvidence) -> Dict[str, List[str]]:
        """Curated entities where they exist; keyword matching only as a
        last resort.

        PubMed and ClinicalTrials.gov both hand over indexed entities --
        MeSH descriptors, ChemicalList substances, subheading qualifiers,
        check tags, the registry's own condition and intervention fields.
        Those are assigned by human indexers and are strictly better than
        `extract_entities_from_text`, whose nine regexes put the words
        "treatment", "therapy" and "outcome" into the graph as clinical
        entities.

        The regex path survives only for a record that arrives with no
        curated metadata at all, and what it produces is marked as such
        so a reader can tell the two apart.
        """
        curated = item.entities or {}
        if any(curated.get(axis) for axis in self.AXES):
            return {axis: list(dict.fromkeys(curated.get(axis) or []))
                    for axis in self.AXES}

        text = " ".join(filter(None, [item.title, item.abstract]))
        if not text:
            return {axis: [] for axis in self.AXES}

        found = extract_entities_from_text(text)
        logger.info(
            f"{item.id} carried no indexed entities; fell back to keyword "
            f"matching, which is much coarser")
        return {
            "conditions": list(dict.fromkeys(
                (item.mesh_terms or []) + found.get("conditions", []))),
            "interventions": found.get("interventions", []),
            "outcomes": found.get("outcomes", []),
            "populations": found.get("populations", []),
        }

    def parse_normalize(self, evidence: List[MedicalEvidence]) -> List[Dict[str, Any]]:
        """Normalise records into the shape the graph service consumes.

        Entities come from the record's own text and MeSH headings. Where
        a field is genuinely absent — a trial with no summary, a paper
        with no MeSH terms — the corresponding list is empty because
        nothing was found, not because nothing was looked for.
        """
        normalized = []
        for item in evidence:
            entities = self._entities_for(item)

            record = {
                "id": item.id,
                "source": item.source,
                "type": "clinical_trial" if item.nct_id else "publication",
                "title": item.title,
                "timestamp": item.pub_date or "",
                "entities": entities,
                "metadata": {
                    "journal": item.journal,
                    "authors": item.authors,
                    "pmid": item.pmid,
                    "nct_id": item.nct_id,
                    "has_abstract": bool(item.abstract),
                    # Randomised trial, meta-analysis, case report: the
                    # strongest evidence-quality signal either source gives.
                    "publication_types": item.publication_types or [],
                },
            }
            normalized.append(record)

        return normalized

    def map_ontologies(self, normalized_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Attach ontology mappings that are actually available.

        MeSH descriptors come back with every PubMed record, so those are
        real. SNOMED-CT and ICD require a UMLS licence and a terminology
        service; there is none configured, so those keys are absent rather
        than present-and-empty. An empty list reads as "we looked and
        found no codes", which is a different claim from "we cannot look".
        """
        for item in normalized_data:
            mappings: Dict[str, Any] = {
                "mesh_terms": item["entities"]["conditions"],
                "unavailable": ["snomed_ct", "icd"],
                "unavailable_reason": (
                    "no UMLS terminology service is configured; these "
                    "vocabularies cannot be mapped without one"
                ),
            }
            item["ontology_mappings"] = mappings
        return normalized_data

    def store_metadata(self, processed_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarise what was ingested.

        In-memory only; there is no metadata store wired up yet, and this
        says so rather than printing "Storing..." as though there were.
        """
        with_abstract = sum(1 for i in processed_data if i["metadata"]["has_abstract"])
        by_source: Dict[str, int] = {}
        for item in processed_data:
            by_source[item["source"]] = by_source.get(item["source"], 0) + 1

        summary = {
            "total": len(processed_data),
            "by_source": by_source,
            "with_abstract": with_abstract,
            "persisted": False,
            "note": "no metadata store configured; this summary is in-memory only",
        }
        logger.info(f"Ingest summary: {summary}")
        return summary

    def emit_evidence_ingested(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Build the `evidence.ingested` event.

        `delivered: False` because no event bus is configured. The previous
        version returned `{"status": "completed"}` for an event that was
        never sent, describing data that was never fetched.
        """
        event = {
            "event": "evidence.ingested",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": summary,
            "delivered": False,
            "note": "no event bus configured; event constructed but not published",
        }
        logger.info(f"Constructed event: {event['event']} ({summary['total']} records)")
        return event

    async def run(
        self, search_terms: List[str], max_per_source: int = 5
    ) -> Dict[str, Any]:
        """The full pipeline: fetch, normalise, map, summarise, emit."""
        evidence = await self.fetch_sources(search_terms, max_per_source)
        normalized = self.parse_normalize(evidence)
        mapped = self.map_ontologies(normalized)
        summary = self.store_metadata(mapped)
        event = self.emit_evidence_ingested(summary)
        return {"records": mapped, "summary": summary, "event": event}


async def main():
    """Run the ingestion pipeline against the live APIs."""
    print("Starting Evidence Ingestion Service...")

    async with EvidenceIngestionService() as service:
        for source in service.define_sources():
            print(f"  source: {source.name} ({source.source_type})")

        result = await service.run(["metformin cardiovascular outcomes"], max_per_source=3)

        print(f"\nIngested {result['summary']['total']} records "
              f"{result['summary']['by_source']}")
        for record in result["records"]:
            print(f"\n  [{record['source']}] {record['title'][:80]}")
            print(f"    conditions:    {record['entities']['conditions'][:4]}")
            print(f"    interventions: {record['entities']['interventions'][:4]}")
            print(f"    outcomes:      {record['entities']['outcomes'][:4]}")

        print(f"\nEvent: {result['event']['event']} "
              f"(delivered={result['event']['delivered']})")


if __name__ == "__main__":
    asyncio.run(main())
