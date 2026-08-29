"""MeSH descriptor classification by tree number.

Curated extraction routed every non-chemical MeSH descriptor onto the
condition axis. Measured against the indexed corpus, roughly half the
high-degree `Condition` nodes were not conditions: Primary Prevention,
Stroke Volume, Treatment Outcome, Drug Therapy Combination, Double-Blind
Method, Glomerular Filtration Rate, Kidney, Follow-Up Studies, Quality of
Life. Those are interventions, measurements, outcomes, study designs and
anatomy, and filing them as diseases put them where nothing could use
them and where the graph retriever would match the wrong axis.

MeSH already classifies every descriptor. Its tree numbers are a
hierarchy whose first letters name the category -- C is Diseases, D is
Chemicals and Drugs, E is Techniques, A is Anatomy, N is Health Care --
so the classification does not have to be guessed. NCBI serves them.

Lookups are cached on disk. A descriptor whose tree cannot be fetched is
recorded as unknown rather than defaulted onto an axis: defaulting is the
behaviour being fixed.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


logger = logging.getLogger(__name__)

DEFAULT_CACHE = Path("data/mesh_tree_cache.json")
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

# Axis each tree prefix belongs to. Longest prefix wins, so E02 (Therapeutics)
# is an intervention while E05 (Investigative Techniques) is not.
#
# Only four axes reach the graph as entity nodes; the rest are named so
# that "we know what this is and it is not a condition" is representable.
# Silence there is what produced the defect.
TREE_AXES = {
    "C": "condition",            # Diseases
    "F03": "condition",          # Mental Disorders
    "D": "intervention",         # Chemicals and Drugs
    "E02": "intervention",       # Therapeutics
    "E04": "intervention",       # Surgical Procedures
    "E01": "diagnostic",         # Diagnosis
    "E05": "study_method",       # Investigative Techniques
    "E06": "intervention",       # Dentistry
    "E07": "intervention",       # Equipment and Supplies
    "N": "health_care",          # Health Care
    "A": "anatomy",
    "B": "organism",
    "G": "process",              # Phenomena and Processes
    "M": "population",           # Named Groups
    "V": "publication_type",
    "Z": "geographic",
    "F01": "behaviour",
    "F02": "process",
    "F04": "study_method",
    "H": "discipline",
    "I": "social",
    "J": "technology",
    "K": "humanities",
    "L": "information_science",
}

# Which axes become entity nodes, and under which label.
GRAPH_LABELS = {
    "condition": "Condition",
    "intervention": "Intervention",
    "population": "Population",
}

# Everything else stays on the Evidence node's mesh_terms, so it remains
# searchable in OpenSearch without claiming to be a clinical entity.
UNCLASSIFIED = "unknown"


@dataclass
class MeshClassifier:
    """Maps a descriptor name to an axis, cached on disk.

    `offline` refuses network lookups. An ingest that cannot reach NCBI
    still runs; unknown descriptors are marked unknown and kept off the
    entity axes rather than guessed onto one.
    """
    cache_path: Path = DEFAULT_CACHE
    offline: bool = False
    min_interval: float = 0.35

    def __post_init__(self) -> None:
        self._cache: Dict[str, List[str]] = {}
        self._last_request = 0.0
        self.lookups = 0
        self.cache_hits = 0
        self._load_cache()

    def _load_cache(self) -> None:
        path = Path(self.cache_path)
        if not path.exists():
            return
        try:
            self._cache = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Could not read MeSH cache {path}: {e}; starting empty")

    def save_cache(self) -> None:
        path = Path(self.cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._cache, indent=1, sort_keys=True) + "\n")

    def _fetch_tree_numbers(self, descriptor: str) -> List[str]:
        """Ask NCBI for a descriptor's tree numbers."""
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request = time.monotonic()

        term = urllib.parse.quote(f"{descriptor}[MeSH Terms]")
        try:
            with urllib.request.urlopen(
                    f"{EUTILS}esearch.fcgi?db=mesh&term={term}&retmode=json",
                    timeout=20) as response:
                uids = json.loads(response.read())["esearchresult"]["idlist"]
            if not uids:
                return []

            time.sleep(self.min_interval)
            with urllib.request.urlopen(
                    f"{EUTILS}efetch.fcgi?db=mesh&id={uids[0]}"
                    f"&rettype=full&retmode=text", timeout=20) as response:
                text = response.read().decode("utf-8", "replace")
        except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
            logger.warning(f"MeSH lookup failed for {descriptor!r}: {e}")
            return []

        for line in text.splitlines():
            if line.strip().startswith("Tree Number"):
                return [part.strip() for part in
                        line.split(":", 1)[1].split(",") if part.strip()]
        return []

    def tree_numbers(self, descriptor: str) -> List[str]:
        key = descriptor.strip().lower()
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        if self.offline:
            return []

        self.lookups += 1
        numbers = self._fetch_tree_numbers(descriptor)
        self._cache[key] = numbers
        return numbers

    @staticmethod
    def axis_for_tree(tree_number: str) -> Optional[str]:
        """Longest matching prefix wins."""
        for length in (3, 1):
            axis = TREE_AXES.get(tree_number[:length])
            if axis:
                return axis
        return None

    def classify(self, descriptor: str) -> str:
        """The axis a descriptor belongs on, or 'unknown'.

        A descriptor in several trees takes the first axis that maps,
        preferring disease over the technique and health-care trees a
        clinical concept is often cross-listed in.
        """
        numbers = self.tree_numbers(descriptor)
        axes = [self.axis_for_tree(n) for n in numbers]
        axes = [a for a in axes if a]
        if not axes:
            return UNCLASSIFIED

        for preferred in ("condition", "intervention", "population"):
            if preferred in axes:
                return preferred
        return axes[0]

    def graph_label(self, descriptor: str) -> Optional[str]:
        """The node label for a descriptor, or None to keep it off the graph."""
        return GRAPH_LABELS.get(self.classify(descriptor))

    def split_descriptors(self, descriptors: List[str]) -> Dict[str, List[str]]:
        """Route descriptors onto axes.

        Anything not on an entity axis lands in `other`, which callers keep
        on the evidence record so it stays searchable without claiming to
        be a clinical entity.
        """
        routed: Dict[str, List[str]] = {
            "conditions": [], "interventions": [], "populations": [], "other": []}
        for descriptor in descriptors:
            axis = self.classify(descriptor)
            if axis == "condition":
                routed["conditions"].append(descriptor)
            elif axis == "intervention":
                routed["interventions"].append(descriptor)
            elif axis == "population":
                routed["populations"].append(descriptor)
            else:
                routed["other"].append(descriptor)
        return routed


async def warm_cache_from_neo4j(cache_path: Path = DEFAULT_CACHE) -> Dict[str, int]:
    """Look up every descriptor in the indexed graph, once.

    Run after ingesting new topics. The cache ships with the repository so
    ingestion classifies offline; without a cached entry a descriptor is
    recorded as unknown rather than guessed onto an axis.
    """
    import neo4j

    from src.integration import load_database_config

    config = load_database_config()["neo4j"]
    driver = neo4j.AsyncGraphDatabase.driver(
        config["uri"], auth=(config["username"], config["password"]))
    try:
        async with driver.session() as session:
            result = await session.run(
                "MATCH (n) WHERE n.name IS NOT NULL RETURN DISTINCT n.name AS name")
            names = [row["name"] for row in await result.data()]
    finally:
        await driver.close()

    classifier = MeshClassifier(cache_path)
    counts: Dict[str, int] = {}
    for name in names:
        axis = classifier.classify(name)
        counts[axis] = counts.get(axis, 0) + 1
    classifier.save_cache()

    logger.info(
        f"Classified {len(names)} descriptors "
        f"({classifier.lookups} fetched, {classifier.cache_hits} cached)")
    return counts


async def main():
    import asyncio  # noqa: F401

    logging.basicConfig(level=logging.INFO)
    counts = await warm_cache_from_neo4j()
    for axis, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {axis:20} {n}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
