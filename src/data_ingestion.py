"""
Data ingestion module to fetch real medical evidence from public APIs
"""
import asyncio
import aiohttp
import time
import urllib.parse
import xml.etree.ElementTree as ET
from typing import List, Dict, Any
import json
import logging
from datetime import datetime
from dataclasses import dataclass


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# NCBI allows roughly 3 requests per second without an API key. Firing
# requests back to back earns a 429, which -- now that a 429 raises rather
# than returning [] -- fails the whole ingest rather than silently
# under-populating it. Pace the requests instead.
PUBMED_MIN_INTERVAL = 0.36
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


class RateLimiter:
    """Enforces a minimum interval between requests."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def wait(self) -> None:
        async with self._lock:
            elapsed = time.monotonic() - self._last
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self._last = time.monotonic()


class RetryableStatus(Exception):
    """Internal signal: the server asked us to come back later."""

    def __init__(self, status: int):
        self.status = status
        super().__init__(f"HTTP {status}")


@dataclass
class MedicalEvidence:
    """Represents a piece of medical evidence"""
    id: str
    title: str
    abstract: str
    pub_date: str
    authors: List[str]
    journal: str
    source: str
    pmid: str = None
    nct_id: str = None
    mesh_terms: List[str] = None
    entities: Dict[str, List[str]] = None


class PubMedFetcher:
    """Fetches medical evidence from PubMed API"""
    
    def __init__(self, max_retries: int = 3, retry_backoff: float = 0.5,
                 min_interval: float = PUBMED_MIN_INTERVAL):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.session = None
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.rate_limiter = RateLimiter(min_interval)

    async def _get(self, url: str, read):
        """Fetch `url`, retrying transient statuses with backoff.

        `read` turns the response into a value. A 429 or 5xx is retried
        because it means "not now", not "nothing found"; every other
        non-200 raises immediately.
        """
        last_status = None
        for attempt in range(self.max_retries + 1):
            await self.rate_limiter.wait()
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await read(response)
                last_status = response.status
                if response.status not in RETRYABLE_STATUSES:
                    break
            if attempt < self.max_retries:
                delay = self.retry_backoff * (2 ** attempt)
                logger.warning(
                    f"HTTP {last_status} from {url[:60]}...; retrying in {delay:.1f}s "
                    f"({attempt + 1}/{self.max_retries})")
                await asyncio.sleep(delay)
        raise RetryableStatus(last_status)
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _build_search_url(self, query: str, retmax: int = 10) -> str:
        """Build PubMed search URL.

        Parameters are percent-encoded. A raw query with spaces or an
        ampersand in it silently changes the search being run.
        """
        params = urllib.parse.urlencode({
            "db": "pubmed",
            "term": query,
            "retmax": retmax,
            "retmode": "json",
            "sort": "relevance",
        })
        return f"{self.base_url}esearch.fcgi?{params}"
    
    def _build_fetch_url(self, id_list: List[str]) -> str:
        """Build PubMed fetch URL"""
        params = urllib.parse.urlencode({
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "xml",
        })
        return f"{self.base_url}efetch.fcgi?{params}"
    
    async def search_pubmed(self, query: str, max_results: int = 10) -> List[str]:
        """Search PubMed and return PMIDs.

        Raises RuntimeError on any transport or API failure. An empty list is
        reserved for the one case where it is true: the query matched nothing.
        """
        try:
            data = await self._get(
                self._build_search_url(query, max_results),
                lambda response: response.json())
            id_list = data.get("esearchresult", {}).get("idlist", [])
            logger.info(f"Found {len(id_list)} results for query: {query}")
            return id_list
        except RetryableStatus as e:
            # A 429 or 5xx used to return [] here, which is
            # indistinguishable from an exhaustive search that found
            # nothing -- and silently under-populates the evidence graph
            # everything downstream reasons over.
            raise RuntimeError(
                f"PubMed search for {query!r} returned HTTP {e.status} "
                f"after {self.max_retries} retries") from e
        except RuntimeError:
            raise
        except Exception as e:
            # An empty list means "this query matched no articles", which is a
            # real and common answer. Returning it on a network or API failure
            # makes a broken ingest indistinguishable from an exhaustive search
            # that found nothing.
            logger.error(f"Error searching PubMed: {e}")
            raise RuntimeError(f"PubMed search failed for {query!r}: {e}") from e

    @staticmethod
    def _text_of(element) -> str:
        """Flatten an element's text, including inline markup.

        PubMed titles and abstracts carry <i>, <sup>, <b> and similar, so
        reading `.text` alone truncates at the first tag.
        """
        if element is None:
            return ""
        return " ".join("".join(element.itertext()).split())

    @classmethod
    def _parse_authors(cls, article) -> List[str]:
        """Author names from an <AuthorList>.

        The given name lives in <ForeName>. An earlier version looked for
        <FirstName>, which does not exist in the NLM DTD, so the guard never
        passed and every ingested article had an empty author list.
        """
        authors = []
        for author in article.findall(".//AuthorList/Author"):
            collective = cls._text_of(author.find("CollectiveName"))
            if collective:
                authors.append(collective)
                continue

            last_name = cls._text_of(author.find("LastName"))
            if not last_name:
                continue

            # Fall back to initials when the given name is not supplied.
            given = cls._text_of(author.find("ForeName")) or cls._text_of(
                author.find("Initials"))
            authors.append(f"{given} {last_name}".strip())
        return authors

    @classmethod
    def _parse_abstract(cls, article) -> str:
        """Join every section of a structured abstract.

        Reading only the first <AbstractText> dropped everything after
        BACKGROUND -- including CONCLUSIONS, which is where the clinical
        finding actually lives.
        """
        sections = []
        for section in article.findall(".//Abstract/AbstractText"):
            text = cls._text_of(section)
            if not text:
                continue
            label = section.get("Label")
            sections.append(f"{label}: {text}" if label else text)
        return " ".join(sections)

    @classmethod
    def _parse_pub_date(cls, article) -> str:
        """Publication year, falling back to the free-text MedlineDate."""
        year = cls._text_of(article.find(".//Journal//PubDate/Year"))
        if year:
            return year
        return cls._text_of(article.find(".//Journal//PubDate/MedlineDate"))

    async def fetch_pubmed_articles(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """Fetch full article details from PubMed."""
        if not pmids:
            return []

        try:
            content = await self._get(
                self._build_fetch_url(pmids),
                lambda response: response.text())
            root = ET.fromstring(content)


            articles = []
            for article in root.findall(".//PubmedArticle"):
                # Scope the PMID lookup to the citation: a bare .//PMID
                # can also match ids inside reference and comment lists.
                pmid = self._text_of(article.find("./MedlineCitation/PMID"))

                articles.append({
                    "pmid": pmid,
                    "title": self._text_of(article.find(".//ArticleTitle")),
                    "abstract": self._parse_abstract(article),
                    "journal": self._text_of(article.find(".//Journal/Title")),
                    "pub_date": self._parse_pub_date(article),
                    "authors": self._parse_authors(article),
                    "mesh_terms": [
                        self._text_of(term)
                        for term in article.findall(".//MeshHeading/DescriptorName")
                    ],
                    "source": "PubMed",
                })

            logger.info(f"Fetched {len(articles)} articles from PubMed")
            return articles
        except RetryableStatus as e:
            raise RuntimeError(
                f"PubMed fetch for {len(pmids)} pmids returned HTTP {e.status} "
                f"after {self.max_retries} retries") from e
        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"Error fetching PubMed articles: {e}")
            raise RuntimeError(
                f"PubMed fetch failed for {len(pmids)} pmids: {e}") from e


class ClinicalTrialsFetcher(PubMedFetcher):
    """Fetches clinical trial data from ClinicalTrials.gov API.

    Inherits the retry and pacing behaviour; only the endpoints differ.
    """
    
    def __init__(self, max_retries: int = 3, retry_backoff: float = 0.5,
                 min_interval: float = 0.2):
        super().__init__(max_retries, retry_backoff, min_interval)
        self.base_url = "https://clinicaltrials.gov/api/"
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _build_search_url(self, query: str, max_results: int = 10) -> str:
        """Build ClinicalTrials.gov v2 search URL.

        Two bugs lived here. The page size parameter is `pageSize`, not
        `page.size`, and the query was interpolated unencoded. The API
        answered 400 to every request as a result, and `search_trials`
        turned that into an empty list -- so trial ingestion returned
        nothing, indistinguishable from a search that matched nothing,
        for as long as this code has existed.
        """
        params = urllib.parse.urlencode({
            "query.term": query,
            "pageSize": max_results,
            "format": "json",
        })
        return f"{self.base_url}v2/studies?{params}"
    
    async def search_trials(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search ClinicalTrials.gov and return study details.

        As with PubMed, an empty list means the query matched no studies.
        Any transport or API failure raises.
        """
        try:
            data = await self._get(
                self._build_search_url(query, max_results),
                lambda response: response.json())
            studies = data.get("studies", [])

            logger.info(f"Found {len(studies)} clinical trials for query: {query}")
            return studies
        except RetryableStatus as e:
            raise RuntimeError(
                f"ClinicalTrials.gov search for {query!r} returned HTTP "
                f"{e.status} after {self.max_retries} retries") from e
        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"Error searching ClinicalTrials.gov: {e}")
            raise RuntimeError(f"ClinicalTrials.gov search failed: {e}") from e


async def ingest_medical_evidence(search_terms: List[str], max_per_source: int = 5) -> List[MedicalEvidence]:
    """Ingest medical evidence from multiple sources"""
    logger.info(f"Starting ingestion for search terms: {search_terms}")
    
    all_evidence = []
    
    # Fetch from PubMed
    async with PubMedFetcher() as pubmed_fetcher:
        for term in search_terms:
            logger.info(f"Searching PubMed for: {term}")
            pmids = await pubmed_fetcher.search_pubmed(term, max_per_source)
            if pmids:
                articles = await pubmed_fetcher.fetch_pubmed_articles(pmids[:max_per_source])
                
                for article in articles:
                    evidence = MedicalEvidence(
                        id=f"pubmed_{article['pmid']}",
                        title=article['title'],
                        abstract=article['abstract'],
                        pub_date=article['pub_date'],
                        authors=article['authors'],
                        journal=article['journal'],
                        source=article['source'],
                        pmid=article['pmid'],
                        mesh_terms=article['mesh_terms'],
                        entities={}  # Will be populated later
                    )
                    all_evidence.append(evidence)
    
    # Fetch from ClinicalTrials.gov
    async with ClinicalTrialsFetcher() as trials_fetcher:
        for term in search_terms:
            logger.info(f"Searching ClinicalTrials.gov for: {term}")
            trials = await trials_fetcher.search_trials(term, max_per_source)
            
            for trial in trials:
                study = trial.get("protocolSection", {})
                identification = study.get("identificationModule", {})
                description = study.get("descriptionModule", {})
                status = study.get("statusModule", {})
                
                # Extract available information
                nct_id = identification.get("nctId", "")
                title = identification.get("briefTitle", "")
                description_text = description.get("briefSummary", "")
                condition = description.get("conditions", [])
                
                # Format the trial data similar to PubMed articles
                evidence = MedicalEvidence(
                    id=f"trial_{nct_id}",
                    title=title,
                    abstract=description_text,
                    pub_date=status.get("startDateStruct", {}).get("date", "")[:4],  # Just year
                    authors=[],  # Trials don't typically have personal authors like papers
                    journal="ClinicalTrials.gov",  # Placeholder
                    source="ClinicalTrials.gov",
                    nct_id=nct_id,
                    mesh_terms=condition,
                    entities={"conditions": condition} if condition else {}
                )
                all_evidence.append(evidence)
    
    logger.info(f"Ingested {len(all_evidence)} pieces of medical evidence")
    return all_evidence


def extract_entities_from_text(text: str) -> Dict[str, List[str]]:
    """
    Simple entity extraction from text (in real implementation, use NER models)
    This is a simplified version for demonstration purposes
    """
    # This would be replaced with proper NLP/NER in a real implementation
    # For now, we'll just look for some common medical terms
    import re
    
    # Common medical terms for demonstration purposes
    # In a real implementation, use BioBERT, Med7, or similar
    conditions = []
    interventions = []
    outcomes = []
    
    text_lower = text.lower()
    
    # Look for some common condition patterns
    condition_patterns = [
        r'diabetes', r'cancer', r'heart', r'cardiac', r'asthma', 
        r'hypertension', r'depression', r'anxiety', r'arthritis'
    ]
    
    intervention_patterns = [
        r'metformin', r'insulin', r'aspirin', r'statin', r'therapy', 
        r'treatment', r'drug', r'medication', r'surgery', r'procedure'
    ]
    
    outcome_patterns = [
        r'mortality', r'survival', r'remission', r'response', r'improvement',
        r'complication', r'adverse', r'side effect', r'outcome'
    ]
    
    for pattern in condition_patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            conditions.extend(list(set(matches)))
    
    for pattern in intervention_patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            interventions.extend(list(set(matches)))
    
    for pattern in outcome_patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            outcomes.extend(list(set(matches)))
    
    return {
        "conditions": list(set(conditions)),
        "interventions": list(set(interventions)),
        "outcomes": list(set(outcomes))
    }


async def main():
    """Main function to test evidence ingestion"""
    logger.info("Starting medical evidence ingestion...")
    
    # Define search terms for testing
    search_terms = [
        "diabetes treatment",
        "cancer therapy", 
        "cardiovascular outcomes",
        "COVID-19 treatment"
    ]
    
    # Ingest medical evidence
    evidence = await ingest_medical_evidence(search_terms, max_per_source=3)
    
    logger.info(f"Successfully ingested {len(evidence)} pieces of evidence")
    
    # Show first few results
    for i, ev in enumerate(evidence[:5]):
        print(f"\n{i+1}. {ev.title[:100]}...")
        print(f"   Source: {ev.source}")
        print(f"   ID: {ev.id}")
        print(f"   Date: {ev.pub_date}")
        print(f"   Abstract: {ev.abstract[:100]}...")
    
    logger.info("Evidence ingestion completed successfully!")
    
    return evidence


if __name__ == "__main__":
    asyncio.run(main())