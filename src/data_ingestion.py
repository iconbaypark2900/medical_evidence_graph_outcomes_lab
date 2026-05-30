"""
Data ingestion module to fetch real medical evidence from public APIs
"""
import asyncio
import aiohttp
import xml.etree.ElementTree as ET
from typing import List, Dict, Any
import json
import logging
from datetime import datetime
from dataclasses import dataclass


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    
    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _build_search_url(self, query: str, retmax: int = 10) -> str:
        """Build PubMed search URL"""
        return (f"{self.base_url}esearch.fcgi?"
                f"db=pubmed&term={query}&retmax={retmax}&retmode=json&sort=relevance")
    
    def _build_fetch_url(self, id_list: List[str]) -> str:
        """Build PubMed fetch URL"""
        ids = ",".join(id_list)
        return (f"{self.base_url}efetch.fcgi?"
                f"db=pubmed&id={ids}&retmode=xml")
    
    async def search_pubmed(self, query: str, max_results: int = 10) -> List[str]:
        """Search PubMed and return PMIDs"""
        try:
            search_url = self._build_search_url(query, max_results)
            async with self.session.get(search_url) as response:
                if response.status == 200:
                    data = await response.json()
                    id_list = data.get("esearchresult", {}).get("idlist", [])
                    logger.info(f"Found {len(id_list)} results for query: {query}")
                    return id_list
                else:
                    logger.error(f"PubMed search failed with status {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error searching PubMed: {e}")
            return []
    
    async def fetch_pubmed_articles(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """Fetch full article details from PubMed"""
        if not pmids:
            return []
        
        try:
            fetch_url = self._build_fetch_url(pmids)
            async with self.session.get(fetch_url) as response:
                if response.status == 200:
                    content = await response.text()
                    root = ET.fromstring(content)
                    
                    articles = []
                    for article in root.findall(".//PubmedArticle"):
                        pmid_elem = article.find(".//PMID")
                        title_elem = article.find(".//ArticleTitle")
                        abstract_elem = article.find(".//AbstractText")
                        journal_elem = article.find(".//Journal/Title")
                        pub_date_elem = article.find(".//PubDate/Year")
                        
                        # Extract authors
                        authors = []
                        for author in article.findall(".//Author"):
                            last_name = author.find("LastName")
                            first_name = author.find("FirstName")
                            if last_name is not None and first_name is not None:
                                authors.append(f"{first_name.text} {last_name.text}")
                        
                        # Extract MeSH terms
                        mesh_terms = []
                        for mesh_term in article.findall(".//MeshHeading/DescriptorName"):
                            mesh_terms.append(mesh_term.text)
                        
                        article_data = {
                            "pmid": pmid_elem.text if pmid_elem is not None else "",
                            "title": title_elem.text if title_elem is not None else "",
                            "abstract": abstract_elem.text if abstract_elem is not None else "",
                            "journal": journal_elem.text if journal_elem is not None else "",
                            "pub_date": pub_date_elem.text if pub_date_elem is not None else "",
                            "authors": authors,
                            "mesh_terms": mesh_terms,
                            "source": "PubMed"
                        }
                        
                        articles.append(article_data)
                    
                    logger.info(f"Fetched {len(articles)} articles from PubMed")
                    return articles
                else:
                    logger.error(f"PubMed fetch failed with status {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching PubMed articles: {e}")
            return []


class ClinicalTrialsFetcher:
    """Fetches clinical trial data from ClinicalTrials.gov API"""
    
    def __init__(self):
        self.base_url = "https://clinicaltrials.gov/api/"
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _build_search_url(self, query: str, max_results: int = 10) -> str:
        """Build ClinicalTrials.gov search URL"""
        return (f"{self.base_url}v2/studies?query.term={query}"
                f"&page.size={max_results}&format=json")
    
    async def search_trials(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search ClinicalTrials.gov and return study details"""
        try:
            search_url = self._build_search_url(query, max_results)
            async with self.session.get(search_url) as response:
                if response.status == 200:
                    data = await response.json()
                    studies = data.get("studies", [])
                    
                    logger.info(f"Found {len(studies)} clinical trials for query: {query}")
                    return studies
                else:
                    logger.error(f"ClinicalTrials search failed with status {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error searching ClinicalTrials.gov: {e}")
            return []


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