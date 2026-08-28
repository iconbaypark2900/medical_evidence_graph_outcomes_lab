"""PubMed / ClinicalTrials.gov ingestion.

No network access: the fetchers are driven with a fake aiohttp session so
the XML and JSON parsing can be checked against the shapes the real APIs
return. `tests/conftest.py::PUBMED_EFETCH_XML` uses the real NLM tag names.
"""
from __future__ import annotations

import aiohttp
import pytest

from src.data_ingestion import ClinicalTrialsFetcher, MedicalEvidence, PubMedFetcher
from tests.conftest import PUBMED_EFETCH_XML


# --------------------------------------------------------------------------
# URL construction
# --------------------------------------------------------------------------

def test_pubmed_search_url_targets_esearch():
    fetcher = PubMedFetcher()

    url = fetcher._build_search_url("heart failure", retmax=25)

    assert url.startswith("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?")
    assert "db=pubmed" in url
    assert "term=heart+failure" in url
    assert "retmax=25" in url


def test_pubmed_search_url_encodes_the_query():
    """An unencoded & or = silently changes the search being run."""
    url = PubMedFetcher()._build_search_url("SGLT2 & heart failure", retmax=5)

    assert "term=SGLT2+%26+heart+failure" in url
    # Exactly one retmax parameter -- the query must not inject its own.
    assert url.count("retmax=") == 1


def test_pubmed_fetch_url_joins_ids_with_commas():
    fetcher = PubMedFetcher()

    url = fetcher._build_fetch_url(["1", "2", "3"])

    # Percent-encoded comma; verified against the live efetch endpoint.
    assert "id=1%2C2%2C3" in url
    assert "retmode=xml" in url


def test_clinical_trials_search_url_uses_v2_api():
    """Regression guard for a URL the API answered 400 to every time.

    The parameter is `pageSize`, not `page.size`, and the query was
    interpolated unencoded. Because search_trials returned [] on a non-2xx
    response, trial ingestion silently produced nothing.
    """
    fetcher = ClinicalTrialsFetcher()

    url = fetcher._build_search_url("atrial fibrillation", max_results=5)

    assert url.startswith("https://clinicaltrials.gov/api/v2/studies?")
    assert "query.term=atrial+fibrillation" in url
    assert "pageSize=5" in url
    assert "page.size" not in url


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------

async def test_search_pubmed_extracts_the_pmid_list(fake_session, fake_response):
    fetcher = PubMedFetcher()
    fetcher.session = fake_session(
        fake_response(200, payload={"esearchresult": {"idlist": ["31234567", "98765432"]}})
    )

    pmids = await fetcher.search_pubmed("heart failure", max_results=2)

    assert pmids == ["31234567", "98765432"]


async def test_search_pubmed_raises_rather_than_returning_an_empty_list(
    fake_session, fake_response
):
    """Regression guard for commit 54e4b8f.

    [] means "this query matched nothing", a real answer. A transport
    failure must not be reported as one.
    """
    fetcher = PubMedFetcher()
    fetcher.session = fake_session(raises=aiohttp.ClientError("connection reset"))

    with pytest.raises(RuntimeError, match="PubMed search failed"):
        await fetcher.search_pubmed("heart failure")


async def test_search_trials_extracts_the_studies_list(fake_session, fake_response):
    fetcher = ClinicalTrialsFetcher()
    fetcher.session = fake_session(
        fake_response(200, payload={"studies": [{"protocolSection": {}}, {"protocolSection": {}}]})
    )

    studies = await fetcher.search_trials("atrial fibrillation")

    assert len(studies) == 2


# --------------------------------------------------------------------------
# PubMed EFetch XML parsing
# --------------------------------------------------------------------------

@pytest.fixture
async def parsed_article(fake_session, fake_response):
    fetcher = PubMedFetcher()
    fetcher.session = fake_session(fake_response(200, text=PUBMED_EFETCH_XML))
    articles = await fetcher.fetch_pubmed_articles(["31234567"])
    assert len(articles) == 1
    return articles[0]


async def test_fetch_parses_identifiers_and_provenance(parsed_article):
    assert parsed_article["pmid"] == "31234567"
    assert parsed_article["source"] == "PubMed"
    assert parsed_article["pub_date"] == "2019"


async def test_fetch_parses_title_and_journal(parsed_article):
    assert parsed_article["title"].startswith("Dapagliflozin in Patients with Heart Failure")
    assert parsed_article["journal"] == "The New England Journal of Medicine"


async def test_fetch_parses_mesh_terms(parsed_article):
    """MeSH terms are the ontology hook the evidence graph is built on."""
    assert parsed_article["mesh_terms"] == [
        "Heart Failure",
        "Sodium-Glucose Transporter 2 Inhibitors",
    ]


async def test_fetch_short_circuits_on_an_empty_pmid_list(fake_session):
    """No ids is a legitimate no-op, not a failure -- no request is made."""
    fetcher = PubMedFetcher()
    fetcher.session = fake_session(raises=AssertionError("should not be called"))

    assert await fetcher.fetch_pubmed_articles([]) == []


async def test_fetch_raises_on_malformed_xml(fake_session, fake_response):
    fetcher = PubMedFetcher()
    fetcher.session = fake_session(fake_response(200, text="<PubmedArticleSet><broken"))

    with pytest.raises(RuntimeError, match="PubMed fetch failed"):
        await fetcher.fetch_pubmed_articles(["31234567"])


def test_medical_evidence_defaults_are_none_not_empty():
    """Distinguishing "no MeSH terms recorded" from "not yet populated"."""
    evidence = MedicalEvidence(
        id="pubmed_1",
        title="t",
        abstract="a",
        pub_date="2019",
        authors=[],
        journal="j",
        source="PubMed",
    )

    assert evidence.pmid is None
    assert evidence.nct_id is None
    assert evidence.mesh_terms is None


# --------------------------------------------------------------------------
# Parsing details that were silently wrong until fixed
#
# The author list was empty on every article (<FirstName> is not an NLM
# element; the given name is in <ForeName>), and structured abstracts were
# truncated to their first section.
# --------------------------------------------------------------------------

async def test_fetch_parses_authors(parsed_article):
    assert parsed_article["authors"] == ["John J V McMurray", "Scott D Solomon"]


async def test_fetch_falls_back_to_initials_when_there_is_no_forename(
    fake_session, fake_response
):
    xml = PUBMED_EFETCH_XML.replace("<ForeName>Scott D</ForeName>\n", "")
    fetcher = PubMedFetcher()
    fetcher.session = fake_session(fake_response(200, text=xml))

    article = (await fetcher.fetch_pubmed_articles(["31234567"]))[0]

    assert article["authors"] == ["John J V McMurray", "SD Solomon"]


async def test_fetch_keeps_collective_authors(fake_session, fake_response):
    """Trial groups are credited as a CollectiveName with no LastName."""
    xml = PUBMED_EFETCH_XML.replace(
        "        </AuthorList>",
        "          <Author ValidYN=\"Y\">\n"
        "            <CollectiveName>DAPA-HF Investigators</CollectiveName>\n"
        "          </Author>\n"
        "        </AuthorList>",
    )
    fetcher = PubMedFetcher()
    fetcher.session = fake_session(fake_response(200, text=xml))

    article = (await fetcher.fetch_pubmed_articles(["31234567"]))[0]

    assert "DAPA-HF Investigators" in article["authors"]


async def test_fetch_captures_the_whole_structured_abstract(parsed_article):
    abstract = parsed_article["abstract"]

    # Every section, in order, each tagged with its label.
    for label in ("BACKGROUND:", "METHODS:", "RESULTS:", "CONCLUSIONS:"):
        assert label in abstract
    assert abstract.index("BACKGROUND:") < abstract.index("CONCLUSIONS:")
    # The conclusion is the part carrying the clinical finding.
    assert "worsening heart failure was lower" in abstract


async def test_fetch_reads_through_inline_markup(fake_session, fake_response):
    """Titles carry <i>/<sup>; reading .text alone truncates at the tag."""
    xml = PUBMED_EFETCH_XML.replace(
        "<ArticleTitle>Dapagliflozin in Patients with Heart Failure and Reduced Ejection Fraction</ArticleTitle>",
        "<ArticleTitle>Dapagliflozin and <i>Na</i><sup>+</sup> Handling</ArticleTitle>",
    )
    fetcher = PubMedFetcher()
    fetcher.session = fake_session(fake_response(200, text=xml))

    article = (await fetcher.fetch_pubmed_articles(["31234567"]))[0]

    assert article["title"] == "Dapagliflozin and Na+ Handling"


async def test_fetch_falls_back_to_medline_date(fake_session, fake_response):
    """Older records carry a free-text MedlineDate instead of a Year."""
    xml = PUBMED_EFETCH_XML.replace(
        "<Year>2019</Year>", "<MedlineDate>2019 Nov-Dec</MedlineDate>"
    )
    fetcher = PubMedFetcher()
    fetcher.session = fake_session(fake_response(200, text=xml))

    article = (await fetcher.fetch_pubmed_articles(["31234567"]))[0]

    assert article["pub_date"] == "2019 Nov-Dec"


# --------------------------------------------------------------------------
# HTTP status handling
#
# Commit 54e4b8f fixed the exception paths; these are the non-2xx paths it
# did not reach, where a 429 or 500 still returned [].
# --------------------------------------------------------------------------

async def test_search_pubmed_raises_on_an_http_error_status(fake_session, fake_response):
    fetcher = PubMedFetcher(max_retries=0, min_interval=0)
    fetcher.session = fake_session(fake_response(429, payload={}))

    with pytest.raises(RuntimeError):
        await fetcher.search_pubmed("heart failure")


async def test_fetch_pubmed_articles_raises_on_an_http_error_status(
    fake_session, fake_response
):
    fetcher = PubMedFetcher(max_retries=0, min_interval=0)
    fetcher.session = fake_session(fake_response(500, text=""))

    with pytest.raises(RuntimeError):
        await fetcher.fetch_pubmed_articles(["31234567"])


async def test_search_trials_raises_on_an_http_error_status(fake_session, fake_response):
    fetcher = ClinicalTrialsFetcher(max_retries=0, min_interval=0)
    fetcher.session = fake_session(fake_response(503, payload={}))

    with pytest.raises(RuntimeError):
        await fetcher.search_trials("atrial fibrillation")


# --------------------------------------------------------------------------
# Rate limiting and retry
#
# NCBI allows roughly 3 requests/second unauthenticated. Now that a 429
# raises instead of returning [], an unpaced ingest fails outright rather
# than silently under-populating -- which is better, but the right answer
# is to not earn the 429.
# --------------------------------------------------------------------------

class SequenceSession:
    """Returns a scripted sequence of responses, one per request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def get(self, url, *args, **kwargs):
        self.calls += 1
        return self.responses.pop(0) if self.responses else self.responses[-1]


async def test_a_rate_limited_search_is_retried_and_succeeds(fake_response):
    fetcher = PubMedFetcher(max_retries=3, retry_backoff=0.001, min_interval=0)
    fetcher.session = SequenceSession([
        fake_response(429, payload={}),
        fake_response(429, payload={}),
        fake_response(200, payload={"esearchresult": {"idlist": ["1", "2"]}}),
    ])

    assert await fetcher.search_pubmed("heart failure") == ["1", "2"]
    assert fetcher.session.calls == 3


async def test_retries_are_bounded_and_then_it_raises(fake_response):
    fetcher = PubMedFetcher(max_retries=2, retry_backoff=0.001, min_interval=0)
    fetcher.session = SequenceSession([fake_response(429, payload={})] * 5)

    with pytest.raises(RuntimeError, match="after 2 retries"):
        await fetcher.search_pubmed("heart failure")

    assert fetcher.session.calls == 3  # the first attempt plus two retries


async def test_a_client_error_is_not_retried(fake_response):
    """400 means the request is wrong; sending it again cannot help."""
    fetcher = PubMedFetcher(max_retries=3, retry_backoff=0.001, min_interval=0)
    fetcher.session = SequenceSession([fake_response(400, payload={})] * 5)

    with pytest.raises(RuntimeError):
        await fetcher.search_pubmed("heart failure")

    assert fetcher.session.calls == 1


async def test_requests_are_paced_to_the_minimum_interval(fake_response):
    import time
    fetcher = PubMedFetcher(min_interval=0.05)
    fetcher.session = SequenceSession(
        [fake_response(200, payload={"esearchresult": {"idlist": []}})] * 4)

    start = time.monotonic()
    for _ in range(3):
        await fetcher.search_pubmed("heart failure")
    elapsed = time.monotonic() - start

    assert elapsed >= 0.10, f"three requests took {elapsed:.3f}s; not paced"


def test_clinical_trials_fetcher_inherits_the_pacing_behaviour():
    fetcher = ClinicalTrialsFetcher()

    assert fetcher.max_retries == 3
    assert fetcher.rate_limiter.min_interval > 0
    assert fetcher.base_url.startswith("https://clinicaltrials.gov/")
