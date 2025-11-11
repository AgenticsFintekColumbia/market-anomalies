# applications/market_anomalies/connectors/sec_edgar.py
from __future__ import annotations
from typing import Optional, List, Dict, Any, Iterable, Literal, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import os
import time
import logging
import requests
import pandas as pd
from pydantic import BaseModel, Field, HttpUrl
from enum import Enum

# Reuse project IO helpers (paths, save_data/save_schema/date_range)
from ._wrds_base import WRDSDataIngestor

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOGLEVEL","INFO"))

# ------------------------------------------------------------------------------
# SEC EDGAR API config (respect rate limits and identify yourself)
# ------------------------------------------------------------------------------
SEC_BASE = "https://data.sec.gov"
SEC_SEARCH = "https://efts.sec.gov/LATEST/search-index"
# Required: identify your app + contact (SEC fair use policy)
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "YourAppName/1.0 (your.email@domain)")
SEC_CONTACT = os.getenv("SEC_CONTACT_EMAIL", "your.email@domain")

# ------------------------------------------------------------------------------
# Models (normalized “event” compatible with your anomaly pipeline)
# ------------------------------------------------------------------------------
class SECEventCategory(str, Enum):
    governance = "governance"
    mna = "mna"
    restructuring = "restructuring"
    other = "other"

class SECEvent(BaseModel):
    provider: Literal["sec_edgar"] = "sec_edgar"
    provider_event_id: Optional[str] = None  # accession number

    # entity
    cik: Optional[str] = None
    ticker: Optional[str] = None
    company_name: Optional[str] = None
    gvkey: Optional[str] = None
    cusip: Optional[str] = None
    isin: Optional[str] = None
    master_id: Optional[str] = None

    # filing metadata
    form_type: Optional[str] = None         # “8-K”, “10-K”, etc.
    filing_date: Optional[datetime] = None
    acceptance_datetime: Optional[datetime] = None
    filing_href: Optional[HttpUrl] = None   # primary document/index
    document_href: Optional[HttpUrl] = None # specific 8-K doc if available
    title: Optional[str] = None

    # event semantics (for 8-K)
    category: SECEventCategory = SECEventCategory.other
    item_codes: List[str] = Field(default_factory=list)   # e.g., ["5.02","1.01"]
    headline: Optional[str] = None
    description: Optional[str] = None

    # timing (no look-ahead: prefer acceptance time)
    event_dt: Optional[datetime] = None
    availability_dt: Optional[datetime] = None

    # passthrough
    raw: Dict[str, Any] = Field(default_factory=dict)

# ------------------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------------------
def _to_cik10(cik: str | int | None) -> Optional[str]:
    if cik is None:
        return None
    s = str(cik).strip().lstrip("0")
    if not s.isdigit():
        return None
    return f"{int(s):010d}"

def _infer_category(items: List[str]) -> SECEventCategory:
    s = " ".join(items)
    if any(x in s for x in ["5.02"]):  # Departure/election of directors/officers
        return SECEventCategory.governance
    if any(x in s for x in ["1.01","2.01"]):  # Material agreements, acquisition/completion
        return SECEventCategory.mna
    if any(x in s for x in ["1.03","2.05","2.06"]):  # Bankruptcy, exit/disposal, impairments
        return SECEventCategory.restructuring
    return SECEventCategory.other

def _parse_acceptance(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    # formats vary: "2024-05-07T06:45:00.000Z" or "20240507T064500"
    try:
        return datetime.fromisoformat(ts.replace("Z","+00:00"))
    except Exception:
        try:
            return datetime.strptime(ts, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        except Exception:
            return None

# ------------------------------------------------------------------------------
# HTTP client (retry/backoff + headers)
# ------------------------------------------------------------------------------
@dataclass
class _Retry:
    attempts: int = 5
    base_sleep: float = 0.5
    max_sleep: float = 8.0

class _SECClient:
    def __init__(self, user_agent: str, contact: str, timeout: int = 30):
        self.hdrs = {
            "User-Agent": f"{user_agent} {contact}",
            "Accept": "application/json",
        }
        self.timeout = timeout
        self._retry = _Retry()

    def _post(self, url: str, json_body: Dict[str, Any]) -> Dict[str, Any]:
        last_exc = None
        for i in range(self._retry.attempts):
            try:
                r = requests.post(url, headers=self.hdrs, json=json_body, timeout=self.timeout)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"Transient {r.status_code}: {r.text}")
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_exc = e
                sleep = min(self._retry.max_sleep, self._retry.base_sleep * (2 ** i) + 0.1 * i)
                logger.warning("SEC POST failed (%s). Retry %d/%d in %.1fs", e, i+1, self._retry.attempts, sleep)
                time.sleep(sleep)
        raise RuntimeError(f"SEC request failed after retries: {last_exc}") from last_exc

    def _get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        last_exc = None
        for i in range(self._retry.attempts):
            try:
                r = requests.get(url, headers=self.hdrs, params=params or {}, timeout=self.timeout)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"Transient {r.status_code}: {r.text}")
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_exc = e
                sleep = min(self._retry.max_sleep, self._retry.base_sleep * (2 ** i) + 0.1 * i)
                logger.warning("SEC GET failed (%s). Retry %d/%d in %.1fs", e, i+1, self._retry.attempts, sleep)
                time.sleep(sleep)
        raise RuntimeError(f"SEC request failed after retries: {last_exc}") from last_exc

# ------------------------------------------------------------------------------
# Ingestor (EDGAR free API)
# ------------------------------------------------------------------------------
class SECEdgarIngestor(WRDSDataIngestor):
    """
    Free SEC EDGAR connector (replaces paid EventVestor).
    Two workflows:
      1) search_filings: Full-text search index (filter by formType and date).
      2) get_company_filings: By CIK using /submissions endpoint.

    For 8-K anomalies:
      - Filter formType="8-K"
      - Optionally filter on items: ["5.02","1.01","2.01","2.05","1.03"]
    """

    def __init__(self, wrds_username: str | None = None):
        # We don't need WRDS here; we just reuse IO methods.
        self.conn = None
        self.client = _SECClient(SEC_USER_AGENT, SEC_CONTACT)

    # ========== Public API ==========

    def fetch_8k_events(
        self,
        start_date: str,
        end_date: str,
        tickers_or_ciks: Optional[List[str]] = None,
        item_codes: Optional[Iterable[str]] = None,
        limit: int = 2500,
    ) -> pd.DataFrame:
        """
        Fetch normalized 8-K events from EDGAR search-index and/or submissions endpoints.
        If tickers_or_ciks provided → search per-entity; else → broad search window.
        """
        rows: List[Dict[str, Any]] = []

        # Strategy:
        #  A) If no tickers provided, use search-index filtered by date+formType (broad).
        #  B) If tickers/ciks provided, iterate them (search-index filter by CIK).
        targets = tickers_or_ciks or [None]

        for target in targets:
            # Build search body
            query = self._build_search_query(start_date, end_date, target)
            payload = self.client._post(SEC_SEARCH, query)
            # Each hit is a filing; normalize only 8-K and extract item codes from description/title if present
            for hit in payload.get("hits", {}).get("hits", []):
                src = hit.get("_source", {})
                if src.get("formType") != "8-K":
                    continue
                ev = self._normalize_hit(src, wanted_items=item_codes)
                if ev:
                    rows.append(ev.dict())

        df = pd.DataFrame(rows)
        if df.empty:
            logger.warning("SEC EDGAR: no 8-K events matched the filters.")
            return df

        # Convenience columns
        df["has_doc_link"] = df["document_href"].notna().astype("Int8")
        df["is_board_change"] = df["item_codes"].apply(lambda xs: 1 if "5.02" in (xs or []) else 0).astype("Int8")
        df["is_mna_related"]  = df["item_codes"].apply(lambda xs: 1 if any(x in (xs or []) for x in ("1.01","2.01")) else 0).astype("Int8")
        df["is_restructuring"]= df["item_codes"].apply(lambda xs: 1 if any(x in (xs or []) for x in ("1.03","2.05","2.06")) else 0).astype("Int8")
        return df

    # ========== Helpers ==========

    def _build_search_query(self, start_date: str, end_date: str, ticker_or_cik: Optional[str]) -> Dict[str, Any]:
        """
        Build the JSON body for /LATEST/search-index.
        Filters: formType=8-K, filingDate range, optional ticker/CIK term.
        """
        must = [
            {"query_string": {"default_field": "formType", "query": "8-K"}},
            {"range": {"filedAt": {"gte": f"{start_date}T00:00:00", "lte": f"{end_date}T23:59:59"}}},
        ]
        if ticker_or_cik:
            term = str(ticker_or_cik).strip()
            # try CIK10 first
            cik10 = _to_cik10(term)
            if cik10:
                must.append({"term": {"cik": cik10}})
            else:
                # fall back to ticker match (case-insensitive)
                must.append({"query_string": {"default_field": "ticker", "query": f'"{term.upper()}"'}})

        return {
            "from": 0,
            "size": 2500,
            "query": {"bool": {"must": must}},
            "sort": [{"filedAt": {"order": "desc"}}],
            "source": [
                "cik","ticker","displayNames","companyName",
                "formType","filedAt","acceptedDate","primaryDocument","linkToFilingDetails","linkToHtml","linkToTxt",
                "description","documentFormatFiles","title","accNo"
            ],
        }

    def _normalize_hit(self, src: Dict[str, Any], wanted_items: Optional[Iterable[str]]) -> Optional[SECEvent]:
        """
        Normalize a search-index hit into SECEvent and filter by item codes if provided.
        """
        cik = _to_cik10(src.get("cik"))
        ticker = (src.get("ticker") or None)
        company = src.get("companyName") or (src.get("displayNames") or [None])[0]
        acc_no = (src.get("accNo") or "").replace("-", "")
        filed_at = src.get("filedAt")
        accepted = src.get("acceptedDate")

        filing_href = src.get("linkToFilingDetails") or src.get("linkToHtml") or src.get("linkToTxt")
        doc_href = None
        # pick the first “documentFormatFiles” html if present
        dff = src.get("documentFormatFiles") or []
        for f in dff:
            if f.get("documentFormat") == "html" and f.get("documentUrl"):
                doc_href = f.get("documentUrl"); break
        if not doc_href:
            doc_href = src.get("linkToHtml") or src.get("linkToTxt") or filing_href

        # crude item-code extraction from description/title text (common in index)
        text = " ".join([str(src.get("description") or ""), str(src.get("title") or "")])
        codes = []
        for code in ("1.01","2.01","2.05","2.06","5.02","1.03"):
            if code in text:
                codes.append(code)

        # If user asked for specific item codes, filter here
        if wanted_items:
            if not any(c in codes for c in wanted_items):
                return None

        event_dt = _parse_acceptance(accepted) or (datetime.fromisoformat(filed_at.replace("Z","+00:00")) if filed_at else None)
        availability_dt = event_dt  # acceptance is the public timestamp

        ev = SECEvent(
            provider_event_id=acc_no or None,
            cik=cik, ticker=ticker, company_name=company,
            form_type=src.get("formType"),
            filing_date=(datetime.fromisoformat(filed_at.replace("Z","+00:00")) if filed_at else None),
            acceptance_datetime=event_dt,
            filing_href=filing_href,
            document_href=doc_href,
            title=src.get("title"),
            category=_infer_category(codes),
            item_codes=codes,
            headline=src.get("title"),
            description=src.get("description"),
            event_dt=event_dt,
            availability_dt=availability_dt,
            raw=src,
        )
        return ev

    @staticmethod
    def schema_doc() -> Dict[str, Any]:
        return {
            "dataset": "SEC EDGAR (free) 8-K events",
            "library": "sec_edgar_api",
            "endpoints": {
                "search-index": "https://efts.sec.gov/LATEST/search-index",
                "company-submissions": "https://data.sec.gov/submissions/CIK##########.json"
            },
            "date_fields": {"filing_date": "FiledAt", "acceptance_datetime": "AcceptedDate (availability)"},
            "identifier_fields": ["cik","ticker"],
            "fields": {
                "form_type": "SEC form (8-K, 10-K, 10-Q, …)",
                "item_codes": "8-K item codes parsed from description/title (best-effort)",
                "filing_href": "Link to filing detail",
                "document_href": "HTML/TXT document link",
                "category": "Derived category: governance/mna/restructuring/other",
                "headline": "Filing title",
                "description": "Index description"
            },
            "common_queries": [
                "All 8-K Item 5.02 (board/officer) in last 90 days",
                "M&A-related items (1.01/2.01) for selected tickers",
                "Restructuring/bankruptcy items (1.03/2.05/2.06)"
            ]
        }

    # Hand-through IO helpers
    save_data = WRDSDataIngestor.save_data
    save_schema = WRDSDataIngestor.save_schema
    date_range = WRDSDataIngestor.date_range
