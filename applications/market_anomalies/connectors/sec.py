from __future__ import annotations
import os
from typing import Dict, Iterable, List, Optional
import pandas as pd
from ._wrds_base import WRDSDataIngestor

# Configure library/table names via env (override here if your site differs)
SEC_LIB           = os.getenv("WRDS_SEC_LIB", "sec_analytics")
SEC_FILINGS_TBL   = os.getenv("WRDS_SEC_FILINGS", "filings")         # general filings metadata
SEC_8K_ITEMS_TBL  = os.getenv("WRDS_SEC_8K_ITEMS", "form8k_items")   # item-level 8-K rows (item_code)

ITEM_SEVERITY = {
    "5.02": 0.9,  # Departure/election of directors or certain officers
    "1.01": 0.8,  # Entry into a material definitive agreement
    "2.01": 1.0,  # Completion of acquisition or disposition of assets
    "2.05": 1.0,  # Costs associated with exit or disposal activities (restructuring)
    "2.06": 0.9,  # Material impairments
    "1.03": 1.0,  # Bankruptcy or receivership
}

class SECIngestor(WRDSDataIngestor):
    """
    WRDS SEC Analytics Suite ingestor.
    - fetch_8k_items: returns item-level events for anomaly detection (governance/M&A/restructuring)
    - fetch_filings:  returns general filings metadata for volume/surge analysis
    """

    # noinspection SqlNoDataSourceInspection,SqlDialectInspection
    def fetch_8k_items(
        self,
        start_date: str,
        end_date: str,
        item_codes: Optional[Iterable[str]] = None,    # e.g. ["5.02","1.01","2.01"]
        ciks: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """
        Fetch item-level 8-K events from SEC Analytics.

        Returns columns (typical): cik, company_name, filing_date, form_type, item_code, item_desc, accession_no, file_url
        Exact column names may vary by site; adjust SELECT aliases if needed.
        """
        item_filter = ""
        if item_codes:
            safe = ",".join("'" + ic.replace("'", "''") + "'" for ic in item_codes)
            item_filter = f" AND i.item_code IN ({safe}) "

        cik_filter = ""
        if ciks:
            safe = ",".join("'" + str(c).replace("'", "''") + "'" for c in ciks)
            cik_filter = f" AND i.cik IN ({safe}) "

        q = f"""
        SELECT 
            i.cik,
            i.company_name,
            i.filing_date,
            i.form_type,
            i.item_code,
            i.item_desc,
            i.accession_number,
            i.file_url
        FROM {SEC_LIB}.{SEC_8K_ITEMS_TBL} i
        WHERE i.filing_date BETWEEN '{start_date}' AND '{end_date}'
          AND i.form_type = '8-K'
          {item_filter}
          {cik_filter}
        ORDER BY i.filing_date DESC, i.cik
        """
        df = self.conn.raw_sql(q)
        if df.empty:
            return df

        # Standardized fields for downstream
        df.rename(
            columns={
                "accession_number": "accession_no",
                "file_url": "document_url",
            },
            inplace=True,
        )

        # Basic severity heuristic from item_code (fallback 0.6)
        df["event_severity"] = df.get("item_code", pd.Series(dtype=str)).map(ITEM_SEVERITY).fillna(0.6)

        # Convenience flags for common anomaly categories
        df["is_board_officer_change"] = (df["item_code"] == "5.02").astype("Int8")
        df["is_mna_related"]          = df["item_code"].isin(["1.01", "2.01"]).astype("Int8")
        df["is_restructuring"]        = df["item_code"].isin(["2.05", "1.03"]).astype("Int8")

        return df

    # noinspection SqlNoDataSourceInspection,SqlDialectInspection
    def fetch_filings(
        self,
        start_date: str,
        end_date: str,
        forms: Optional[Iterable[str]] = None,   # e.g., ["10-K","10-Q","8-K"]
        ciks: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """
        Fetch general filings metadata (volume/surge analysis, doc links).
        Useful for "information flow" anomalies (sudden surge in filings).
        """
        form_filter = ""
        if forms:
            safe = ",".join("'" + f.replace("'", "''") + "'" for f in forms)
            form_filter = f" AND f.form_type IN ({safe}) "

        cik_filter = ""
        if ciks:
            safe = ",".join("'" + str(c).replace("'", "''") + "'" for c in ciks)
            cik_filter = f" AND f.cik IN ({safe}) "

        q = f"""
        SELECT 
            f.cik,
            f.company_name,
            f.filing_date,
            f.form_type,
            f.accession_number,
            f.file_url
        FROM {SEC_LIB}.{SEC_FILINGS_TBL} f
        WHERE f.filing_date BETWEEN '{start_date}' AND '{end_date}'
          {form_filter}
          {cik_filter}
        ORDER BY f.filing_date DESC, f.cik
        """
        df = self.conn.raw_sql(q)
        if df.empty:
            return df

        df.rename(
            columns={
                "accession_number": "accession_no",
                "file_url": "document_url",
            },
            inplace=True,
        )
        return df

    @staticmethod
    def schema_doc() -> Dict:
        """
        Minimal schema doc (good for Text2SQL). Add more fields if your site exposes them.
        """
        return {
            "dataset": "WRDS SEC Analytics Suite",
            "library": os.getenv("WRDS_SEC_LIB", SEC_LIB),
            "tables": {
                os.getenv("WRDS_SEC_8K_ITEMS", SEC_8K_ITEMS_TBL): "Item-level rows for 8-K filings (item_code, item_desc)",
                os.getenv("WRDS_SEC_FILINGS", SEC_FILINGS_TBL): "General filings metadata (10-K/10-Q/8-K, links)",
            },
            "date_field": "filing_date",
            "identifier_fields": ["cik"],
            "key_fields": {
                "form_type": "SEC form (8-K, 10-K, 10-Q, etc.)",
                "item_code": "8-K item code (e.g., 5.02, 1.01, 2.01)",
                "item_desc": "Item description (if available)",
                "accession_no": "SEC accession number",
                "document_url": "URL to filing document",
                "event_severity": "Heuristic based on item_code (0..1)",
                "is_board_officer_change": "1 if 8-K Item 5.02",
                "is_mna_related": "1 if 8-K Items 1.01 or 2.01",
                "is_restructuring": "1 if 8-K Items 2.05 or 1.03"
            },
            "common_queries": [
                "8-K Item 5.02 events in last 90 days (board/officer changes)",
                "M&A-related 8-K items for a given CIK list",
                "Filing surge counts by form_type for last quarter"
            ]
        }
