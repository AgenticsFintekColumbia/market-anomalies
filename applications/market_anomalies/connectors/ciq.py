from __future__ import annotations
import logging
from typing import Dict, Tuple
import pandas as pd
from ._wrds_base import WRDSDataIngestor

logger = logging.getLogger(__name__)

# Default CIQ Key Development event classes
DEFAULT_EVENT_IDS: Tuple[int, ...] = (16, 81, 232)   # Exec Changes, M&A Announce, M&A Canceled


class CIQIngestor(WRDSDataIngestor):
    """
    Capital IQ (CIQ) Key Developments ingestor for WRDS.

    Provides:
      - fetch_keydev_events(start, end, event_ids)
      - fetch_if_needed(name, start, end)  <-- NEW (caching)
    """

    library = "ciq"

    # noinspection SqlNoDataSourceInspection,SqlDialectInspection
    def fetch_keydev_events(
        self,
        start_date: str,
        end_date: str,
        event_ids: Tuple[int, ...] = DEFAULT_EVENT_IDS,
        only_primary_us_tickers: bool = True,
    ) -> pd.DataFrame:

        if not event_ids:
            event_ids = DEFAULT_EVENT_IDS

        event_filter = "(" + ",".join(str(i) for i in event_ids) + ")"
        primary_flag_clause = "AND t.primaryflag = 1" if only_primary_us_tickers else ""

        # Ensure ticker is active on or before the event date
        ticker_date_filter = "AND (t.enddate IS NULL OR t.enddate >= kd.announcedate)"

        q = f"""
        SELECT 
            kd.keydevid,
            kd.companyid,
            kd.companyname,
            kd.gvkey,
            kd.announcedate AS event_date,
            kd.keydeveventtypeid AS event_type_id,
            kd.eventtype AS event_type_name,
            kd.headline,
            t.ticker
        FROM {self.library}.wrds_keydev AS kd
        JOIN {self.library}.wrds_ticker AS t
             ON kd.companyid = t.companyid
        WHERE 
            kd.keydeveventtypeid IN {event_filter}
            AND kd.announcedate BETWEEN '{start_date}' AND '{end_date}'
            AND kd.gvkey IS NOT NULL
            {primary_flag_clause}
            {ticker_date_filter}
        ORDER BY kd.announcedate DESC
        """

        logger.info(
            "CIQ: fetching KeyDev events %s → %s (event_ids=%s)",
            start_date, end_date, event_ids,
        )

        df = self.conn.raw_sql(q, date_cols=['event_date'])
        if df.empty:
            logger.warning("CIQ KeyDev returned 0 rows.")
            return df

        # Simple event severity heuristic
        severity_map = {16: 0.8, 81: 0.7, 232: 1.0}
        df["event_severity"] = df["event_type_id"].map(severity_map).fillna(0.6)

        # Boolean flags
        df["is_exec_change"]  = (df["event_type_id"] == 16).astype("Int8")
        df["is_mna_announce"] = (df["event_type_id"] == 81).astype("Int8")
        df["is_mna_canceled"] = (df["event_type_id"] == 232).astype("Int8")

        # De-dupe to avoid duplicate event/ticker pairs
        df = df.drop_duplicates(
            subset=["keydevid", "ticker", "event_type_id", "event_date", "headline"]
        )

        logger.info("CIQ: %d KeyDev events after filtering.", len(df))
        return df

    # -------------------------------------------------------------
    # CACHING LOGIC (same pattern as CRSP & Compustat)
    # -------------------------------------------------------------
    def fetch_if_needed(
        self,
        name: str,
        start: str,
        end: str,
        event_ids: Tuple[int, ...] = DEFAULT_EVENT_IDS,
        only_primary_us_tickers: bool = True,
    ) -> pd.DataFrame:

        if self.data_exists(name):
            logger.info("CIQ: loading cached data → %s", name)
            return self.load_data(name)

        logger.info("CIQ: fetching from WRDS (no cached file found)...")
        df = self.fetch_keydev_events(
            start_date=start,
            end_date=end,
            event_ids=event_ids,
            only_primary_us_tickers=only_primary_us_tickers
        )
        self.save_data(df, name)
        return df

    # -------------------------------------------------------------
    # SCHEMA DOCUMENTATION
    # -------------------------------------------------------------
    @staticmethod
    def get_schema_documentation() -> Dict:
        return {
            "dataset": "CIQ Key Developments (WRDS)",
            "library": "ciq",
            "tables": [
                "wrds_keydev",
                "ciqkeydeveventtype",
                "wrds_ticker"
            ],
            "identifier_fields": ["keydevid", "companyid", "gvkey", "ticker"],
            "date_field": "event_date",
            "fields": {
                "event_type_id": "CIQ event type ID (e.g. 16 Exec Change, 81 M&A, 232 Canceled)",
                "event_type_name": "Human-readable event type",
                "headline": "CIQ event headline",
                "event_severity": "Severity heuristic (0.6–1.0)",
                "is_exec_change": "Flag for executive change events",
                "is_mna_announce": "Flag for M&A announcement events",
                "is_mna_canceled": "Flag for M&A cancellation events",
            },
            "notes": [
                "Only events with gvkey != NULL are included for CRSP/Compustat linking.",
                "Only primary US tickers selected by default (t.primaryflag = 1)."
            ],
            "common_queries": [
                "Executive changes in last 90 days",
                "M&A announcements by ticker",
                "Canceled/withdrawn deals by company",
            ]
        }
