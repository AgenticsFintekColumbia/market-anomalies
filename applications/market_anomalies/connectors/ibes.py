from ._wrds_base import WRDSDataIngestor
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class IBESIngestor(WRDSDataIngestor):
    """
    Ingestor for IBES analyst data (EPS summary + recommendations).
    Uses WRDSDataIngestor caching helpers: data_exists / load_data / save_data.
    """

    # ------------------------ EPS SUMMARY ------------------------ #
    # noinspection SqlNoDataSourceInspection,SqlDialectInspection
    def fetch_eps_summary(self, start: str, end: str) -> pd.DataFrame:
        """
        Fetch IBES EPS summary statistics from statsum_epsus for a date window.

        NOTE: We restrict to:
          - fpi = '1' (quarterly)
          - measure = 'EPS'
        """
        q = f"""
            SELECT
                ticker,
                cusip,
                statpers  AS estimate_date,             -- fiscal statement date
                fpedats   AS fiscal_period_end,         -- fiscal period end (aligns w/ Compustat datadate)
                fpi,
                measure,
                meanest   AS consensus_estimate,
                medest    AS median_estimate,
                numest    AS num_analysts,
                stdev     AS estimate_std,
                actual    AS actual_eps
            FROM ibes.statsum_epsus
            WHERE statpers BETWEEN '{start}' AND '{end}'
              AND fpi = '1'                -- quarterly EPS
              AND measure = 'EPS'
            ORDER BY ticker, statpers
        """
        df = self.conn.raw_sql(q)
        # Make sure date-like columns are proper datetimes
        date_cols = ["estimate_date", "fiscal_period_end", "actual_announcement_date"]
        for c in date_cols:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c])
        return df

    def fetch_eps_if_needed(self, name: str, start: str, end: str) -> pd.DataFrame:
        """
        Check if EPS summary data already exists on disk; if so, load it.
        Otherwise fetch from WRDS, save, and return.
        """
        if self.data_exists(name):
            logger.info("Loading cached IBES EPS summary: %s", name)
            return self.load_data(name)

        logger.info("Fetching IBES EPS summary from WRDS...")
        df = self.fetch_eps_summary(start, end)
        self.save_data(df, name)
        return df

    # --------------------- RECOMMENDATIONS ----------------------- #
    # noinspection SqlNoDataSourceInspection,SqlDialectInspection
    def fetch_recommendations(self, start: str, end: str) -> pd.DataFrame:
        """
        Fetch analyst recommendation events from recddet.

        Uses your WRDS schema where the analyst column is named 'analyst'.
        """
        q = f"""
            SELECT
                ticker,
                cusip,
                anndats   AS recommendation_date,  -- event date
                analyst   AS analyst_id,           -- analyst column
                ereccd      AS recommendation_code,  -- 1..5 recommendation code
                revdats   AS relation_date
            FROM ibes.recddet
            WHERE anndats BETWEEN '{start}' AND '{end}'
            ORDER BY ticker, anndats, analyst
        """

        df = self.conn.raw_sql(q)

        # normalize date columns
        for c in ["recommendation_date", "relation_date"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c])

        return df

    def fetch_recs_if_needed(self, name: str, start: str, end: str) -> pd.DataFrame:
        """
        Same caching pattern for recommendations.
        """
        if self.data_exists(name):
            logger.info("Loading cached IBES recommendations: %s", name)
            return self.load_data(name)

        logger.info("Fetching IBES recommendations from WRDS...")
        df = self.fetch_recommendations(start, end)
        self.save_data(df, name)
        return df

    # -------------------- SCHEMA DOCUMENTATION ------------------- #
    @staticmethod
    def get_schema_documentation() -> dict:
        """
        High-level schema documentation for IBES usage in the project.
        Raw fields only; engineered metrics (earnings_surprise, coverage_score, etc.)
        are computed downstream.
        """
        return {
            "dataset": "IBES Analyst Estimates & Recommendations",
            "library": "ibes",
            "tables": {
                "recddet": "Analyst-level recommendation events",
                "statsum_epsus": "Consensus EPS summary statistics (US)"
            },
            "identifier_fields": {
                "ticker": "IBES ticker (transient; later mapped to PERMNO/GVKEY)",
                "cusip": "CUSIP as reported by IBES"
            },
            "date_fields": {
                "estimate_date": "statpers: fiscal statement date (end of reporting period)",
                "fiscal_period_end": "fpedats: fiscal period end (aligns with Compustat datadate)",
                "actual_announcement_date": "anndats: date of actual EPS announcement",
                "recommendation_date": "anndats: date of recommendation event"
            },
            "fields_eps_summary": {
                "consensus_estimate": "Mean EPS estimate across analysts (meanest)",
                "median_estimate": "Median EPS estimate (medest)",
                "num_analysts": "Number of contributing analysts (numest)",
                "estimate_std": "Standard deviation of estimates (stdev)",
                "actual_eps": "Reported EPS for the fiscal period (actual)",
                "fpi": "Forecast period indicator (1=quarterly, 2=annual, etc.)",
                "measure": "Measure code (EPS, etc.)"
            },
            "fields_recommendations": {
                "recommendation_code": "IBES recommendation code (1=Strong Buy … 5=Sell)",
                "rec_change": "IBES change flag (sign indicates upgrade/downgrade)",
                "analyst_id": "IBES analyst identifier (analys)"
            },
            "common_queries": [
                "Recent analyst downgrades by ticker",
                "EPS negative surprises in last quarter (using EPS + CRSP later)",
                "Estimate revisions around event dates",
                "Coverage-filtered earnings surprise screens"
            ]
        }
