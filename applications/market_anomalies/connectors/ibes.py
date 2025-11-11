from __future__ import annotations
from typing import Dict
import pandas as pd
from ._wrds_base import WRDSDataIngestor

class IBESIngestor(WRDSDataIngestor):
    """
    IBES analyst recommendations and EPS estimates/surprises.
    """

    # noinspection SqlNoDataSourceInspection,SqlDialectInspection
    def fetch_analyst_recommendations(self, start_date: str, end_date: str) -> pd.DataFrame:
        q = f"""
        SELECT 
            a.ticker,
            a.anndats AS announce_date,
            a.analys  AS analyst_id,
            a.ireccd  AS recommendation_code
        FROM ibes.recddet a
        WHERE a.anndats BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY a.ticker, a.analys, a.anndats
        """
        df = self.conn.raw_sql(q)
        if df.empty:
            return df

        # Per-analyst changes (upgrade/downgrade)
        df = df.sort_values(["ticker", "analyst_id", "announce_date"])
        df["prev_rec"] = df.groupby(["ticker", "analyst_id"])["recommendation_code"].shift(1)
        df["rec_change"] = df["recommendation_code"] - df["prev_rec"]
        df["is_upgrade"] = (df["rec_change"] < 0).astype("Int8")
        df["is_downgrade"] = (df["rec_change"] > 0).astype("Int8")
        return df

    # noinspection SqlNoDataSourceInspection,SqlDialectInspection
    def fetch_eps_estimates(self, start_date: str, end_date: str) -> pd.DataFrame:
        q = f"""
        SELECT 
            ticker,
            statpers  AS estimate_date,
            fpedats   AS fiscal_period_end,
            numest    AS num_analysts,
            meanest   AS consensus_estimate,
            stdev     AS estimate_std,
            medest    AS median_estimate,
            actual    AS actual_eps
        FROM ibes.statsum_epsus
        WHERE statpers BETWEEN '{start_date}' AND '{end_date}'
          AND fpi = '1'         -- quarterly
          AND measure = 'EPS'
        ORDER BY ticker, statpers
        """
        df = self.conn.raw_sql(q)
        if df.empty:
            return df

        # Surprise metric
        denom = df["consensus_estimate"].abs().replace(0.0, pd.NA)
        df["earnings_surprise"] = (df["actual_eps"] - df["consensus_estimate"]) / denom
        df["is_neg_surprise_5pct"] = (df["earnings_surprise"] < -0.05).astype("Int8")

        # Coverage score (0..1), simple normalization
        df["coverage_score"] = (df["num_analysts"] / 10.0).clip(0, 1)
        return df

    @staticmethod
    def schema_doc() -> Dict:
        return {
            "dataset": "IBES Analyst Estimates & Recommendations",
            "library": "ibes",
            "tables": {
                "recddet": "Analyst recommendations (per-analyst events)",
                "statsum_epsus": "Consensus EPS time series (US)"
            },
            "date_fields": {"recommendations": "anndats", "estimates": "statpers"},
            "identifier_fields": ["ticker"],
            "fields": {
                "recommendation_code": "1=Strong Buy … 5=Sell",
                "rec_change": "Positive means downgrade, negative upgrade",
                "num_analysts": "Coverage count",
                "consensus_estimate": "Mean EPS estimate",
                "actual_eps": "Reported EPS",
                "earnings_surprise": "(actual - consensus)/|consensus|",
                "coverage_score": "min(num_analysts/10, 1)"
            },
            "common_queries": [
                "Recent analyst downgrades by ticker",
                "EPS negative surprises in last quarter",
                "Estimate revisions around event dates"
            ]
        }
