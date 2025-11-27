from __future__ import annotations
from typing import Dict
import pandas as pd
from ._wrds_base import WRDSDataIngestor, logger

class CompustatIngestor(WRDSDataIngestor):
    """
    Quarterly fundamentals (Compustat North America)
    Produces sector-relative margins (z-scores) for anomaly detection.
    """

    # noinspection SqlNoDataSourceInspection,SqlDialectInspection
    def fetch_quarterly_fundamentals(self, start_date: str, end_date: str) -> pd.DataFrame:
        q = f"""
        SELECT 
            a.gvkey, -- need this to merge with CRSP
            a.datadate,
            a.fyearq,
            a.fqtr,
            b.conm  AS company_name,
            b.sic,
            -- Income statement (quarterly)
            a.revtq AS revenue,
            a.niq   AS net_income,
            a.oibdpq AS ebitda,
            -- Balance sheet (quarterly)
            a.atq   AS total_assets,
            a.ltq   AS total_liabilities,
            a.seqq  AS shareholders_equity
            
        FROM comp.fundq a
        LEFT JOIN comp.company b
          ON a.gvkey = b.gvkey
        WHERE a.datadate BETWEEN '{start_date}' AND '{end_date}'
          AND a.indfmt = 'INDL'
          AND a.datafmt = 'STD'
          AND a.popsrc = 'D'
          AND a.consol = 'C'
        ORDER BY a.gvkey, a.datadate
        """
        df = self.conn.raw_sql(q)
        if df.empty:
            return df

        # Basic margins
        df["net_margin"]   = df["net_income"] / df["revenue"].replace({0.0: pd.NA})
        df["ebitda_margin"] = df["ebitda"] / df["revenue"].replace({0.0: pd.NA})

        # Broad sector by first 2 digits of SIC (simple, fast)
        df["sector2"] = df["sic"].astype(str).str[:2]

        # Sector-relative z-scores by quarter date
        for m in ["net_margin", "ebitda_margin"]:
            grp = df.groupby(["sector2", "datadate"])[m]
            df[f"sector_avg_{m}"] = grp.transform("mean")
            df[f"sector_std_{m}"] = grp.transform("std")
            df[f"{m}_z"] = (df[m] - df[f"sector_avg_{m}"]) / (df[f"sector_std_{m}"].replace(0.0, pd.NA))
            df[f"is_{m}_anomaly"] = (df[f"{m}_z"] < -2).astype("Int8")

        return df

    def fetch_if_needed(self, name: str, start: str, end: str) -> pd.DataFrame:
        if self.data_exists(name):
            logger.info("Loading cached Compustat quarterly data: %s", name)
            return self.load_data(name)

        logger.info("Fetching Compustat quarterly data from WRDS...")
        df = self.fetch_quarterly(start, end)
        self.save_data(df, name)
        return df

    @staticmethod
    def get_schema_documentation() -> Dict:
        return {
            "dataset": "Compustat Quarterly Fundamentals",
            "library": "comp",
            "primary_table": "fundq",
            "date_field": "datadate",
            "identifier_fields": ["gvkey", "ticker"],
            "fields": {
                "revenue": "Quarterly revenue",
                "net_income": "Quarterly net income",
                "ebitda": "Quarterly EBITDA",
                "total_assets": "Total assets (quarterly)",
                "total_liabilities": "Total liabilities (quarterly)",
                "shareholders_equity": "Shareholders' equity (quarterly)",
                "net_margin": "net_income / revenue",
                "ebitda_margin": "ebitda / revenue",
                "net_margin_z": "Sector-relative z-score (by SIC2 & date)",
                "ebitda_margin_z": "Sector-relative z-score (by SIC2 & date)"
            },
            "common_queries": [
                "Companies underperforming sector margins by >2σ in last quarter",
                "Quarter-over-quarter margin deterioration",
                "Revenue growth vs sector average"
            ]
        }
