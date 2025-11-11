# applications/market_anomalies/connectors/sdc.py
from ._wrds_base import WRDSDataIngestor
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class SDCIngestor(WRDSDataIngestor):
    """Corporate transactions from SDC Platinum (via WRDS)."""

    def __init__(self, wrds_username: str):
        super().__init__(wrds_username)
        self.library = 'sdc'

    # noinspection SqlNoDataSourceInspection,SqlDialectInspection
    def fetch_mna_deals(self, start_date: str, end_date: str) -> pd.DataFrame:
        query = f"""
        SELECT dealno, anncdt as announce_date, compname, targetname,
               dealvalue, percentofacq, status, dealform, country, sic
        FROM sdc.mna
        WHERE anncdt BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY anncdt DESC
        """
        logger.info(f"Fetching M&A deals {start_date}–{end_date}")
        df = self.conn.raw_sql(query)
        df['is_large_deal'] = (df['dealvalue'] > df['dealvalue'].median()).astype(int)
        return df

    def get_schema_documentation(self):
        return {
            "dataset": "SDC Platinum M&A",
            "library": "sdc",
            "date_field": "anncdt",
            "identifier_fields": ["dealno", "compname"],
            "key_fields": {
                "dealvalue": "Transaction value (USD millions)",
                "percentofacq": "Percent acquired",
                "status": "Deal completion status",
                "is_large_deal": "Binary large-deal anomaly"
            },
            "common_queries": [
                "Recent M&A spikes by sector",
                "Aborted or withdrawn deals",
                "Cross-border transaction surges"
            ]
        }
