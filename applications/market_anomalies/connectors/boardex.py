# applications/market_anomalies/connectors/boardex.py
from ._wrds_base import WRDSDataIngestor
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class BoardExIngestor(WRDSDataIngestor):
    """BoardEx leadership data (executive changes & governance)."""

    def __init__(self, wrds_username: str):
        super().__init__(wrds_username)
        self.library = 'boardex'

    # noinspection SqlNoDataSourceInspection,SqlDialectInspection
    def fetch_board_changes(self, start_date: str, end_date: str) -> pd.DataFrame:
        query = f"""
        SELECT companyid, personid, role, board_join_date, board_leave_date,
               current_flag, companyname, ticker
        FROM boardex.directors
        WHERE board_join_date BETWEEN '{start_date}' AND '{end_date}'
           OR board_leave_date BETWEEN '{start_date}' AND '{end_date}'
        """
        df = self.conn.raw_sql(query)
        df['event_type'] = df.apply(lambda r: 'join' if pd.notna(r.board_join_date) else 'leave', axis=1)
        return df

    def get_schema_documentation(self):
        return {
            "dataset": "BoardEx Directors",
            "library": "boardex",
            "date_field": "board_join_date / board_leave_date",
            "identifier_fields": ["companyid", "personid"],
            "key_fields": {
                "role": "Director role/title",
                "current_flag": "Active board member flag",
                "event_type": "Join or Leave indicator"
            },
            "common_queries": [
                "CEO/CFO turnover in last quarter",
                "Board composition volatility",
                "Average tenure of directors by sector"
            ]
        }
