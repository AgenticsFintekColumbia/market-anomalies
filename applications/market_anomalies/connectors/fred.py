from ._wrds_base import WRDSDataIngestor
import pandas as pd
from fredapi import Fred
import os, logging

logger = logging.getLogger(__name__)

class FREDIngestor(WRDSDataIngestor):
    """Macro indicators from the Federal Reserve (FRED API)."""

    def __init__(self, api_key: str | None = None):
        self.fred = Fred(api_key=api_key or os.getenv("FRED_API_KEY"))

    def fetch_indicators(self, series: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        data = {}
        for s in series:
            logger.info(f"Fetching FRED series {s}")
            ser = self.fred.get_series(s, observation_start=start_date, observation_end=end_date)
            data[s] = ser
        df = pd.DataFrame(data)
        df.index.name = "date"
        return df.reset_index()
