from ._wrds_base import WRDSDataIngestor
import pandas as pd
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

class YahooFinanceIngestor(WRDSDataIngestor):
    """Quick price signals from Yahoo Finance."""

    def fetch_prices(self, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        df = yf.download(tickers, start=start_date, end=end_date, group_by="ticker", auto_adjust=True)
        frames = []
        for t in tickers:
            x = df[t].reset_index()
            x["ticker"] = t
            frames.append(x)
        return pd.concat(frames)
