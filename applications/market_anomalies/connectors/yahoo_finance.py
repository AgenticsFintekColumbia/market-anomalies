import yfinance as yf
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

class YahooFinanceIngestor:
    source = "yahoo_finance"

    def __enter__(self):
        # no WRDS connection needed
        return self

    def __exit__(self, exc_type, exc, tb):
        # nothing to close
        return False

    def fetch(self, tickers, start_date, end_date) -> pd.DataFrame:
        logger.info(
            "YahooFinanceIngestor: fetching prices for %s from %s to %s",
            tickers,
            start_date,
            end_date,
        )
        df = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            group_by="ticker",
            auto_adjust=False,
            progress=False,
        )

        if isinstance(tickers, str):
            tickers = [tickers]

        frames = []
        for t in tickers:
            x = df[t].reset_index()
            x = x.rename(columns={"Date": "date", "Adj Close": "adj_close", "Volume": "volume"})
            x["ticker"] = t
            x["date"] = pd.to_datetime(x["date"]).dt.tz_localize(None)
            frames.append(x[["date", "ticker", "adj_close", "volume"]])

        out = pd.concat(frames, ignore_index=True)
        out["source"] = self.source
        return out

    def save_data(self, df: pd.DataFrame, path: str):
        """
        Save data to data/<path>.parquet, creating directories if necessary.
        Example path: "external/yahoo_prices"
        """
        full_path = f"data/{path}.parquet"
        parent_dir = os.path.dirname(full_path)

        if not os.path.exists(parent_dir):
            logger.info(f"Creating directory {parent_dir}")
            os.makedirs(parent_dir, exist_ok=True)

        logger.info(f"Saving Yahoo Finance data to {full_path}")
        df.to_parquet(full_path, index=False)
