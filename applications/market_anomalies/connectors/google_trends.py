import logging
import pandas as pd
from pytrends.request import TrendReq
import os
import warnings

logger = logging.getLogger(__name__)


class GoogleTrendsIngestor:
    """Public attention signal via Google Trends."""

    source = "google_trends"

    def __init__(self, hl: str = "en-US", tz: int = 360):
        """
        hl: host language (e.g., 'en-US')
        tz: timezone offset in minutes (e.g., 360 for UTC-6)
        """
        self.trends = TrendReq(hl=hl, tz=tz)

    # Optional context manager support
    def __enter__(self):
        """No WRDS connection required; just return the instance."""
        return self

    def __exit__(self, exc_type, exc, tb):
        """Nothing special to clean up."""
        return False

    def fetch_interest(
        self,
        keywords: list[str],
        start_date,
        end_date,
    ) -> pd.DataFrame:
        """
        Fetch Google Trends interest-over-time and return tidy long format:
        columns = [date, ticker, trend_score, source]
        """
        logger.info(
            "GoogleTrendsIngestor: fetching trends for %s from %s to %s",
            keywords, start_date, end_date
        )

        # Build request payload
        self.trends.build_payload(
            kw_list=keywords,
            timeframe=f"{start_date} {end_date}",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self.trends.build_payload(
                kw_list=keywords,
                timeframe=f"{start_date} {end_date}",
            )

        df = self.trends.interest_over_time().reset_index()
        df = df.drop(columns=["isPartial"], errors="ignore")

        # Ensure date column is timezone-naive
        if pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = df["date"].dt.tz_localize(None)

        # Convert wide → long format
        df_long = df.melt(
            id_vars=["date"],
            var_name="ticker",
            value_name="trend_score"
        )
        df_long["source"] = self.source

        return df_long[["date", "ticker", "trend_score", "source"]]

    # Generic interface for orchestrator
    def fetch(self, keywords: list[str], start_date, end_date) -> pd.DataFrame:
        return self.fetch_interest(keywords, start_date, end_date)

    def save_data(self, df: pd.DataFrame, path: str):
        """
        Save data to data/<path>.parquet, creating directories if necessary.
        Example path: "external/google_trends"
        """
        full_path = f"data/{path}.parquet"
        parent_dir = os.path.dirname(full_path)

        # Create parent directory if missing
        if not os.path.exists(parent_dir):
            logger.info(f"Creating directory {parent_dir}")
            os.makedirs(parent_dir, exist_ok=True)

        logger.info(f"Saving Google Trends data to {full_path}")
        df.to_parquet(full_path, index=False)
