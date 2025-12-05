import pandas as pd
import numpy as np
from datetime import timedelta

from ._wrds_base import WRDSDataIngestor


class CRSPIngestor(WRDSDataIngestor):
    """
    Ingestor for CRSP monthly stock data joined with the security names file.

    - Uses crsp.msf (monthly stock file) for prices/returns.
    - Joins crsp.dsenames for ticker + CUSIP with date-validity logic.
    - Filters to common shares on NYSE / AMEX / NASDAQ.
    """

    library = "crsp"

    # ------------------------------------------------------------------
    # 1) Date window helper used by the orchestrator
    # ------------------------------------------------------------------
    def get_valid_window(self, days_back: int = 365) -> tuple[str, str]:
        """
        Return a (start_date, end_date) tuple as YYYY-MM-DD strings
        aligned to the latest date actually available in CRSP.msf.

        This prevents asking WRDS for data beyond the last CRSP load
        (which would return 0 rows).
        """
        # Find the max available date in the monthly stock file
        q = "SELECT MAX(date) AS max_date FROM crsp.msf"
        df = self.conn.raw_sql(q, date_cols=["max_date"])

        max_date = df["max_date"].iloc[0]
        if pd.isna(max_date):
            # Fallback: just use today-ish if something is very wrong
            end = pd.Timestamp.today().normalize()
        else:
            end = pd.to_datetime(max_date).normalize()

        start = end - timedelta(days=days_back)

        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    # If you still use get_date_range elsewhere, you can keep this alias:
    def get_date_range(self, days_back: int = 365) -> tuple[str, str]:
        return self.get_valid_window(days_back)

    # ------------------------------------------------------------------
    # 2) Actual data fetch: msf + dsenames
    # ------------------------------------------------------------------
    # noinspection SqlDialectInspection,SqlNoDataSourceInspection
    def fetch_monthly_with_names(self, start: str, end: str) -> pd.DataFrame:
        """
        Fetch monthly CRSP data with ticker & cusip from msf + dsenames.

        Returns at least:
          permno, permco, date, price, ret, volume, ticker, cusip,
          shrcd, exchcd, namedt, nameendt
        """
        q = f"""
        SELECT
            msf.permno,
            msf.permco,
            msf.date,
            msf.prc        AS price,
            msf.ret        AS ret,
            msf.vol        AS volume,
            names.ticker,
            names.ncusip   AS cusip,
            names.shrcd,
            names.exchcd,
            names.namedt,
            names.nameendt
        FROM crsp.msf AS msf
        LEFT JOIN crsp.dsenames AS names
          ON msf.permno = names.permno
         AND names.namedt <= msf.date
         AND msf.date   <= names.nameendt
        WHERE msf.date BETWEEN '{start}' AND '{end}'
          AND names.exchcd IN (1, 2, 3)   -- NYSE / AMEX / NASDAQ
          AND names.shrcd  IN (10, 11)    -- common shares
        ORDER BY msf.permno, msf.date
        """

        df = self.conn.raw_sql(
            q,
            date_cols=["date", "namedt", "nameendt"],
        )

        df["ticker"] = df["ticker"].astype(str).str.strip()
        df["cusip"] = df["cusip"].astype(str).str.strip()

        for col in ["price", "ret", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    # ------------------------------------------------------------------
    # 3) Backwards-compatibility aliases
    # ------------------------------------------------------------------
    def fetch_daily_stock_data(self, start: str, end: str) -> pd.DataFrame:
        """
        Kept for backward compatibility with older code that expects
        a 'daily' method name. It actually returns monthly msf data.
        """
        return self.fetch_monthly_with_names(start, end)

    def fetch_if_needed(self, rel_path: str, start: str, end: str) -> pd.DataFrame:
        """
        Cache-aware fetch: look for a parquet under data_dir / rel_path*.
        If not present, fetch from WRDS and save.
        """
        from pathlib import Path
        import os

        rel = rel_path.replace("\\", "/")
        subdir = Path(self.data_dir) / Path(rel).parent
        subdir.mkdir(parents=True, exist_ok=True)

        fname = f"{Path(rel).name}_{start}_{end}.parquet"
        fpath = subdir / fname

        if fpath.exists():
            return pd.read_parquet(fpath)

        df = self.fetch_monthly_with_names(start, end)
        if not df.empty:
            df.to_parquet(fpath, index=False)

        return df

    # ------------------------------------------------------------------
    # 4) Schema doc for your LLM/Text2SQL layer
    # ------------------------------------------------------------------
    @staticmethod
    def get_schema_documentation() -> dict:
        return {
            "dataset": "CRSP Monthly Stock File (with names)",
            "library": "crsp",
            "primary_table": "msf",
            "date_field": "date",
            "identifier_fields": ["permno", "permco", "ticker", "cusip"],
            "key_metrics": {
                "price": "CRSP monthly price (prc, signed)",
                "ret": "CRSP monthly return (ret)",
                "volume": "Monthly trading volume (vol)",
            },
            "notes": [
                "Joined crsp.msf with crsp.dsenames on permno + date validity.",
                "Filtered to common shares (shrcd 10,11) on NYSE/AMEX/NASDAQ (exchcd 1,2,3).",
                "Use get_valid_window(days_back) to get a CRSP-consistent date range.",
            ],
        }
