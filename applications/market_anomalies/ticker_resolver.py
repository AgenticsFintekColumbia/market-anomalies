import sqlite3
from datetime import datetime
import pandas as pd
from pathlib import Path
from typing import Optional, Dict

class TickerResolver:
    def __init__(self, db_path: str, master_parquet: Optional[str] = None):
        self.db_path = str(Path(db_path))
        db_dir = Path(self.db_path).resolve().parent
        default_master = db_dir / "wrds" / "master_db.parquet"
        master_path = Path(master_parquet) if master_parquet else default_master

        if not master_path.exists():
            raise FileNotFoundError(f"master_db.parquet not found at {master_path}")

        # Load CRSP master (dsenames-based) mapping
        self.master = pd.read_parquet(master_path)

        # Normalize
        self.master["ticker"] = self.master["ticker"].astype(str).str.upper().str.strip()
        self.master["permno"] = self.master["permno"].astype(str)
        self.master["cusip"] = self.master["cusip"].astype(str)
        self.master["namedt"] = pd.to_datetime(self.master["namedt"])
        self.master["nameendt"] = pd.to_datetime(self.master["nameendt"])

    def _get_latest_crsp_date(self) -> str:
        with sqlite3.connect(self.db_path) as conn:
            row = pd.read_sql_query("SELECT MAX(date) AS max_date FROM crsp_daily", conn)
        return str(row["max_date"].iloc[0])

    def resolve(self, ticker: str, as_of: Optional[str] = None) -> Dict[str, Optional[str]]:
        t = ticker.strip().upper()
        if as_of is None:
            as_of = self._get_latest_crsp_date()
        as_of_dt = datetime.fromisoformat(str(as_of))

        # Filter by ticker & active name period
        df = self.master[self.master["ticker"] == t].copy()
        if df.empty:
            return {"ticker": t, "permno": None, "cusip": None}

        active = df[
            (df["namedt"] <= as_of_dt)
            & ((df["nameendt"].isna()) | (df["nameendt"] >= as_of_dt))
        ]

        if active.empty:
            # fall back to the latest name
            active = df.sort_values("namedt").tail(1)

        row = active.iloc[0]
        return {
            "ticker": t,
            "permno": row["permno"],
            "cusip": row["cusip"],
            "gvkey": row["gvkey"],
        }
