from __future__ import annotations
import os, logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict
import yaml, pandas as pd, wrds
t
logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOGLEVEL","INFO"))


PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR    = Path(os.getenv("DATA_DIR",    PROJECT_ROOT / "data"))
SCHEMA_DIR  = Path(os.getenv("SCHEMA_DIR",  PROJECT_ROOT / "schemas"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
SCHEMA_DIR.mkdir(parents=True, exist_ok=True)

class WRDSDataIngestor:
    """Base class for WRDS data ingestion with common utilities."""

    def __init__(self, wrds_username: str | None = None):
        self.conn = wrds.Connection(wrds_username=wrds_username)  # uses .pgpass if present

    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb):
        try: self.conn.close()
        finally: return False

    @staticmethod
    def date_range(days_back: int = 90) -> tuple[str,str]:
        end = datetime.now(); start = end - timedelta(days=days_back)
        return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')

    @staticmethod
    def save_data(df: pd.DataFrame, filename: str, fmt: str = "parquet") -> Path:
        path = DATA_DIR / f"{filename}.{fmt}"
        if fmt == "parquet": df.to_parquet(path, index=False)
        elif fmt == "csv":   df.to_csv(path, index=False)
        else: raise ValueError("fmt must be parquet or csv")
        logger.info("Saved %s rows → %s", len(df), path)
        return path

    @staticmethod
    def save_schema(schema: Dict, dataset_name: str) -> Path:
        path = SCHEMA_DIR / f"{dataset_name}_schema.yaml"
        with open(path, "w") as f: yaml.safe_dump(schema, f, sort_keys=False)
        logger.info("Saved schema → %s", path)
        return path
