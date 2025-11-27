from __future__ import annotations

import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional

import yaml
import pandas as pd
import wrds


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
# reuse the same logger across data ingestors
logger = logging.getLogger("wrds_ingestor")

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(os.getenv("LOGLEVEL", "INFO"))


# ------------------------------------------------------------------
# Base WRDS ingestor
# ------------------------------------------------------------------
class WRDSDataIngestor:
    """
    Base class for WRDS data ingestion with:
      - WRDS connection management
      - local caching via data_dir
      - schema saving via schema_dir
      - convenience date-range helpers
    """

    def __init__(
        self,
        wrds_username: Optional[str] = None,
        data_dir: Optional[str | Path] = None,
        schema_dir: Optional[str | Path] = None,
    ):
        # Open WRDS connection (uses ~/.pgpass if present)
        self.conn = wrds.Connection(wrds_username=wrds_username)

        # --------- data_dir resolution ---------
        # Priority:
        # 1. explicit data_dir argument
        # 2. DATA_DIR env var
        # 3. PROJECT_ROOT/data (where PROJECT_ROOT can come from env or be inferred)
        project_root_default = Path(__file__).resolve().parents[2]
        project_root = Path(os.getenv("PROJECT_ROOT", project_root_default))

        if data_dir is not None:
            self.data_dir = Path(data_dir)
        else:
            env_data_dir = os.getenv("DATA_DIR")
            self.data_dir = Path(env_data_dir) if env_data_dir else project_root / "data"

        self.data_dir.mkdir(parents=True, exist_ok=True)

        # --------- schema_dir resolution ---------
        if schema_dir is not None:
            self.schema_dir = Path(schema_dir)
        else:
            env_schema_dir = os.getenv("SCHEMA_DIR")
            self.schema_dir = (
                Path(env_schema_dir) if env_schema_dir else project_root / "schemas"
            )

        self.schema_dir.mkdir(parents=True, exist_ok=True)

    # --------- context manager support ---------
    def __enter__(self) -> "WRDSDataIngestor":
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.conn.close()
        finally:
            # return False so exceptions are NOT suppressed
            return False

    # --------- path + caching helpers ---------
    def _resolve_path(self, name: str) -> Path:
        """
        Convert a logical dataset name like 'wrds/compustat_quarterly'
        into a concrete file path under data_dir, defaulting to .parquet.
        """
        path = self.data_dir / name
        if not path.suffix:
            path = path.with_suffix(".parquet")
        return path

    def data_exists(self, name: str) -> bool:
        """Check if a cached dataset already exists on disk."""
        return self._resolve_path(name).exists()

    def load_data(self, name: str) -> pd.DataFrame:
        """Load a cached dataset from disk as a DataFrame."""
        path = self._resolve_path(name)
        logger.info("Loading cached data from %s", path)
        return pd.read_parquet(path)

    def save_data(self, df: pd.DataFrame, name: str) -> Path:
        """
        Save a dataset under data_dir using a logical name like
        'wrds/crsp_daily' or 'wrds/compustat_quarterly'.
        """
        path = self._resolve_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info("Saved %s rows → %s", len(df), path)
        return path

    # Optional generic file saver if you ever need explicit filenames/formats
    def save_file(self, df: pd.DataFrame, filename: str, fmt: str = "parquet") -> Path:
        """
        Save to a specific filename (no logical name resolution) under data_dir.
        Mostly for ad-hoc exports; ingestion caching should use save_data().
        """
        path = self.data_dir / f"{filename}.{fmt}"
        path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "parquet":
            df.to_parquet(path, index=False)
        elif fmt == "csv":
            df.to_csv(path, index=False)
        else:
            raise ValueError("fmt must be 'parquet' or 'csv'")

        logger.info("Saved file → %s", path)
        return path

    # --------- date helpers ---------
    @staticmethod
    def get_date_range(days_back: int = 90) -> tuple[str, str]:
        """
        Convenience helper: returns (start_date, end_date) as 'YYYY-MM-DD' strings
        for a backward-looking window.
        """
        end = datetime.now()
        start = end - timedelta(days=days_back)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    # --------- schema helper ---------
    def save_schema(self, schema: Dict, dataset_name: str) -> Path:
        """
        Save a YAML schema file under schema_dir, e.g.
        <schema_dir>/wrds_compustat_schema.yaml
        """
        path = self.schema_dir / f"{dataset_name}_schema.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(schema, f, sort_keys=False)
        logger.info("Saved schema → %s", path)
        return path
