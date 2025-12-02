from __future__ import annotations
import os
from pathlib import Path
import sqlite3

import pandas as pd


def get_project_root() -> Path:
    """
    Heuristic: assume this file lives in applications/market_anomalies/,
    so project root is two levels up.
    """
    here = Path(__file__).resolve()
    return here.parents[2]  # .../Agentics/


def build_master_db():
    project_root = get_project_root()
    data_dir = project_root / "applications" / "data"
    wrds_dir = data_dir / "wrds"
    wrds_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load core WRDS parquet files ---

    # CRSP monthly (or 'crsp_daily' if you used that name for msf)
    # Adjust filename if your parquet is named slightly differently.
    crsp_path = next(wrds_dir.glob("crsp_*.parquet"), None)
    if crsp_path is None:
        raise FileNotFoundError(f"No CRSP parquet found in {wrds_dir}")

    crsp = pd.read_parquet(crsp_path)

    # Compustat quarterly
    comp_path = next(wrds_dir.glob("compustat_quarterly*.parquet"), None)
    if comp_path is None:
        raise FileNotFoundError(f"No Compustat parquet found in {wrds_dir}")
    comp = pd.read_parquet(comp_path)

    # IBES EPS summary
    ibes_eps_path = next(wrds_dir.glob("ibes_eps_summary*.parquet"), None)
    if ibes_eps_path is None:
        raise FileNotFoundError(f"No IBES EPS parquet found in {wrds_dir}")
    ibes_eps = pd.read_parquet(ibes_eps_path)

    # IBES recommendations
    ibes_recs_path = next(wrds_dir.glob("ibes_recommendations*.parquet"), None)
    if ibes_recs_path is None:
        raise FileNotFoundError(f"No IBES recommendations parquet found in {wrds_dir}")
    ibes_recs = pd.read_parquet(ibes_recs_path)

    # CIQ key developments (optional but useful for gvkey/ticker crosswalk)
    ciq_path = next(wrds_dir.glob("ciq_keydev*.parquet"), None)
    ciq = pd.read_parquet(ciq_path) if ciq_path is not None else pd.DataFrame()

    # --- 2. Normalize identifiers in each dataset ---

    # CRSP: we only need unique permno/(permco?) and maybe ticker if present
    crsp_ids_cols = [c for c in ["permno", "permco", "ticker", "cusip"] if c in crsp.columns]
    crsp_ids = crsp[crsp_ids_cols].drop_duplicates().copy()
    # make IDs strings for safer joins
    for c in crsp_ids_cols:
        crsp_ids[c] = crsp_ids[c].astype(str)

    # Compustat: gvkey + ticker
    comp_ids = comp[[c for c in ["gvkey", "ticker"] if c in comp.columns]].drop_duplicates().copy()
    for c in comp_ids.columns:
        comp_ids[c] = comp_ids[c].astype(str)

    # IBES EPS: ticker + cusip
    ibes_eps_ids = ibes_eps[[c for c in ["ticker", "cusip"] if c in ibes_eps.columns]].drop_duplicates().copy()
    for c in ibes_eps_ids.columns:
        ibes_eps_ids[c] = ibes_eps_ids[c].astype(str)

    # IBES Recs: ticker + cusip (if present)
    ibes_recs_ids = ibes_recs[[c for c in ["ticker", "cusip"] if c in ibes_recs.columns]].drop_duplicates().copy()
    for c in ibes_recs_ids.columns:
        ibes_recs_ids[c] = ibes_recs_ids[c].astype(str)

    # CIQ: gvkey + ticker
    if not ciq.empty:
        ciq_ids = ciq[[c for c in ["gvkey", "ticker"] if c in ciq.columns]].drop_duplicates().copy()
        for c in ciq_ids.columns:
            ciq_ids[c] = ciq_ids[c].astype(str)
    else:
        ciq_ids = pd.DataFrame(columns=["gvkey", "ticker"])

    # --- 3. Merge identifier spaces into a security_master ---

    # Start from CRSP permno (your main time series anchor)
    security_master = crsp_ids.copy()

    # Bring in CIQ gvkey/ticker (permno ←→ gvkey via shared ticker, if present)
    # First, merge CRSP & CIQ on ticker (if ticker exists in both)
    if "ticker" in security_master.columns and not ciq_ids.empty:
        security_master = security_master.merge(
            ciq_ids,
            on="ticker",
            how="left",
            suffixes=("", "_ciq"),
        )

    # Bring in Compustat info (gvkey/ticker)
    # We try both joins:
    #   1) on ticker (if present),
    #   2) on gvkey (if already filled from CIQ)
    if "ticker" in security_master.columns and "ticker" in comp_ids.columns:
        security_master = security_master.merge(
            comp_ids.rename(columns={"ticker": "ticker_comp"}),
            left_on="ticker",
            right_on="ticker_comp",
            how="left",
        )

    if "gvkey" in security_master.columns and "gvkey" in comp_ids.columns:
        security_master = security_master.merge(
            comp_ids.rename(columns={"ticker": "ticker_comp2"}),
            on="gvkey",
            how="left",
            suffixes=("", "_comp2"),
        )

    # Bring in IBES IDs (ticker/cusip)
    # Here ticker alignment is the main bridge
    if "ticker" in security_master.columns and "ticker" in ibes_eps_ids.columns:
        security_master = security_master.merge(
            ibes_eps_ids.rename(columns={"cusip": "ibes_cusip"}),
            on="ticker",
            how="left",
        )

    # Also try join on cusip (if CRSP had cusip)
    if "cusip" in security_master.columns and "cusip" in ibes_eps_ids.columns:
        security_master = security_master.merge(
            ibes_eps_ids.rename(columns={"ticker": "ibes_ticker_from_cusip"}),
            on="cusip",
            how="left",
            suffixes=("", "_from_cusip"),
        )

    # Drop obvious duplicate helper cols and tidy up
    # Keep a minimal but useful set of columns
    keep_cols = []
    for c in [
        "permno",
        "permco",
        "ticker",
        "cusip",
        "gvkey",
        "ticker_comp",
        "ibes_cusip",
        "ibes_ticker_from_cusip",
    ]:
        if c in security_master.columns:
            keep_cols.append(c)

    security_master = security_master[keep_cols].drop_duplicates().reset_index(drop=True)

    # Simple de-dup heuristic: if ticker_comp exists and ticker is NaN, fill ticker from ticker_comp
    if "ticker" in security_master.columns and "ticker_comp" in security_master.columns:
        security_master["ticker"] = security_master["ticker"].fillna(security_master["ticker_comp"])

    # Same for cusip vs ibes_cusip
    if "cusip" in security_master.columns and "ibes_cusip" in security_master.columns:
        security_master["cusip"] = security_master["cusip"].fillna(security_master["ibes_cusip"])

    # Drop helper columns if present
    security_master = security_master.drop(
        columns=[c for c in ["ticker_comp", "ibes_cusip", "ibes_ticker_from_cusip"] if c in security_master.columns],
        errors="ignore",
    )

    # --- 4. Save as Parquet ---
    master_path = wrds_dir / "master_db.parquet"
    security_master.to_parquet(master_path, index=False)
    print(f"✅ Saved security_master parquet to {master_path} ({len(security_master)} rows).")

    # --- 5. Also store as SQLite table for the app ---
    db_path = os.getenv("SQL_DB_PATH", str(data_dir / "market_anomalies.db"))
    with sqlite3.connect(db_path) as conn:
        security_master.to_sql("security_master", conn, if_exists="replace", index=False)
    print(f"✅ Saved security_master as SQLite table 'security_master' in {db_path}.")


if __name__ == "__main__":
    build_master_db()
