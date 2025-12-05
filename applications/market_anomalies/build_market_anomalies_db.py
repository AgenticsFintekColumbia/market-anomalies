from pathlib import Path
import os
import sqlite3
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # 'applications'
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
WRDS_DIR = DATA_DIR / "wrds"

DB_PATH = DATA_DIR / "market_anomalies.db"

def load_parquet(name: str) -> pd.DataFrame:
    """
    Load a parquet file from applications/data/wrds/ using either:
      - exact match: name="crsp_daily" → crsp_daily.parquet
      - prefix match: name="crsp_daily_" → crsp_daily_*.parquet

    Returns a pandas DataFrame.
    """

    # first try finding an exact file match
    exact_path = WRDS_DIR / f"{name}.parquet"
    if exact_path.exists():
        print("Loading:", exact_path)
        return pd.read_parquet(exact_path)

    # otherwise treat name as a prefix (the CRSP parquet)
    pattern = f"{name}*.parquet"
    matches = list(WRDS_DIR.glob(pattern))

    if not matches:
        raise FileNotFoundError(f"No parquet files match prefix '{name}' in {WRDS_DIR}")

    # choose the most recent date match for CRSP
    matches = sorted(matches)
    chosen = matches[-1]

    print(f"Loading latest matching file: {chosen}")
    return pd.read_parquet(chosen)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    WRDS_DIR.mkdir(parents=True, exist_ok=True)

    print("Using DATA_DIR:", DATA_DIR)
    print("SQLite DB path:", DB_PATH)

    conn = sqlite3.connect(DB_PATH.as_posix())

    crsp = load_parquet("crsp_daily")
    crsp.to_sql("crsp_daily", conn, if_exists="replace", index=False)

    comp = load_parquet("compustat_quarterly")
    comp.to_sql("compustat_quarterly", conn, if_exists="replace", index=False)

    ibes_eps = load_parquet("ibes_eps_summary")
    ibes_eps.to_sql("ibes_eps_summary", conn, if_exists="replace", index=False)

    ibes_rec = load_parquet("ibes_recommendations")
    ibes_rec.to_sql("ibes_recommendations", conn, if_exists="replace", index=False)

    ciq = load_parquet("ciq_keydev")
    ciq.to_sql("ciq_keydev", conn, if_exists="replace", index=False)

    # this is the connecting parquet to link everything together!
    master = load_parquet("master_db")
    master.to_sql("crsp_master", conn, if_exists="replace", index=False)

    conn.commit()
    conn.close()
    print(f"✅ Built SQLite DB at {DB_PATH} (including crsp_master)")


if __name__ == "__main__":
    main()
