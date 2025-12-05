from __future__ import annotations
import wrds
import pandas as pd
from pathlib import Path

# noinspection SqlNoDataSourceInspection,SqlDialectInspection
def build_master_db(wrds_username: str):
    conn = wrds.Connection(wrds_username=wrds_username)
    q = """
        SELECT msf.permno,
               msf.permco,
               names.ticker,
               names.ncusip AS cusip,
               msf.date,
               msf.ret,
               msf.prc      AS price,
               msf.vol      AS volume,
               names.shrcd,
               names.exchcd,
               names.namedt,
               names.nameendt,
               names.comnam AS company_name
        FROM crsp.msf AS msf
        LEFT JOIN crsp.dsenames AS names
          ON msf.permno = names.permno
         AND names.namedt <= msf.date
         AND msf.date <= names.nameendt
        WHERE names.exchcd IN (1, 2, 3)
          AND names.shrcd IN (10, 11)
        ORDER BY msf.permno, msf.date;
    """
    print("Fetching full CRSP security master with price data...")
    df = conn.raw_sql(q)
    conn.close()

    df["permno"] = df["permno"].astype(int).astype(str)
    df["permco"] = df["permco"].astype(int).astype(str)
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["cusip"] = df["cusip"].astype(str).str.strip()
    df["namedt"] = pd.to_datetime(df["namedt"])
    df["nameendt"] = pd.to_datetime(df["nameendt"])

    project_root = Path(__file__).resolve().parents[2]
    wrds_dir = project_root / "applications" / "data" / "wrds"
    wrds_dir.mkdir(parents=True, exist_ok=True)

    out_path = wrds_dir / "master_db.parquet"
    df.to_parquet(out_path, index=False)
    print(f"✅ Saved master_db.parquet with {len(df)} rows to {out_path}")



if __name__ == "__main__":
    build_master_db("andreaoclark")
