from __future__ import annotations
from typing import Dict, Iterable, List, Optional
import pandas as pd
import numpy as np
from ._wrds_base import WRDSDataIngestor

class BetaIngestor(WRDSDataIngestor):
    """
    Rolling CAPM beta and idiosyncratic volatility from CRSP + Fama-French daily factors.
    Produces tidy (date, permno, beta_252, resid_vol_252) for each security.
    """

    # noinspection SqlNoDataSourceInspection,SqlDialectInspection
    def fetch_crsp_daily(self, start_date: str, end_date: str, permnos: Optional[Iterable[int]] = None) -> pd.DataFrame:
        where_ids = ""
        if permnos:
            ids = ",".join(str(int(p)) for p in permnos)
            where_ids = f" AND a.permno IN ({ids}) "

        q = f"""
        SELECT a.permno, a.date, a.ret
        FROM crsp.dsf a
        WHERE a.date BETWEEN '{start_date}' AND '{end_date}'
        {where_ids}
        ORDER BY a.permno, a.date
        """
        df = self.conn.raw_sql(q)
        # ret often comes as float already; ensure numeric
        df["ret"] = pd.to_numeric(df["ret"], errors="coerce")
        return df

    # noinspection SqlNoDataSourceInspection,SqlDialectInspection
    def fetch_ff_factors_daily(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Assumes WRDS library 'ff' (Fama-French) with daily factors, columns: date, mktrf, smb, hml, rf.
        If your site uses a different table name, parameterize it here.
        """
        q = f"""
        SELECT date, mktrf, rf
        FROM ff.factors_daily
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY date
        """
        ff = self.conn.raw_sql(q)
        ff["mktrf"] = pd.to_numeric(ff["mktrf"], errors="coerce")
        ff["rf"]    = pd.to_numeric(ff["rf"], errors="coerce")
        return ff

    def compute_rolling_beta(self, crsp_df: pd.DataFrame, ff_df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
        """
        CAPM beta via rolling Cov(R_i - Rf, MKT-Rf)/Var(MKT-Rf).
        Also returns residual (idiosyncratic) volatility over the window.
        """
        if crsp_df.empty or ff_df.empty:
            return pd.DataFrame()

        # Merge on date
        df = crsp_df.merge(ff_df, on="date", how="inner")
        df["excess_ret"] = df["ret"] - df["rf"]

        # pre-compute market stats
        df = df.sort_values(["permno", "date"])
        df["mktrf_var"] = df["mktrf"].rolling(window).var()

        # Rolling covariance per permno
        def _roll_cov(x):
            return x["excess_ret"].rolling(window).cov(x["mktrf"])

        df["cov_ex_mkt"] = df.groupby("permno", group_keys=False).apply(_roll_cov)

        # Beta
        df["beta_capm_252"] = df["cov_ex_mkt"] / df["mktrf_var"].replace(0.0, np.nan)

        # Residuals from simple CAPM (using rolling beta & zero alpha approx)
        # For a tighter approach, you could refit OLS per window; this is fast & close.
        df["residual"] = df["excess_ret"] - df["beta_capm_252"] * df["mktrf"]
        df["idio_vol_252"] = df.groupby("permno")["residual"].rolling(window).std().reset_index(level=0, drop=True)

        out = df[["date", "permno", "beta_capm_252", "idio_vol_252"]].dropna().reset_index(drop=True)
        return out

    def fetch_and_compute(
        self, start_date: str, end_date: str, permnos: Optional[List[int]] = None, window: int = 252
    ) -> pd.DataFrame:
        crsp = self.fetch_crsp_daily(start_date, end_date, permnos=permnos)
        ff   = self.fetch_ff_factors_daily(start_date, end_date)
        return self.compute_rolling_beta(crsp, ff, window=window)

    @staticmethod
    def schema_doc() -> Dict:
        return {
            "dataset": "Rolling Beta (CRSP + Fama-French)",
            "library": {"prices": "crsp.dsf", "factors": "ff.factors_daily"},
            "date_field": "date",
            "identifier_fields": ["permno"],
            "fields": {
                "beta_capm_252": "Rolling 252-day CAPM beta vs (MKT-RF)",
                "idio_vol_252": "Rolling 252-day residual standard deviation"
            },
            "notes": [
                "Returns are excess (RET - RF)",
                "Window length configurable (default 252 trading days)"
            ],
            "common_queries": [
                "Names with surging idiosyncratic volatility",
                "Large changes in rolling beta over 90 days"
            ]
        }
