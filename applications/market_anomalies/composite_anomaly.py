from __future__ import annotations
from ticker_resolver import TickerResolver
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import is_dataclass, asdict


import numpy as np
import pandas as pd
import sqlite3
import logging
logger = logging.getLogger(__name__)


# ============================================================
# Config + small Bayesian helper
# ============================================================

@dataclass
class AnomalyConfig:
    """Configuration for CRSP-based anomaly detection."""
    vol_window: int = 6      # works with 12 monthly obs
    ret_lookback: int = 6
    min_obs: int = 3         # don't require 252 obs for monthly data


@dataclass
class ComponentWeights:
    """Weights for composite anomaly score."""
    w_crsp: float = 0.35
    w_compustat: float = 0.20
    w_ibes_eps: float = 0.20
    w_ibes_recs: float = 0.15
    w_ciq: float = 0.07
 

class BayesianConfidenceAssessment:
    """
    Tiny helper for mapping z-scores to anomaly probabilities.
    This is intentionally simple; you can swap in your richer
    Bayesian logic later.
    """

    @staticmethod
    def z_to_two_sided_p(z: float) -> float:
        """Approximate two-sided p-value for a z-score."""
        # use a simple normal CDF approximation
        z = float(z)
        # guard
        if np.isnan(z):
            return 1.0
        # tail prob ~ exp(-z^2/2) / (|z|*sqrt(2π)), here we just use exp part
        tail = np.exp(-0.5 * z * z)
        # two-sided, but we squash to [0,1]
        return float(min(1.0, max(0.0, 2 * tail)))

    @staticmethod
    def anomaly_from_p(p: float) -> float:
        """
        Map p-value in [0,1] to anomaly probability in [0,1].
        Small p → high anomaly.
        """
        p = float(np.clip(p, 1e-6, 1.0))
        # simple transform: anomaly = 1 - sqrt(p)
        return float(1.0 - np.sqrt(p))


# ============================================================
# CRSP core: permno-based monthly anomaly features
# ============================================================

class CRSPAnomalyCore:
    """
    Computes CRSP-based anomaly features for a security (by permno):
      - monthly return
      - rolling volatility
      - z-scores vs baseline
      - per-period CRSP anomaly score & normalized price index
    """

    def __init__(self, db_path: str, config: Optional[AnomalyConfig] = None):
        self.db_path = str(Path(db_path).expanduser().resolve())
        self.config = config or AnomalyConfig()
        self.bayes = BayesianConfidenceAssessment()

    def _load_crsp_window(self, permno: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load CRSP monthly data for a permno, including a long history window.

        Assumes SQLite table 'crsp_daily' actually stores your CRSP
        monthly stock file (msf-like) with columns such as:
          - date
          - permno
          - final_ret (preferred) or ret
        """
        if end_date is None:
            with sqlite3.connect(self.db_path) as conn:
                row = pd.read_sql_query("SELECT MAX(date) AS max_date FROM crsp_daily", conn)
            end_date = row["max_date"].iloc[0]

        end_dt = datetime.fromisoformat(str(end_date))
        start_dt = end_dt - timedelta(days=self.config.history_days)
        start_str = start_dt.strftime("%Y-%m-%d")
        end_str = end_dt.strftime("%Y-%m-%d")

        q = """
        SELECT *
        FROM crsp_daily
        WHERE permno = ?
          AND date BETWEEN ? AND ?
        ORDER BY date
        """
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(q, conn, params=(permno, start_str, end_str))

        if df.empty:
            return df

        df["date"] = pd.to_datetime(df["date"])
        df["permno"] = df["permno"].astype(str)

        # Choose return column: prefer final_ret, fallback to ret
        cols_lower = {c.lower(): c for c in df.columns}
        if "final_ret" in cols_lower:
            ret_col = cols_lower["final_ret"]
        elif "ret" in cols_lower:
            ret_col = cols_lower["ret"]
        else:
            raise ValueError("CRSP table has no 'final_ret' or 'ret' column.")

        df["ret"] = pd.to_numeric(df[ret_col], errors="coerce")

        return df

    def compute_crsp_features(self, permno: str | int) -> pd.DataFrame:
        """
        Pull CRSP time series for a given permno from `crsp_daily`
        and compute simple anomaly features (rolling volatility + z-scores).

        We build a synthetic price index from returns.
        """
        p_int = int(permno)

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                """
                SELECT date, permno, ret
                FROM crsp_daily
                WHERE permno = ?
                ORDER BY date
                """,
                conn,
                params=(p_int,),
            )

        if df.empty:
            return df

        # Ensure datetime + sorted
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Clean returns
        df["ret"] = pd.to_numeric(df["ret"], errors="coerce").fillna(0.0)

        # --- Synthetic price index (starts at 1.0) ---
        df["price_index"] = (1.0 + df["ret"]).cumprod()

        # If too few observations, return basic series with neutral anomaly
        if len(df) < self.config.min_obs:
            df["volatility"] = np.nan
            df["ret_z"] = np.nan
            df["vol_z"] = np.nan
            df["anomaly_crsp"] = 0.5
            return df

        # Rolling volatility
        df["volatility"] = (
            df["ret"]
            .rolling(self.config.vol_window, min_periods=self.config.min_obs)
            .std()
        )

        # z-scores
        vol_mean = df["volatility"].mean(skipna=True)
        vol_std = df["volatility"].std(skipna=True)
        df["vol_z"] = (df["volatility"] - vol_mean) / (vol_std + 1e-10)

        ret_mean = df["ret"].mean(skipna=True)
        ret_std = df["ret"].std(skipna=True)
        df["ret_z"] = (df["ret"] - ret_mean) / (ret_std + 1e-10)

        # map vol_z into [0,1] anomaly-ish score
        df["anomaly_crsp"] = 0.5 + 0.5 * np.tanh(df["vol_z"].fillna(0.0))

        return df


# ============================================================
# Composite detector using master_db.parquet
# ============================================================

class CompositeAnomalyDetector:
    ...

    def __init__(
        self,
        db_path: str,
        master_parquet_path: Optional[str] = None,
        config: Optional[AnomalyConfig] = None,
        weights: Optional[ComponentWeights] = None,
    ):
        # Convert to absolute path ALWAYS
        self.db_path = str(Path(db_path).expanduser().resolve())

        # Parent dir (project root discovery)
        db_dir = Path(self.db_path).resolve().parent

        self.config = config or AnomalyConfig()
        self.weights = weights or ComponentWeights()
        self.crsp_core = CRSPAnomalyCore(self.db_path, self.config)

        # --- Centralized CRSP ID resolver ---
        self.ticker_resolver = TickerResolver(
            self.db_path,
            master_parquet_path
        )

        # to have access to the raw master mapping
        self.master = self.ticker_resolver.master


    # ---------------------------
    # ID resolution
    # ---------------------------


    def resolve_ids(self, ticker: str) -> Dict[str, Optional[str]]:
        t = ticker.strip().upper()

        # 1) First try CRSP/master_db via TickerResolver
        ids_crsp = self.ticker_resolver.resolve(t)

        ids: Dict[str, Optional[str]] = {
            "ticker": ids_crsp.get("ticker", t),
            "permno": ids_crsp.get("permno"),
            "cusip": ids_crsp.get("cusip"),
            "gvkey": ids_crsp.get("gvkey"),
        }

        # 2) If we still don't have permno/gvkey, try IBES → CUSIP → master_db
        try:
            needs_permno = ids["permno"] is None
            needs_gvkey = ids["gvkey"] is None
            needs_cusip = ids["cusip"] is None

            if needs_permno or needs_gvkey or needs_cusip:
                with sqlite3.connect(self.db_path) as conn:
                    info = pd.read_sql_query("PRAGMA table_info(ibes_eps_summary);", conn)
                    if not info.empty:
                        cols = {c.lower(): c for c in info["name"]}

                        tic_col = cols.get("ticker") or cols.get("tic")
                        cusip_col = cols.get("cusip")

                        if tic_col and cusip_col:
                            q = f"""
                                SELECT {tic_col} AS ticker,
                                    {cusip_col} AS cusip
                                FROM ibes_eps_summary
                                WHERE UPPER({tic_col}) = ?
                                ORDER BY estimate_date DESC
                                LIMIT 1
                            """
                            ibes_map = pd.read_sql_query(q, conn, params=(t,))
                            if not ibes_map.empty:
                                cusip = str(ibes_map.loc[0, "cusip"]).strip()
                                if not ids["cusip"]:
                                    ids["cusip"] = cusip

                                # Now map CUSIP → permno/gvkey via master_db
                                master = self.ticker_resolver.master
                                # master["cusip"] is already normalized to str
                                rows = master[master["cusip"] == cusip]
                                if not rows.empty:
                                    # pick the most recent name
                                    row = rows.sort_values("namedt").tail(1).iloc[0]
                                    if ids["permno"] is None and "permno" in row.index:
                                        ids["permno"] = str(row["permno"])
                                    if ids["gvkey"] is None and "gvkey" in row.index and pd.notna(row["gvkey"]):
                                        ids["gvkey"] = str(row["gvkey"])
        except Exception as e:
            logger.warning("[resolve_ids] IBES→CUSIP fallback failed for %s: %s", t, e)

        logger.info("[resolve_ids] %s -> %s", t, ids)
        return ids


    # ---------------------------
    # Dataset-specific loaders
    # ---------------------------

    def _load_latest_compustat(self, gvkey: Optional[str]) -> Optional[pd.Series]:
        if gvkey is None:
            return None
        q = """
        SELECT *
        FROM compustat_quarterly
        WHERE gvkey = ?
        ORDER BY datadate DESC
        LIMIT 1
        """
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(q, conn, params=(gvkey,))
        if df.empty:
            return None
        return df.iloc[0]

    def _load_latest_ibes_eps(self, ticker: str) -> Optional[pd.Series]:
        """
        Load the most recent IBES EPS summary row for a given ticker,
        based on `estimate_date` in `ibes_eps_summary`.
        """
        t = ticker.strip().upper()

        q = """
            SELECT *
            FROM ibes_eps_summary
            WHERE UPPER(ticker) = ?
            ORDER BY estimate_date DESC LIMIT 1 \
            """
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(q, conn, params=(t,))

        if df.empty:
            return None

        return df.iloc[0]

    def _load_recent_ibes_recs(
            self, ticker: str, window_days: int = 365
    ) -> Optional[pd.DataFrame]:
        """
        Load recent IBES recommendations for a ticker from ibes_recommendations.

        We DO NOT assume the date column is called 'anndats'.
        We inspect the table and pick a likely date column:
        - 'anndats'          (raw WRDS)
        - 'announce_date'    (common rename)
        - 'recommendation_date'
        - 'date'
        """
        t = ticker.strip().upper()

        with sqlite3.connect(self.db_path) as conn:
            info = pd.read_sql_query("PRAGMA table_info(ibes_recommendations);", conn)
            colnames = {c.lower(): c for c in info["name"]}

            # Pick a date column
            if "anndats" in colnames:
                date_col = colnames["anndats"]
            elif "announce_date" in colnames:
                date_col = colnames["announce_date"]
            elif "recommendation_date" in colnames:
                date_col = colnames["recommendation_date"]
            elif "date" in colnames:
                date_col = colnames["date"]
            else:
                # No obvious date column → just return all recs for this ticker
                df = pd.read_sql_query(
                    "SELECT * FROM ibes_recommendations WHERE UPPER(ticker) = ?",
                    conn,
                    params=(t,),
                )
                return df if not df.empty else None

            # Find the latest date for this ticker
            q_max = f"""
                SELECT MAX({date_col}) AS max_date
                FROM ibes_recommendations
                WHERE UPPER(ticker) = ?
            """
            max_df = pd.read_sql_query(q_max, conn, params=(t,))
            if max_df.empty or max_df["max_date"].isna().all():
                return None

            latest = pd.to_datetime(max_df.loc[0, "max_date"])
            cutoff = latest - pd.Timedelta(days=window_days)

            # Pull window of recs
            q = f"""
                SELECT *
                FROM ibes_recommendations
                WHERE UPPER(ticker) = ?
                  AND {date_col} BETWEEN ? AND ?
                ORDER BY {date_col} DESC
            """
            df = pd.read_sql_query(
                q,
                conn,
                params=(
                    t,
                    cutoff.strftime("%Y-%m-%d"),
                    latest.strftime("%Y-%m-%d"),
                ),
            )

        return df if not df.empty else None

    def _load_recent_ibes_recs(self, ticker: str, window_days: int = 180) -> Optional[pd.DataFrame]:
        """
        Load IBES recommendation records for a ticker within a recent time window.
        We DO NOT assume the column is called 'anndats' — we detect the correct date column.
        """
        t = ticker.strip().upper()

        with sqlite3.connect(self.db_path) as conn:
            # --- 1) Inspect schema ---
            info = pd.read_sql_query("PRAGMA table_info(ibes_recommendations);", conn)
            colnames = {c.lower(): c for c in info["name"]}

            # --- 2) Choose best available date column ---
            if "anndats" in colnames:
                date_col = colnames["anndats"]
            elif "announce_date" in colnames:
                date_col = colnames["announce_date"]
            elif "recommendation_date" in colnames:
                date_col = colnames["recommendation_date"]
            elif "date" in colnames:
                date_col = colnames["date"]
            else:
                # No recognizable date column → fallback: return all recs for ticker
                df = pd.read_sql_query(
                    "SELECT * FROM ibes_recommendations WHERE UPPER(ticker) = ?",
                    conn,
                    params=(t,),
                )
                return df if not df.empty else None

            # --- 3) Determine available max date ---
            q_max = f"""
                SELECT MAX({date_col}) AS max_date
                FROM ibes_recommendations
                WHERE UPPER(ticker) = ?
            """
            max_row = pd.read_sql_query(q_max, conn, params=(t,))
            max_date = max_row["max_date"].iloc[0]

            if max_date is None:
                return None

            end_date = pd.to_datetime(max_date)
            start_date = end_date - timedelta(days=window_days)

            # --- 4) Pull window of recs ---
            q = f"""
                SELECT *
                FROM ibes_recommendations
                WHERE UPPER(ticker) = ?
                  AND {date_col} BETWEEN ? AND ?
                ORDER BY {date_col} DESC
            """

            df = pd.read_sql_query(
                q,
                conn,
                params=(
                    t,
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                ),
            )

        return df if not df.empty else None

    def _load_recent_ciq_events(
            self, gvkey: Optional[str], window_days: int = 365
    ) -> Optional[pd.DataFrame]:
        """
        Load recent CIQ key developments for a company identified by gvkey.

        We try to infer a date column and then keep only events in a trailing window.
        """
        if gvkey is None:
            return None

        with sqlite3.connect(self.db_path) as conn:
            # Inspect schema to find likely date column
            info = pd.read_sql_query("PRAGMA table_info(ciq_keydev);", conn)
            if info.empty:
                return None

            colnames = {c.lower(): c for c in info["name"]}

            if "announce_date" in colnames:
                date_col = colnames["announce_date"]
            elif "date" in colnames:
                date_col = colnames["date"]
            elif "event_date" in colnames:
                date_col = colnames["event_date"]
            else:
                # No obvious date column
                return None

            q = f"""
                SELECT *
                FROM ciq_keydev
                WHERE gvkey = ?
            """
            df = pd.read_sql_query(q, conn, params=(gvkey,))

        if df.empty or date_col not in df.columns:
            return None

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        if df.empty:
            return None

        latest_date = df[date_col].max()
        window_start = latest_date - timedelta(days=window_days)
        mask = df[date_col].between(window_start, latest_date)
        window_df = df.loc[mask]

        return window_df if not window_df.empty else None


    # ---------------------------
    # Component scoring helpers
    # ---------------------------
    @staticmethod
    def _logistic(x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-x)))

    @staticmethod
    def _score_crsp_from_row(row: pd.Series) -> float:
        """
        Turn CRSP volatility z-scores into a single [0,1] anomaly score.

        Assumes `row["vol_z"]` is a z-score of recent volatility vs its own history.
        Mapping (two-sided on |z|):
          - |z| <= 0.5  → ~0.0  (very normal)
          - |z| = 1.5   → ~0.33
          - |z| = 2.0   → ~0.5
          - |z| >= 3.0  → 1.0   (strong volatility anomaly)
        """
        z = row.get("vol_z", 0.0)

        try:
            z = float(z)
        except (TypeError, ValueError):
            return 0.0  # if it's completely unusable, treat as no anomaly

        if np.isnan(z):
            return 0.0  # "no CRSP anomaly" rather than 0.5

        mag = abs(z)

        # Ignore tiny noise: anything below 0.5 std dev → 0 anomaly
        if mag <= 0.5:
            return 0.0

        # Clamp at 3 standard deviations
        mag_clamped = min(mag, 3.0)

        # Linearly map [0.5, 3.0] → [0.0, 1.0]
        # shift by 0.5 so 0.5 → 0, 3.0 → 1
        score = (mag_clamped - 0.5) / (3.0 - 0.5)
        return float(np.clip(score, 0.0, 1.0))


    def _score_compustat(self, row: Optional[pd.Series]) -> Optional[float]:
        """
        Score fundamentals anomaly using Compustat margin z-scores.

        Heuristic:
            - reads `net_margin_z` and `ebitda_margin_z` (defaults to 0)
            - uses max(|z|) and maps it into [0,1] with a smooth curve
        """

        if row is None or not isinstance(row, pd.Series) or row.empty:
            return None

        z_net = float(row.get("net_margin_z", 0.0))
        z_ebitda = float(row.get("ebitda_margin_z", 0.0))

        # anomaly based on magnitude of deviation from sector average
        z_mag = max(abs(z_net), abs(z_ebitda))

        # map |z| into [0,1] with a smooth curve
        score = 1.0 - np.exp(- (z_mag / 2.0) ** 2)
        return float(np.clip(score, 0.0, 1.0))


    def _score_ibes_eps(self, row: Optional[pd.Series]) -> Optional[float]:
        """
        Score latest EPS row into an anomaly score in [0,1].

        Heuristic:
            - earnings_surprise = (actual - consensus) / |consensus|
            - Large negative surprise => high anomaly
            - Large positive surprise => lower anomaly
            - Higher analyst coverage => more confident score
        """
        if row is None or not isinstance(row, pd.Series) or row.empty:
            return None

        # if actual EPS is available, use surprise as anomaly
        actual = row.get("actual_eps")
        if pd.notna(actual):
            consensus = row.get("consensus_estimate")
            if pd.notna(consensus) and consensus != 0:
                surprise = (actual - consensus) / abs(consensus)
                # map surprise magnitude into [0,1]
                return float(np.clip(abs(surprise), 0.0, 1.0))

        # Otherwise, fall back to dispersion as a measure of "uncertainty anomaly"
        est_std = row.get("estimate_std")
        num_analysts = row.get("num_analysts")

        if pd.isna(est_std) or pd.isna(num_analysts) or num_analysts < 3:
            return None

        # higher dispersion -> higher anomaly
        dispersion = float(est_std)
        score = 1.0 - np.exp(-dispersion * 10.0)  # scale factor tunable
        return float(np.clip(score, 0.0, 1.0))


    @staticmethod
    def _score_ibes_recs(df: Optional[pd.DataFrame]) -> Optional[float]:
        """
        Turn a recent recommendations window into an anomaly score in [0,1].

        Heuristic:
          - recommendation_code: 1=Strong Buy ... 5=Sell (higher = worse)
          - More bearish on average => higher anomaly.
        """
        if df is None or df.empty:
            return None

        cols = list(df.columns)
        lower_map = {c.lower(): c for c in cols}

        # 1) Try exact / strong matches first
        if "recommendation_code" in lower_map:
            rec_col = lower_map["recommendation_code"]
        elif "ireccd" in lower_map:
            rec_col = lower_map["ireccd"]
        elif "reccd" in lower_map:
            rec_col = lower_map["reccd"]
        else:
            # 2) Fallback: any column with "code" in the name, but NOT date / ticker
            candidates = [
                c
                for c in cols
                if ("code" in c.lower())
                   and ("date" not in c.lower())
                   and ("time" not in c.lower())
                   and ("ticker" not in c.lower())
            ]
            if not candidates:
                return None
            rec_col = candidates[0]

        # Convert that column to numeric safely
        codes = pd.to_numeric(df[rec_col], errors="coerce").dropna()
        if codes.empty:
            return None

        # Map average code from [1,5] -> [0,1]
        avg_code = codes.mean()
        score = (avg_code - 1.0) / 4.0
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _score_ciq_events(df: Optional[pd.DataFrame]) -> Optional[float]:
        """
        Turn a window of CIQ events into an anomaly score in [0,1].

        Simple Poisson-like intensity: more events in the window => higher anomaly.
        """
        if df is None or df.empty:
            return None

        n_events = len(df)

        # Treat 2 events per window as 'typical' and scale around that.
        typical = 2.0
        lam = n_events / typical
        score = 1.0 - float(np.exp(-lam))  # 0 events ~0, many events ~1

        return float(np.clip(score, 0.0, 1.0))

    # ---------------------------
    # Public API
    # ---------------------------

    def compute_composite_for_ticker(
            self, ticker: str
    ) -> Tuple[pd.DataFrame, float, Dict[str, Any]]:
        """
        Given a TICKER, build:
          - CRSP time series (if available),
          - Compustat / IBES static components (if available),
          - a composite anomaly score in [0,1],
          - a components dict for interpretability.

        Never hard-fails just because one component is missing.
        """
        ids = self.resolve_ids(ticker)
        permno = ids.get("permno")
        gvkey = ids.get("gvkey")
        resolved_ticker = ids.get("ticker")

        # --------------------------------------------------
        # 1) CRSP time series (OPTIONAL)
        # --------------------------------------------------
        ts = pd.DataFrame()
        crsp_score = 0.5  # neutral if no CRSP
        crsp_details: Dict[str, Any] = {
            "latest_date": None,
            "vol_z": None,
            "ret_z": None,
            "anomaly_crsp": None,
            "note": "CRSP not used (no permno mapping or no time series rows).",
        }

        if permno is not None:
            ts_candidate = self.crsp_core.compute_crsp_features(permno)
            if ts_candidate is not None and not ts_candidate.empty:
                ts = ts_candidate
                latest_row = ts.iloc[-1]
                # helper you define; simple mapping from vol_z to [0,1]
                crsp_score = float(
                    0.5 + 0.5 * np.tanh(latest_row.get("vol_z", 0.0))
                )
                crsp_details = {
                    "latest_date": str(
                        getattr(latest_row["date"], "date", lambda: latest_row["date"])()
                    ),
                    "vol_z": float(latest_row.get("vol_z", np.nan)),
                    "ret_z": float(latest_row.get("ret_z", np.nan)),
                    "anomaly_crsp": crsp_score,
                    "note": None,
                }
            else:
                crsp_details["note"] = f"No CRSP rows for permno {permno} in crsp_daily."

        # --------------------------------------------------
        # 2) Compustat (OPTIONAL, via gvkey)
        # --------------------------------------------------
        comp_row = self._load_latest_compustat(gvkey) if gvkey is not None else None
        comp_score = self._score_compustat(comp_row)

        # --------------------------------------------------
        # 3) IBES EPS (OPTIONAL, via ticker)
        # --------------------------------------------------
        ibes_eps_row = (
            self._load_latest_ibes_eps(resolved_ticker) if resolved_ticker else None
        )
        ibes_eps_score = self._score_ibes_eps(ibes_eps_row)

        # --------------------------------------------------
        # 4) IBES Recs (OPTIONAL, via ticker)
        # --------------------------------------------------
        ibes_recs_df = (
            self._load_recent_ibes_recs(resolved_ticker) if resolved_ticker else None
        )
        ibes_recs_score = self._score_ibes_recs(ibes_recs_df)

        # --------------------------------------------------
        # 5) CIQ Key Developments (OPTIONAL, via gvkey)
        # --------------------------------------------------
        ciq_df = self._load_recent_ciq_events(gvkey) if gvkey is not None else None
        ciq_score = self._score_ciq_events(ciq_df)

     

        # --------------------------------------------------
        # Composite score: use ONLY available (non-None) components
        # --------------------------------------------------
        w = self.weights

        component_scores = {
            "crsp": crsp_score,
            "compustat": comp_score,
            "ibes_eps": ibes_eps_score,
            "ibes_recs": ibes_recs_score,
            "ciq": ciq_score
        }
        component_weights = {
            "crsp": w.w_crsp,
            "compustat": w.w_compustat,
            "ibes_eps": w.w_ibes_eps,
            "ibes_recs": w.w_ibes_recs,
            "ciq": w.w_ciq
        }

        # Filter to sources that actually have a score
        available_keys = [k for k, s in component_scores.items() if s is not None]

        if available_keys:
            total_w = sum(component_weights[k] for k in available_keys)
            latest_composite = sum(
                component_weights[k] * component_scores[k] for k in available_keys
            ) / total_w
        else:
            latest_composite = None  # no usable signals


        if not ts.empty:
            ts = ts.copy()
            crsp_series = ts.get("anomaly_crsp", pd.Series(0.5, index=ts.index)).fillna(0.5)

            total_w = (
                w.w_crsp
                + w.w_compustat
                + w.w_ibes_eps
                + w.w_ibes_recs
                + w.w_ciq
            )


            if not ts.empty:
                ts = ts.copy()
                crsp_series = ts.get("anomaly_crsp", pd.Series(index=ts.index, dtype=float))

                w = self.weights

                # numerator and denominator for weighted average
                num = 0.0
                den = 0.0

                # CRSP: time-varying series
                if crsp_series is not None and not crsp_series.isna().all():
                    num = num + w.w_crsp * crsp_series.fillna(0.0)
                    den = den + w.w_crsp

                # Compustat: scalar score, include only if present
                if comp_score is not None:
                    num = num + w.w_compustat * comp_score
                    den = den + w.w_compustat

                # IBES EPS
                if ibes_eps_score is not None:
                    num = num + w.w_ibes_eps * ibes_eps_score
                    den = den + w.w_ibes_eps

                # IBES Recs
                if ibes_recs_score is not None:
                    num = num + w.w_ibes_recs * ibes_recs_score
                    den = den + w.w_ibes_recs

                # CIQ events
                if ciq_score is not None:
                    num = num + w.w_ciq * ciq_score
                    den = den + w.w_ciq


                if den > 0:
                    ts["composite_score"] = num / den
                else:
                    ts["composite_score"] = np.nan


        components: Dict[str, Any] = {
            "ids": ids,
            "crsp": crsp_details,
            "compustat": {
                "score": float(comp_score) if comp_score is not None else None,
                "raw_row": comp_row.to_dict()
                if isinstance(comp_row, pd.Series)
                else None,
            },
            "ibes_eps": {
                "score": float(ibes_eps_score)
                if ibes_eps_score is not None
                else None,
                "raw_row": ibes_eps_row.to_dict()
                if isinstance(ibes_eps_row, pd.Series)
                else None,
            },
            "ibes_recs": {
                "score": float(ibes_recs_score)
                if ibes_recs_score is not None
                else None,
                "num_records": int(len(ibes_recs_df))
                if isinstance(ibes_recs_df, pd.DataFrame)
                else 0,
            },
            "ciq": {
                "score": float(ciq_score) if ciq_score is not None else None,
                "num_events": int(len(ciq_df))
                if isinstance(ciq_df, pd.DataFrame)
                else 0,
            },
            "weights": {
                "crsp": w.w_crsp,
                "compustat": w.w_compustat,
                "ibes_eps": w.w_ibes_eps,
                "ibes_recs": w.w_ibes_recs,
                "ciq": w.w_ciq
            },
            "latest_composite_score": float(latest_composite),
            }

        return ts, latest_composite, components

    def _weights_as_dict(self) -> dict:
        """
        Safely turn self.weights into a plain dict, whether it's
        a Pydantic model, dataclass, or simple namespace.
        """
        w = self.weights

        # Pydantic v2
        if hasattr(w, "model_dump"):
            return w.model_dump()

        # Pydantic v1 style
        if hasattr(w, "dict"):
            return w.dict()

        # Dataclass
        if is_dataclass(w):
            return asdict(w)

        # Already a dict-like
        try:
            return dict(w)
        except Exception:
            # Last resort: introspect __dict__
            return {
                k: getattr(w, k)
                for k in dir(w)
                if not k.startswith("_") and isinstance(getattr(w, k), (int, float))
            }
