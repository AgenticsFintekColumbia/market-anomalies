import streamlit as st
import pandas as pd
import asyncio
import aiosqlite
import sqlite3
import os
import sys
import json
import yaml
import logging
import glob
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, ConfigDict, Field
from composite_anomaly import CompositeAnomalyDetector


# --- 1. Page Config (MUST BE FIRST) ---
st.set_page_config(page_title="WRDS Market Anomaly Hunter", page_icon="📈", layout="wide")


# --- 2. Environment & Setup ---
def setup_path_and_env():
    """
    Robustly finds the project root and the 'agentics' library.
    Traverses up the directory tree to find the folder containing 'agentics' or 'src/agentics'.
    """
    current_file = Path(__file__).resolve()
    current_dir = current_file.parent

    found_lib_path = None
    project_root = None

    # Search up to 4 levels up
    for i in range(5):
        # Handle current dir (i=0) and parents
        if i == 0:
            candidate = current_dir
        else:
            # safety check for index out of bounds
            if i - 1 < len(current_dir.parents):
                candidate = current_dir.parents[i - 1]
            else:
                break

        # Check 1: Is 'agentics' directly here?
        if (candidate / "agentics").is_dir():
            found_lib_path = candidate
            project_root = candidate
            break

        # Check 2: Is 'src/agentics' here?
        if (candidate / "src" / "agentics").is_dir():
            found_lib_path = candidate / "src"
            project_root = candidate
            break

        # Check 3: Is .env here? (Store as potential root, but keep looking for lib)
        if (candidate / ".env").exists() and project_root is None:
            project_root = candidate

    # If we found the library path, add it to sys.path
    if found_lib_path:
        lib_str = str(found_lib_path)
        if lib_str not in sys.path:
            sys.path.insert(0, lib_str)  # Insert at 0 to prioritize local source over installed packages
            print(f"✅ Added to sys.path: {lib_str}")

    # Load .env from project root if found
    if project_root:
        from dotenv import load_dotenv
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)

    return project_root


PROJECT_ROOT = setup_path_and_env()

# Import Agentics
try:
    # Try importing directly (standard notebook usage)
    from agentics import Agentics as AG
except ImportError:
    try:
        # Fallback for some versions/forks
        from agentics import AG
    except ImportError:
        st.error(
            f"⚠️ `agentics` library is missing.\n\nProject Root detected: `{PROJECT_ROOT}`\n\nPlease ensure the `agentics` folder is in that directory.")
        st.stop()

# --- 3. Database Configuration ---
# Use the environment variable or fall back to your specific path
if PROJECT_ROOT:
    default_db_path = PROJECT_ROOT / "applications" / "data" / "market_anomalies.db"
else:
    # Fallback default
    default_db_path = Path("market_anomalies.db")

DB_PATH = os.getenv("SQL_DB_PATH", str(default_db_path))


# --- 4. Data Orchestrator Integration ---

def build_database(db_target_path: str):
    """
    Orchestrates fetching of WRDS data via connectors and building the SQLite DB.
    Prioritizes existing Parquet files in applications/data/wrds to avoid re-fetching.
    """
    status_container = st.status("🏗️ Building Database from Data...", expanded=True)

    try:
        # Ensure data directory exists
        db_path_obj = Path(db_target_path)
        # Assuming structure: applications/data/market_anomalies.db
        # Data is in: applications/data/wrds/
        data_dir = db_path_obj.parent
        wrds_data_dir = data_dir / "wrds"

        db_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Function to find latest parquet file for a given prefix
        def find_latest_parquet(prefix):
            if not wrds_data_dir.exists():
                return None
            # Search for files like crsp_daily_2020-01-01_2023-12-31.parquet
            files = list(wrds_data_dir.glob(f"{prefix}*.parquet"))
            # Also check exact matches if the date stamping wasn't used yet
            if not files:
                exact_match = wrds_data_dir / f"{prefix}.parquet"
                if exact_match.exists():
                    return exact_match
                return None

            # Sort by modification time to get the newest
            files.sort(key=os.path.getmtime, reverse=True)
            return files[0]

        # Load Config for credentials (only needed if we fetch fresh)
        config_path = Path(__file__).resolve().parent / "config.yaml"
        config = None
        if config_path.exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

        with sqlite3.connect(db_target_path) as conn:

            # --- 1. CRSP ---
            status_container.write("📥 Checking CRSP Daily Stock data...")
            parquet_file = find_latest_parquet("crsp_daily")

            if parquet_file:
                status_container.write(f"📂 Found cached CRSP: {parquet_file.name}")
                df = pd.read_parquet(parquet_file)
                df.to_sql("crsp_daily", conn, if_exists="replace", index=False)
                status_container.write(f"✅ CRSP loaded from cache ({len(df)} rows)")
            else:
                # Fallback to fetching
                if not config:
                    status_container.error("Config missing, cannot fetch CRSP.")
                else:
                    status_container.write("🌐 Fetching CRSP from WRDS (Cache missing)...")
                    from connectors.crsp import CRSPIngestor
                    with CRSPIngestor(config["wrds"]["username"]) as crsp:
                        s, e = crsp.get_date_range(90)
                        crsp_df = crsp.fetch_if_needed("wrds/crsp_daily", s, e)
                        if crsp_df is not None and not crsp_df.empty:
                            crsp_df.to_sql("crsp_daily", conn, if_exists="replace", index=False)
                            status_container.write(f"✅ CRSP saved ({len(crsp_df)} rows)")
                        else:
                            status_container.warning("⚠️ CRSP fetch returned empty.")

            # --- 2. Compustat ---
            status_container.write("📥 Checking Compustat Quarterly data...")
            parquet_file = find_latest_parquet("compustat_quarterly")

            if parquet_file:
                status_container.write(f"📂 Found cached Compustat: {parquet_file.name}")
                df = pd.read_parquet(parquet_file)
                df.to_sql("compustat_quarterly", conn, if_exists="replace", index=False)
                status_container.write(f"✅ Compustat loaded from cache ({len(df)} rows)")
            else:
                if config:
                    from connectors.compustat import CompustatIngestor
                    with CompustatIngestor(config["wrds"]["username"]) as comp:
                        s, e = comp.get_date_range(180)
                        comp_df = comp.fetch_if_needed("wrds/compustat_quarterly", s, e)
                        if comp_df is not None:
                            comp_df.to_sql("compustat_quarterly", conn, if_exists="replace", index=False)
                            status_container.write(f"✅ Compustat saved")

            # --- 3. IBES EPS ---
            status_container.write("📥 Checking IBES Estimates...")
            parquet_file = find_latest_parquet("ibes_eps_summary")
            if parquet_file:
                df = pd.read_parquet(parquet_file)
                df.to_sql("ibes_eps_summary", conn, if_exists="replace", index=False)
                status_container.write(f"✅ IBES EPS loaded from cache")
            elif config:
                from connectors.ibes import IBESIngestor
                with IBESIngestor(config["wrds"]["username"]) as ibes:
                    s, e = ibes.get_date_range(365 * 5)
                    eps_df = ibes.fetch_eps_if_needed("wrds/ibes_eps_summary", s, e)
                    if eps_df is not None:
                        eps_df.to_sql("ibes_eps_summary", conn, if_exists="replace", index=False)

            # --- 4. IBES Recommendations ---
            parquet_file = find_latest_parquet("ibes_recommendations")
            if parquet_file:
                df = pd.read_parquet(parquet_file)
                df.to_sql("ibes_recommendations", conn, if_exists="replace", index=False)
                status_container.write(f"✅ IBES Recs loaded from cache")
            # (Skipping fetch fallback for brevity, follows same pattern)

            # --- 5. CIQ ---
            status_container.write("📥 Checking Capital IQ data...")
            parquet_file = find_latest_parquet("ciq_keydev")
            if parquet_file:
                df = pd.read_parquet(parquet_file)
                df.to_sql("ciq_keydev", conn, if_exists="replace", index=False)
                status_container.write(f"✅ CIQ loaded from cache")
            elif config:
                from connectors.ciq import CIQIngestor
                with CIQIngestor(config["wrds"]["username"]) as ciq:
                    s, e = ciq.get_date_range(365)
                    df_ciq = ciq.fetch_if_needed("wrds/ciq_keydev", start=s, end=e, event_ids=(16, 81, 232),
                                                 only_primary_us_tickers=True)
                    if df_ciq is not None:
                        df_ciq.to_sql("ciq_keydev", conn, if_exists="replace", index=False)

        status_container.update(label="✅ Database Built Successfully!", state="complete", expanded=False)
        return True

    except Exception as e:
        status_container.update(label="❌ Orchestration Failed", state="error")
        st.error(f"Error building database: {e}")
        import traceback
        st.code(traceback.format_exc())
        return False


# --- 5. Scaffolding: Data Models ---
class Text2sqlQuestion(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    question: Optional[str] = None
    db_id: Optional[str] = None
    query: Optional[str] = None
    reasoning_type: Optional[str] = None
    commonsense_knowledge: Optional[str] = None
    # RENAMED schema -> db_schema to avoid conflict with Pydantic internal attribute
    db_schema: Optional[str] = None
    generated_query: Optional[str] = Field(
        None, description="The query generated by AI"
    )
    system_output_df: Optional[str] = None
    gt_output_df: Optional[str] = None
    final_report: Optional[str] = Field(
        None, description="The final natural language report answering the user's question"
    )

class AnomalySummaryState(BaseModel):
    ticker: str
    composite_score: float
    components_json: str  # JSON dump of the components dict to load
    summary_markdown: Optional[str] = Field(
        default=None,
        description="Human-readable anomaly summary to show in the UI.",
    )



# --- 6. Helper Functions ---
@st.cache_data(show_spinner="Reading Schema...")
def get_schema_cached(db_path):
    """
    Extracts schema from SQLite DB. Cached to prevent re-reading on every click.
    """
    if not os.path.exists(db_path):
        return {"error": f"Database not found at {db_path}"}

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        schema_json = {}
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name});")
            schema = cursor.fetchall()

            schema_json[table_name] = {
                col[1]: {
                    "type": col[2],
                    "notnull": col[3],
                    "dflt_value": col[4],
                }
                for col in schema
            }
        conn.close()
        return schema_json
    except Exception as e:
        return {"error": str(e)}


def get_data_range_stats(db_path):
    """
    Helper to find the time range of the data.
    Crucial for helping the LLM understand it is looking at historical data.
    """
    if not os.path.exists(db_path):
        return "DB not found.", "1900-01-01"

    stats = []
    max_date_overall = "1900-01-01"  # Default old date

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        for (table_name,) in tables:
            # Heuristic: find columns with 'date' in the name
            cursor.execute(f"PRAGMA table_info({table_name});")
            cols = [c[1] for c in cursor.fetchall()]
            date_cols = [c for c in cols if 'date' in c.lower() or 'time' in c.lower()]

            if date_cols:
                # Pick the first date column to check range
                target_col = date_cols[0]
                cursor.execute(f"SELECT MIN({target_col}), MAX({target_col}), COUNT(*) FROM {table_name}")
                min_d, max_d, count = cursor.fetchone()
                stats.append(f"Table '{table_name}': {count} rows. Range: {min_d} to {max_d} (Column: {target_col})")

                # Track max date found
                if max_d and str(max_d) > max_date_overall:
                    max_date_overall = str(max_d)
            else:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                stats.append(f"Table '{table_name}': {count} rows. (No date column identified)")

        conn.close()
    except Exception as e:
        return f"Error getting stats: {e}", "2024-01-01"

    return "\n".join(stats), max_date_overall


async def async_execute_sql(sql_query: str, db_path: str) -> str:
    """
    Executes SQL asynchronously using aiosqlite.

    Guardrails:
    - Rewrites DATE('now') / CURRENT_DATE to use the latest date present in the DB,
      so we never ask for rows beyond the data's time coverage.
    """
    if not sql_query:
        return "Error: No query generated"

    # Clean up markdown code blocks if the LLM included them
    cleaned_query = sql_query.replace("```sql", "").replace("```", "").strip()

    # Get the actual max date from the DB
    _, max_db_date = get_data_range_stats(db_path)
    anchor_date = max_db_date or "1900-01-01"

    # Guardrail rewrites for dynamic date functions
    # Handle DATE('now', ...) → DATE('<anchor>', ...)
    cleaned_query = cleaned_query.replace("DATE('now'", f"DATE('{anchor_date}'")
    cleaned_query = cleaned_query.replace("date('now'", f"DATE('{anchor_date}'")

    # Handle bare DATE('now')
    cleaned_query = cleaned_query.replace("DATE('now')", f"DATE('{anchor_date}')")
    cleaned_query = cleaned_query.replace("date('now')", f"DATE('{anchor_date}')")

    # CURRENT_DATE → DATE('<anchor>')
    cleaned_query = cleaned_query.replace("CURRENT_DATE", f"DATE('{anchor_date}')")
    cleaned_query = cleaned_query.replace("current_date", f"DATE('{anchor_date}')")

    try:
        async with aiosqlite.connect(db_path) as db:
            # Also normalize double quotes to single quotes for SQLite
            sql_to_run = cleaned_query.replace('"', "'")
            async with db.execute(sql_to_run) as cursor:
                try:
                    columns = [description[0] for description in cursor.description]
                except TypeError:
                    return "[]"  # Handle cases with no return (DDL, etc.)

                rows = await asyncio.wait_for(cursor.fetchall(), timeout=10)
                df = pd.DataFrame(rows, columns=columns)
                return df.to_json(orient='records')
    except Exception as e:
        return f"Error: {str(e)}"



# --- 7. Maps (Manual Call) ---

async def get_schema_map(state: Text2sqlQuestion) -> Text2sqlQuestion:
    # 1. Get raw SQLite structure (truth on ground)
    raw_schema = get_schema_cached(DB_PATH)
    data_stats, max_db_date = get_data_range_stats(DB_PATH)

    # 2. Get rich documentation from ALL Connectors
    rich_docs = {}
    try:
        # Try dynamic relative imports first (if running from same folder)
        try:
            from connectors.crsp import CRSPIngestor
            from connectors.compustat import CompustatIngestor
            from connectors.ibes import IBESIngestor
            from connectors.ciq import CIQIngestor
        except ImportError:
            # Fallback for nested module structures
            from applications.market_anomalies.connectors.crsp import CRSPIngestor
            from applications.market_anomalies.connectors.compustat import CompustatIngestor
            from applications.market_anomalies.connectors.ibes import IBESIngestor
            from applications.market_anomalies.connectors.ciq import CIQIngestor

        # Retrieve docs
        if hasattr(CRSPIngestor, 'get_schema_documentation'):
            rich_docs["crsp_daily"] = CRSPIngestor.get_schema_documentation()

        if hasattr(CompustatIngestor, 'get_schema_documentation'):
            rich_docs["compustat_quarterly"] = CompustatIngestor.get_schema_documentation()

        if hasattr(IBESIngestor, 'get_schema_documentation'):
            rich_docs["ibes_eps_summary"] = IBESIngestor.get_schema_documentation()
            rich_docs["ibes_recommendations"] = IBESIngestor.get_schema_documentation()

        if hasattr(CIQIngestor, 'get_schema_documentation'):
            rich_docs["ciq_keydev"] = CIQIngestor.get_schema_documentation()

    except Exception as e:
        # Non-critical: Proceed with raw schema if connectors fail to load
        print(f"Schema Enrichment Warning: {e}")
        pass

    # 3. Domain Knowledge Cheatsheet (Anomaly Detection)
    anomaly_cheatsheet = """
    **Market Anomaly Guide:**
    1. **Momentum:** Look for stocks with high returns (`ret`) over past 6-12 months in `crsp_daily`.
    2. **Post-Earnings Announcement Drift (PEAD):** Look for positive earnings surprises. Compare `compustat_quarterly.eps` (actual) vs `ibes_eps_summary.meanest` (consensus). Join on ticker/date.
    3. **Value/Reversal:** Low P/E or P/B ratios, or recent significant price drops. Use `compustat_quarterly` for book value/earnings and `crsp_daily` for market cap/price.
    4. **Analyst Sentiment:** Look for rating upgrades in `ibes_recommendations`.
    5. **Corporate Events:** Significant events in `ciq_keydev` (e.g. M&A, buybacks).
    """

    # 4. Combine structural schema with semantic documentation
    full_context = {
        "sqlite_database_structure": raw_schema,
        "dataset_semantic_documentation": rich_docs,
        "data_availability_summary": data_stats,
        "latest_data_date": max_db_date,
        "domain_knowledge": anomaly_cheatsheet,
        "general_guidance": f"""
        - **IMPORTANT: The database contains historical data ending on {max_db_date}.**
        - **CRITICAL RULE:** DO NOT use `DATE('now')` or `DATE('now', '-3 months')` because the database has NO future data.
        - **INSTEAD USE:** `DATE('{max_db_date}')` as the anchor. For "last 3 months", use `DATE('{max_db_date}', '-3 months')`.
        - Use 'dataset_semantic_documentation' to understand column meanings.
        - Use 'sqlite_database_structure' for exact column names.
        """
    }

    state.db_schema = json.dumps(full_context, indent=2)
    return state


async def execute_query_map(state: Text2sqlQuestion) -> Text2sqlQuestion:
    """Step 3: Execute the generated SQL"""
    if state.generated_query:
        state.system_output_df = await async_execute_sql(state.generated_query, DB_PATH)
    return state

async def generate_anomaly_summary(
    ticker: str,
    composite_score: float,
    components: Dict[str, Any],
) -> str:
    """
    Use Agentics + Gemini to compress composite anomaly info into a short, readable report.
    """
    payload = {
        "ticker": ticker,
        "composite_score": composite_score,
        "ids": components.get("ids"),
        "crsp": components.get("crsp"),
        "compustat": components.get("compustat"),
        "ibes_eps": components.get("ibes_eps"),
        "ibes_recs": components.get("ibes_recs"),
        "weights": components.get("weights", {}),
        "latest_composite_score": components.get("latest_composite_score", composite_score),
    }
    components_json = json.dumps(payload, default=str, indent=2)

    state = AnomalySummaryState(
        ticker=ticker,
        composite_score=composite_score,
        components_json=components_json,
    )

    try:
        agent = AG(states=[state], atype=AnomalySummaryState)
    except Exception:
        agent = AG()
        agent.states = [state]

    agent.llm = AG.get_llm_provider()

    agent = await agent.self_transduction(
        ["ticker", "composite_score", "components_json"],
        ["summary_markdown"],
        instructions="""
        You are a senior equity analyst writing a brief anomaly assessment for a portfolio manager.

        You receive a JSON object in `components_json` with fields:
          - ticker
          - composite_score in [0,1]
          - ids: mapping with identifiers (ticker, permno, cusip, gvkey)
          - crsp: CRSP-specific diagnostics (may include a score and/or time series stats;
                  if available, it may contain a volatility z-score under a key like `vol_z`
                  and a CRSP anomaly score in [0,1]).
          - compustat: { score, raw_row } where:
                • score in [0,1] is a fundamentals anomaly score
                • raw_row has quarterly fundamentals and sector-relative z-scores
          - ibes_eps: { score, raw_row } where:
                • score in [0,1] is an earnings-surprise anomaly score
                • raw_row has consensus_estimate, actual_eps, num_analysts, etc.
          - ibes_recs: { score, num_records } where:
                • score in [0,1] reflects bearish/bullish analyst sentiment
                • num_records is the number of recent recommendation records
          - weights: { crsp, compustat, ibes_eps, ibes_recs } giving the
                contribution of each dataset to the composite score.
          - latest_composite_score: the most recent composite anomaly.

        INTERPRETATION OF composite_score:
          • 0.0–0.3 → "low anomaly"
          • 0.3–0.6 → "moderate anomaly"
          • 0.6–0.8 → "high anomaly"
          • 0.8–1.0 → "extreme anomaly"

        IMPORTANT: The CRSP volatility anomaly score is derived from a volatility z-score (vol_z)
        using this approximate mapping from |z| to anomaly level:
          - |z| ≤ 0.5  → anomaly score ≈ 0.0  (very normal volatility)
          - |z| ≈ 1.5  → anomaly score ≈ 0.33 (mildly elevated volatility)
          - |z| ≈ 2.0  → anomaly score ≈ 0.5  (moderate volatility anomaly)
          - |z| ≥ 3.0  → anomaly score ≈ 1.0  (strong volatility anomaly, tail event)

        If a CRSP volatility z-score (e.g. `vol_z`) is present in the JSON, explicitly mention it
        and briefly relate the reported CRSP anomaly score to this mapping so the reader understands
        how "unusual" the volatility is in standard deviation units.

        TASK:
        - Produce a SHORT Markdown report (max ~10 lines) with:

          1. One-sentence headline stating overall anomaly level for the ticker
             based on composite_score.

          2. A bullet list of 3–5 key points:
             - CRSP: explain whether market behavior (returns/volatility) is normal,
               mildly unusual, or strongly unusual. If a volatility z-score is present,
               mention its approximate value (e.g., "volatility is about 2.1 standard
               deviations above its own history") and connect it to the mapping above
               (e.g., "this corresponds to a moderate volatility anomaly").
             - Compustat: interpret fundamentals anomaly from compustat.score and
               the raw fundamentals in compustat.raw_row (e.g. margins vs sector,
               leverage, profitability trends).
             - IBES EPS: describe earnings surprises and analyst coverage based on
               ibes_eps.score and ibes_eps.raw_row (consensus vs actual EPS,
               num_analysts, etc.).
             - IBES Recommendations: describe analyst sentiment based on
               ibes_recs.score and ibes_recs.num_records (e.g. more bearish,
               neutral, or supportive), if data is present.

          3. A final line explaining that the composite anomaly score is a
             weighted combination of these signals, explicitly referencing the
             weights (e.g. "most weight on CRSP and Compustat").

        IMPORTANT RULES:
        - Use qualitative language only (e.g. "elevated volatility", "mildly weak
          fundamentals", "strongly negative earnings surprise").
        - If a dataset has score = null/None, say:
             "No reliable signal from Compustat" (or the relevant source).
        - Refer to the weights to describe which sources are most influential;
          e.g., if weights.crsp is largest, say that market behavior is the
          primary driver of the anomaly.
        - DO NOT output JSON; return only well-formatted Markdown.
        """,
    )

    result_state = agent.states[0]
    if isinstance(result_state, tuple):
        result_state = result_state[0]

    return result_state.summary_markdown or "Summary generation failed."

# --- Main App Logic ---

with st.sidebar:
    st.header("⚙️ Configuration")

    # Try to get key from env, otherwise ask user
    env_api_key = os.getenv("GEMINI_API_KEY")
    if env_api_key:
        api_key = env_api_key
        st.success(f"🔑 API Key loaded from env")
    else:
        api_key = st.text_input("Gemini API Key", type="password")
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key

    st.info("Using `IB Agentics` Framework")
    st.markdown(f"**Database Path:**\n`{DB_PATH}`")

    # --- New Debug Section ---
    if os.path.exists(DB_PATH):
        with st.expander("🔎 Database Inspector", expanded=False):
            stats_text, max_d = get_data_range_stats(DB_PATH)
            st.text(stats_text)


st.title("📊 WRDS Market Anomaly Hunter")
st.markdown("Find market anomalies (Momentum, PEAD, Reversals) using natural language.")

# --- Database Check & Orchestration ---
db_needs_build = False
if not os.path.exists(DB_PATH):
    db_needs_build = True
else:
    # Intelligent check: If CRSP is missing or has 0 rows, we force a rebuild
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            # Check if crsp_daily table exists
            cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='crsp_daily'")
            if cursor.fetchone()[0] > 0:
                # Check if it has data
                cursor.execute("SELECT count(*) FROM crsp_daily")
                if cursor.fetchone()[0] == 0:
                    db_needs_build = True
            else:
                db_needs_build = True
    except Exception:
        db_needs_build = True

if db_needs_build:
    st.warning("⚠️ Database incomplete (Missing CRSP data). Attempting to build from WRDS...")

    # Run the orchestrator logic to fetch and build the DB
    success = build_database(DB_PATH)

    if success:
        st.success("Database created! Reloading...")
        st.rerun()
    else:
        st.error("Could not build the database. Please check your WRDS credentials and connection.")
        st.stop()

# --- TABS: Text2SQL Agent (Tab 1) + Company Drilldown (Tab 2) ---
tab_chat, tab_drill = st.tabs(["💬 Text2SQL Agent", "🔍 Company Drilldown"])

# =========================
# TAB 1: Text2SQL Chat Agent
# =========================
with tab_chat:
    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": (
                "I am connected to your WRDS database. Try asking: "
                "'Find stocks with the highest momentum over the last 3 months' "
                "or 'Which companies beat earnings estimates last quarter?'"
            )
        }]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("E.g., Which stocks had the highest returns last quarter?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        if not api_key:
            st.error("Please provide a Gemini API Key.")
            st.stop()

        async def run_agentic_workflow():
            with st.status("🚀 Agentics Workflow Running...", expanded=True) as status:

                # 1. Initialize State
                state = Text2sqlQuestion(
                    question=prompt,
                    db_id="market_anomalies"
                )

                # --- Step 1: Get Schema (Manual Call) ---
                st.write("🔍 Identifying Tables & Schema...")
                state = await get_schema_map(state)

                # --- Step 2: Generate SQL (Agentics Transduction) ---
                st.write("🧠 Generating SQL Query...")

                # Create fresh agent specifically for this step
                try:
                    agent = AG(states=[state], atype=Text2sqlQuestion)
                except Exception:
                    agent = AG()
                    agent.states = [state]

                agent.llm = AG.get_llm_provider()

                # DEFENSIVE: Ensure we don't have tuples before transduction
                if agent.states and isinstance(agent.states[0], tuple):
                    st.warning("Detected tuple state, attempting to unwrap...")
                    agent.states = [agent.states[0][0]]

                # Updated prompt to leverage the new schema context
                agent = await agent.self_transduction(
                    ["question", "db_schema"],
                    ["generated_query"],
                    instructions="""
                    You are an expert financial data scientist. Use the provided schema JSON in `db_schema` to answer the question.

                    The `db_schema` JSON contains:
                      - `sqlite_database_structure`  → actual tables/columns
                      - `dataset_semantic_documentation` → human descriptions
                      - `data_availability_summary`  → per-table min/max date ranges
                      - `latest_data_date`           → the MAX date present in ANY table

                    ABSOLUTE CRITICAL RULES ABOUT DATES:
                    - The database only contains historical data up to `latest_data_date`.
                    - NEVER use dynamic date functions such as DATE('now'), date('now', ...),
                      CURRENT_DATE, or variations of those.
                    - Instead, ALWAYS treat `latest_data_date` as "today".
                    - For example, if the user asks for "last 3 months":
                        • Use a predicate like:
                          WHERE some_date_column BETWEEN DATE('<latest_data_date>', '-3 months')
                                                     AND DATE('<latest_data_date>')
                        • Replace <latest_data_date> with the literal string from the JSON.

                    Other rules:
                    1. Use `dataset_semantic_documentation` to decide which tables are relevant.
                    2. Use `sqlite_database_structure` for exact column names.
                    3. Be careful with identifiers (permno vs ticker vs gvkey). If no mapping
                       table is available, only join on identifiers that exist in BOTH tables.
                    4. Return ONLY valid SQLite SQL (no comments, no markdown fences).
                    """,
                )

                # Extract state back from agent (handling potential tuple return)
                result_state = agent.states[0]
                if isinstance(result_state, tuple):
                    result_state = result_state[0]

                state.generated_query = result_state.generated_query
                st.code(state.generated_query, language="sql")

                # --- Step 3: Execute SQL (Manual Call) ---
                st.write("⚡ Executing Query...")
                state = await execute_query_map(state)

                results = state.system_output_df

                # --- Step 4: Generate Custom Report (Agentics Transduction) ---
                final_report = ""
                if results and not results.startswith("Error") and results != "[]":
                    st.write("📝 Synthesizing Report...")

                    # Update agent with results
                    agent.states = [state]

                    agent = await agent.self_transduction(
                        ["question", "generated_query", "system_output_df"],
                        ["final_report"],
                        instructions="""
                        You are a hedge fund analyst. 
                        1. Read the user's question and the SQL results.
                        2. Identify if there is a market anomaly (e.g., significant abnormal returns, earnings surprise).
                        3. Write a concise report highlighting the top findings.
                        4. Suggest a follow-up query if the data is inconclusive.
                        """
                    )

                    # Extract final report (again, handling tuples defensively)
                    final_state_obj = agent.states[0]
                    if isinstance(final_state_obj, tuple):
                        final_state_obj = final_state_obj[0]

                    if hasattr(final_state_obj, 'final_report'):
                        final_report = final_state_obj.final_report
                    else:
                        final_report = "Error: Report generation failed."
                else:
                    final_report = (
                        f"I could not retrieve data. The query execution returned: {results}\n\n"
                        "**Hint:** Check the 'Database Inspector' in the sidebar to ensure your "
                        "query date range matches the available data."
                    )

                status.update(label="✅ Workflow Complete", state="complete", expanded=False)
                return final_report

        # Run Async Loop
        try:
            response_text = asyncio.run(run_agentic_workflow())
            st.write(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
        except Exception as e:
            st.error(f"Workflow failed: {e}")

# ================================
# TAB 2: Company Anomaly Drilldown
# ================================
with tab_drill:
    st.header("🔍 Company Anomaly Drilldown")
    st.markdown(
        "Enter a ticker (e.g. `HEI`, `CCL`) to see its normalized price index and a "
        "composite anomaly score combining CRSP, Compustat, and IBES."
    )

    ticker_input = st.text_input("Ticker symbol:", value="HEI").upper().strip()

    if st.button("Analyze Company"):
        if not ticker_input:
            st.error("Please enter a ticker.")
        else:
            try:
                detector = CompositeAnomalyDetector(DB_PATH)  # master_db default path is inferred
                ts_df, composite_score, components = detector.compute_composite_for_ticker(ticker_input)

            except FileNotFoundError as e:
                st.error(str(e))
                st.stop()
            except Exception as e:
                st.error(f"Composite anomaly computation failed: {e}")
                st.stop()

            if ts_df.empty:
                st.error(components.get("error", f"No data found for ticker '{ticker_input}'."))
            else:
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.subheader(f"Price Index & Composite Anomaly — {ticker_input.upper()}")
                    chart_df = ts_df.set_index("date")[["price_index", "composite_score"]]
                    st.line_chart(chart_df)

                with col2:
                    st.subheader("Composite Anomaly Score")
                    st.metric(
                        label="Current composite anomaly (0–1)",
                        value=f"{composite_score:.2f}",
                        help="0 = normal, 1 = highly anomalous (CRSP + Compustat + IBES)."
                    )

                    st.subheader("🧾 Anomaly Summary")
                    try:
                        summary_md = asyncio.run(
                            generate_anomaly_summary(
                                ticker=ticker_input.upper(),
                                composite_score=composite_score,
                                components=components,
                            )
                        )
                        st.markdown(summary_md)
                    except Exception as e:
                        st.error(f"Failed to generate anomaly summary: {e}")
                        st.write("Composite score:", composite_score)

                    # Optional: tuck raw debug info away
                    with st.expander("🔍 Detailed component breakdown (debug)", expanded=False):
                        st.json(components)
            # if ts_df.empty:
            #     st.error(components.get("error", f"No data found for ticker '{ticker_input}'."))
            # else:
            #     col1, col2 = st.columns([2, 1])
            #
            #     with col1:
            #         st.subheader(f"Price Index & Composite Anomaly — {ticker_input}")
            #         chart_df = ts_df.set_index("date")[["price_index", "composite_score"]]
            #         st.line_chart(chart_df)
            #
            #     with col2:
            #         st.subheader("Composite Anomaly Score")
            #         st.metric(
            #             label="Current composite anomaly (0–1)",
            #             value=f"{composite_score:.2f}",
            #             help="0 = normal, 1 = highly anomalous (CRSP + Compustat + IBES)."
            #         )
            #         st.markdown("**Component breakdown:**")
            #         st.json(components)

