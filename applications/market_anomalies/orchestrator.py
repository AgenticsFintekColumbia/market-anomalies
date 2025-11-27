import logging
import os
from datetime import datetime
from pathlib import Path
import yaml
import pandas as pd

from connectors.crsp import CRSPIngestor
from connectors.compustat import CompustatIngestor
from connectors.ibes import IBESIngestor
from connectors.ciq import CIQIngestor
from connectors.boardex import BoardExIngestor
from connectors.fred import FREDIngestor
from connectors.yahoo_finance import YahooFinanceIngestor
from connectors.google_trends import GoogleTrendsIngestor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

WRDS_USER = CONFIG["wrds"]["username"]
DAYS_BACK = CONFIG["date_range"]["days_back"]
FRED_KEY  = CONFIG["fred"]["api_key"]


def main():
    logger.info("=== Market Anomalies Data Ingestion Started ===")
    start_time = datetime.now()

    with CRSPIngestor(WRDS_USER) as crsp:
        s, e = crsp.get_date_range(90)
        crsp_df = crsp.fetch_if_needed("wrds/crsp_daily", s, e)
        crsp.save_schema(crsp.get_schema_documentation(), "wrds_crsp")
        print("CRSP data successfully retrieved!")

    with CompustatIngestor(WRDS_USER) as comp:
        s, e = comp.get_date_range(180)
        compustat_df = comp.fetch_if_needed("wrds/compustat_quarterly", s, e)
        comp.save_schema(comp.get_schema_documentation(), "wrds_compustat")
        print("Quarterly Compustat data successfully retrieved!")

    with IBESIngestor(WRDS_USER) as ibes:
        s, e = ibes.get_date_range(365 * 5)  # e.g. last 5 years

        eps_df = ibes.fetch_eps_if_needed("wrds/ibes_eps_summary", s, e)
        recs_df = ibes.fetch_recs_if_needed("wrds/ibes_recommendations", s, e)

        ibes.save_schema(ibes.get_schema_documentation(), "wrds_ibes")

        print("IBES EPS summary + recommendations successfully saved (or loaded from cache)!")

    with CIQIngestor(WRDS_USER) as ciq:
        s, e = ciq.get_date_range(365)
        df_ciq = ciq.fetch_if_needed(
            "wrds/ciq_keydev",
            start=s,
            end=e,
            event_ids=(16, 81, 232),
            only_primary_us_tickers=True
        )
        ciq.save_schema(ciq.get_schema_documentation(), "wrds_ciq")

        print("CIQ KeyDev data successfully saved (or loaded from cache)!")

    # with BoardExIngestor(WRDS_USER) as bx:
    #     s, e = bx.get_date_range(180)
    #     df = bx.fetch_board_changes(s, e)
    #     bx.save_data(df, "wrds/boardex_changes")
    #     bx.save_schema(bx.get_schema_documentation(), "wrds_boardex")
    #
    # fred = FREDIngestor(FRED_KEY)
    # s, e = CRSPIngestor(WRDS_USER).get_date_range(180)
    # macro = fred.fetch_indicators(["VIXCLS", "FEDFUNDS", "SP500"], s, e)
    # fred.save_data(macro, "external/fred_macro")

    # yahoo = YahooFinanceIngestor()
    # yf_df = yahoo.fetch_prices(["^GSPC", "^VIX"], s, e)
    # yahoo.save_data(yf_df, "external/yahoo_indices")

    # google = GoogleTrendsIngestor()
    # trends = google.fetch_interest(["Nvidia", "Apple", "Tesla"], s, e)
    # google.save_data(trends, "external/google_trends")

    logger.info("Ingestion completed in %s seconds", (datetime.now() - start_time).seconds)

if __name__ == "__main__":
    main()
