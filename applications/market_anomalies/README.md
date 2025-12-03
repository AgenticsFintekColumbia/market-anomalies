# WRDS Market Anomaly Hunter

LLM-powered anomaly detection using CRSP, Compustat, and IBES

This application identifies market anomalies for any U.S. publicly traded company by combining three WRDS datasets:

- CRSP (market microstructure anomalies; volatility z-scores)

- Compustat (fundamental performance vs. sector)

- IBES (analyst earnings expectations + analyst recommendations)

A composite anomaly score synthesizes all available information into a single interpretable signal, and an LLM-based summary agent provides a natural-language explanation citing evidence from each datasource.

## Installation

Clone the repository

```
git clone https://github.com/aclarkse/Agentics
cd Agentics/applications/market_anomalies

```

Create a virtual environment

```
python3 -m venv .venv
source .venv/bin/activate
```

Install Python dependencies

```
pip install -r requirements.txt
```

# Environment Variables & Configuration

Add a `.env` file in `application/market_anomalies ` with:

```
WRDS_USERNAME=your_wrds_username
GEMINI_API_KEY=your_gemini_api_key
SQL_DB_PATH=applications/data/market_anomalies.db
```

Configure the ingestion in `config.yaml`:

```
wrds:
  username: your_wrds_username

date_range:
  days_back: 365

fred:
  api_key: ""

paths:
  data_dir: "applications/data"
```

## Build the database
Before running the app, you need to ingest the WRDS data and build the SQLite database.
To do so, cd into the `applications/market_anomalies/` directory and run:

```
python orchestrator.py
```

Once the database is built, it should be cached and usable until the next CRSP update, for
which you can simply run the data orchestrator again to pull the latest time window. To run the
application:

```
streamlit run app.py
```

