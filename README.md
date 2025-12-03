# 📈 WRDS Market Anomaly Hunter

LLM-powered anomaly detection using CRSP, Compustat, and IBES from [WRDS](https://wrds-www.wharton.upenn.edu/).

This application identifies market anomalies for any U.S. publicly traded company by combining three WRDS datasets:

- 🏛️ CRSP (market microstructure anomalies; volatility z-scores)

- 🏢 Compustat (fundamental performance vs. sector)

- 👥 IBES (analyst earnings expectations + analyst recommendations)

A composite anomaly score synthesizes all available information into a single interpretable signal, and an LLM-based
summary agent provides a natural-language explanation citing evidence from each datasource.

## 🛠️ Installation

1. Clone the repository fork:

```
git clone https://github.com/AgenticsFintekColumbia/market-anomalies
cd AgenticsFintekColumbia/applications/market_anomalies
```

2. Create a virtual environment:

```
python3 -m venv .venv
source .venv/bin/activate
```

3. Install Python dependencies:

```
pip install -r requirements.txt
```

## ⚙️Environment Variables & Configuration

Add a `.env` file in `applications/market_anomalies` with:

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

## 🏗️ Build the database
Before running the app, you need to ingest the WRDS data and build the SQLite database.

1. Navigate to the directory:

```
applications/market_anomalies/
```

2. Run the orchestrator:

```
python orchestrator.py
```

Once the database is built, it is cached. You only need to run the orchestrator again if you want to pull a fresh time window (e.g., next month).

## 🚀 Run the Application

Launch the Streamlit interface:

```
streamlit run app.py
```

## 👨‍💻 Authors & Contributors
Andrea Clark (aoc2111@columbia.edu)

