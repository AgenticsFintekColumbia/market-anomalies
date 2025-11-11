# scripts/dow_health_report.py
# Run: uv run python scripts/dow_health_report.py --sector health --lookback 5 --min-change 1 --limit 10
import os, argparse, time
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import yfinance as yf
import yaml

from pydantic import BaseModel, Field
from crewai import Agent, Crew, Task
from crewai_tools import MCPServerAdapter
from agentics import AG
from mcp import StdioServerParameters  # For Stdio Server


# -----------------------------
# Pydantic structs for the report
# -----------------------------
class Citation(BaseModel):
    url: Optional[str] = None
    authors: Optional[List[str]] = None
    title: Optional[str] = None
    relevant_text: Optional[str] = None

class WebSearchReport(BaseModel):
    title: Optional[str] = None
    abstract: Optional[str] = None
    full_report: Optional[str] = Field(
        None,
        description="Markdown report of findings."
    )
    citations: List[Citation] = Field(default_factory=list, description="Citations to relevant sources")


# -----------------------------
# Yahoo Finance helpers (Dow-only universe)
# -----------------------------
def get_dow_tickers() -> List[str]:
    """Return list of Dow 30 tickers; fallback to Wikipedia if yfinance helper missing."""
    if hasattr(yf, "tickers_dow"):
        return yf.tickers_dow()
    tables = pd.read_html("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average")
    tickers = tables[1]["Symbol"].tolist()
    return [t.replace(".", "-") for t in tickers]

def get_sector_map(tickers: List[str]) -> Dict[str, str]:
    """Map ticker -> sector using Yahoo metadata (best-effort)."""
    sector = {}
    for i, t in enumerate(tickers, 1):
        try:
            info = yf.Ticker(t).get_info()
            sector[t] = info.get("sector", "") or ""
        except Exception:
            sector[t] = ""
        if i % 10 == 0:
            time.sleep(0.25)
    return sector

def compute_pct_change(prices: pd.DataFrame, lookback: int) -> pd.Series:
    """Percent change of Close over N trading days."""
    close = prices["Close"]
    return (close.iloc[-1] / close.iloc[-lookback] - 1.0) * 100.0

def yahoo_quote_url(ticker: str) -> str:
    return f"https://finance.yahoo.com/quote/{ticker}"

def build_table_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No matching tickers found for the selected filters._"
    cols = ["Ticker", "Name", "Sector", "Price", "MarketCap", "Change(%)"]
    show = df[cols].copy()
    # pretty formatting
    show["Price"] = show["Price"].map(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
    show["MarketCap"] = show["MarketCap"].map(lambda x: f"${x:,.0f}" if pd.notna(x) else "")
    show["Change(%)"] = show["Change(%)"].map(lambda x: f"{x:,.2f}%")
    return show.to_markdown(index=False)


# -----------------------------
# Core pipeline that fetches data and returns a WebSearchReport
# -----------------------------
def fetch_dow_sector_risers(sector_kw: str, lookback: int, min_change: float, limit: int) -> WebSearchReport:
    tickers = get_dow_tickers()

    # Batch prices
    hist = yf.download(
        tickers=tickers,
        period=f"{max(lookback + 2, 7)}d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )

    # Normalize to a (field -> df) panel
    if isinstance(hist.columns, pd.MultiIndex):
        prices = pd.concat({"Close": hist.xs("Close", axis=1, level=1)}, axis=1)
    else:
        prices = pd.concat({"Close": hist["Close"]}, axis=1)

    chg = compute_pct_change(prices, lookback).dropna()
    chg.name = f"%chg_{lookback}d"

    sector_map = get_sector_map(chg.index.tolist())

    # Assemble rows and add Yahoo links for citations
    records, cits = [], []
    for t in chg.sort_values(ascending=False).index:
        sector_val = sector_map.get(t, "")
        if sector_kw.lower() not in sector_val.lower():
            continue
        if float(chg[t]) < min_change:
            continue
        try:
            tk = yf.Ticker(t)
            info = tk.get_info()
            price = getattr(tk, "fast_info", {}).get("last_price", np.nan) if hasattr(tk, "fast_info") else np.nan
            records.append({
                "Ticker": t,
                "Name": info.get("shortName", ""),
                "Sector": sector_val,
                "Price": price,
                "MarketCap": info.get("marketCap", np.nan),
                "Change(%)": float(chg[t]),
                "URL": yahoo_quote_url(t),
            })
            cits.append(Citation(url=yahoo_quote_url(t), title=f"Yahoo Finance: {t}"))
        except Exception:
            # still include change/sector if price/meta failed
            records.append({
                "Ticker": t,
                "Name": "",
                "Sector": sector_val,
                "Price": np.nan,
                "MarketCap": np.nan,
                "Change(%)": float(chg[t]),
                "URL": yahoo_quote_url(t),
            })
            cits.append(Citation(url=yahoo_quote_url(t), title=f"Yahoo Finance: {t}"))

        if len(records) >= limit:
            break

    df = pd.DataFrame(records).sort_values(by="Change(%)", ascending=False)
    table_md = build_table_markdown(df)

    title = f"Rising Dow Stocks in Sector ~ '{sector_kw}' (lookback={lookback} trading days, min Δ={min_change:.2f}%)"
    abstract = (
        f"This report screens **Dow Jones (30)** constituents for the **{sector_kw}** sector and ranks by "
        f"**{lookback}-day percent change** using Yahoo Finance data. "
        f"Only tickers with ≥ {min_change:.2f}% change are shown (top {limit})."
    )
    full_report = f"""\
# {title}

**Method**
- Universe: Dow Jones Industrial Average (DJIA, 30 stocks).
- Data source: Yahoo Finance via `yfinance` (batch price history & metadata).
- Metric: {lookback}-trading-day percent change in closing price.
- Sector filter: substring match on Yahoo “sector” field → `{sector_kw}`.
- Threshold: ≥ {min_change:.2f}% change. Limited to top {limit} results.

**Results**

{table_md}

**Notes**
- Sector metadata from Yahoo can be noisy; we match by substring (case-insensitive).
- Price change uses trading days (not calendar days).
"""
    # Add an overall Dow page citation for completeness
    cits.append(Citation(url="https://finance.yahoo.com/quote/%5EDJI", title="Yahoo Finance: ^DJI (Dow Jones Industrial Average)"))

    return WebSearchReport(
        title=title,
        abstract=abstract,
        full_report=full_report,
        citations=cits,
    )


# -----------------------------
# CrewAI wiring: we still include your MCP web search capability.
# The agent can use it to enrich context if desired (news, profiles).
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sector", default="health", help="Sector substring filter, e.g. 'health'")
    ap.add_argument("--lookback", type=int, default=5, help="Trading days lookback")
    ap.add_argument("--min-change", type=float, default=0.0, help="Minimum percent change")
    ap.add_argument("--limit", type=int, default=10, help="Top N results")
    args = ap.parse_args()

    # Generate the core report data from Yahoo Finance
    base_report = fetch_dow_sector_risers(
        sector_kw=args.sector,
        lookback=args.lookback,
        min_change=args.min_change,
        limit=args.limit,
    )

    # Connect to community MCP servers using uvx (crew_ai_web_search style)
    fetch_params = StdioServerParameters(
        command="uvx",
        args=["mcp-server-fetch"],
        env={"UV_PYTHON": "3.12", **os.environ},
    )
    # Connect to your own MCP server by providing the path (optional)
    mcp_path = os.getenv("MCP_SERVER_PATH")
    search_params = StdioServerParameters(
        command="python3",
        args=[mcp_path] if mcp_path else [],
        env={"UV_PYTHON": "3.12", **os.environ},
    )

    # Build agent with MCP tools; it can use web search to add context/citations.
    with MCPServerAdapter(fetch_params) as fetch_tools, \
         MCPServerAdapter(search_params) as search_tools if mcp_path else MCPServerAdapter(fetch_params) as _noop:

        tools = fetch_tools + (search_tools if mcp_path else [])

        # Report-writer agent
        doc_agent = Agent(
            role="Market Web Researcher",
            goal=(
                "Use web tools to validate and add context (profiles, news) to the provided "
                "Yahoo-finance-based Dow sector screen. Produce a clear markdown report."
            ),
            backstory=(
                "You summarize market screens and add links & brief justifications. "
                "Prefer official or reputable sources. Avoid speculation."
            ),
            tools=tools,
            reasoning=False,
            reasoning_steps=6,
            memory=False,
            verbose=True,
            llm=AG.get_llm_provider(),
        )

        # We pass the computed table + seed citations; the agent may add more citations via web tools.
        prompt = f"""\
You are given a preliminary report (markdown) and citations built from Yahoo Finance screening:

---BEGIN PRELIM REPORT---
{base_report.full_report}
---END PRELIM REPORT---

Seed citations (JSON):
{yaml.dump([c.model_dump() for c in base_report.citations], sort_keys=False)}

TASK:
1) Verify tickers/companies exist with quick web lookups using your tools (Yahoo/official pages).
2) Add/merge citations where helpful (avoid duplicates). Prefer a citation per listed ticker.
3) Keep the table unchanged (it reflects computed values). You may add a one-paragraph commentary.
4) Output the final, polished markdown report and the consolidated citations.
"""

        doc_task = Task(
            description=prompt,
            expected_output="A WebSearchReport with polished markdown and consolidated citations.",
            agent=doc_agent,
            output_pydantic=WebSearchReport,
        )

        crew = Crew(agents=[doc_agent], tasks=[doc_task], verbose=True)
        result = crew.kickoff()

    # Fallback: if the agent didn't return pydantic (e.g., MCP not available), emit the base report
    final_report = result.pydantic or base_report
    print(yaml.dump(final_report.model_dump(), sort_keys=False))


if __name__ == "__main__":
    main()