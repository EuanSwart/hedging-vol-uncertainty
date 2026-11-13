"""download spy and vix daily closes (2010-2025) from yahoo finance into data/spy_vix.csv.

the committed csv is the file all reported results use: yahoo adjusts historical prices, so a fresh download can
differ slightly. run this only to refresh or audit the data."""
import pandas as pd
import yfinance as yf

from hedging.config import SPY_CSV

START, END = "2010-01-01", "2026-01-01"


def close(ticker: str) -> pd.Series:
    df = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    s = df["Close"]
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


if __name__ == "__main__":
    out = pd.concat({"spy": close("SPY"), "vix": close("^VIX")}, axis=1).dropna()
    out.index.name = "date"
    out.to_csv(SPY_CSV)
    print(f"wrote {len(out)} rows to {SPY_CSV}")
