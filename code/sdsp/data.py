"""Data loading, cleaning and the chronological split.

Notation (as in the report):
    p[n]  daily price (USD), n = 0..N-1
    x[n]  = ln p[n]                      log-price
    r[n]  = x[n] - x[n-1]                log-return (defined for n >= 1)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COURSE_CSV = ROOT / "data" / "course" / "BTC-USD.csv"
RECENT_CSV = ROOT / "data" / "recent" / "BTC-USD_2025-10-01_2026-10-15.csv"
FULL_CSV = ROOT / "data" / "full" / "BTC-USD_close_2014-09-17_2026-10-15.csv"
PRICE_COLS = ["Open", "High", "Low", "Close"]

# The last row of the course file (2021-08-06) was saved before the day closed: its Close is
# 40907.74 USD while Yahoo's final value is 42816.50 USD (-4.7 %). The course data are therefore
# stopped at 2021-08-05. Other revised days are listed in data/course/yahoo_crosscheck.csv.
COURSE_LAST_COMPLETE_DAY = "2021-08-05"


def load_raw(path: str | Path = COURSE_CSV) -> pd.DataFrame:
    """Read a Yahoo Finance CSV. Yahoo writes missing days as the string 'null'."""
    df = pd.read_csv(path, parse_dates=["Date"], na_values=["null"])
    df = df.set_index("Date").sort_index()
    df = df.drop(columns=["Adj Close"], errors="ignore")  # identical to Close for BTC-USD
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Put the series on a gap-free daily grid and fill missing days.

    Prices are interpolated linearly in the *log* domain (a constant return across the gap);
    volume is interpolated linearly. The column 'interpolated' marks the filled days so they
    can be excluded from evaluation.
    """
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    out = df.reindex(full_idx)
    out.index.name = "Date"
    out["interpolated"] = out["Close"].isna()
    for c in PRICE_COLS:
        out[c] = np.exp(np.log(out[c]).interpolate(method="time"))
    out["Volume"] = out["Volume"].interpolate(method="time")
    return out


def add_returns(df: pd.DataFrame, col: str = "Close") -> pd.DataFrame:
    """Add log-price x[n] and log-return r[n] for the chosen price column."""
    out = df.copy()
    out["logp"] = np.log(out[col])
    out["r"] = out["logp"].diff()
    return out


def load_dataset(path: str | Path = COURSE_CSV, col: str = "Close",
                 end: str | None = COURSE_LAST_COMPLETE_DAY) -> pd.DataFrame:
    """Load, clean and add returns in one call. `end` drops rows after that date (None keeps all)."""
    df = load_raw(path)
    if end is not None:
        df = df.loc[:end]
    return add_returns(clean(df), col=col)


def load_full(path: str | Path = FULL_CSV) -> pd.Series:
    """Full daily history (close rounded to cents, no gaps). Returns the log-return series r[n]."""
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    assert df.index.to_series().diff().dropna().eq(pd.Timedelta("1D")).all(), "gap in full history"
    return np.log(df["Close"]).diff().dropna().rename("r")


@dataclass
class Series:
    """Arrays shared by all experiments on one price column, with the chronological split.

    The first 80 % of the days form the training block, the last 20 % the test block. The last
    20 % of the training block is the validation block; the rest is the fit block.
    """
    df: pd.DataFrame
    col: str = "Close"
    train_frac: float = 0.8
    val_frac: float = 0.2
    x: np.ndarray = field(init=False)       # log price
    r: np.ndarray = field(init=False)       # log return, r[0] := 0 (undefined)
    p: np.ndarray = field(init=False)       # price
    mask: np.ndarray = field(init=False)    # True = real (not interpolated) observation
    val: np.ndarray = field(init=False)     # positions of validation days
    test: np.ndarray = field(init=False)    # positions of test days

    def __post_init__(self):
        self.p = self.df[self.col].to_numpy(float)
        self.x = np.log(self.p)
        self.r = np.r_[0.0, np.diff(self.x)]
        self.mask = ~self.df["interpolated"].to_numpy() if "interpolated" in self.df else np.ones(len(self.p), bool)
        N = len(self.p)
        n_train = int(round(self.train_frac * N))
        n_fit = int(round((1 - self.val_frac) * n_train))
        self.val = np.arange(n_fit, n_train)
        self.test = np.arange(n_train, N)
        self.sigma2_fit = float(np.var(self.r[1:n_fit]))   # used to scale step sizes / noise

    @property
    def dates(self):
        return self.df.index

    def mse_r(self, pos: np.ndarray, rhat: np.ndarray) -> float:
        m = self.mask[pos]
        return float(np.mean((self.r[pos][m] - np.asarray(rhat)[m]) ** 2))
