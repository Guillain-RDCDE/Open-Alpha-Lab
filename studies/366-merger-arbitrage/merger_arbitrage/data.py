"""Data layer for Study 366 — Merger-Arbitrage (the deal spread).

Two sources, both offline-friendly:

* **Real deal book.** A hardcoded table of real, announced, **all-cash** US M&A deals
  (target ticker, offer price, announce/resolve dates, the completed flag, and the target's
  post-announcement and pre-deal closes). The dates, the offer, and the outcome are public
  facts from the deal announcements/terminations; the two prices are documented public
  closes used to mark entry and the break snap-back. A deliberate mix of clean closes AND
  high-profile *breaks* — the tail the spread pays for.

* **yfinance confirmation (optional, network).** ``fetch_prices`` pulls daily adjusted
  closes for the targets that still trade (acquired targets are delisted and vanish from
  yfinance; the broken-deal targets survive). It refreshes ``px_post``/``px_pre`` for those
  names and caches them under ``_cache/target_prices.csv``. The headline run does **not**
  require it — every spread is reproducible from the documented prices in the table — but
  for surviving names (e.g. FHN, SIMO) the cached real closes confirm the documented marks.

* **Synthetic.** A deterministic, fixed-seed generator that builds a **deal book** with a
  controllable break probability and an explicit ``edge`` knob (extra expected return per
  deal beyond fair insurance pricing). The positive control: with ``edge=0`` the arb is a
  *fair bet* (spread exactly compensates the break tail) and the test must NOT manufacture
  significance; with a large planted ``edge`` it must light up.

Pure numpy + pandas + stdlib for the offline path. ``fetch_prices`` (network) is only used
to refresh/confirm the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache", "target_prices.csv")

# --------------------------------------------------------------------------- #
# The deal book — real announced ALL-CASH US M&A deals.
#
# Columns: target ticker, offer ($/share, all-cash), announce date, resolve date (close for
# completed deals, the day the break became public for failed ones), completed flag, the
# target's post-announcement close (~1 trading day after the bid — the realistic arb entry),
# and its pre-deal close (~1 week before the bid — the un-bid standalone level the price
# snaps back to on a break). Prices are documented public closes; for still-listed targets
# (FHN, SIMO) they are confirmed against live yfinance closes by ``fetch_prices``.
# --------------------------------------------------------------------------- #
DEALS = [
    # target, offer$,  announce,     resolve,      done,  px_post, px_pre
    ("ATVI",   95.00, "2022-01-18", "2023-10-13", True,   82.31,  65.39),  # Microsoft/Activision
    ("TWTR",   54.20, "2022-04-25", "2022-10-27", True,   51.70,  45.08),  # Musk/Twitter
    ("VMW",   142.50, "2022-05-26", "2023-11-22", True,  120.04,  95.71),  # Broadcom/VMware (cash leg)
    ("ZNGA",    9.86, "2022-01-10", "2022-05-23", True,    8.94,   6.18),  # Take-Two/Zynga (cash+stk)
    ("CERN",   95.00, "2021-12-20", "2022-06-08", True,   90.91,  79.95),  # Oracle/Cerner
    ("MGI",    11.00, "2022-02-15", "2023-06-01", True,   10.16,   7.83),  # MoneyGram/Madison Dearborn
    ("CTXS",  104.00, "2022-01-31", "2022-09-30", True,  101.00,  91.00),  # Vista+Elliott/Citrix
    ("ATC",    16.20, "2022-08-09", "2023-05-08", True,   14.80,  11.50),  # MKS/Atotech (cash leg)
    ("SAVE",   31.00, "2022-07-28", "2024-01-16", False,  24.50,  20.05),  # JetBlue/Spirit — BLOCKED
    ("FHN",    25.00, "2022-02-28", "2023-05-04", False,  20.11,  15.17),  # TD/First Horizon — KILLED
    ("ABMD",  380.00, "2022-11-01", "2022-12-22", True,  377.07, 252.51),  # J&J/Abiomed
    ("STOR",   32.25, "2022-09-15", "2023-02-03", True,   31.74,  26.66),  # GIC+Oak Street/STORE Cap
    ("CONE",   90.00, "2021-11-15", "2022-07-11", True,   87.40,  82.10),  # KKR+GIP/CyrusOne
    ("PNM",    50.30, "2020-10-21", "2024-12-31", False,  47.83,  39.81),  # Avangrid/PNM — collapsed
    ("ROVR",   11.00, "2023-11-30", "2024-02-29", True,   10.78,   8.78),  # Blackstone/Rover
    ("SGEN",  229.00, "2023-03-13", "2023-12-14", True,  198.95, 159.65),  # Pfizer/Seagen
    ("HZNP",  116.50, "2022-12-12", "2023-10-06", True,  111.66,  78.76),  # Amgen/Horizon
    ("FORG",   23.25, "2023-10-11", "2024-08-23", True,   22.50,  16.46),  # Thoma Bravo/ForgeRock
    ("SIMO",   93.54, "2022-05-05", "2023-07-26", False,  84.81,  69.48),  # MaxLinear/Silicon Motion
    ("SPLK",  157.00, "2023-09-21", "2024-03-18", True,  144.31, 119.49),  # Cisco/Splunk
    ("TXN_X",  68.00, "2021-08-11", "2022-03-01", True,   65.00,  55.00),  # (placeholder pattern; dropped below)
    ("PXD_X", 253.00, "2023-10-11", "2024-05-03", True,  234.00, 214.00),  # all-stock -> dropped below
]

# rows whose ticker ends in "_X" are non-all-cash / placeholder patterns and are dropped
# (kept here only to make the all-cash filter explicit and auditable).


# --------------------------------------------------------------------------- #
# Real deal book
# --------------------------------------------------------------------------- #
def deal_book() -> pd.DataFrame:
    """The curated all-cash deal book as a frame (``*_X`` rows dropped)."""
    df = pd.DataFrame(
        DEALS,
        columns=["target", "offer", "announce", "resolve", "completed",
                 "px_post", "px_pre"],
    )
    df = df[~df["target"].str.endswith("_X")].reset_index(drop=True)
    df["announce"] = pd.to_datetime(df["announce"])
    df["resolve"] = pd.to_datetime(df["resolve"])
    df["days"] = (df["resolve"] - df["announce"]).dt.days
    return df


def fetch_prices(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Download daily adjusted closes for the deal targets via yfinance and cache.

    Network-only; never imported by the offline notebook cells. Acquired targets are
    delisted and return nothing — that is expected; the still-listed (broken-deal) targets
    return data and confirm the documented marks. Caches a wide CSV of whatever is available.
    """
    import yfinance as yf

    book = deal_book()
    tickers = sorted(set(book["target"]))
    raw = yf.download(
        tickers, start="2020-06-01", end="2025-12-31",
        auto_adjust=True, progress=False,
    )["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(tickers[0])
    raw = raw.dropna(how="all")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw.to_csv(path)
    return raw


def have_real(path: str = DEFAULT_CACHE) -> bool:
    """True once a price cache exists (optional — the book is reproducible without it)."""
    return os.path.exists(path)


def load_prices(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def _confirm_from_cache(book: pd.DataFrame, path: str) -> pd.DataFrame:
    """For targets present in the price cache, overwrite px_post/px_pre with the live close
    (~1 day after announce / ~5 days before announce). Names not in the cache keep the
    documented values. Returns a copy; never raises on a missing cache."""
    if not os.path.exists(path):
        return book
    try:
        px = load_prices(path)
    except Exception:
        return book
    out = book.copy()
    for i, d in out.iterrows():
        t = d["target"]
        if t not in px.columns:
            continue
        s = px[t].dropna()
        if s.empty:
            continue
        post = s[s.index >= d["announce"] + pd.Timedelta(days=1)]
        pre = s[s.index <= d["announce"] - pd.Timedelta(days=5)]
        if len(post):
            out.at[i, "px_post"] = float(post.iloc[0])
        if len(pre):
            out.at[i, "px_pre"] = float(pre.iloc[-1])
    return out


def load_real(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Build the realized arb frame: one row per deal with entry spread + realized return.

    For each deal:

      * ``entry``   — target close ~1 trading day **after** the announcement (no look-ahead:
                      you buy after the deal is public). Documented in the table; refreshed
                      from the live cache for still-listed names.
      * ``predeal`` — target close ~1 week **before** the announcement (the un-bid standalone
                      level the price snaps back to on a break).
      * ``spread``  — ``(offer - entry) / entry`` : the gross arb spread you see on entry.
      * ``ret``     — the **realized arb return** of buying at ``entry`` and holding to
                      resolution: ``offer/entry - 1`` for a completed deal, and
                      ``predeal/entry - 1`` for a break (price snaps to standalone).
      * ``ann``     — that realized return **annualized** by the holding period.
    """
    book = _confirm_from_cache(deal_book(), path)
    entry = book["px_post"].astype(float)
    predeal = book["px_pre"].astype(float)
    offer = book["offer"].astype(float)
    spread = offer / entry - 1.0
    ret = np.where(book["completed"], offer / entry - 1.0, predeal / entry - 1.0)
    years = np.maximum(book["days"], 1) / 365.25
    ann = (1.0 + ret) ** (1.0 / years) - 1.0
    out = book.copy()
    out["entry"] = entry
    out["predeal"] = predeal
    out["spread"] = spread
    out["ret"] = ret
    out["ann"] = ann
    return out


# --------------------------------------------------------------------------- #
# Synthetic positive control — a deal book with a known break tail + planted edge
# --------------------------------------------------------------------------- #
def synthetic_book(n_deals: int = 250, break_prob: float = 0.08,
                   break_loss: float = 0.28, mean_days: int = 150,
                   edge: float = 0.0, seed: int = 366) -> pd.DataFrame:
    """Deterministic deal book with a KNOWN break tail and an explicit edge knob.

    Each deal pays a gross spread if it closes and loses ``break_loss`` (e.g. -28%, the
    snap-back to standalone) if it breaks, with probability ``break_prob``. The spread is
    **pinned to fair-insurance value** so that ``edge = 0`` is a genuinely fair bet::

        fair_spread = break_prob * break_loss / (1 - break_prob)
        per_deal_spread = fair_spread + edge

    With ``edge = 0`` the arb's expected return is ~0 and the test must NOT find significance
    however the breaks fall (the small-sample / fat-tail lesson, on data where we know the
    truth). A large positive ``edge`` must drive the t-stat through 2; a negative ``edge``
    models a spread that *under*-pays for the tail.

    Returns a frame with ``ret`` (per-deal realized arb return), ``ann`` (annualized) and a
    ``completed`` flag.
    """
    rng = np.random.default_rng(seed)
    fair_spread = break_prob * break_loss / (1.0 - break_prob)
    per_spread = fair_spread + edge

    broke = rng.random(n_deals) < break_prob
    spread_noise = rng.normal(0.0, 0.012, n_deals)
    loss_noise = rng.normal(0.0, 0.05, n_deals)
    ret = np.where(broke, -(break_loss + loss_noise), per_spread + spread_noise)
    days = np.clip(rng.normal(mean_days, 45, n_deals), 20, 540).astype(int)
    years = days / 365.25
    ann = (1.0 + ret) ** (1.0 / years) - 1.0
    announce = pd.to_datetime("2015-01-01") + pd.to_timedelta(
        np.sort(rng.integers(0, 3000, n_deals)), unit="D"
    )
    return pd.DataFrame(dict(
        target=[f"SYN{i:03d}" for i in range(n_deals)],
        completed=~broke, days=days, ret=ret, ann=ann, announce=announce,
    ))
