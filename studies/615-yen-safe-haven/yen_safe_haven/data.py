"""Data layer for Study 615 — Yen Safe Haven.

Two sources, both offline-friendly once cached:

* **Real tape.** Daily yfinance closes for ``JPY=X`` (the USD/JPY spot rate — yen per
  dollar, so the *yen*'s return against the dollar is **minus** the USDJPY percent
  change), ``SPY`` (total-return, auto-adjusted), ``^IRX`` (13-week US T-bill discount
  yield, the carry / funding-rate proxy — Japanese short rates sit at ~0 for almost the
  whole sample, so the US bill IS the USD-JPY rate differential to first order) and
  ``^TNX`` (US 10-year yield, for the rate-shock vs growth-scare split). Cached wide
  under the study's own ``_cache/ysh_tape.csv``.

* **Synthetic world.** A deterministic, seeded daily generator with a TUNABLE planted
  crash-hedge: the synthetic yen earns ``+hedge x |SPY|`` on SPY down days and nothing
  systematic otherwise, plus independent FX noise. ``hedge = 0`` is the null — the
  machinery must NOT find a safe haven in it; a planted ``hedge`` must light up. A
  machinery proof only, never market evidence.

Pure numpy + pandas + stdlib on the offline path. ``fetch_tape`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
TAPE_CACHE = os.path.join(CACHE_DIR, "ysh_tape.csv")

TICKERS = {"JPY=X": "usdjpy", "SPY": "spy", "^IRX": "irx", "^TNX": "tnx"}

# Frozen as-of: last complete calendar month at build time (2026-07-03).
AS_OF = "2026-06-30"

# ---------------------------------------------------------------------------
# Event windows (dates hardcoded; the RETURNS are always computed from the tape).
# Sources: SPY peak/trough dates from the daily tape itself; episode framing:
#   * GFC deleveraging   — Lehman weekend (2008-09-12) through year-end forced carry
#     unwind (USDJPY ~107 -> ~90). BIS Quarterly Review, Dec 2008.
#   * Aug-2015           — CNY devaluation (2015-08-11) + the 2015-08-24 flash crash.
#   * COVID risk-off     — SPY peak 2020-02-19 -> the initial JPY rally leg (USDJPY
#     ~112 -> ~102 by 2020-03-09), then the dollar-scramble reversal into the SPY
#     trough 2020-03-23 (BIS Bulletin No.2, "The dollar funding squeeze", Apr 2020).
#   * 2022 bear          — SPY top 2022-01-03 -> bear-market trough 2022-10-12; the Fed
#     hiking cycle WIDENED the rate differential all the way down (USDJPY 115 -> 146).
#   * Aug-2024 unwind    — BOJ hike 2024-07-31 -> the 2024-08-05 global carry-unwind
#     crash (Nikkei -12.4% in a day; USDJPY ~153 -> ~144).
# ---------------------------------------------------------------------------
EVENTS = [
    ("GFC deleveraging (Lehman -> year-end)", "2008-09-12", "2008-12-31"),
    ("Aug-2015 (CNY deval + flash crash)", "2015-08-10", "2015-08-25"),
    ("COVID risk-off leg (SPY peak -> 03-09)", "2020-02-19", "2020-03-09"),
    ("COVID dollar scramble (03-09 -> trough)", "2020-03-09", "2020-03-23"),
    ("2022 bear market (SPY top -> trough)", "2022-01-03", "2022-10-12"),
    ("Aug-2024 carry unwind (BOJ hike -> crash)", "2024-07-31", "2024-08-05"),
]


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_tape(start: str = "1996-01-01", end: str | None = None,
               path: str = TAPE_CACHE, retries: int = 3) -> pd.DataFrame:
    """Download USDJPY / SPY / ^IRX / ^TNX daily closes and cache them (network, once)."""
    import yfinance as yf

    raw = None
    for _ in range(retries):
        try:
            raw = yf.download(list(TICKERS), start=start, end=end,
                              auto_adjust=True, progress=False)["Close"]
            if raw is not None and len(raw) > 0:
                break
        except Exception:
            time.sleep(2.0)
    tape = raw.rename(columns=TICKERS)[list(TICKERS.values())].sort_index()
    tape.index = pd.DatetimeIndex(tape.index).tz_localize(None)
    tape = tape.dropna(how="all")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tape.to_csv(path)
    return tape


def have_real(path: str = TAPE_CACHE) -> bool:
    return os.path.exists(path)


def load_real(path: str = TAPE_CACHE, asof: str | None = AS_OF) -> pd.DataFrame:
    """Cached wide daily tape (usdjpy, spy, irx, tnx), sliced to the frozen as-of."""
    tape = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if asof is not None:
        tape = tape[tape.index <= pd.Timestamp(asof)]
    return tape


# --------------------------------------------------------------------------- #
# Synthetic world — a machinery proof with a TUNABLE planted crash hedge
# --------------------------------------------------------------------------- #
def synthetic_world(hedge: float = 0.0, n_days: int = 7500, seed: int = 615,
                    spy_mu: float = 4e-4, spy_sig: float = 0.012,
                    fx_sig: float = 0.006) -> pd.DataFrame:
    """Deterministic daily world with a planted safe-haven coefficient ``hedge``.

    ``spy`` is a noisy equity return; the synthetic yen earns ``hedge x (-spy)`` on SPY
    **down days only** (the carry-unwind mechanism, planted) plus independent noise —
    with ``hedge = 0`` the yen is pure noise and the detector must stay flat. Returns a
    frame shaped like the real tape (usdjpy/spy/irx/tnx price levels) so every real-tape
    function runs on it unchanged. Span ~30 years (well under the 250-year ns cap).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("1996-01-02", periods=n_days)
    spy_ret = rng.normal(spy_mu, spy_sig, n_days)
    jpy_ret = hedge * np.where(spy_ret < 0, -spy_ret, 0.0) \
        + rng.normal(0.0, fx_sig, n_days)
    spy = 100.0 * np.exp(np.cumsum(np.log1p(spy_ret)))
    usdjpy = 110.0 * np.exp(-np.cumsum(np.log1p(jpy_ret)))   # yen up => USDJPY down
    irx = np.full(n_days, 3.0) + rng.normal(0, 0.02, n_days).cumsum() * 0.01
    tnx = np.full(n_days, 4.0) + rng.normal(0, 0.03, n_days).cumsum() * 0.01
    return pd.DataFrame({"usdjpy": usdjpy, "spy": spy,
                         "irx": np.clip(irx, 0.0, None),
                         "tnx": np.clip(tnx, 0.0, None)}, index=idx)
