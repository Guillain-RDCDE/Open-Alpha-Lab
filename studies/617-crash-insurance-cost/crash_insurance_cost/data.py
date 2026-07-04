"""Data layer for Study 617 — Crash-Insurance-Cost.

Two sources, both offline-friendly once cached:

* **Real tape.** Daily yfinance closes for ``TAIL`` (the Cambria Tail Risk ETF — the
  live, buyable crash-insurance product: ~90% intermediate US Treasuries + a ladder of
  out-of-the-money SPX puts, ~1%/quarter option budget, launched 2017-04-05),
  ``IEF`` (7-10y Treasury ETF — a clean proxy for TAIL's collateral sleeve),
  ``SPY`` (the thing being insured) and ``^VIX`` (the price of the insurance).
  TAIL/IEF/SPY are **total-return** (auto-adjusted, distributions folded in); ^VIX is a
  level. Cached wide under the study's own ``_cache/cic_tape.csv``.

* **Synthetic world.** A deterministic, seeded daily generator shaped exactly like the
  real tape, with a TUNABLE planted **net insurance bleed**: the synthetic fund holds a
  bond sleeve plus a put sleeve that pays a premium continuously and collects rare crash
  jumps. ``bleed = 0`` is the *fair-insurance* null (expected jump payoffs exactly repay
  the premium — the decomposition must find **no** drag); a planted ``bleed`` (premium
  paid in excess of expected payoff) must light up. A machinery proof only, never market
  evidence.

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
TAPE_CACHE = os.path.join(CACHE_DIR, "cic_tape.csv")

TICKERS = {"TAIL": "tail", "IEF": "ief", "SPY": "spy", "^VIX": "vix"}

# Frozen as-of: last complete calendar month at build time (2026-07-03).
AS_OF = "2026-06-30"

# TAIL facts (fund page: cambriafunds.com/tail; prospectus): inception 2017-04-05,
# expense ratio 0.59%/yr (inside the NAV), design = laddered OTM SPX puts (~1%/quarter
# budget) on a US-intermediate-Treasury collateral sleeve.
TAIL_INCEPTION = "2017-04-05"
TAIL_EXPENSE_BPS = 59.0

# COVID episode anchors (the one time the insurance paid): SPY's 2020 peak / trough.
COVID_PEAK = "2020-02-19"    # SPY all-time high before the crash
COVID_TROUGH = "2020-03-23"  # SPY bear-market low


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_tape(start: str = "1990-01-01", end: str | None = None,
               path: str = TAPE_CACHE, retries: int = 3) -> pd.DataFrame:
    """Download TAIL / IEF / SPY / ^VIX daily closes and cache them (network, once)."""
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
    """Cached wide daily tape (tail, ief, spy, vix), sliced to the frozen as-of."""
    tape = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if asof is not None:
        tape = tape[tape.index <= pd.Timestamp(asof)]
    return tape


# --------------------------------------------------------------------------- #
# Synthetic world — a machinery proof with a TUNABLE planted insurance bleed
# --------------------------------------------------------------------------- #
def synthetic_world(bleed: float = 0.0, n_days: int = 7500, seed: int = 617,
                    payoff_ann: float = 0.05, jump_prob: float = 1.0 / 252.0,
                    bond_mu: float = 1.2e-4, bond_sig: float = 0.0035,
                    spy_mu: float = 4e-4, spy_sig: float = 0.011,
                    sleeve_noise: float = 8e-4, bond_beta: float = 0.90) -> pd.DataFrame:
    """Deterministic daily world with a planted net insurance bleed ``bleed`` (per year).

    The synthetic fund is ``bond_beta`` x a bond sleeve plus a put sleeve. The put
    sleeve pays a continuous premium of ``(payoff_ann + bleed)``/yr and collects crash
    jumps worth ``payoff_ann``/yr in expectation (size ``payoff_ann/(252*jump_prob)``
    with probability ``jump_prob`` per day). With ``bleed = 0`` the insurance is
    actuarially FAIR — the decomposition-vs-bonds alpha must sit at ~0; a planted
    ``bleed`` (e.g. 0.05 = pay 5%/yr more than you collect) must show up as a
    significantly negative alpha. Returns a frame shaped like the real tape
    (tail/ief/spy/vix levels) so every real-tape function runs on it unchanged.
    Span ~30 years (well under the 250-year ns cap).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("1996-01-02", periods=n_days)
    bond_ret = rng.normal(bond_mu, bond_sig, n_days)
    spy_ret = rng.normal(spy_mu, spy_sig, n_days)
    jump_size = payoff_ann / (252.0 * jump_prob)
    jumps = np.where(rng.random(n_days) < jump_prob, jump_size, 0.0)
    sleeve = -(payoff_ann + bleed) / 252.0 + jumps + rng.normal(0.0, sleeve_noise, n_days)
    fund_ret = bond_beta * bond_ret + sleeve
    ief = 100.0 * np.exp(np.cumsum(np.log1p(bond_ret)))
    spy = 100.0 * np.exp(np.cumsum(np.log1p(spy_ret)))
    tail = 25.0 * np.exp(np.cumsum(np.log1p(fund_ret)))
    vix = np.clip(18.0 + rng.normal(0, 0.5, n_days).cumsum() * 0.05, 10.0, 80.0)
    return pd.DataFrame({"tail": tail, "ief": ief, "spy": spy, "vix": vix}, index=idx)
