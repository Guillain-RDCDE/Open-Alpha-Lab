"""Data layer for Study 386 — NFCI-Conditions (financial-conditions proxy + S&P 500).

Two sources, both offline-friendly:

* **Real tape.** A weekly **financial-conditions proxy** built from yfinance instruments,
  cached under ``_cache/nfci_proxy.csv``. The true Chicago-Fed NFCI lives only on FRED
  (``NFCI``), which is not reachable from this environment, so we construct a transparent
  stand-in from tradable risk gauges and z-score it to NFCI's sign convention
  (**positive => tighter conditions**):

    - equity volatility       ``^VIX``               (tight when high)
    - rates volatility        ``^MOVE``              (tight when high)
    - corporate credit spread ``LQD`` vs ``IEF``     (IG total-return underperformance of
                                                      duration-matched Treasuries => wider
                                                      spread => tight)
    - broad-dollar momentum   ``UUP``                (a rising dollar tightens global $-funding)

  Each leg is converted to a weekly level, standardised (z-score over the common window),
  and averaged into one index ``fci`` (high = tight). The cached file is a small weekly CSV
  with columns ``spy`` (S&P 500 weekly close, from ``^GSPC``) and ``fci``.

* **Synthetic.** A deterministic, fixed-seed generator producing a controllable conditions
  index with a PLANTED forward edge knob: ``edge`` is the extra forward drift that loose
  (low-FCI) weeks earn over tight (high-FCI) weeks. ``edge = 0`` must NOT manufacture a
  significant timing edge; a large ``edge`` must light up. This is the positive control — a
  machinery proof, never market evidence.

Pure numpy + pandas + stdlib for the offline path. ``fetch_proxy`` (network) is only used
once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "..", "..", "_cache", "nfci_proxy.csv")

# Instruments used to build the conditions proxy. ^VIX/^GSPC reach back to 1990; the credit
# and rates legs (LQD/IEF/^MOVE) start 2002, UUP 2007 — the index is z-scored over the
# common overlapping window so each leg contributes on the same scale.
PROXY_TICKERS = ["^GSPC", "^VIX", "^MOVE", "LQD", "IEF", "UUP"]


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_proxy(start: str = "2002-01-01", end: str | None = None,
                path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Download the proxy instruments via yfinance and cache a weekly conditions index.

    Network-only; used once to build ``_cache/nfci_proxy.csv``. Never imported by the
    offline notebook cells. Returns the weekly frame (``spy``, ``fci`` and the leg z-scores).
    """
    import yfinance as yf

    raw = yf.download(PROXY_TICKERS, start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    weekly = raw.resample("W-FRI").last()
    # drop the in-progress final week (house rule: no partial bar in a stamped run)
    weekly = weekly.iloc[:-1] if len(weekly) else weekly
    frame = build_fci(weekly)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frame.to_csv(path)
    return frame


def build_fci(weekly: pd.DataFrame) -> pd.DataFrame:
    """Assemble the weekly financial-conditions proxy from instrument closes.

    ``weekly`` is a wide frame of Friday closes (columns = PROXY_TICKERS). Returns a frame
    indexed by week with columns:
      * ``spy``  — S&P 500 weekly close (from ``^GSPC``)
      * ``fci``  — the conditions proxy (z-scored, **high = tight**), NFCI's sign
      * one z-scored column per leg (``z_vix``, ``z_move``, ``z_credit``, ``z_dollar``)
    """
    spy = weekly["^GSPC"]

    # --- legs, each oriented so that HIGHER => TIGHTER conditions ---
    vix = weekly["^VIX"]                                   # high vol = tight
    move = weekly["^MOVE"]                                 # high rates vol = tight
    # credit spread proxy: 26-week relative drawdown of IG credit vs duration-matched
    # Treasuries. When LQD underperforms IEF, spreads are widening => tighter credit.
    rel = np.log(weekly["LQD"]) - np.log(weekly["IEF"])
    credit = -(rel - rel.rolling(26, min_periods=8).mean())   # high when IG lags => tight
    # broad-dollar momentum: 26-week dollar return; a rising dollar tightens $-funding
    dollar = np.log(weekly["UUP"]) - np.log(weekly["UUP"].shift(26))

    def _z(s: pd.Series) -> pd.Series:
        m, sd = s.mean(), s.std(ddof=0)
        return (s - m) / sd if sd and np.isfinite(sd) else s * 0.0

    z_vix, z_move, z_credit, z_dollar = _z(vix), _z(move), _z(credit), _z(dollar)
    legs = pd.concat([z_vix, z_move, z_credit, z_dollar], axis=1)
    legs.columns = ["z_vix", "z_move", "z_credit", "z_dollar"]
    # average the available legs each week (early weeks before all legs exist still get a
    # value from the legs that do); require at least the VIX leg present.
    fci = legs.mean(axis=1, skipna=True)
    # re-standardise the composite so the index itself is a clean z-score (high = tight)
    fci = (fci - fci.mean()) / fci.std(ddof=0)

    out = pd.concat([spy.rename("spy"), fci.rename("fci"), legs], axis=1)
    out = out.dropna(subset=["spy", "fci"])
    return out


def have_real(path: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(path)


def load_real(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load the cached weekly conditions-proxy frame (index = week, ``spy`` + ``fci`` + legs)."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return df


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_fci(n_weeks: int = 1_200, edge: float = 0.0, seed: int = 386,
                  mu_weekly: float = 0.0015, sig_weekly: float = 0.020,
                  rho: float = 0.80) -> pd.DataFrame:
    """Deterministic conditions index + S&P-like price with a PLANTED forward edge.

    The conditions index is the sum of two pieces:
      * a **persistent** part ``base`` (AR(1), ``rho``) — the slow regime;
      * a **transient** tightness shock ``e`` (i.i.d.) — a one-week spike.

    The price couples to each piece differently, which is the whole point of the control:
      * **Contemporaneous (mechanical, NOT predictive).** This week's return falls with the
        *transient* shock ``e`` — exactly the real index's tautology (it embeds equity vol),
        so tight weeks *are* down weeks. Because ``e`` is i.i.d., this link carries **zero
        forward information**: a one-week-lagged timing rule cannot harvest it.
      * **Forward (the planted edge).** Next week's drift falls with ``edge`` × the
        *persistent* tightness — i.e. tight regimes genuinely predict lower forward returns by
        a knob we set.

    With ``edge = 0`` only the contemporaneous link exists: the forward Welch *t* must stay
    near 0 (no false positive — a step-out rule earns nothing predictive) even though the
    contemporaneous correlation is strongly negative, just like the real tape. A large ``edge``
    must drive the forward *t* well past 2. That is the positive control — demonstrated on data
    where we know the truth.

    A decorative weekly ``period_range`` index avoids any OutOfBoundsDatetime on long spans.
    """
    rng = np.random.default_rng(seed)

    # persistent regime (AR(1)) + transient i.i.d. tightness shock
    base = np.empty(n_weeks)
    base[0] = 0.0
    for t in range(1, n_weeks):
        base[t] = rho * base[t - 1] + rng.normal(0, np.sqrt(1 - rho ** 2))
    e = rng.normal(0.0, 1.0, size=n_weeks)
    fci = base + 0.8 * e
    fci = (fci - fci.mean()) / fci.std()

    ret = rng.normal(mu_weekly, sig_weekly, size=n_weeks)
    ret += -0.010 * e                                  # contemporaneous (transient, no foresight)
    ret[1:] += -edge * base[:-1]                        # forward edge on the PERSISTENT regime

    price = 100.0 * np.exp(np.cumsum(ret))
    idx = pd.period_range("1990-01-07", periods=n_weeks, freq="W-FRI")
    return pd.DataFrame({"spy": price, "fci": fci}, index=idx)
