"""Cross-sectional breadth — is the overnight effect a firm-level fact?

Ten ETFs could be a fluke or a data quirk. The decisive credibility test is the
**firm level**: does the overnight-vs-intraday split hold across hundreds of
individual stocks (cf. Lou, Polk, and Skouras 2019)? This module reproduces that
on the current S&P 500 constituents.

SURVIVORSHIP GUARD — read before anything cross-sectional
----------------------------------------------------------
This engine starts from *current* S&P 500 membership projected backwards:
delisted names (bankruptcies, buy-outs, deletions) are absent, so any
cross-sectional backtest built on it is biased **upward** — the losers were
removed before the test ever saw them. The qualitative breadth result is
robust to this; precise magnitudes are not. Mirroring
``quantlab.hf_data.load_panel``, the cross-sectional entry points here
(:func:`sp500_symbols`, :func:`download_panel`,
:func:`cross_section_decompose`) **refuse to run** unless you pass
``allow_survivorship_bias=True`` — an explicit, documented opt-in, with a
warning so the caveat survives into logs.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .decompose import decompose, summary
from .hf_data import SurvivorshipBiasError  # one guard class for the whole desk

_CACHE = Path(os.environ.get("OVERNIGHT_CACHE", "_cache"))
_WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

__all__ = [
    "SurvivorshipBiasError",
    "sp500_symbols",
    "panel_cache_path",
    "download_panel",
    "cross_section_decompose",
]


_MEMBERSHIP_NOTE = (
    "is built on *current* S&P 500 membership projected backwards, so delisted "
    "names are missing and any cross-sectional backtest on it is biased upward"
)


def _require_optin(flag: bool, what: str, note: str = _MEMBERSHIP_NOTE) -> None:
    """Refuse cross-sectional use of the survivor universe without the opt-in."""
    if flag:
        warnings.warn(
            f"{what} {note}. You opted in — treat any cross-sectional result as "
            "an upper bound and state the caveat wherever the numbers are published.",
            stacklevel=3,
        )
        return
    raise SurvivorshipBiasError(
        f"Refusing: {what} {note}. Pass allow_survivorship_bias=True to "
        "acknowledge the bias (and document it where you publish), or use a "
        "point-in-time membership source."
    )


def sp500_symbols(use_cache: bool = True, allow_survivorship_bias: bool = False) -> list[str]:
    """Current S&P 500 tickers (Yahoo format). Wikipedia, cached to JSON.

    GUARDED: the list is *today's* membership — feeding it to any historical
    backtest injects survivorship bias, which is how every caller in this repo
    uses it. Requires ``allow_survivorship_bias=True`` (see module docstring).

    Class shares use '-' on Yahoo (e.g. BRK-B), so '.' is translated.
    """
    _require_optin(allow_survivorship_bias, "sp500_symbols()")
    _CACHE.mkdir(parents=True, exist_ok=True)
    cache = _CACHE / "sp500_symbols.json"
    if use_cache and cache.exists():
        return json.loads(cache.read_text())
    # Wikipedia 403s the default urllib UA; fetch with a browser UA, then parse.
    import io
    import urllib.request

    req = urllib.request.Request(_WIKI, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310 (trusted URL)
        html = r.read().decode("utf-8", "ignore")
    table = pd.read_html(io.StringIO(html))[0]  # needs lxml
    syms = [str(s).replace(".", "-").strip() for s in table["Symbol"].tolist()]
    cache.write_text(json.dumps(syms))
    return syms


def panel_cache_path(symbols: list[str], start: str = "2010-01-01") -> Path:
    """Where :func:`download_panel` caches the batched download for ``symbols``.

    The key embeds a 12-hex sha256 of the *sorted* symbol list, so two
    different lists that happen to share a length no longer collide (the old
    key was ``panel_<len>_<start>``: any 503-name list silently reused any
    other 503-name list's parquet). Exposed so cache-first study loaders can
    probe for the file without importing yfinance.
    """
    digest = hashlib.sha256(",".join(sorted(symbols)).encode("utf-8")).hexdigest()[:12]
    return _CACHE / f"panel_{len(symbols)}_{digest}_{start}.parquet"


def download_panel(
    symbols: list[str],
    start: str = "2010-01-01",
    use_cache: bool = True,
    allow_survivorship_bias: bool = False,
) -> dict:
    """Batched OHLC download for many tickers (one threaded call), cached.

    GUARDED: a multi-name panel keyed to today's membership is the textbook
    survivorship trap; requires ``allow_survivorship_bias=True``.

    Uses ``auto_adjust=True`` (total-return prices) for a clean, simple breadth
    study — documented choice. Returns ``{ticker: DataFrame[Open, High, Low,
    Close]}``, skipping tickers Yahoo returns empty. Cache key includes a hash
    of the sorted symbol list (see :func:`panel_cache_path`); a legacy
    ``panel_<len>_<start>.parquet`` is adopted (renamed) once if present, with
    a warning, since its symbol list cannot be verified.
    """
    _require_optin(allow_survivorship_bias, f"download_panel({len(symbols)} names)")
    import yfinance as yf  # lazy

    _CACHE.mkdir(parents=True, exist_ok=True)
    cache = panel_cache_path(symbols, start)
    legacy = _CACHE / f"panel_{len(symbols)}_{start}.parquet"
    if use_cache and not cache.exists() and legacy.exists():
        warnings.warn(
            f"Adopting legacy panel cache {legacy.name} as {cache.name}; the old key "
            "did not record WHICH symbols it held, so verify it matches this list "
            "(delete the file to force a clean re-download).",
            stacklevel=2,
        )
        os.replace(legacy, cache)
    if use_cache and cache.exists():
        raw = pd.read_parquet(cache)
    else:
        raw = yf.download(
            symbols, start=start, auto_adjust=True, progress=False,
            group_by="ticker", threads=True,
        )
        if use_cache:
            raw.to_parquet(cache)

    panel = {}
    for s in symbols:
        if s not in raw.columns.get_level_values(0):
            continue
        df = raw[s][["Open", "High", "Low", "Close"]].dropna()
        if not df.empty:
            panel[s] = df
    return panel


def cross_section_decompose(
    panel: dict, min_days: int = 750, allow_survivorship_bias: bool = False
) -> dict:
    """Per-symbol night/day stats + equal-weight aggregate across the cross-section.

    GUARDED: this function cannot see where ``panel`` came from, and in this
    repo it is fed survivor-universe panels; requires
    ``allow_survivorship_bias=True``. If your panel is genuinely point-in-time
    or synthetic, the flag is your explicit acknowledgment that you checked.

    Returns a dict with:
      ``per_symbol``   DataFrame indexed by ticker (overnight/intraday cum & Sharpe, n)
      ``ew_overnight`` / ``ew_intraday``  equal-weight daily cross-sectional mean returns
      ``frac_overnight_wins``  share of stocks whose overnight Sharpe > intraday Sharpe
      ``n_stocks``
    """
    _require_optin(
        allow_survivorship_bias,
        f"cross_section_decompose({len(panel)} names)",
        note=(
            "cannot verify its panel's provenance, and in this repo it is fed "
            "survivor-universe panels (delisted names absent, cross-sectional "
            "results biased upward)"
        ),
    )
    rows, on_cols, id_cols = {}, {}, {}
    for s, ohlc in panel.items():
        dec = decompose(ohlc)
        if len(dec) < min_days:
            continue
        sm = summary(dec)
        rows[s] = {
            "overnight_cum": sm.loc["overnight", "cum_return"],
            "intraday_cum": sm.loc["intraday", "cum_return"],
            "overnight_sharpe": sm.loc["overnight", "sharpe"],
            "intraday_sharpe": sm.loc["intraday", "sharpe"],
            "n": int(len(dec)),
        }
        on_cols[s] = dec["r_overnight"]
        id_cols[s] = dec["r_intraday"]

    per = pd.DataFrame(rows).T
    ew_on = pd.DataFrame(on_cols).mean(axis=1).dropna()
    ew_id = pd.DataFrame(id_cols).mean(axis=1).dropna()
    frac = float((per["overnight_sharpe"] > per["intraday_sharpe"]).mean()) if len(per) else np.nan
    return {
        "per_symbol": per,
        "ew_overnight": ew_on,
        "ew_intraday": ew_id,
        "frac_overnight_wins": frac,
        "n_stocks": int(len(per)),
    }
