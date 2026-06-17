"""Data layer for Study 216 (Hemline-Index).

Two tapes, one shape (a decade-level panel linking hemline proxy to market returns):

- ``synthetic_decades`` -- a *deterministic, offline* generator.  A ``hemline_boost``
  knob injects a correlation between the hemline direction and market returns, so tests
  can confirm the engine detects a planted association and finds nothing when the knob
  is zero.  ``hemline_boost = 0.0`` is the pure null: fashion and equities are
  independent.

- ``load_real_tape`` -- the hardcoded historical table.  Hemline direction is proxied
  decade by decade using fashion-history consensus (short = bullish, long = bearish);
  S&P 500 / broad-market decade returns come from the Shiller shared cache if available,
  or fall back to pre-computed numbers embedded here.  n = 10 decades (1920s-2010s) --
  a brutally small sample that is the whole point of the teardown.

No look-ahead issue: hemline *direction* (rising vs falling) is only observable at the
end of a fashion cycle, not its start -- making even a real correlation untradeable.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SHILLER_CACHE = os.path.abspath(
    os.path.join(HERE, "..", "..", "..", "_cache", "shiller_sp500.parquet")
)

# ---------------------------------------------------------------------------
# Hardcoded decade-level hemline proxy table
# ---------------------------------------------------------------------------
# Sources: fashion-history consensus from Economist (2010), Wolfe (2006),
# Bhardwaj et al. (2016), and popular accounts.
# Hemline: +1 = skirts rising/short (bullish claim), -1 = falling/long (bearish claim).
# Decade S&P 500 total return (price-only, approximate, from Shiller Dec closes).
# The market return column is used only if real data is unavailable; otherwise
# we compute from Shiller parquet.
#
# Note on the proxy: "hemline direction" is inherently subjective and contested.
# Fashion historians disagree on the 1930s, 1970s, 1990s.  We use the most
# commonly cited encoding; alternative encodings are discussed in references.md.

HEMLINE_TABLE = pd.DataFrame(
    {
        "decade_start": [1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010],
        "decade_label": [
            "1920s", "1930s", "1940s", "1950s", "1960s",
            "1970s", "1980s", "1990s", "2000s", "2010s",
        ],
        # Hemline direction for each decade:
        # +1 = rising/short hemlines dominated (bullish claim)
        # -1 = falling/long hemlines dominated (bearish claim)
        "hemline_dir": [1, -1, -1, 1, 1, -1, 1, 1, -1, -1],
        # Fashion description (used in notebooks / display)
        "fashion_note": [
            "Flappers; rising hemlines, Charleston era",
            "Depression; hemlines fall to floor (Maxi)",
            "WWII practical; moderate/falling (fabric rationing)",
            "Postwar prosperity; hemlines modestly rising",
            "Miniskirt revolution; sharply rising",
            "Maxi, midi fashion resurgence; falling hemlines",
            "Power dressing; moderate/rising (mixed)",
            "Grunge then feminism; rising trend overall",
            "Boho/maxi revival; falling hemlines",
            "Maxi & midi resurgence; falling hemlines",
        ],
        # Approximate decade S&P 500 price-only return (frozen fallback only)
        # Computed from Shiller Dec prices: (P_end / P_start - 1) * 100
        "sp500_decade_pct": [
            219.0,   # 1920s: massive bull
            -41.0,   # 1930s: Depression crash
            34.0,    # 1940s: WWII recovery
            268.0,   # 1950s: postwar bull
            53.0,    # 1960s: moderate growth
            17.0,    # 1970s: stagflation
            227.0,   # 1980s: Reagan bull
            315.0,   # 1990s: dot-com bull
            -24.0,   # 2000s: lost decade
            190.0,   # 2010s: QE bull
        ],
    }
).set_index("decade_start")


def load_real_tape(
    shiller_cache: str = SHILLER_CACHE,
) -> pd.DataFrame:
    """Decade-level panel: hemline direction + S&P 500 decade return.

    If the Shiller cache is available the decade returns are computed directly
    from December-close prices (same method as Studies 161, 136).  Otherwise
    the frozen values from HEMLINE_TABLE are used.

    Returns a DataFrame indexed by ``decade_start`` with columns:
    ``decade_label``, ``hemline_dir``, ``fashion_note``, ``sp500_decade_pct``,
    ``market_positive`` (bool).
    """
    df = HEMLINE_TABLE.copy()
    if os.path.exists(shiller_cache):
        try:
            raw = pd.read_parquet(shiller_cache)
            raw["Date"] = pd.to_datetime(raw["Date"])
            raw = raw[raw["SP500"] > 0].copy()
            raw["year"] = raw["Date"].dt.year
            raw["month"] = raw["Date"].dt.month
            dec = raw[raw["month"] == 12].groupby("year")["SP500"].last()
            for d_start in df.index:
                d_end = d_start + 9
                if d_start in dec.index and d_end in dec.index:
                    ret = (dec[d_end] / dec[d_start] - 1.0) * 100.0
                    df.loc[d_start, "sp500_decade_pct"] = ret
        except Exception:
            pass  # Fall back to frozen values
    df["market_positive"] = df["sp500_decade_pct"] > 0
    df["market_bullish"] = df["hemline_dir"] == 1  # claimed direction
    return df.sort_index()


# ---------------------------------------------------------------------------
# Synthetic tape -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_decades(
    n_decades: int = 50,
    base_positive_prob: float = 0.65,
    hemline_boost: float = 0.0,
    seed: int = 216,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible decade-level synthetic tape with optional hemline-market link.

    Hemline directions are drawn i.i.d. from Bernoulli(0.5) (+1 / -1).
    Market direction (positive / negative decade) is drawn from Bernoulli(base_positive_prob).
    When ``hemline_boost > 0`` the market is MORE likely to be positive in rising-hemline
    decades: P(market_up | hemline_up) = base_positive_prob + hemline_boost.
    ``hemline_boost = 0.0`` is the pure null.

    Returns ``(df, truth)`` where ``df`` has columns
    ``hemline_dir``, ``market_positive``, ``market_bullish``, ``decade_return_proxy``.
    """
    rng = np.random.default_rng(seed)
    # Hemline: +1 or -1, each with probability 0.5
    hemline_dirs = rng.choice([-1, 1], size=n_decades)
    # Market return sign -- boosted when hemline is rising
    p_up = np.where(
        hemline_dirs == 1,
        np.clip(base_positive_prob + hemline_boost, 0.0, 1.0),
        np.clip(base_positive_prob - hemline_boost, 0.0, 1.0),
    )
    market_positive = rng.random(n_decades) < p_up
    decade_returns = np.where(market_positive, 1.0, -1.0) * (
        20.0 + rng.exponential(60.0, n_decades)
    )

    df = pd.DataFrame(
        {
            "hemline_dir": hemline_dirs,
            "market_positive": market_positive,
            "market_bullish": hemline_dirs == 1,
            "decade_return_proxy": decade_returns,
        }
    )
    truth = {
        "n_decades": n_decades,
        "base_positive_prob": base_positive_prob,
        "hemline_boost": hemline_boost,
        "seed": seed,
        "n_rising": int((hemline_dirs == 1).sum()),
        "n_falling": int((hemline_dirs == -1).sum()),
    }
    return df, truth


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of the real tape (sp500_decade_pct column)."""
    arr = df["sp500_decade_pct"].to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
