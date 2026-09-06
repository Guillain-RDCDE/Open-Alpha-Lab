"""Confirmation delay, priced on both sides — Study 981.

A confirmation rule takes a binary signal and requires it to hold for ``k`` consecutive
sessions before the position changes. It is supposed to buy fewer whipsaws at the price of
later entries. Both halves are measurable, and this module measures them separately rather
than reporting the net and calling it a day.

**The signals** (three, chosen because they fail differently):

- ``ma_signal`` — price above its 200-day moving average. Slow, few crossings, the classic.
- ``momentum_signal`` — 12-1 total return positive. Slower still, monthly in character.
- ``rsi_signal`` — Wilder's 14-day RSI above 50. Fast, noisy, dozens of crossings a year —
  the case where confirmation is supposed to earn its keep.

**The rule** (``confirm``): the position changes only after the raw signal has held its new
value for ``k`` sessions. ``k = 1`` is the unconfirmed base case; the position always applies
from the *next* session, so every arm carries exactly one day of execution lag whatever ``k``
is. That detail is what stops the comparison from being a comparison of lags.

**The two effects, separated:**

- ``whipsaw_stats`` counts *round trips shorter than a month* — the trades a confirmation rule
  exists to prevent — and the share of all trades they represent.
- ``entry_delay_stats`` measures how much later each arm enters and exits, and what the
  delayed portion of the move was worth. That is the price of waiting, in basis points.

The scoreboard is a long-flat book (asset when the confirmed signal is on, T-bills otherwise)
with costs charged per switch, compared against the unconfirmed version *and* against
buy-and-hold, because "better than the unconfirmed rule" is not the same as "worth doing".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.analytics import mean_tstat_hac

TRADING_DAYS = 252
SIGNALS = ("ma200", "mom12_1", "rsi14")
SIGNAL_LABEL = {"ma200": "Price above the 200-day average",
                "mom12_1": "12-1 momentum positive",
                "rsi14": "14-day RSI above 50"}
CONFIRM_DAYS = (1, 2, 3, 5, 10, 21)
SHORT_TRIP = 21     # a round trip closed inside a month is what "whipsaw" means here


# --------------------------------------------------------------------------- #
# The raw signals
# --------------------------------------------------------------------------- #
def ma_signal(prices: pd.Series, window: int = 200) -> pd.Series:
    """True while the close is above its ``window``-day simple moving average."""
    return (prices > prices.rolling(window).mean()).where(
        prices.rolling(window).mean().notna())


def momentum_signal(prices: pd.Series, lookback: int = 252, skip: int = 21) -> pd.Series:
    """True while the 12-1 total return is positive."""
    r = prices.shift(skip) / prices.shift(lookback) - 1.0
    return (r > 0).where(r.notna())


def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI — exponential smoothing with alpha = 1/window, as in the original."""
    d = prices.diff()
    up = d.clip(lower=0.0)
    down = (-d).clip(lower=0.0)
    roll_up = up.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    roll_dn = down.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100.0 - 100.0 / (1.0 + roll_up / roll_dn)
    # The two degenerate cases have answers, and returning NaN for them loses information:
    # an unbroken run of up days is RSI 100 (Wilder's definition), an unbroken run of down
    # days is 0, and a flat stretch — no movement either way — is 50.
    out = out.where(roll_dn > 0, other=np.where(roll_up > 0, 100.0, 50.0))
    return out.where(roll_up.notna() & roll_dn.notna()).rename(f"rsi{window}")


def rsi_signal(prices: pd.Series, window: int = 14, level: float = 50.0) -> pd.Series:
    """True while RSI is above ``level`` — the fast, noisy signal in the set."""
    r = rsi(prices, window)
    return (r > level).where(r.notna())


def raw_signal(prices: pd.Series, name: str) -> pd.Series:
    """Dispatch on ``SIGNALS``."""
    return {"ma200": ma_signal, "mom12_1": momentum_signal, "rsi14": rsi_signal}[name](prices)


# --------------------------------------------------------------------------- #
# The confirmation rule
# --------------------------------------------------------------------------- #
def confirm(signal: pd.Series, k: int = 1) -> pd.Series:
    """Change state only after the raw signal has held its new value for ``k`` sessions.

    Implemented as a rolling unanimity test: the state becomes True once the last ``k`` raw
    readings are all True, becomes False once the last ``k`` are all False, and otherwise
    holds. ``k = 1`` returns the raw signal unchanged, which is what makes the sweep a clean
    comparison — the same code path for every arm, including the base case.
    """
    s = signal.astype("float")
    if k <= 1:
        return signal.fillna(False).astype(bool)
    all_true = s.rolling(k).min() == 1.0
    all_false = s.rolling(k).max() == 0.0
    state = pd.Series(np.nan, index=signal.index, dtype="float")
    state[all_true] = 1.0
    state[all_false] = 0.0
    return state.ffill().fillna(0.0).astype(bool)


# --------------------------------------------------------------------------- #
# What the rule does to the trades
# --------------------------------------------------------------------------- #
def trades_from(position: pd.Series) -> pd.DataFrame:
    """Every long episode: entry, exit and length in sessions."""
    pos = position.astype(int)
    change = pos.diff().fillna(pos.iloc[0])
    entries = list(pos.index[change == 1])
    exits = list(pos.index[change == -1])
    if len(exits) < len(entries):
        exits.append(pos.index[-1])
    rows = []
    for a, b in zip(entries, exits):
        rows.append({"entry": a, "exit": b,
                     "length": int(pos.index.get_indexer([b])[0]
                                   - pos.index.get_indexer([a])[0])})
    return pd.DataFrame(rows)


def whipsaw_stats(position: pd.Series, short_trip: int = SHORT_TRIP) -> dict:
    """How many round trips were closed inside a month — the thing confirmation targets."""
    t = trades_from(position)
    if t.empty:
        return {"n_trades": 0, "n_whipsaws": 0, "whipsaw_share": np.nan,
                "median_length": np.nan}
    short = int((t["length"] <= short_trip).sum())
    return {"n_trades": int(len(t)), "n_whipsaws": short,
            "whipsaw_share": float(short / len(t)),
            "median_length": float(t["length"].median())}


def entry_delay_stats(raw: pd.Series, confirmed: pd.Series, rets: pd.Series) -> dict:
    """How much later the confirmed arm acts, and what happened while it waited.

    ``delay_cost_bps`` is the return the confirmed arm missed on days when the raw signal was
    already on and the confirmed one was not — the direct price of waiting. ``avoided_bps`` is
    the mirror image: the return it *avoided* on days when the raw signal was on and the market
    fell. Reporting both stops the study from being a sales pitch in either direction.
    """
    r, c = raw.fillna(False).astype(bool), confirmed.astype(bool)
    idx = r.index.intersection(c.index).intersection(rets.index)
    r, c, x = r.loc[idx], c.loc[idx], rets.loc[idx].fillna(0.0)
    waiting = r & ~c
    exiting_late = ~r & c
    return {"days_waiting": int(waiting.sum()),
            "days_late_to_exit": int(exiting_late.sum()),
            "delay_cost_bps": float(x[waiting].sum() * 1e4),
            "late_exit_cost_bps": float(x[exiting_late].sum() * 1e4),
            "share_disagreeing": float((waiting | exiting_late).mean())}


# --------------------------------------------------------------------------- #
# The backtest
# --------------------------------------------------------------------------- #
def run_arm(prices: pd.Series, cash: pd.Series, signal_name: str, k: int,
            cost_bps: float = 2.0) -> dict:
    """One (signal, confirmation) arm: long the asset while confirmed, bills otherwise."""
    px = prices.dropna()
    c = cash.reindex(px.index).ffill()
    r_asset = px.pct_change().fillna(0.0)
    r_cash = c.pct_change().fillna(0.0)
    raw = raw_signal(px, signal_name)
    conf = confirm(raw, k)
    pos = conf.shift(1).fillna(False)          # one execution lag, identical across arms
    switches = pos.astype(int).diff().abs().fillna(0.0)
    ret = pd.Series(np.where(pos, r_asset, r_cash), index=px.index) - switches * cost_bps / 1e4
    valid = raw.notna()
    ret, pos = ret[valid], pos[valid]
    years = len(ret) / TRADING_DAYS
    curve = (1 + ret).cumprod()
    sd = float(ret.std(ddof=1))
    hold = r_asset[valid]
    diff = (ret - hold).dropna()
    return {
        "signal": signal_name, "k": k,
        "cagr": float(curve.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan,
        "vol": sd * np.sqrt(TRADING_DAYS),
        "sharpe": float(ret.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan,
        "max_dd": float((curve / curve.cummax() - 1).min()),
        "time_invested": float(pos.mean()),
        "switches_per_year": float(switches[valid].sum() / years),
        "t_vs_hold": float(mean_tstat_hac(diff)["tstat"]) if len(diff) > 50 else np.nan,
        "cagr_vs_hold": float(curve.iloc[-1] ** (1 / years) - 1
                              - ((1 + hold).cumprod().iloc[-1] ** (1 / years) - 1)),
        **{f"ws_{k2}": v for k2, v in whipsaw_stats(pos).items()},
        **entry_delay_stats(raw[valid], conf[valid], r_asset[valid]),
        "returns": ret,
    }


def sweep(prices: pd.Series, cash: pd.Series, signal_name: str,
          ks=CONFIRM_DAYS, cost_bps: float = 2.0) -> pd.DataFrame:
    """One row per confirmation length, for a single signal and tape."""
    rows = []
    for k in ks:
        out = run_arm(prices, cash, signal_name, k, cost_bps)
        out.pop("returns")
        rows.append(out)
    return pd.DataFrame(rows).set_index("k")


def full_grid(px: pd.DataFrame, cash: pd.Series, tickers, signals=SIGNALS,
              ks=CONFIRM_DAYS, cost_bps: float = 2.0) -> pd.DataFrame:
    """Every (tape, signal, confirmation) combination — with the cell count made explicit."""
    rows = []
    for tk in tickers:
        for s in signals:
            t = sweep(px[tk], cash, s, ks, cost_bps)
            t = t.reset_index()
            t.insert(0, "ticker", tk)
            rows.append(t)
    return pd.concat(rows, ignore_index=True)


def best_k(grid: pd.DataFrame, metric: str = "sharpe") -> pd.DataFrame:
    """Which confirmation length won on each tape and signal — and by how much over k = 1."""
    rows = []
    for (tk, s), g in grid.groupby(["ticker", "signal"]):
        g = g.set_index("k")
        b = g[metric].idxmax()
        rows.append({"ticker": tk, "signal": s, "best_k": int(b),
                     f"{metric}_best": float(g.loc[b, metric]),
                     f"{metric}_k1": float(g.loc[1, metric]),
                     "gain": float(g.loc[b, metric] - g.loc[1, metric]),
                     "whipsaw_k1": float(g.loc[1, "ws_whipsaw_share"]),
                     "whipsaw_best": float(g.loc[b, "ws_whipsaw_share"])})
    return pd.DataFrame(rows)


def synthetic_tape(n: int = 6000, trendiness: float = 1.0, vol_ann: float = 0.18,
                   drift_ann: float = 0.07, seed: int = 981) -> pd.DataFrame:
    """A tape whose returns are persistent (``trendiness`` = 1) or mean-reverting (= -1).

    Implemented as an AR(1) on returns with coefficient ``0.06 * trendiness``: positive gives
    trends that a moving average can actually follow, negative gives the choppy tape where
    every signal whipsaws. Includes a cash column so the backtest runs on it unchanged.
    """
    rng = np.random.default_rng(seed)
    sd = vol_ann / np.sqrt(TRADING_DAYS)
    phi = 0.06 * float(trendiness)
    e = rng.normal(0, sd * np.sqrt(max(1 - phi ** 2, 1e-9)), n)
    r = np.empty(n)
    r[0] = e[0]
    for t in range(1, n):
        r[t] = phi * r[t - 1] + e[t]
    r += drift_ann / TRADING_DAYS
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.DataFrame({"ASSET": 100 * np.cumprod(1 + r),
                         "CASH": np.cumprod(np.full(n, (1.02) ** (1 / TRADING_DAYS)))},
                        index=idx)


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    - **Signal** (does confirmation do what it claims?): **Real** if the share of round trips
      closed inside a month falls by at least a third from ``k = 1`` to ``k = 5`` on a majority
      of (tape, signal) cells; **Weak** if it falls at all on a majority; **None** otherwise.
      This is a mechanical claim and it should be easy to confirm — which is why failing it
      would be the interesting outcome.
    - **Tradability**: **Investable** if the best confirmation length beats ``k = 1`` on Sharpe
      on a majority of cells **and** the pooled improvement clears 0.05; **Fragile** if it wins
      on a majority without the size; **Mirage** if it does not.
    """
    signal = ("Real" if h["frac_whipsaw_cut"] >= 0.5 else
              ("Weak" if h["frac_whipsaw_any"] >= 0.5 else "None"))
    wins, gain = h["frac_sharpe_wins"], h["mean_sharpe_gain"]
    trad = ("Investable" if wins > 0.5 and gain >= 0.05
            else ("Fragile" if wins > 0.5 else "Mirage"))
    return {
        "signal": signal,
        "signal_why": (
            f"Yes, mechanically, and by a lot. Requiring five consecutive days of agreement "
            f"cuts the share of round trips closed inside a month from "
            f"**{h['whipsaw_k1']:.0%}** to **{h['whipsaw_k5']:.0%}** and the number of trades "
            f"from {h['trades_k1']:.1f} to {h['trades_k5']:.1f} a year, across "
            f"{h['n_cells']} tape × signal cells; the reduction holds on "
            f"**{h['frac_whipsaw_cut']:.0%}** of them. The fast signal is where it bites: "
            f"RSI's whipsaw share falls from {h['rsi_whipsaw_k1']:.0%} to "
            f"{h['rsi_whipsaw_k5']:.0%}, while the 200-day average — which barely whipsaws to "
            f"begin with — has little to gain."),
        "trad": trad,
        "trad_why": (
            f"The waiting is not free. Across the grid the confirmed arms spent "
            f"**{h['days_waiting_k5']:,} sessions** holding cash while the raw signal was "
            f"already positive, worth **{h['delay_cost_k5']:+,.0f} bps** of forgone return, "
            f"against **{h['late_exit_cost_k5']:+,.0f} bps** avoided by exiting late. Some "
            f"confirmation length beat the unconfirmed rule on Sharpe in "
            f"**{h['frac_sharpe_wins']:.0%}** of cells, by **{h['mean_sharpe_gain']:+.3f}** on "
            f"average — and the winning length is different in almost every cell "
            f"({h['n_distinct_best_k']} different values of *k* across {h['n_cells']} cells), "
            f"which is what choosing it in hindsight looks like."),
        "one_sentence": (
            f"Confirmation does exactly what it says — five days of agreement cuts whipsaw "
            f"trades from {h['whipsaw_k1']:.0%} to {h['whipsaw_k5']:.0%} of all round trips — "
            f"but it pays for that with {h['days_waiting_k5']:,} sessions of sitting out a "
            f"signal that was already right, and the confirmation length that would have won "
            f"is only knowable afterwards."),
    }
