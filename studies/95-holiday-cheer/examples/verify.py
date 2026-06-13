"""Headline run for Study 95 (Holiday-Cheer) on the real tapes.

Measures the pre-holiday effect two ways - SPY total-return (1993+, the fair tape) and
^GSPC price-only (1950+, the long sample the decay story needs) - derives the
pre-holiday calendar straight from the trading index, runs the pre-holiday-only book
against buy-and-hold, splits pre/post-1990 with a bootstrap test of the decay, and stamps
the as-of date + content fingerprint. Re-run to reproduce; match the fingerprint to
confirm you hold the same tape.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from holiday_cheer import data, strategy  # noqa: E402

AS_OF = "2026-06-12"
COST_BPS = 1.0


def _run(ticker: str, mode: str, start: str, label: str) -> None:
    frame = data.load_real(ticker, start=start, mode=mode).loc[:AS_OF]
    close = frame["close"]
    fp = data.fingerprint(frame)
    is_pre = data.pre_holiday_mask(close.index)
    n_hol = len(data.derive_market_holidays(close.index))

    print(f"\n=== {label} : {ticker} ({mode}) ===")
    print(f"[data] {len(close):,} rows  {close.index[0].date()} -> {close.index[-1].date()}  "
          f"as-of {AS_OF}  fingerprint={fp}")
    print(f"[calendar] {n_hol} market holidays derived from gaps; "
          f"{int(is_pre.sum())} pre-holiday days")

    e = strategy.effect_stats(close, is_pre)
    print(f"  pre-holiday mean = {e['mu_pre']*1e4:6.2f} bps/day   "
          f"rest mean = {e['mu_rest']*1e4:5.2f} bps/day   "
          f"ratio = {e['ratio']:.1f}x")
    print(f"  gap = {e['gap']*1e4:+.2f} bps   HAC t(pre)={e['t_pre']:+.2f}   "
          f"HAC t(gap)={e['t_gap']:+.2f}")
    wp = e["wilson_pre"]; wr = e["wilson_rest"]
    print(f"  win-rate pre = {e['win_pre']*100:.1f}% [{wp[0]*100:.1f}, {wp[1]*100:.1f}]   "
          f"rest = {e['win_rest']*100:.1f}% [{wr[0]*100:.1f}, {wr[1]*100:.1f}]")

    # Pre-holiday-only book vs buy-and-hold.
    book = strategy.pre_holiday_only(close, is_pre, cost_bps=COST_BPS)
    bh = strategy.buy_and_hold(close)
    print(f"  [pre-only book] days invested = {book['days_invested']} "
          f"({book['time_in_market']*100:.1f}% of tape)   "
          f"mean/invested-day = {book['mean_invested_day']*1e4:.2f} bps")
    print(f"  [pre-only book] CAGR = {book['cagr']*100:.2f}%   "
          f"Sharpe(whole tape) = {book['sharpe']:.3f}   "
          f"Sharpe(per invested day) = {book['sharpe_invested']:.3f}")
    print(f"  [buy & hold]    CAGR = {bh['cagr']*100:.2f}%   Sharpe = {bh['sharpe']:.3f}")

    # Sub-period contrast pre/post-1990.
    c = strategy.subperiod_contrast(close, is_pre, split_date="1990-01-01")
    print(f"  [pre/post-1990] gap pre-1990 = {c['gap_early']*1e4:.2f} bps "
          f"(t={c['t_early']:+.2f}, n_pre={c['n_pre_early']})   "
          f"gap post-1990 = {c['gap_late']*1e4:.2f} bps "
          f"(t={c['t_late']:+.2f}, n_pre={c['n_pre_late']})")
    print(f"  [pre/post-1990] decay = {c['decay']*1e4:.2f} bps   "
          f"P(decay>0)={c['p_decay_gt0']:.3f}   "
          f"95% CI=[{c['decay_ci'][0]*1e4:.2f}, {c['decay_ci'][1]*1e4:.2f}] bps")


def main() -> None:
    # SPY total return (fair tape, 1993+) - note: too short to split pre/post-1990 well.
    _run("SPY", "total_return", "1993-01-29", "SPY total return (fair tape, no pre-1990)")
    # ^GSPC price-only (long sample, 1950+) - the decay story.
    _run("^GSPC", "split_only", "1950-01-01", "GSPC PRICE-ONLY (long sample, 1950+)")
    print("\nNote: ^GSPC is PRICE-ONLY (no dividends) - quoted as such. SPY is total-return.")


if __name__ == "__main__":
    main()
