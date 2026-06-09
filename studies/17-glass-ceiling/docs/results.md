# Real-data results — Study 17 (Glass-Ceiling)

*The frozen real-tape run behind the notebooks' quoted numbers. The notebooks **execute** on the
offline synthetic core (where the answer is baked in and provable); this page is the **measurement**
on the market, produced by [`examples/verify.py`](../examples/verify.py). Reproduce with
`python examples/verify.py` (cache-only) — a reader who matches the fingerprints holds the same tape.*

## Provenance

| Tape | Rows | Window | Fingerprint |
|---|---|---|---|
| BTC-USD 5m | 16,981 | 2026-04-11 → 2026-06-09 | `83da23506854` |
| SPY 5m | 4,680 | 2026-03-13 → 2026-06-08 | `001fb04576e1` |
| QQQ 5m | 4,680 | 2026-03-13 → 2026-06-08 | `3f14b74247a8` |

Source: Yahoo! Finance intraday OHLCV via `yfinance` (`auto_adjust=True`), the deepest history Yahoo
serves at 5-minute resolution (~60 days). **This is the named limitation:** every real win rate rests
on tens of trades, so the intervals are wide *by construction* — the verdict is carried by the
synthetic core, and this leg only checks that the coin flip is visible in the wild.

## The breakout bracket on real intraday bars

Setup: long on `confirm=2` closes above a 30-bar trailing-high resistance; stop at the swing low
(floored at 1%); target at **1R**. Cost charged at **2 bps round-trip** (optimistic for a liquid
venue; crypto and CFDs are worse). Win rate carries a Wilson 95% interval.

| Tape | Trades | Win % | 95% CI | Gross E[R] | Break-even win % | Net E[R] |
|---|---|---|---|---|---|---|
| **BTC-USD** | 84 | **45.2** | **[35, 56]** | −0.095 | 50.9 | **−0.112** |
| **SPY** | 19 | 63.2 | [41, 81] | +0.263 | 50.9 | +0.246 |
| **QQQ** | 28 | 57.1 | [39, 73] | +0.143 | 50.8 | +0.127 |

**How to read it.** Every 95% interval **straddles 50%** — not one tape can reject the coin flip. The
**deepest, most relevant sample** (BTC-USD, 84 trades — Koroush's own 24/7 market) sits *below* 50%
and is **net-negative** once it pays 2 bps on both legs. SPY and QQQ point estimates land above 50%,
but on 19 and 28 trades their intervals reach from the low-40s into the 70s–80s: consistent with
anything from a losing strategy to a strong one. That irreducible ambiguity on a 60-day window *is*
the finding — it is exactly why a wall of winning screenshots proves nothing.

## The filters: selection illusion, made literal

| Tape | Filter lift (win %) | Kept fraction | Trades surviving all 3 filters |
|---|---|---|---|
| BTC-USD | **−0.452** | 11% | 9 (won 0%) |
| SPY | +0.368 | 5% | **1** (won 100%) |
| QQQ | +0.095 | 11% | 3 (won 67%) |

The "A-grade" subset that passes the staircase + volume + clean-trend filters is **1 to 9 trades**.
On SPY it is a *single* trade that happened to win — a "100% win rate" that is one coin landing heads.
On BTC the filtered subset went 0-for-9. The lifts swing from −45 to +37 points with no stability,
because they are computed on a handful of trades: this is sampling noise dressed as selectivity, the
precise mechanism by which a filtered highlight reel manufactures a track record.

## Verdict

- **Signal `NONE`** — the breakout win rate is statistically a coin flip on every real tape (all CIs
  contain 50%), echoing the synthetic null.
- **Tradability `MIRAGE`** — at 1:1 the break-even win rate is ~50.9% at just 2 bps; the deepest real
  sample is net-negative, and a real 1-minute crypto/CFD spread is wider than 2 bps.
- **Do the filters help? `NOT SUPPORTED`** — no stable win-rate lift; the filtered subsets are too
  small to carry any conclusion except the selection illusion itself.

*As-of freeze: 2026-06-09. Regenerate with `python examples/verify.py --fetch` (refills the cache),
then `python examples/verify.py` (cache-only) to reproduce the fingerprints above.*
