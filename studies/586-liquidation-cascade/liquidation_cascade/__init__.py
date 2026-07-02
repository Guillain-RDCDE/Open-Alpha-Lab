"""Study 586 — Liquidation-Cascade (crypto capitulation-bottom folklore).

The folklore: when a big wave of **forced liquidations** hits crypto — over-leveraged longs
getting margin-called and dumped by the exchange en masse — the selling is *mechanical*, not
informed, so the price overshoots to the downside and then **snaps back**. A large liquidation
event, the story goes, is a *capitulation bottom* — a mean-reversion (buy-the-blood) signal.

We build a deterministic synthetic BTC-style price path with an explicit ``liquidation`` channel
(a single knob, ``bounce_alpha``, controls whether a liquidation spike is followed by a bounce, a
continuation, or nothing) and run an **event study**: forward returns after large-liquidation days
vs the unconditional baseline, a two-sample *t*, a **label-shuffle placebo**, a multi-horizon /
multi-threshold robustness sweep, honest costs, and a seed-robust synthetic positive control.

**Data-availability limitation (named on the SIGNAL axis).** Real per-exchange forced-liquidation
tapes (Binance/Bybit/OKX aggregate liquidation USD, e.g. the series Coinglass sells) are **not**
available on a free, no-key retail stack with any usable history. So this study is
**synthetic-only**: it can never earn `REAL` (that needs a robust *t* ≥ 2 on a real tape) — it is
capped at `WEAK`/`NONE`, and its job is to (a) show what the effect *would* look like if real, and
(b) demonstrate the engine that a reader with a Coinglass/Amberdata feed could point at live data.

Distinct from the desk's other crypto studies: [133 crypto-seasonality] / [175 crypto-weekend]
(calendar), [210 crypto-trend] / [251 crypto-reversal] (price-only momentum/reversal on the tape),
[325 crypto-fear-greed] (a sentiment index). This one is a **microstructure event study** keyed on
*forced-liquidation flow*, a channel none of those touch.
"""
