"""Study 559 — Dark-Pool-Ratio.

The microstructure folklore: when a larger fraction of a stock's volume prints **off-exchange**
(in dark pools, ATSs and internalisers) rather than on the lit exchanges, that is supposedly
*informed accumulation* — patient institutions building a position quietly — and the stock is said
to drift up afterwards. We build a synthetic dark-pool-ratio (DPR) panel against forward returns,
sort on the ratio, measure the information coefficient (IC), run a label-shuffle placebo, and prove
the engine with a seed-robust positive control.

**Data availability.** There is *no free* per-name daily dark-pool-ratio tape a no-key retail stack
can reach: FINRA publishes ATS ("dark pool") volume only weekly, at a two-to-four-week lag, and
off-ATS internalised volume is not in it at all — so a true point-in-time DPR panel joined to
forward returns is not buildable offline here. This study is therefore **synthetic-only**: it
proves the *machinery* (can an IC/sort test catch a planted DPR→return effect and stay flat at the
null?) and states the tape gap openly on the SIGNAL axis. A synthetic-only study can never earn
`REAL` (that needs a robust t ≥ 2 on a real tape); it is capped at `WEAK`/`NONE`.

Distinct from this desk's other flow/imbalance studies: [Study 376 — MOC-Imbalance](../376-moc-imbalance/)
tests the *market-on-close order imbalance*; [Study 418 — Money-Flow-Index](../418-money-flow-index/)
and [Study 419 — Chaikin-Money-Flow](../419-chaikin-money-flow/) are price×volume oscillators.
Study 559 is the **venue-of-execution** signal — *where* the volume prints, not its price impact.
"""
