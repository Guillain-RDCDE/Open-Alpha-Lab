"""Reproduce the Study 340 (Bank-Loans) headline run on the real total-return tapes.

    PYTHONIOENCODING=utf-8 python examples/verify.py

Pins BKLN (Invesco Senior Loan ETF) against TLT (long Treasuries), IEF (intermediate
Treasuries) and SPY (equity) on return/vol/drawdown, the rate-sensitivity test (every
>5% TLT, rate-driven drawdown) and the credit-risk test (every >10% equity drawdown).
As-of 2026-05-31 (last complete month); match the fingerprint to confirm you hold the
same tapes.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from bank_loans import data, strategy  # noqa: E402

AS_OF = "2026-05-31"


def main() -> int:
    prices = data.load_real(("BKLN", "TLT", "IEF", "SPY")).loc[:AS_OF]
    rets = strategy.to_returns(prices)
    print(f"panel: {len(prices):,} rows  {prices.index[0].date()} -> "
          f"{prices.index[-1].date()}  fingerprint={data.fingerprint(prices)}")

    print("\n--- the four arms (total return) ---")
    for c in ("BKLN", "TLT", "IEF", "SPY"):
        s = strategy.stats(rets[c])
        print(f"{c}: CAGR {s['cagr']*100:6.2f}%  vol {s['vol']*100:5.1f}%  "
              f"Sharpe {s['sharpe']:.3f}  maxDD {s['max_dd']*100:6.1f}%")

    print("\n--- rate sensitivity: how much of a rate move does BKLN inherit? ---")
    b_tlt, t_tlt = strategy.hac_beta_t(rets["BKLN"], rets["TLT"])
    b_ief, t_ief = strategy.hac_beta_t(rets["BKLN"], rets["IEF"])
    print(f"beta BKLN~TLT = {b_tlt:+.3f}  HAC t = {t_tlt:+.2f}")
    print(f"beta BKLN~IEF = {b_ief:+.3f}  HAC t = {t_ief:+.2f}")

    print("\n--- where the risk hides: BKLN vs equities ---")
    b_spy, t_spy = strategy.hac_beta_t(rets["BKLN"], rets["SPY"])
    print(f"beta BKLN~SPY = {b_spy:+.3f}  HAC t = {t_spy:+.2f}")
    print(f"downside beta to SPY (worst 10% SPY days) = "
          f"{strategy.downside_beta(rets['BKLN'], rets['SPY']):.3f}")
    print(f"downside beta to TLT (worst 10% TLT days) = "
          f"{strategy.downside_beta(rets['BKLN'], rets['TLT']):.3f}")

    print("\n--- the rate test: every >5% TLT (rate-driven) drawdown ---")
    eps = strategy.asset_drawdowns(rets[["TLT", "BKLN", "IEF", "SPY"]], "TLT", thresh=-0.05)
    rows = [{"peak": e["peak"].date(), "trough": e["trough"].date(),
             "TLT %": e["driver_loss"]*100, "BKLN %": e["others"]["BKLN"]*100,
             "IEF %": e["others"]["IEF"]*100} for e in eps]
    print(pd.DataFrame(rows).round(1).to_string(index=False))

    print("\n--- the credit test: every >10% equity drawdown ---")
    eps = strategy.asset_drawdowns(rets[["SPY", "BKLN", "TLT", "IEF"]], "SPY", thresh=-0.10)
    rows = [{"peak": e["peak"].date(), "trough": e["trough"].date(),
             "SPY %": e["driver_loss"]*100, "BKLN %": e["others"]["BKLN"]*100,
             "TLT %": e["others"]["TLT"]*100} for e in eps]
    print(pd.DataFrame(rows).round(1).to_string(index=False))
    bkln_fell = sum(1 for e in eps if e["others"]["BKLN"] < 0)
    print(f"BKLN FELL in {bkln_fell}/{len(eps)} equity crashes.")

    boot = strategy.bootstrap_beta_diff(rets["BKLN"], rets["TLT"], rets["SPY"],
                                        block=21, n_boot=2000, seed=340)
    print(f"\nbeta-to-equity minus beta-to-TLT = {boot['point']:+.3f}  "
          f"95% CI [{boot['ci95'][0]:+.3f}, {boot['ci95'][1]:+.3f}]  "
          f"equity-wins {boot['frac_equity_wins']*100:.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
