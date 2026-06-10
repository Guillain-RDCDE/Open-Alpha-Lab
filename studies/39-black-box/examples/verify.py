"""Reproduce the real-data headline run (docs/results.md) — liquid crypto, 2017–today.

    python examples/verify.py            # cache-only (offline); prints if the crypto cache is present
    python examples/verify.py --fetch    # download daily closes from Yahoo! Finance, then run

For each coin it fits the MLP and prints the IN-SAMPLE vs WALK-FORWARD out-of-sample Sharpe (gross and
net of cost), the directional accuracy, the cost sweep, the break-even cost, and the shuffled-label
overfitting control — then the as-of date and the inputs fingerprint that docs/results.md quotes.
"""

from __future__ import annotations

import os
import sys
import warnings

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
warnings.filterwarnings("ignore")

from black_box import costs, data, extension, strategy

AS_OF = "2026-06-10"
# The documented headline run: full net, a 2-year initial train and ~6-month expanding OOS blocks.
NET = dict(hidden=(32, 16), max_iter=400)
WF = dict(min_train=504, step=189, **NET)


def main(fetch: bool) -> None:
    closes = data.fetch_crypto(fetch=fetch)
    if closes.empty:
        print("No cached crypto. Re-run with --fetch (needs network) to download daily closes.")
        return
    try:
        from quantlab import repro
        closes = repro.as_of(closes, AS_OF)
    except Exception:
        pass
    closes = closes.dropna(how="all")

    coins = list(closes.columns)
    print(f"\nCrypto daily closes, {closes.index[0].date()} - {closes.index[-1].date()} "
          f"({len(closes)} days, coins: {', '.join(coins)})\n")

    for coin in coins:
        c = closes[coin].dropna()
        gap = extension.insample_vs_oos(c, cost_bps=10.0, **WF)
        is_sh = gap.loc["in_sample", "sharpe_gross"]
        is_acc = gap.loc["in_sample", "accuracy"]
        oos_sh = gap.loc["walk_forward_oos", "sharpe_gross"]
        oos_net = gap.loc["walk_forward_oos", "sharpe_net"]
        oos_acc = gap.loc["walk_forward_oos", "accuracy"]
        pos = strategy.walk_forward_predictions(c, **WF)
        be = costs.breakeven_cost_bps(c, pos)
        tpd = costs.turnover(pos)
        print(f"  {coin:9}  in-sample Sharpe {is_sh:6.2f} (acc {is_acc:.3f})  |  "
              f"OOS Sharpe {oos_sh:6.2f} (acc {oos_acc:.3f})  net@10bp {oos_net:6.2f}  "
              f"turnover {tpd:.2f}/day  break-even {be:.2f} bp")

    # Detailed view on the flagship coin.
    flag = "BTC-USD" if "BTC-USD" in coins else coins[0]
    c = closes[flag].dropna()
    print(f"\n--- {flag} detail ---")
    gap = extension.insample_vs_oos(c, cost_bps=10.0, **WF)
    print("In-sample vs walk-forward OOS:\n" + gap.round(3).to_string())
    pos = strategy.walk_forward_predictions(c, **WF)
    print("\nCost sweep (OOS net Sharpe/CAGR):\n" + costs.cost_sweep(c, pos).round(3).to_string())
    print("\nShuffled-label control (in-sample train accuracy survives meaningless labels):\n"
          + extension.shuffled_label_control(c, n_shuffles=4, **NET).round(3).to_string())

    try:
        from quantlab import repro
        print(f"\nas-of {AS_OF} · inputs fingerprint {repro.fingerprint(closes)}")
    except Exception:
        pass


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
