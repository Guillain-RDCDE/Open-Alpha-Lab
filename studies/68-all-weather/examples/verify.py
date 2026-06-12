"""Reproduce the real run (docs/results.md) — risk parity vs the usual mixes.
    python examples/verify.py   # cache-only (reads the shared cross-asset pull)
"""
from __future__ import annotations
import os, sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from all_weather import data, strategy as st


def main(fetch):
    ret = data.fetch_panel(fetch=fetch)
    if ret.empty:
        print("No cached cross-asset data."); return
    print(f"\nRisk parity vs the usual mixes — SPY/IEF/GLD/DBC, "
          f"{ret.index.min().date()}–{ret.index.max().date()} ({len(ret)} days)\n")
    tbl = st.compare(ret)
    show = tbl.copy()
    show["ann"] = (show["ann"] * 100).round(2); show["vol"] = (show["vol"] * 100).round(2)
    show["max_drawdown"] = (show["max_drawdown"] * 100).round(1); show["sharpe"] = show["sharpe"].round(2)
    print(show[["ann", "vol", "sharpe", "max_drawdown"]].to_string())
    rp, sp, sf = tbl.loc["risk-parity"], tbl.loc["SPY only"], tbl.loc["60/40"]
    print(f"\n  risk parity: Sharpe {rp['sharpe']:.2f} (vs SPY {sp['sharpe']:.2f}, 60/40 {sf['sharpe']:.2f}); "
          f"maxDD {rp['max_drawdown']:.0%} (vs SPY {sp['max_drawdown']:.0%})")
    print(f"  but ann return {rp['ann']:.1%} vs SPY {sp['ann']:.1%} — better Sharpe, far smaller drawdown, ~half the return")
    try:
        from quantlab import repro
        print(f"\ninputs fingerprint {repro.fingerprint(ret)}")
    except Exception: pass


if __name__ == "__main__":
    main("--fetch" in sys.argv)
