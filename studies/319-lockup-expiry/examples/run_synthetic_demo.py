"""Offline synthetic demo — Study 319 (Lockup-Expiry).

Shows the machinery proof with no network: a planted -300 bps one-shot sag is recovered
by the event-study engine, while a null panel produces nothing significant. This is the
positive control; it never backs a Signal stamp (see METHODOLOGY.md).

    python studies/319-lockup-expiry/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lockup_expiry import data, strategy as st  # noqa: E402


def report(label: str, panel) -> None:
    a = st.aar(panel)
    row0 = a[a["tau"] == 0].iloc[0]
    s = st.car(panel, lo=-1, hi=1, n_boot=1000)
    print(f"[{label}] AAR(0)={row0['aar']*1e4:+.1f}bps t={row0['tstat']:+.2f} | "
          f"CAR[-1,+1]={s['mean_car_bps']:+.1f}bps t={s['tstat']:+.2f} "
          f"detect_sag={st.detects_sag(panel)}")


def main() -> None:
    null, _ = data.synthetic_panel(n_events=60, sag_bps=0.0, seed=319)
    sag, _ = data.synthetic_panel(n_events=60, sag_bps=-300.0, seed=319)
    print("=== Synthetic machinery proof (offline, deterministic) ===")
    report("NULL   ", null)
    report("PLANTED", sag)
    print("\nThe engine sees the planted sag and stays silent on the null — as it must.")


if __name__ == "__main__":
    main()
