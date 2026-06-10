"""Study 38 — Chorus: the alpha combo (Kakushadze-Serur §3.20) — the capstone of the desk's first 37 studies.

No single anomaly is impressive alone, but combining several weak, *decorrelated* signals into one
portfolio produces a materially better Sharpe than any component — the Fundamental Law of Active
Management (Grinold-Kahn), and the desk's recurring lesson that **the edge is diversification, not
prediction.** We build three simple, causal, dollar-neutral cross-sectional signals on the cached S&P 500
panel — momentum (Study 24), reversal (Study 33) and a low-vol tilt — and combine them equal-weight and
inverse-vol, asking whether the chorus out-sings every soloist, and whether it survives its own turnover.
"""

from . import costs, data, extension, signals, strategy  # noqa: F401
