"""Study 549 — Spotify-Mood: does the musical *valence* of top-streamed songs lead the market?

A synthetic-only alt-data study. There is no free, historical "aggregate valence of the global
top-streamed songs" tape (the Spotify audio-features API has been closed to new apps since 2024,
and a survivorship-clean monthly panel of chart valence was never published), so the *mood series
is synthetic by construction* — the limitation is stated openly on the SIGNAL axis. We join that
synthetic monthly valence index against **real** ``^GSPC`` monthly returns and run the honest
predictive test.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
