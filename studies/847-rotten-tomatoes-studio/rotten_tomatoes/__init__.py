"""Study 847 — Rotten-Tomatoes -> Studio.

Does the critic reception of a big film (a coarse Rotten-Tomatoes *tier*: fresh vs
rotten) move the *distributing studio's* stock around the release? Curated table of ~40
major wide releases, 2022 -> 2025, each tagged with its distributing studio ticker
(DIS / WBD / PARA / CMCSA / NFLX / SONY), its real opening (or streaming-premiere) date,
and a coarse public critic-score tier. A tier-conditioned event study on the studio's
opening-weekend and following-week abnormal return, honestly graded.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
