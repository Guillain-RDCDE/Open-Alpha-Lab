"""Study 208 — Gold-Miners.

Gold-miners ETF (GDX) is marketed as "leveraged gold" — supposedly amplifying gold
(GLD) moves on the upside while acting as a hedge. This study measures GDX's realized
beta to GLD, tests the leverage-symmetry claim (do miners give MORE upside than downside
amplification?), and evaluates a miners-vs-bullion timing rule against a simple GLD hold.
"""

from . import data, strategy  # noqa: F401
