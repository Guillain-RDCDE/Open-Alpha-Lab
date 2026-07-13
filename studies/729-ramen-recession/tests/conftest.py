"""Test bootstrap for Study 729 (the ramen index) — offline, deterministic.

Puts the study root on ``sys.path`` so ``import ramen_recession`` resolves whether
the suite is run per-study or as part of a cross-study collection from the repo
root. No network, no fixtures needed beyond the import shim.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
