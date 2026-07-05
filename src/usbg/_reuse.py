"""_reuse.py: expose the B-mode renderer geometry (scan grid, tissue/acoustic tables, phantom).

In the original testbed this geometry was imported (not copied) from the sibling
probe_action_recognizer repo. For this standalone SonoSPADE release the same file is vendored
under usbg._vendor.us_render, so nothing external is required. Re-exports us_render plus its
geometry constants for convenience.
"""
from __future__ import annotations

from ._vendor import us_render

# geometry constants (probe imaging plane), re-exported so downstream never hardcodes them
N_LINES = us_render.N_LINES            # scanlines (lateral)
N_SAMPLES = us_render.N_SAMPLES        # samples per scanline (axial/depth)
IMG_DEPTH_M = us_render.IMG_DEPTH_M    # axial depth imaged (m)
IMG_WIDTH_M = us_render.IMG_WIDTH_M    # lateral extent at the face (m)
PROBE = us_render.PROBE                # "convex" | "linear"

__all__ = ["us_render", "N_LINES", "N_SAMPLES", "IMG_DEPTH_M", "IMG_WIDTH_M", "PROBE"]
