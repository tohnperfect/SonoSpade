"""usbg: the SonoSPADE package.

The SonoSPADE per-tissue ultrasound texture stage and its evaluation suite: a physics
B-mode renderer over CT-derived tissue labels, the dual-input SPADE generator and its
CUT/CycleGAN baselines, the frozen segmenter that supplies free supervision, and the
per-tissue metrics. The package name is historical and is kept so imports match the
released code.

The channels/classes/timing/SE(3) geometry contract is provided by the vendored
usbg._vendor package (self-contained; see usbg.contracts_bridge).
"""
from __future__ import annotations

import warnings

# Known false positive: numpy 2.1.x built against Apple's Accelerate BLAS raises spurious
# "overflow / divide by zero / invalid value encountered in matmul" RuntimeWarnings on
# finite, correct inputs (the SIMD remainder path sets FP status flags that numpy surfaces).
# We are pinned to numpy < 2.2 by the sibling us_rl_sim repo, so we cannot take the upstream
# fix; suppress only this specific message and only for matmul, leaving all other warnings.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

__version__ = "0.1.0"
