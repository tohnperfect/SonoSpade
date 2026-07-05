"""usbg: us_bmode_gym.

A closed-loop simulated ultrasound acquisition testbed. Couples LOTUS (CT-label to
B-mode rendering) with MuJoCo (probe pose and contact), driven by an existing RL motion
policy, auto-labeled by an existing MS-TCN recognizer, producing an (image, action)
dataset used to behavior-clone an image-conditioned acquisition policy.

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
