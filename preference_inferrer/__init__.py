#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
"""
We create personalizable agents by splitting an interactive task into two stages: (1) identifying the preferences and then (2) conditioning agents on these preferences. 

This PREDICT and this repository focuses on the first stage and aims to answer the following research questions:

- *RQ1*: Does providing adversarial comparison trajectories lead to more accurate and precise preferences? 
Orthogonal basis of preferences
- *RQ2*: Does breaking down preferences into components lead to faster and more generalizable inference?
Preferences do not exist in a vacuum
- *RQ3*: Does confirming a preference across multiple samples lead to more accurate and precise preferences?
"""

__version__ = "0.1.0"
