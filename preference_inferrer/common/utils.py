#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

"""
Functions for logging trajectories.
"""
import string
from random import seed

import numpy as np
import torch
import transformers

ALPHA_NUMERIC_CHARACTERS = list(string.ascii_lowercase + string.ascii_uppercase + string.digits)


def set_seed_everywhere(random_seed: int):
    """
    Sets the random seed for transformers, random, torch, and numpy
    Args:
        random_seed: the random seed to set
    """
    transformers.set_seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(random_seed)
    seed(random_seed)


def print_in_color(string_to_print, color=None):
    """
    Prints something in color.
    Adapted from https://github.com/gao-g/prelude/blob/main/src/utils/color_utils.py
    Args:
        string_to_print: The string to print
        color: The color to print in
    """
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m"
    }
    print(f"{colors.get(color, '')}{string_to_print}\033[0m")  # Default to no color if invalid color is provided
