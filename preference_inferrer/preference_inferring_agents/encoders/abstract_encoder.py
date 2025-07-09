#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from abc import ABC, abstractmethod

import torch as th


class AbstractEncoder(ABC):
    """
    Abstract base class for objects that turn text into a numerical representation,
    ideally in a way where similar texts are encoded closer in the numerical space
    """
    def __init__(self):
        """Init"""
        if th.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

    @abstractmethod
    def encode(self, text: str) -> th.Tensor:
        """
        Encode text into a numerical tensor
        Args:
            text: The text to encode
        Returns:
            The numerical tensor
        """
