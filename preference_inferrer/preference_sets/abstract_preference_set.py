#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

"""
Defines the structure of a set of preferences.
"""
import typing as t
from abc import ABC, abstractmethod
from types import SimpleNamespace


class PreferenceSet(ABC):
    """
    Defines a set of preferences, and number of utility function around preference sets.
    """

    def __init__(self, config: SimpleNamespace):
        """
        Args:
            config: configurations
        """
        self.config = config

    @abstractmethod
    def in_natural_language(self, mode: str) -> t.Union[t.List[str], str]:
        """
        Returns the set of preferences in natural language.
        NOTE: Sorts attributes to make things deterministic in llm prompts
        Args:
            mode: One of "list", "string", or "json".
                  If "list", return a list of strings where each item describes one preference.
                  If "string", returns a single string that contains all preferences.
                  If "json", return a json string representation (output of json.dumps)
        Returns:
            Preferences in natural language
        """

    @abstractmethod
    def remove(self, preference: str):
        """
        Removes a preference from the set
        Args:
            preference: the preference to remove.
        """

    @abstractmethod
    def update(self, other: "PreferenceSet"):
        """
        Update the given preference set with a new preference set
        Args:
            other: the other preference set to update this preference set with
        """

    @abstractmethod
    def __eq__(self, other: "PreferenceSet"):
        """
        Define the "=" operator to match set behavior
        Args:
            other: the other preference set to compare equality to
        Returns:
            True if preferred and dispreferred attributes are equal, otherwise False
        """

    @abstractmethod
    def __sub__(self, other: "PreferenceSet"):
        """
        Define the subtract (-) operator to match set behavior.
        Args:
            other: the other preference set to subtract
        Returns:
            A new preference set object containing the current preferences minus the "other" preferences
        """

    @abstractmethod
    def __len__(self):
        """
        Define the len() operator to match set behavior.
        Returns:
            The number of preferences in the set
        """

    @abstractmethod
    def is_empty(self):
        """
        Returns:
            True if no preferences are contained, else False
        """

    @abstractmethod
    def get_number_of_contradictions(self) -> int:
        """
        Calculates the number of contradictions in this preference set.
        Contradictions are defined as an attribute that is both preferred and dispreferred.
        Returns:
            the number of contradictions
        """

    @abstractmethod
    def as_single_set(self) -> set:
        """
        Combines the preferred_attributes and dispreferred_attributes into a single set by prefixing them with
        appropriate strings.
        Returns:
            A single set describing the full preference set
        """

    @staticmethod
    @abstractmethod
    def create_empty_preference_set(config: SimpleNamespace) -> "PreferenceSet":
        """
        Generate an empty set of preferences.
        Args:
            config: configurations
        Returns:
             An empty preference set
        """
