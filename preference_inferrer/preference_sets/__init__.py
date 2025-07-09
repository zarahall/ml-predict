#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import typing as t
from types import SimpleNamespace

from preference_inferrer.preference_sets.abstract_preference_set import PreferenceSet
from preference_inferrer.preference_sets.writing_preference_set import WritingPreferenceSet


def get_empty_preference_set(task_name: str, config: SimpleNamespace) -> PreferenceSet:
    """
    Returns an empty preference set for the task specified by task_name
    Args:
        task_name: Specifies the kind of preference set to create
        config: configurations
    Returns:
        An empty preference set
    """
    if task_name in ["summarization", "email_writing"]:
        return WritingPreferenceSet.create_empty_preference_set(config)
    else:
        raise ValueError(f"Unknown task {task_name}.")


def get_preference_set(task_name: str, preferences: t.Union[str, t.List[str]],
                       config: SimpleNamespace) -> PreferenceSet:
    """
    Creates and returns the correct type of preference set with specified preferences
    Args:
        task_name: Specifies the kind of preference set to create
        preferences: The preferences to include in the preference set
        config: configurations
    Returns:
        The created preference set
    """
    if isinstance(preferences, str):
        preferences = {preferences}
    
    return WritingPreferenceSet(preferences, config)
