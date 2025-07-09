#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import typing as t
from abc import ABC, abstractmethod

from preference_inferrer.preference_sets import PreferenceSet
from preference_inferrer.tasks.task_instances import TaskInstance


class Task(ABC):
    """
    Abstract base class for objects representing a task.
    Within this repo, a task is a general type of task (e.g. "summarize articles") and a task instance is a specific
    instance of the task (e.g. "summarize this article: <article>")
    """
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def get_task_instances(self) -> t.Iterable[TaskInstance]:
        """
        Defines a generator that yields task instances
        """
        pass

    @abstractmethod
    def get_metrics(self, task_instance: TaskInstance, pred_preferences: PreferenceSet,
                    true_preferences: PreferenceSet):
        """
        Returns a dictionary containing metrics that measure how successful the agent did.
        Args:
            task_instance: Contains the task content as well as the agent's and user's trajectory that complete it
            pred_preferences: The inferred preferences
            true_preferences: The user's true preferences
        Returns:
            A dictionary with metrics where the key is a string and the values are numeric
        """
