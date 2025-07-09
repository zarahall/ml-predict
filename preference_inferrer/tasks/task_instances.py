#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import typing as t
from abc import ABC


class TaskInstance(ABC):
    """
    Abstract base class for objects representing a task instance.
    Within this repo, a task is a general type of task (e.g. "summarize articles") and a task instance is a specific
    instance of the task (e.g. "summarize this article: <article>")
    """
    def __init__(self, task_name):
        """
        Base task instance class.
        Args:
            task_name: The name of the task
        """
        self.task_name = task_name
        self.context = None
        self.source = None
        self.source_idx = None
        self.agent_completion = None
        self.user_completion = None


class WritingTaskInstance(TaskInstance):
    """
    Task instances for the writing (PLUME / PRELUDE) frameworks
    """
    def __init__(self, task_type: str, task_content: str, source: str, source_idx: int):
        """
        Args:
            task_type: Which task instance this is from. Must be one of "email_writing" or "summarization"
            task_content: The main content to write an email about / summarize
            source: The source of the document. Required to map to user preferences
        """
        super().__init__(task_type)
        self.task_content = task_content
        self.source = source
        self.context = self.task_content
        self.source_idx = source_idx

    def __repr__(self):
        out_str = (f"TASK CONTENT: {self.task_content}\n"
                   f"DRAFT: {self.agent_completion}\n"
                   f"SOURCE: {self.source}\n")
        return out_str
