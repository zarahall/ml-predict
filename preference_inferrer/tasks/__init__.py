#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from types import SimpleNamespace

from preference_inferrer.tasks.task import Task


def get_task(task_name: str, config: SimpleNamespace) -> Task:
    """
    Creates and returns the task specified by task_name
    Args:
        task_name: the task to create
        config: configurations
    Returns:
        The created task
    """
    if task_name == "email_writing":
        from preference_inferrer.tasks.writing.email_writing import EmailWriting
        return EmailWriting(config)
    elif task_name == "summarization":
        from preference_inferrer.tasks.writing.summarization import Summarization
        return Summarization(config)
    else:
        raise ValueError(f"Unknown task {task_name}.")
