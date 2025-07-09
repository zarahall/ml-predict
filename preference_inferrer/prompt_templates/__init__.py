#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from types import SimpleNamespace

from preference_inferrer.prompt_templates.abstract_templates import Templates
from preference_inferrer.prompt_templates.plume_templates import PlumeTemplates
from preference_inferrer.prompt_templates.prelude_templates import PreludeTemplates


def get_prompt_templates(task_type: str, config: SimpleNamespace) -> Templates:
    """
    Creates templates for the provided environment using the config
    Args:
        task_type: Task type to create templates for
        config: Experiment/agent configuration
    Returns:
        Created templates
    """
    if task_type == "plume":
        templates = PlumeTemplates(config)
    elif task_type in ["prelude", "prelude.no_edit"]:
        templates = PreludeTemplates(config)
    else:
        raise ValueError(f"task_type: {task_type} is not recognized. "
                         f"Please select one of plume or prelude.")
    return templates
