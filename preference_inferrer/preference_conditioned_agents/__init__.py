#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from types import SimpleNamespace

from preference_inferrer.preference_conditioned_agents.abstract_preference_conditioned_agent import (
    PreferenceConditionedAgent)
from preference_inferrer.preference_conditioned_agents.preference_conditioned_writing_agent import \
    PreferenceConditionedWritingAgent


def get_conditioned_agent(task_name: str, logger_name: str, config: SimpleNamespace) -> PreferenceConditionedAgent:
    """
    Creates agent for the provided environment using the config
    Args:
        task_name: environment to create agent for
        logger_name: name for the agent's logger
        config: experiment/agent configuration
    Returns:
        Created agent
    """
    if task_name in ["email_writing", "summarization"]:
        agent = PreferenceConditionedWritingAgent(config, logger_name)
    else:
        raise ValueError(f"task_name: {task_name} is not recognized.\n"
                         f"Please select one of 'email_writing', or 'summarization'.")
    return agent
