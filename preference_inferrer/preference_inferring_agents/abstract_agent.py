#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import typing as t
from abc import ABC, abstractmethod
from types import SimpleNamespace

from preference_inferrer.llms import get_llm
from preference_inferrer.preference_conditioned_agents import get_conditioned_agent
from preference_inferrer.preference_sets.abstract_preference_set import PreferenceSet
from preference_inferrer.prompt_templates import get_prompt_templates
from preference_inferrer.tasks.task_instances import TaskInstance
from preference_inferrer.tasks.trajectory_logger import Trajectory


class Agent(ABC):
    """
    An abstract base class the defines the required methods for an agent to be evaluated by the PLUME/PICKUP frameworks.
    """

    def __init__(self, config: SimpleNamespace):
        """
        Args:
            config: configurations
        """
        # CF = counterfactual
        self.pc_agent = get_conditioned_agent(config.task.name, "CF_AGENT", config)
        self.llm = get_llm(config.agent.llm_name, "INFERRER", config)
        self.environment_prompts = get_prompt_templates(config.task.framework, config)
        self.config = config
        self.can_cheat = config.agent.name in {"oracle_preference"}
        self.preference = None
        self.task = None

    def cheat(self, true_preferences: PreferenceSet):
        """
        Used only by the oracle agent, who overwrites this method. All other agents are forbidden from cheating.
        Args:
            true_preferences: The user's true preference
        """
        raise ValueError(f"You are not supposed to cheat with {type(self)}")

    def set_task(self, task: t.Any):
        """
        Provides the agent direct access to the task at hand
        Args:
            task: the task the agent is trying to solve
        """
        self.task = task

    @abstractmethod
    def complete(self, task_instance: TaskInstance) -> t.Tuple[Trajectory, PreferenceSet]:
        """
        Complete the task instance
        Args:
            task_instance: The task instance to complete
        Returns:
            The trajectory (start state and actions) that completes the task.
            The preference set used to complete the task
        """

    @abstractmethod
    def learn(self, task_instance: TaskInstance, agent_trajectory: Trajectory, user_trajectory: Trajectory):
        """
        Learn based on the counterfactual agent and user's trajectory
        """

    def get_metrics(self) -> t.Dict[str, t.Any]:
        """
        Calculate and returns a dictionary of numerically measurable metrics
        Returns:
            A dictionary containing metrics (keys are strings with metric names, values are the numeric metric values).
        """
        return {
            **self.llm.get_metrics(),
            **self.pc_agent.get_metrics()
        }
