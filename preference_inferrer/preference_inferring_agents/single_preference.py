#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import typing as t
from types import SimpleNamespace
from preference_inferrer.tasks.trajectory_logger import Trajectory
from preference_inferrer.preference_inferring_agents.abstract_agent import Agent
from preference_inferrer.preference_sets import PreferenceSet
from preference_inferrer.tasks.task_instances import TaskInstance


class SinglePreferenceAgent(Agent):
    """
    A baseline agent that solves the task using a single ground truth preference set for all examples
    """

    def __init__(self, single_preference: PreferenceSet, config: SimpleNamespace):
        super().__init__(config)
        self.single_preference = single_preference

    def complete(self, task_instance: TaskInstance) -> t.Tuple[Trajectory, PreferenceSet]:
        """
        Gets a preference-conditioned agent to complete the task with the single preferences
        Args:
            task_instance: the task to complete
        Returns:
            The trajectory (start state and actions) that completes the task.
            The preference set used to complete the task
            The number of relevant examples used to complete the task
        """
        return self.pc_agent.solve_task(task_instance, preferences=self.single_preference), self.single_preference

    def learn(self, task_instance: TaskInstance, agent_trajectory: Trajectory, user_trajectory: Trajectory):
        """
         No-op since this agent never learns anything.
         Args:
             task_instance: The task instance that was solved
             agent_trajectory: The agent's attempt at completing the task
             user_trajectory: The user's correct completion of the task
         """
        pass
