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


class OraclePreferenceAgent(Agent):
    """
    A baseline agent that solves the task using the ground truth preference set
    """

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.true_preferences = None

    def cheat(self, true_preferences: PreferenceSet):
        """
        Cheat by storing the ground truth preference set
        Args:
            true_preferences: The ground truth preference set
        """
        self.true_preferences = true_preferences

    def complete(self, task_instance: TaskInstance) -> t.Tuple[Trajectory, PreferenceSet]:
        """
        Gets a preference-conditioned agent to complete the task with the ground truth preferences
        Args:
            task_instance: the task to complete
        Returns:
            The trajectory (start state and actions) completing the task and the true preference set
        """
        return self.pc_agent.solve_task(task_instance, preferences=self.true_preferences), self.true_preferences

    def learn(self, task_instance: TaskInstance, agent_trajectory: Trajectory,
              user_trajectory: Trajectory):
        """
         No-ops since this agent already has oracle knowledge
         Args:
             task_instance: The task instance that was solved
             agent_trajectory: The agent's attempt at completing the task
             user_trajectory: The user's correct completion of the task
         """
        pass
