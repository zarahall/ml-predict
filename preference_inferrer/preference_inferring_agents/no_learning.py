#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import typing as t
from types import SimpleNamespace
from preference_inferrer.tasks.trajectory_logger import Trajectory
from preference_inferrer.preference_inferring_agents.abstract_agent import Agent
from preference_inferrer.preference_sets import PreferenceSet, WritingPreferenceSet
from preference_inferrer.tasks.task_instances import TaskInstance


class NoLearningAgent(Agent):
    """
    A baseline agent that does not learn or use preferences in any way.
    Will always solve the task as if there are no preferences.
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.preference_set = WritingPreferenceSet.create_empty_preference_set(self.config)

    def complete(self, task_instance: TaskInstance) -> t.Tuple[Trajectory, PreferenceSet]:
        """
        Gets a preference-conditioned agent to complete the task with no preference.
        Args:
            task_instance: the task to complete
        Returns:
            The trajectory (start state and actions) completing the task, and an empty preference set
        """
        return self.pc_agent.solve_task(task_instance, preferences=self.preference_set), self.preference_set

    def learn(self, task_instance: TaskInstance, agent_trajectory: Trajectory, user_trajectory: Trajectory):
        """
        No-op since this agent by definition does not learn
        Args:
            task_instance: The task instance that was solved
            agent_trajectory: The agent's attempt at completing the task
            user_trajectory: The user's correct completion of the task
        """
        pass
