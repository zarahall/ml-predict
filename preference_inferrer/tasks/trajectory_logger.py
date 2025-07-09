#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

"""
Class to log trajectories in a format that can create strings to feed to LLMs.
"""
import typing as t
from abc import ABC, abstractmethod
from types import SimpleNamespace

import numpy as np


class Trajectory(ABC):
    """
    Abstract base class for objects that represents a trajectory completed by an agent.
    Includes methods to map the trajectory to natural language
    """
    def __init__(self, config: SimpleNamespace):
        """
        Args:
            config: configurations
        """
        self.config = config
        self.completed_task = None

    @abstractmethod
    def get_start_state_in_natural_language(self) -> str:
        """
        Creates a string (natural language) representation of the start state
        Returns:
            The natural language representation of the start state
        """

    @abstractmethod
    def get_actions_in_natural_language(self, agent_name: t.Union[str, None]) -> str:
        """
        Creates a string (natural language) representation of the actions performed
        Returns:
            The natural language representation of the actions performed
        """

    def in_natural_language(self, agent_name: t.Union[str, None] = None) -> str:
        """
        Creates a natural language description of the trajectory
        Args:
            agent_name: The name of the agent performing the actions. If None, "we" are performing the actions
        Returns:
            Returns a string describing the trajectory that can be fed to an LLM
        """
        out_str = self.get_start_state_in_natural_language()
        out_str += self.get_actions_in_natural_language(agent_name)
        return out_str

    @abstractmethod
    def in_natural_language_for_validation(self, agent_name: t.Union[str, None] = None) -> str:
        """
        Return a user actions in a specific format used in validation.
        Args:
            agent_name: name of the agent completing the task (almost always "user" in this function)
        Returns:
            String describing the trajectory to be used in validation
        """


class WritingTrajectory(Trajectory):
    """
    Represents a "trajectory" in the user edit environment.
    The trajectory is logged such that it can be described in language to an LLM.
    It consists of a start state (i.e. the task at hand)
    and a final state (i.e. the agent's written solution for the task at hand).
    """

    def __init__(self, config: SimpleNamespace, task_type: str, task_content: str, edited: bool, completed_task: str):
        """
        Args:
            config: Configurations
            task_type: The type of the task. either "summarization" or "email writing"
            task_content: The main content of the task (i.e. article to summarize or notes to write an email about)
            edited: Whether the agent edited the output. Only relevant if the agent was provided with a draft.
                    In the agent was provided a draft and edited is False, then the only action is a No-op.
            completed_task: The actions (e.g. generated string) that completes the task. e.g. Final output of the agent.
        """
        super().__init__(config)
        self.task_type = task_type
        self.task_content = task_content
        self.edited = edited
        self.completed_task = completed_task

    def get_start_state_in_natural_language(self) -> str:
        """
        Gets a description of the start state. For the user edit environment, this consists of what the underlying task
        is and base information that needs to be used to complete the task.
        Returns:
            A string describing the start state of the trajectory
        """
        task_description = "write an email about" if self.task_type == "email_writing" else "summarize"
        trajectory_str = (f"The task is to {task_description} the following:"
                          f"\n\"\"\"\n{self.task_content}\n\"\"\"\n")
        return trajectory_str

    def get_actions_in_natural_language(self, agent_name: t.Union[str, None]) -> str:
        """
        Gets a description of the actions performed by agent <agent_name>.
        For the user edit environment, this consists of the writing outputted to complete the task
        If agent name is none, the returned sentence uses first-person plural "we" as the actor.
        Args:
            agent_name: The name of the agent performing the actions. If None, "we" are performing the actions
        Returns:
            A string describing what the task was and how it was completed
        """
        task_output = "email" if self.task_type == "email_writing" else "summary"
        if agent_name is None:
            trajectory_str = (f"we wrote this {task_output}:"
                              f"\n\"\"\"\n{self.completed_task}\n\"\"\"\n")
        else:
            trajectory_str = (f"The {agent_name} wrote this {task_output}:"
                              f"\n\"\"\"\n{self.completed_task}\n\"\"\"\n")
        return trajectory_str

    def in_natural_language_for_validation(self, agent_name: t.Union[str, None] = None) -> str:
        """
        Return a user actions in a specific format used in validation.
        In plume, this is just the completed task
        Args:
            agent_name: name of the agent completing the task (not used in plume's validation)
        Returns:
            String describing the trajectory to be used in validation
        """
        return f"\n\"\"\"\n{self.completed_task}\n\"\"\"\n"

    def __eq__(self, other: "WritingTrajectory"):
        """Returns True if the two WritingTrajectory objects are equal, otherwise False"""
        if other is None:
            return False
        return self.task_content == other.task_content and self.completed_task == other.completed_task
