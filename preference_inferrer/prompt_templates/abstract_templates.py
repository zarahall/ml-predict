#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

"""
Abstract base class for templates to use with LLMs.
"""
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import SimpleNamespace

from preference_inferrer.tasks.trajectory_logger import Trajectory


@dataclass
class Templates(ABC):
    """
    Environment prompts templates. There are four main prompt groups.
    1. A high level system prompt describing the overarching task at hand
    2. A prompt that tasks the LLM to infer preferences from trajectories ("inferring")
    3. A prompt that breaks down preferences into a set of basis preferences ("basis")
    4. A prompt that verifies preferences against trajectories ("validating")

    For groups 2, 3, and 4, there are two parts to the prompts.
    1. the instruction of what to do (infer, break down, validate)
    2. the instruction of how to format the output so that it is parsable.
        - if the parsing fails, we count and log it as an error.

    There are additional prompts used to help in the reasoning process.
    These are all controlled by the config
    """
    # Below variables must be defined in initialize_prompts() function
    # Name of template class
    name: str
    # High leve descriptions of the task at hand
    system_prompt: str = field(init=False)

    # Format and example prompts. Key must be one of "infer", "basis", or "validate" and values are the
    # format/example prompts for the respective preference inference steps.
    format_prompts: t.Dict[str, t.Callable] = field(init=False)
    # General prompts
    other_reasoning: t.List[str] = field(init=False)
    other_val_reasoning: t.List[str] = field(init=False)

    # Prompts related to user edit agents
    get_task_prompt: t.Callable = field(init=False)
    get_task_prompt_icl: t.Callable = field(init=False)
    get_preference_inference_prompt: t.Callable = field(init=False)
    get_majority_preference_prompt: t.Callable = field(init=False)

    def __init__(self, config: SimpleNamespace):
        """
        Args:
            config: experiment configurations used to determine template structure
        """
        self.config = config
        # Dynamic content is generated after initialization to reflect instance-specific properties.
        # This method must be implemented by subclasses to set the appropriate prompts.
        self.initialize_prompts()

    def base_infer_prompt(self, preferences: str, user_traj: Trajectory,
                          counterfactual: t.Optional[Trajectory] = None) -> str:
        """
        Generates a prompt to infer preferences given the user's trajectory, the current preferences and
        optionally a counterfactual trajectory
        This a base inference prompt that doesn't go much further than directly providing the information and task
        It only allows for the addition of new preferences and relies on validation to remove incorrect preferences.
        NOTE: the preferences parameter is technically a string, but represents a list of preferences
        Args:
            preferences: the current set of inferred preferences in string form
            user_traj: The user's trajectory
            counterfactual: the counterfactual trajectory
        Returns:
            The prompt
        """
        user_traj = user_traj.in_natural_language("user")

        action_noun = "writing"
        prompt = f"Infer the user's underlying preferences "
        prompt += f"from the user's {action_noun}: {user_traj}\n"

        if counterfactual is not None:
            counterfactual_actions = counterfactual.get_actions_in_natural_language(None)
            prompt += f"Given the current inferred preferences: {preferences}, {counterfactual_actions}"
        else:
            prompt += f"Here are the currently inferred preferences: {preferences}.\n"
        return prompt

    def refine_preferences_prompt(self, preferences: str, user_traj: Trajectory,
                                  counterfactual: t.Optional[Trajectory] = None) -> str:
        """
        Generates a prompt to refine preferences given the user's trajectory and optionally a counterfactual trajectory
        This version of the prompt asks the llm to directly update/refine the list of preferences allowing for
        adding, removing, and modifying preferences. It is currently the best performing prompt type.
        NOTE: the preferences parameter is technically a string, but represents a list of preferences
        Args:
            preferences: the current set of inferred preferences in string form
            user_traj: The user's trajectory
            counterfactual: the counterfactual trajectory
        Returns:
            The prompt
        """
        action_noun = {
            "summarization": "summary", "email_writing": "email"
        }[self.config.task.name]

        # Get start state
        start_state_string = user_traj.get_start_state_in_natural_language()
        # Get user actions
        user_traj = user_traj.get_actions_in_natural_language("user")
        prompt = f"We received a new task. \n{start_state_string}\n"
        if preferences is not None:
            prompt += f"We have previously identified the following preferences:\n{preferences}\n"
        else:
            prompt += f"We have yet to identify preferences.\n"
        # If counterfactual is provided, include it in prompt
        if counterfactual:
            # Make sure counterfactual has the same start state (e.g. it's the same environment)
            if start_state_string != counterfactual.get_start_state_in_natural_language():
                raise ValueError(f"user start state:\n"
                                 f"{start_state_string}\n"
                                 f"does not equal cf agent start state:\n"
                                 f"{counterfactual.get_start_state_in_natural_language()}")
            counterfactual = counterfactual.get_actions_in_natural_language(None)
            if preferences is not None:
                prompt += f"Based on these preferences, {counterfactual}\n"
            else:
                prompt += f"Currently, {counterfactual}\n"

            prompt += f"However, this differs from the user's {action_noun}. {user_traj}\n"
        else:
            prompt += f"{user_traj}\n"

        # modify the edit instruction based on whether the preferences are represented as a list of components
        if self.config.do_breakdown_preferences_each_iteration:
            prompt += (f"Refine the list of preferences by adding, removing, or updating preferences "
                       f"in order to better imitate the user.\n")
        else:
            prompt += (f"Refine the preference by adding, removing, or updating components "
                       f"in order to better imitate the user.\n")

        return prompt

    @staticmethod
    def basis_instruction_prompt(raw_preference: str) -> str:
        """
        Generates a prompt to break down a raw (unedited) preference string into a list of base preferences.
        Args:
            raw_preference: the preferences to break down
        Returns:
            The prompt
        """
        prompt = f"You inferred the following preference string:\n{raw_preference}\n"
        prompt += f"Format this preference into a concise set of preferences. "
        return prompt

    def validation_instruction_prompt(self, preference: str, user_traj: str, other_preferences: t.List[str]) -> str:
        """
        Generates a prompt to validate a preference against a trajectory. Optionally provides the other inferred
        preferences for context.
        Args:
            preference: the preferences to validate
            user_traj: The user's trajectory to validate it against
            other_preferences: other preferences to provide context
        Returns:
            The prompt
        """
        action_noun = "writing"

        prompt = (f"Validate the following preference: \"{preference}\"\n"
                  f"against the following {action_noun}: \n{user_traj}\n")

        prompt += (f"Does the {action_noun} confirm or contradict the preference? "
                   f"Select one of the following:\n"
                   f"strongly confirms the preference, somewhat confirms the preference, "
                   f"is neutral toward the preference, "
                   f"somewhat contradicts the preference, strongly contradicts the preference. \n")
        if self.config.include_preferences_in_validate:
            prompt += f"For additional context, here are the other inferred preferences: \"{other_preferences}\".\n"
        return prompt

    def coalesce_prompt(self, preferences: str) -> str:
        """
        Defines and returns a prompt used to coalesce preferences.
        Args:
            preferences: The preferences to coalesce
        Returns:
            The prompt
        """
        # modify the instruction based on whether the preferences are represented as a list of components
        if self.config.do_breakdown_preferences_each_iteration:
            return (
                f"We are tasked to curate a prompt to guide a specific style of writing. "
                f"We currently have the following list of preferences related to writing styles: \n{preferences}\n"
                f"Unfortunately, these preferences may overlap or contain redundancies. "
                f"Please review the list and condense it by combining similar or overlapping preferences, "
                f"ensuring that the distinct intent behind each one remains clear so that a writer can easily follow them. "
                f"Ensure the condensed list is concise, non-redundant, and preserves the original level of specificity. "
                f"When applicable, preserve the exact wording. "
                f"Return the revised preferences in the same format as the original list."
            )
        else:
            return (
                f"We are tasked to curate a prompt to guide a specific style of writing. "
                f"We currently have the following list of preferences related to writing styles: \n{preferences}\n"
                f"Unfortunately, these preferences may overlap or contain redundancies. "
                f"Please review the list and condense it into a single preference description, "
                f"ensuring that the distinct intent behind each preference in the list remains clear so that a writer "
                f"can easily follow them. "
                f"Ensure the condensed preference is concise, non-redundant, and preserves the original level of "
                f"specificity. "
                f"When applicable, preserve the exact wording."
            )

    @abstractmethod
    def initialize_prompts(self):
        """
        Subclasses must implement this method to initialize all dynamic content based on instance-specific properties.
        """
        pass
