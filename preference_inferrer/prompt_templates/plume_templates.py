#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

"""
Templates to use for the user edit environment.
"""
import typing as t

from preference_inferrer.tasks.trajectory_logger import WritingTrajectory
from preference_inferrer.prompt_templates.abstract_templates import Templates
from preference_inferrer.tasks.task_instances import WritingTaskInstance


FORMAT_PROMPTS_DO_BREAKDOWN_PREFERENCES_EACH_ITERATION = {
            "infer": lambda line_keyword: (
                f"After reasoning, output the refined set of preferences as a JSON array, "
                f"where each element is a string, on a single new line and prefaced with \"{line_keyword}\".\n"
            ),
            "basis": lambda line_keyword: (
                f"Format the final set of preferences as a JSON list on a single line and prefaced with "
                f"\"{line_keyword}\". Each element in the JSON list should be a string."
                f"The final output should look like:\n"
                f"{line_keyword} [\"<preference 1>\", ...,  \"<preference i>\", ...]"
            ),
            "coalesce": lambda line_keyword: (
                f"Format the subset of preferences as a JSON list on a single line and prefaced with "
                f"\"{line_keyword}\". Each element in the JSON list should be a string."
                f"The final output should look like:\n"
                f"{line_keyword} [\"<preference 1>\", ...,  \"<preference i>\", ...]"
            ),
            "validate": lambda line_keyword: (
                f"Your final decision should be output on a separate line prefaced with \"{line_keyword}\" "
            )
        }

FORMAT_PROMPTS_DO_BREAKDOWN_PREFERENCES_AFTER_REFINEMENT = {
            "infer": lambda line_keyword: (
                f"After reasoning, output the refined set of preferences on a single new line and prefaced "
                f"with \"{line_keyword}\".\n"
            ),
            "basis": lambda line_keyword: (
                f"Format the final set of preferences as a JSON list on a single line and prefaced with "
                f"\"{line_keyword}\". Each element in the JSON list should be a string. "
                f"The final output should look like:\n"
                f"{line_keyword} [\"<preference 1>\", ...,  \"<preference i>\", ...]"
            ),
            "coalesce": lambda line_keyword: (
                f"Format the preferences on a single line and prefaced with \"{line_keyword}\"."
            ),
            "validate": lambda line_keyword: (
                f"Your final decision should be output on a separate line prefaced with \"{line_keyword}\" "
            )
        }

FORMAT_PROMPTS_NO_BREAKDOWN_PREFERENCES = {
            "infer": lambda line_keyword: (
                f"After reasoning, output the refined set of preferences on a single new line and prefaced "
                f"with \"{line_keyword}\"."
            ),
            "basis": lambda line_keyword: "",
            "coalesce": lambda line_keyword: (
                f"Format the preferences on a single line and prefaced with \"{line_keyword}\"."
            ),
            "validate": lambda line_keyword: (
                f"Your final decision should be output on a separate line prefaced with \"{line_keyword}\" "
            )
        }


class PlumeTemplates(Templates):

    def initialize_prompts(self):
        """
        Initialize the framework specific prompts. Done post init to simplify things
        """
        self.system_prompt = (
            f"A user is completing writing tasks. "
            f"The user has an underlying set of preferences that explains why they write the way they do. "
            f"The objective is to identify these underlying preferences so that we can write exactly like the user. "
        )

        if self.config.do_breakdown_preferences_each_iteration:
            self.format_prompts = FORMAT_PROMPTS_DO_BREAKDOWN_PREFERENCES_EACH_ITERATION
        elif self.config.do_breakdown_preferences_after_refinement:
            self.format_prompts = FORMAT_PROMPTS_DO_BREAKDOWN_PREFERENCES_AFTER_REFINEMENT
        else:
            self.format_prompts = FORMAT_PROMPTS_NO_BREAKDOWN_PREFERENCES

        # Additional information and directives that may be beneficial to the llm
        self.other_reasoning = [
            "- Identify and reason about specific differences between our writing and the user's writing.",
            "- Consider the full spectrum of writing traits, from distinct quirks to broader stylistic tendencies.",
            "- Provide a concise set of preferences in the imperative form.",
            "- Be precise; make the fewest possible changes to the preference set.",
            "- Do not qualify, dilute, or soften preferences.",
            "- Only refine the preferences if a clear difference exists. Otherwise, preserve the current preferences."
        ]

        self.other_val_reasoning = []

    # EVERYTHING BELOW ARE PROMPTS FROM THE PRELUDE REPO
    def get_preference_inference_prompt(self,
                                        trajectories: t.List[t.Tuple[WritingTrajectory, WritingTrajectory]]) -> str:
        """
        Creates the prompt to infer preferences given the agent's attempted trajectory and the user's correct trajectory
        Args:
            trajectories: A tuple of trajectories containing the agent's trajectory and the user's trajectory
        Returns:
            The prompt to feed to the LLM agent
        """
        prompt = ""
        output_type = "email" if self.config.task.name == "email_writing" else "summary"
        for agent_trajectory, user_trajectory in trajectories:
            prompt += f"Original {output_type}: {agent_trajectory.completed_task}\n"
            prompt += f"Revised {output_type}: {user_trajectory.completed_task}\n\n"
        prompt += (f"Based on the edits and revisions by this user on the original {output_type} in the above examples,"
                   " what do you find about this user's generic preference in terms of writing style and formatting? "
                   "Please answer in a short phrase and only recommend those preferences that are widely used. ")
        return prompt

    def get_majority_preference_prompt(self, preferences: t.List[str]) -> str:
        """
        Creates the prompt to get an LLM to aggregate a set of preferences.
        Args:
            preferences: A list of preferences to aggregate
        Returns:
            The prompt to feed to the LLM agent
        """
        output_type = "emails" if self.config.task.name == "email_writing" else "summaries"
        prompt = f"List of user preferences successfully being used to generate similar {output_type}: \n"
        for preference in preferences:
            prompt += f'- {preference}\n'
        prompt += ("Based on the above examples, "
                   "please come up with short phrase with the most represented writing preferences of this user. ")
        return prompt
