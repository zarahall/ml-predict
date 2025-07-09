#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import typing as t

from preference_inferrer.tasks.trajectory_logger import WritingTrajectory
from preference_inferrer.prompt_templates.abstract_templates import Templates
from preference_inferrer.tasks.task_instances import WritingTaskInstance


class PreludeTemplates(Templates):
    def initialize_prompts(self):
        self.system_prompt = "You are an AI assistant that helps people find information."

        self.format_prompts = {
            "infer": lambda line_keyword: (
                f"After reasoning, output the full refined preference on a single new line and prefaced "
                f"with \"{line_keyword}\".\n"
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

        # Additional information and directives that may be beneficial to the llm
        self.other_reasoning = [
            "- Identify and reason about specific differences between our writing and the user's writing.",
            "- Consider the full spectrum of writing traits, from distinct quirks to broader stylistic tendencies.",
            "- Provide a concise set of preferences in the imperative form.",
            "- Be precise; make the fewest possible changes to the preference set.",
            "- Do not qualify, dilute, or soften existing preferences.",
            "- Only refine the preferences if a clear difference exists. Otherwise, preserve the current preferences."
        ]

        self.other_val_reasoning = [
            # f"- Let's think step by step.",
            # f"- Before answering, identify, analyze, and reason about specific excerpts that confirm or "
            # f"contradict the preference."
        ]

    # EVERYTHING BELOW ARE PROMPTS FROM THE PRELUDE REPO
    def get_task_prompt_icl(self, task_instance: WritingTaskInstance,
                            trajectories: t.List[t.Tuple[str, WritingTrajectory, WritingTrajectory]]) -> str:
        """
        Creates the prompt to get an LLM agent to complete the task using in-context learning (ICL) examples.
        Args:
            task_instance: The task to solve
            trajectories: The icl examples to use to solve the task.
        Returns:
            The prompt to feed to the LLM agent
        """
        task_content = task_instance.task_content
        prompt = ""
        if self.config.task.name == "summarization":
            for _, agent_trajectory, user_trajectory in trajectories:
                prompt += f"Original summary of an article: {agent_trajectory.completed_task}\n"
                prompt += f"Revised summary by a user: {user_trajectory.completed_task}\n\n"
            prompt += (f"Article: {task_content}\n"
                       f"Based on the edits and revisions by this user on the original summary in the above examples, "
                       f"please summarize the above article: ")

        else:
            for agent_trajectory, user_trajectory in trajectories:
                prompt += f"Original email: {agent_trajectory.completed_task}\n"
                prompt += f"Revised email: {user_trajectory.completed_task}\n\n"
            prompt += (f"Notes: {task_content}\n"
                       f"Based on the edits and revisions by this user on the original email in the above examples, "
                       f"please write an email based on the above notes for this user: ")
        return prompt

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
        if self.config.task.name == "summarization":
            for agent_trajectory, user_trajectory in trajectories:
                prompt += f"Original summary of an article: {agent_trajectory.completed_task}\n"
                prompt += f"Revised summary by a user: {user_trajectory.completed_task}\n\n"
            prompt += (
                "Based on the edits and revisions by this user on the original summary in the above examples, "
                "what do you find about this user's generic preference in terms of writing style and formatting? "
                "Please answer in a short phrase and only recommend those preferences that are widely used. ")
        else:
            for agent_trajectory, user_trajectory in trajectories:
                prompt += f'Original email: {agent_trajectory.completed_task}\n'
                prompt += f'Revised email: {user_trajectory.completed_task}\n\n'
            prompt += (
                "Based on the edits and revisions by this user on the original email in the above examples, "
                "what do you find about this user's generic preference in terms of writing style and formatting? "
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
        if self.config.task.name == "summarization":
            prompt = "List of user preferences successfully being used to generate summaries of similar documents: \n"
            for preference in preferences:
                prompt += f"- {preference}\n"
            prompt += ("Based on the above examples, "
                       "please come up with short phrase with the most represented summarization preferences "
                       "of the user. ")
        else:
            prompt = "List of user preferences successfully being used to generate emails of a similar kind: \n"
            for preference in preferences:
                prompt += f'- {preference}\n'
            prompt += ("Based on the above examples, "
                       "please come up with short phrase with the most represented writing preferences of this user. ")
        return prompt
