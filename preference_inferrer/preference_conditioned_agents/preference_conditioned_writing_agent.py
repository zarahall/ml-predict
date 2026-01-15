#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import re
import typing as t
from pathlib import Path

from preference_inferrer.llms import get_llm
from preference_inferrer.preference_conditioned_agents.abstract_preference_conditioned_agent import \
    PreferenceConditionedAgent
from preference_inferrer.preference_sets.writing_preference_set import (WritingPreferenceSet,
                                                                        EMAIL_SOURCE_TO_PREFERENCE,
                                                                        ARTICLE_SOURCE_TO_PREFERENCE)
from preference_inferrer.tasks.task_instances import WritingTaskInstance
from preference_inferrer.tasks.trajectory_logger import WritingTrajectory


SYSTEM_PROMPTS = {
    "plume": "You are an experienced writer. Adapt your writing to heavily emphasize the provided preferences. ",
    "prelude": "You are an AI assistant that helps people find information. "
}


class PreferenceConditionedWritingAgent(PreferenceConditionedAgent):
    def __init__(self, config, logger_name: str):
        """
        Args:
            config: Configurations
            logger_name: Name to provide the logger. Used to differentiate the output of the user's and agent's LLMs.
        """
        super().__init__(config)
        self.llm = get_llm(config.user.llm_name, logger_name, config)

        # map prelude.no_edit to prelude
        framework = self.config.task.framework.split(".")[0]
        self.system_prompt = SYSTEM_PROMPTS[framework]
        self.article_source_to_preference = ARTICLE_SOURCE_TO_PREFERENCE[framework]
        self.email_source_to_preference = EMAIL_SOURCE_TO_PREFERENCE[framework]

    def get_task_prompt(self, task_name: str, task_content: str, preferences: t.Optional[WritingPreferenceSet] = None,
                        in_context_examples: t.Optional[str] = None) -> str:
        """
        Get prompt to feed to LLM to complete task
        Args:
            task_name: Name of task to complete
            task_content: Notes/Article to write an email about/summarize
            preferences (Optional): Preferences to use while completing task
            in_context_examples (Optional): In context examples to use while completing task
        Returns:
            Prompt to use to complete task
        """
        content_type = "notes" if task_name == "email_writing" else "article"
        these_this = "these" if task_name == "email_writing" else "this"
        output_type = "email" if task_name == "email_writing" else "summary"

        if self.config.task.framework in ["prelude", "prelude.no_edit"]:
            task_prompt = f"{content_type.capitalize()}: {task_content}\n"
            if preferences is None or preferences.is_empty():
                if task_name == "email_writing":
                    task_prompt += f"Write a short email based on your above notes: "
                else:  # summarization
                    task_prompt += f"Please summarize the above article: "
            else:
                preferences = preferences.in_natural_language(mode="string")
                if task_name == "email_writing":
                    task_prompt += (f"These {content_type} are written by a user who prefers "
                                    f"the following style of emails: {preferences}. "
                                    f"Please write a short email based on the above "
                                    f"notes to address those specified preferences.")
                else:  # summarization
                    task_prompt += (f"Assume that you need to summarize the above article for a user, "
                                    f"who prefers the following style: {preferences} "
                                    f"Please write a summary of the above article to address "
                                    f"those specified preferences. ")
        else:
            if preferences is not None and not preferences.is_empty() and in_context_examples is not None:
                preference_list = preferences.in_natural_language(mode="list")
                preference_str = "\n" + ", ".join(preference_list)
                task_prompt = f"You have the following preferences: {preference_str}\n"
                task_prompt += f"You have also previously observed the following examples:\n"
                for i, (content, completion) in enumerate(in_context_examples):
                    task_prompt += (f"--- Example {i + 1} ---\n"
                                    f"{content_type.capitalize()}:\n"
                                    f"[START OF {content_type.upper()}]\n"
                                    f"{task_content}\n"
                                    f"[END OF {content_type.upper()}]\n\n"
                                    f"{output_type.capitalize()}:\n"
                                    f"\n\"\"\"\n{completion}\n\"\"\"\n")
                task_prompt += (f"Using the style of the above examples combined with your preferences, "
                                f"write a short {output_type} about {these_this} {content_type}:\n")
            elif preferences is not None and not preferences.is_empty():
                preference_list = preferences.in_natural_language(mode="list")
                preference_str = "\n" + ", ".join(preference_list)
                task_prompt = (f"You have the following preferences: {preference_str}\n"
                               f"Using these preferences, "
                               f"write a short {output_type} about {these_this} {content_type}:\n")
            elif in_context_examples is not None:
                task_prompt = f"You have previously observed the following examples:\n"
                for i, (content, completion) in enumerate(in_context_examples):
                    task_prompt += (f"--- Example {i + 1} ---\n"
                                    f"{content_type.capitalize()}:\n"
                                    f"[START OF {content_type.upper()}]\n"
                                    f"{task_content}\n"
                                    f"[END OF {content_type.upper()}]\n\n"
                                    f"{output_type.capitalize()}:\n"
                                    f"\n\"\"\"\n{completion}\n\"\"\"\n")
                task_prompt += (f"Using the style of the above examples, "
                                f"write a short {output_type} about {these_this} {content_type}:\n")
            else:
                task_prompt = f"Write a short {output_type} based on {these_this} {content_type}:\n"

            task_prompt += (f"[START OF {content_type.upper()}]\n"
                            f"{task_content}\n"
                            f"[END OF {content_type.upper()}]\n")

            task_prompt += f"Encapsulate the {output_type} in triple quotes \n\"\"\"\n<{output_type}>\n\"\"\"\n"
            if preferences is not None and not preferences.is_empty():
                preference_list = preferences.in_natural_language(mode="list")
                task_prompt += (f"Remember to incorporate all the following preferences:"
                                f"\n")
                task_prompt += "\n".join(preference_list)

        return task_prompt

    @staticmethod
    def get_edit_prompts(task_name: str, task_content: str, draft: str,
                         preferences: t.Optional[WritingPreferenceSet]) -> t.Tuple[str, str]:
        """
        Creates two prompts for the email writing task to feed to the user (e.g. gpt4) to ask
            1) whether the agent's email is satisfactory
            2) if it is not, how the user would edit the draft email given their preferences.
        Args:
            task_name: name of the task to complete
            task_content: The notes/article to write the email/summary about
            draft: A draft email/summary
            preferences: The preference to use when writing
        Returns:
            A tuple consisting of the resolution prompt (should we edit) and the edit prompt (edit this)
        """
        if preferences is None:
            preferences = "the user has no preferences"
        else:
            preferences = preferences.in_natural_language(mode="string")

        if task_name == "email_writing":
            resolution_prompt = (f"Notes: {task_content}\n"
                                 f"Email: {draft}\n"
                                 f"Is the above email based on the above notes good for a user "
                                 f"who wants the following style: {preferences}? Please answer yes or no.")

            edit_prompt = (f"Email: {draft}\n"
                           f"Assume that you prefer {preferences}. "
                           f"Please revise the above email to meet your style: ")

        else:  # summarization
            resolution_prompt = (f"Article: {task_content}\n"
                                 f"Summary: {draft}\n"
                                 f"Is the above summary of the above article good for person "
                                 f"who would love to use the following style: {preferences}? Please answer yes or no.")

            edit_prompt = (f"Summary: {draft}\n"
                           f"Assume that you prefer {preferences}. "
                           f"Please revise the above summary of an article to meet your style: ")
        return resolution_prompt, edit_prompt

    def get_true_preferences(self, task_instance: WritingTaskInstance) -> WritingPreferenceSet:
        """
        Use user level knowledge to retrieve the user's true preferences.
        Used for oracle agent, metrics, and the actual user.
        Args:
            task_instance: the task instance that the preferences apply to.
        Returns:
            The user's true preferences.
        """
        if task_instance.task_name == "summarization":
            preferences = self.article_source_to_preference[task_instance.source]
        elif task_instance.task_name == "email_writing":
            preferences = self.email_source_to_preference[task_instance.source]
        else:
            raise ValueError(f"Unknown task {task_instance.task_name}.")
        
        preferences = [p.strip(" ") for p in preferences.split("; ")]
        return WritingPreferenceSet(preferences, self.config)

    def solve_task(self, task_instance: WritingTaskInstance,
                   is_user: bool = False,
                   preferences: t.Optional[WritingPreferenceSet] = None,
                   in_context_examples: t.Optional[t.List[str]] = None,
    ) -> WritingTrajectory:
        """
        Solve the task instance optionally using provided preferences or in context examples.
        Args:
            task_instance: Task to solve with the provided preferences.
            is_user: True if the user calling this, else false (default)
            preferences (Optional): Preferences to use when solving the task instance
            in_context_examples (Optional): In context examples to use when solving the task instance
        Returns:
            The "trajectory" of the agent. E.g. the start state (task instance) and action (writing) taken by the agent.
        """
        if self.config.task.framework == "prelude" and is_user:
            # Used by the user in prelude
            resolution_prompt, edit_prompt = self.get_edit_prompts(task_instance.task_name, task_instance.task_content,
                                                                   task_instance.agent_completion, preferences)
            self.llm.clear_chat_history(self.system_prompt)
            resolution = self.llm.prompt_llm(resolution_prompt)
            if "yes" in resolution.lower():
                is_edited = False
                final_output = task_instance.agent_completion
            else:  # "no" in resolution.lower():
                is_edited = True
                self.llm.clear_chat_history(self.system_prompt)
                final_output = self.llm.prompt_llm(edit_prompt)
        else:
            task_prompt = self.get_task_prompt(task_instance.task_name, task_instance.task_content,
                                               preferences=preferences, in_context_examples=in_context_examples)
            is_edited = False
            self.llm.clear_chat_history(self.system_prompt)
            final_output = self.llm.prompt_llm(task_prompt)
            if self.config.task.framework == "plume":
                # Extract the output from between the triple quotes
                match = re.search(r'"""(.*?)"""', final_output, re.DOTALL)
                if match:
                    # Extract the matched text and strip leading/trailing whitespace
                    final_output = match.group(1).strip()

        return WritingTrajectory(self.config, task_instance.task_name, task_instance.task_content, is_edited,
                                 final_output)

    def get_metrics(self):
        return self.llm.get_metrics()
