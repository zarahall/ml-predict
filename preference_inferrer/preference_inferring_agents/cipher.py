#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import typing as t
from types import SimpleNamespace
from preference_inferrer.tasks.trajectory_logger import WritingTrajectory
from preference_inferrer.preference_inferring_agents.abstract_agent import Agent
from preference_inferrer.preference_inferring_agents.example_retriever import ExampleRetriever
from preference_inferrer.preference_sets import WritingPreferenceSet
from preference_inferrer.tasks.task_instances import WritingTaskInstance


class Cipher1Agent(Agent):
    """
    One of two main agents from 'Aligning LLM Agents by Learning Latent Preference from User Edits'
    Adapted from https://github.com/gao-g/prelude/blob/main/src/agent/cipher.py
    On each edit, if the agent completion is not accepted, CIPHER probes an LLM for a preference that could explain the
    edits and stores the task context along with the learned preference. When completing a new task, CIPHER identifies
    the most similar previously solved task context, and uses the learned preference to solve the new task.
    """

    def __init__(self, config: SimpleNamespace):
        """
        Args:
            config: Configurations
        """
        super().__init__(config)
        # Create an encoder model to encode documents
        self.example_retriever = ExampleRetriever(config)
        # Current context - used as an assertion check that complete was always called before learn
        self.current_context = None
        self.preference = WritingPreferenceSet(None, self.config)
        
        self.system_prompt = self.environment_prompts.system_prompt

    def complete(self, task_instance: WritingTaskInstance) -> t.Tuple[WritingTrajectory, WritingPreferenceSet]:
        """
        Complete the task instance
        Args:
            task_instance: the task instance to complete
        Returns:
            The trajectory (start state and actions) completing the task, and the preference set used to guide actions
        """
        # Retrieve the most similar examples if one exists, and use the preference learned from that example as the
        # starting point.

        examples = self.example_retriever.get_most_similar_examples(task_instance.context, top_k=1,
                                                                    doc_source=task_instance.source)
        if len(examples) > 0:
            _, _, _, self.preference = examples[0]
        else:
            self.preference = WritingPreferenceSet.create_empty_preference_set(self.config)
        # Prompt the llm to complete the task with the learned preferences
        trajectory = self.pc_agent.solve_task(task_instance, preferences=self.preference)
        return trajectory, self.preference

    def learn(self, task_instance: WritingTaskInstance, agent_trajectory: WritingTrajectory,
              user_trajectory: WritingTrajectory):
        """
        Learn from the completed (agent trajectory) and optionally edited (user trajectory) task instance
        Args:
            task_instance: The task instance that was solved
            agent_trajectory: The agent's attempt at completing the task
            user_trajectory: The user's correct completion of the task
        """
        # If the user edited the trajectory, prompt the llm to learn a preference that could explain the edits
        # so that we can use that preference in the future.
        # If edits were not made, we assume the starting preference was satisfactory, so we keep it
        if agent_trajectory != user_trajectory:
            prompt = self.environment_prompts.get_preference_inference_prompt([(agent_trajectory, user_trajectory)])
            self.llm.clear_chat_history(self.system_prompt)
            preference = self.llm.prompt_llm(prompt)
            self.preference = WritingPreferenceSet(preference, self.config)
        # Store the context and preference for future use
        self.example_retriever.add(task_instance.context, task_instance.source, self.preference)

    def get_metrics(self) -> t.Dict[str, t.Any]:
        """
        Calculate and returns a dictionary of numerically measurable metrics
        Returns:
            A dictionary containing metrics (keys are strings with metric names, values are the numeric metric values).
        """
        return {
            **self.llm.get_metrics(),
            **self.pc_agent.get_metrics(),
            **self.example_retriever.get_metrics()
        }


class CipherNAgent(Agent):
    """
    One of two main agents from 'Aligning LLM Agents by Learning Latent Preference from User Edits'
    Adapted from https://github.com/gao-g/prelude/blob/main/src/agent/cipher.py
    NOTE: The main difference between this class and the above class is that this class aggregates the preferences
          from multiple previous examples rather than just using a single previous example
    On each edit, if the agent completion is not accepted, CIPHER probes an LLM for a preferences that could explain the
    edits and stores the task context along with the learned preference. When completing a new task, CIPHER identifies
    the most similar previously solved task contexts, aggregates the preferences learned in those tasks,
     and uses the aggregated preference to solve the new task.
    """

    def __init__(self, config: SimpleNamespace, icl_count=3):
        super().__init__(config)
        # Create an encoder model to encode documents
        self.example_retriever = ExampleRetriever(config)
        self.icl_count = icl_count
        # Current doc
        self.preference = None
        self.pref_aggregated = None
        self.system_prompt = self.environment_prompts.system_prompt

    def complete(self, task_instance: WritingTaskInstance) -> t.Tuple[WritingTrajectory, WritingPreferenceSet]:
        """
        Complete the task instance
        Args:
            task_instance: the task instance to complete
        Returns:
            The trajectory (start state and actions) completing the task, and the preference set used to guide actions
        """
        # Retrieve the most similar examples and their associated preferences
        preferences = None
        related_examples = self.example_retriever.get_most_similar_examples(task_instance.context,
                                                                            top_k=self.icl_count,
                                                                            doc_source=task_instance.source)
        if len(related_examples) > 0:
            preferences = [pref.in_natural_language(mode="string") for _, _, _, pref in related_examples]
        # Aggregate the preferences retrieved above
        if preferences:
            prompt = self.environment_prompts.get_majority_preference_prompt(preferences)
            self.llm.clear_chat_history(self.system_prompt)
            pref_aggregated_str = self.llm.prompt_llm(prompt)
            self.pref_aggregated = WritingPreferenceSet(pref_aggregated_str, self.config)
        else:
            self.pref_aggregated = WritingPreferenceSet.create_empty_preference_set(self.config)
        # Prompt the llm to complete the task with the aggregated preferences
        traj = self.pc_agent.solve_task(task_instance, preferences=self.pref_aggregated)
        return traj, self.pref_aggregated

    def learn(self, task_instance: WritingTaskInstance, agent_trajectory: WritingTrajectory,
              user_trajectory: WritingTrajectory):
        """
        Learn from the completed (agent trajectory) and optionally edited (user trajectory) task instance
        Args:
            task_instance: The task instance that was solved
            agent_trajectory: The agent's attempt at completing the task
            user_trajectory: The user's correct completion of the task
        """
        # If the user edited the trajectory, prompt the llm to learn a preference that could explain the edits
        # so that we can use that preference in the future.
        # If edits were not made, we assume the starting aggregated preference was satisfactory, so we keep it
        self.preference = self.pref_aggregated
        if agent_trajectory != user_trajectory:
            self.llm.clear_chat_history(self.system_prompt)
            prompt = self.environment_prompts.get_preference_inference_prompt([(agent_trajectory, user_trajectory)])
            preference = self.llm.prompt_llm(prompt)
            self.preference = WritingPreferenceSet(preference, self.config)

        # Store the context and preference for future use
        self.example_retriever.add(task_instance.context, task_instance.source, self.preference)

    def get_metrics(self) -> t.Dict[str, t.Any]:
        """
        Calculate and returns a dictionary of numerically measurable metrics
        Returns:
            A dictionary containing metrics (keys are strings with metric names, values are the numeric metric values).
        """
        return {
            **self.llm.get_metrics(),
            **self.pc_agent.get_metrics(),
            **self.example_retriever.get_metrics()
        }

