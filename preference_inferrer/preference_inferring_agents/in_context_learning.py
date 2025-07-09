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


class InContextLearningAgent(Agent):
    def __init__(self, config: SimpleNamespace):
        """
        Adapted from https://github.com/gao-g/prelude/blob/main/src/agent/icl_edit.py
        Retrieves an icl_count number of (task, user_output) pairs.
        Feed these pairs directly into the prompt as in-context learning examples to complete new tasks.
        NOTE: This agent only works in PLUME
        """
        super().__init__(config)
        self.example_retriever = ExampleRetriever(config)
        self.icl_count = self.config.num_other_sequences
        self.system_prompt = self.environment_prompts.system_prompt

    def complete(self, task_instance: WritingTaskInstance) -> t.Tuple[WritingTrajectory, WritingPreferenceSet]:
        """
        Retrieve an icl_count number of related (task, user completion) pairs from history.
        Uses these pairs as in context examples to complete the new task instance
        Args:
            task_instance: The new task to complete.
        Returns:
            The trajectory (start state and actions) used to solve the task.
        """
        examples = self.example_retriever.get_most_similar_examples(task_instance.context, top_k=self.icl_count,
                                                                    doc_source=task_instance.source)
        if any(examples):
            examples = [(task_content, info) for task_content, _, _, info in examples]
        else:
            examples = None

        # Prompt the llm to complete the task with the learned preferences
        trajectory = self.pc_agent.solve_task(task_instance, in_context_examples=examples)
        return trajectory, WritingPreferenceSet.create_empty_preference_set(self.config)

    def learn(self, task_instance: WritingTaskInstance, agent_trajectory: WritingTrajectory,
              user_trajectory: WritingTrajectory):
        """
        Add the (agent and user completions) pair to our history. Return an empty preference set since no preferences
        are ever inferred.
        Args:
            task_instance: The task instance that was solved
            agent_trajectory: The agent's attempt at completing the task
            user_trajectory: The user's correct completion of the task
        Returns:
            An empty preference set
        """
        self.example_retriever.add(task_instance.context, task_instance.source, user_trajectory.completed_task)
