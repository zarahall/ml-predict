#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import typing as t
from itertools import chain, islice

import numpy as np
import pandas as pd

from preference_inferrer.preference_sets.writing_preference_set import (WritingPreferenceSet,
                                                                        EMAIL_SOURCE_TO_PREFERENCE)
from preference_inferrer.tasks.writing.dataset_helpers import load_data, OurInputExample
from preference_inferrer.tasks.writing.writing_quality_assesor import WritingQualityAssessor
from preference_inferrer.tasks.task import Task
from preference_inferrer.tasks.task_instances import WritingTaskInstance


class EmailWriting(Task):
    """
    Email writing task - one of the two tasks in the user edit environment
    Given an article from a specific source, write an email about the article according to some initially unknown
    preferences.
    """
    ALL_DATASETS = ["slf5k", "ccby", "ampere", "paper_tweet"]

    def __init__(self, config):
        super().__init__(config)
        self.data = self.get_dataset()
        # map prelude.no_edit to prelude
        framework = self.config.task.framework.split(".")[0]
        self.possible_preferences = list(EMAIL_SOURCE_TO_PREFERENCE[framework].values())
        self.quality_assessor = WritingQualityAssessor(self.config)

    def get_dataset(self) -> t.List[OurInputExample]:
        """
        Loads examples from several datasets and combines them into a single dataset.
        Returns:
            A list where each element contains the required information for a single instance of a dataset
        """
        set_of_datasets = (self.config.task.datasets or self.ALL_DATASETS)

        combined_dataset = []
        num_examples_per_source = self.config.task.number_of_examples_per_preference_set
        for dataset in set_of_datasets:
            combined_dataset.append(list(load_data(dataset=dataset, num_ex=-1, split="train")))
        rng = np.random.default_rng(seed=self.config.seed)
        for dataset in combined_dataset:
            rng.shuffle(dataset)
        combined_dataset = list(chain.from_iterable(
            map(lambda inner_dataset: islice(inner_dataset, num_examples_per_source), combined_dataset))
        )
        if self.config.retrieval_mode != "exact_match":
            rng.shuffle(combined_dataset)
        return combined_dataset

    def get_task_instances(self) -> t.Iterable[WritingTaskInstance]:
        """
        Iterate over dataset instances
        """
        for d in self.data:
            source_idx = self.ALL_DATASETS.index(d.doc_type)
            yield WritingTaskInstance("email_writing", d.article.strip(), d.doc_type, source_idx)

    def get_metrics(self, task_instance: WritingTaskInstance, pred_preferences: WritingPreferenceSet,
                    true_preferences: WritingPreferenceSet) -> t.Dict[str, t.Any]:
        """
        Returns a dictionary containing metrics that measure how successful the agent did.
        Args:
            task_instance: Contains the task content as well as the agent's and user's trajectory that complete it
            pred_preferences: The inferred preferences
            true_preferences: The user's true preferences
        Returns:
            A dictionary with metrics where the key is a string and the values are numeric
        """
        agent_completion, user_completion = task_instance.agent_completion, task_instance.user_completion
        # Calculate metrics related to the summaries
        writing_cost, writing_n_cost = self.quality_assessor.calculate_levenshtein_distance(agent_completion,
                                                                                            user_completion)
        writing_bs, _, _ = self.quality_assessor.get_bertscore(agent_completion, user_completion)
        writing_llm_pp_match = self.quality_assessor.llm_judges_preference_matching(task_instance,
                                                                                    true_preferences,
                                                                                    per_preference=True)
        writing_llm_fp_match = self.quality_assessor.llm_judges_preference_matching(task_instance,
                                                                                    true_preferences,
                                                                                    per_preference=False)
        writing_llm_av_match = (writing_llm_pp_match + writing_llm_fp_match) / 2

        # Calculate metrics relating to how well preferences were inferred
        true_preferences = true_preferences.in_natural_language(mode="string")
        pred_preferences = pred_preferences.in_natural_language(mode="string") or ""
        pred_preferences = pred_preferences or "None"

        correct_idx = self.possible_preferences.index(true_preferences)
        f1, precision, recall, accuracy = self.quality_assessor.get_bertscore(
            [pred_preferences] * len(self.possible_preferences),
            self.possible_preferences,
            correct_idx
        )

        metrics = {
            "writing_ldist": writing_cost,
            "writing_n_ldist": writing_n_cost,
            "writing_BS": writing_bs,
            "writing_ppm": writing_llm_pp_match,
            "writing_fpm": writing_llm_fp_match,
            "writing_avm": writing_llm_av_match,
            "preference_acc": accuracy,
            "preference_f1": f1,
            "preference_pre": precision,
            "preference_rec": recall,
            "preference_string_length": self.quality_assessor.preference_len(pred_preference=pred_preferences),
            "preference_llm_judge_similarity": self.quality_assessor.score_preference_similarity(
                pred_preference=pred_preferences, true_preference=true_preferences),
            **self.quality_assessor.get_metrics()
        }
        return metrics

    def get_llm_judge_question_and_answer_df(self) -> pd.DataFrame:
        """
        Save LLM as a Judge questions and answer to csv to enable human comparison
        """
        return self.quality_assessor.get_llm_judge_question_and_answer_df()
