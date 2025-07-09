#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import logging
import typing as t

import editdistance
import nltk
import numpy as np
import pandas as pd
from evaluate import load
from nltk.tokenize import word_tokenize

from preference_inferrer.llms import get_llm
from preference_inferrer.preference_sets import PreferenceSet
from preference_inferrer.tasks.task_instances import TaskInstance

nltk.download("punkt_tab")


class WritingQualityAssessor:
    """
    Class that contains several method to evaluate the quality of writing with reference to a ground truth text
    or ground truth set of preferences.
    """
    bertscore = None
    bleu = None
    llm = None

    def __init__(self, config):
        """
        Args:
            config: configurations
        """
        self.config = config
        self.logger = logging.getLogger("JUDGE")
        columns = ["question", "llm_score"]
        self.df = pd.DataFrame(columns=columns)

    @staticmethod
    def calculate_levenshtein_distance(agent_completion: str, user_completion: str) -> t.Tuple[int, float]:
        """
        Calculate the Levenshtein distance between the agent's completed summary and the user's completed summary.
        This functions returns both the un-normalized the normalized cost
        Args:
            agent_completion: The agent's summary
            user_completion: The user's summary
        Returns:
            The levenshtein distance as an int, normalized levenshtein distance as a float
        """
        tokenized_agent_completion = word_tokenize(agent_completion)
        tokenized_user_completion = word_tokenize(user_completion)
        l_dist = editdistance.eval(tokenized_agent_completion, tokenized_user_completion)
        normalize_l_dist = l_dist / max(len(tokenized_user_completion), len(tokenized_agent_completion))
        return l_dist, normalize_l_dist

    def get_bertscore(self, predictions: t.Union[str, t.List[str]], references: t.Union[str, t.List[str]],
                      correct_idx: t.Optional[int] = None) -> t.Union[t.Tuple[float, float, float],
                                                                      t.Tuple[float, float, float, int]]:
        """
        Returns the BERTScore f1, precision, and recall between the prediction and reference(s).
        If a "correct_idx" is provided, it will additionally calculate the accuracy of the prediction
        (i.e. is the prediction closest to the correct reference within a list of references.)
        Args:
            predictions: The generated sample
            references: The reference(s) (ground-truth) sample
            correct_idx (Optional): The index of the true reference
        Returns:
            BERTScore f1, precision, and recall between the predictions and references.
            If correct_idx is provided, also returns if the prediction was closest (BERTScore) to the correct reference
        """
        if WritingQualityAssessor.bertscore is None:
            WritingQualityAssessor.bertscore = load("bertscore")
        if isinstance(predictions, str):
            predictions = [predictions]
        if isinstance(references, str):
            references = [references]

        writing_similarity = WritingQualityAssessor.bertscore.compute(predictions=predictions,
                                                                      references=references,
                                                                      model_type=self.config.bertscore_model,
                                                                      rescale_with_baseline=True, lang="en")
        if correct_idx is not None:
            correct = int(np.argmax(writing_similarity["f1"]) == correct_idx)
            return (writing_similarity["f1"][correct_idx], writing_similarity["precision"][correct_idx],
                    writing_similarity["recall"][correct_idx], correct)

        return writing_similarity["f1"][0], writing_similarity["precision"][0], writing_similarity["recall"][0]

    def score_adherence_to_preference_component(self, task_instance: TaskInstance, agent_completion: str,
                                                preference: str) -> int:
        """
        Uses LLM-as-a-Judge to calculate how well a writing sample (agent_completion) exhibits a preference on a
        5-point likert scale.
        Args:
            task_instance: The task that the writing sample came from
            agent_completion: The writing sample to evaluate
            preference: The preference to evaluate the writing sample on
        Returns:
            A integer score between -2 and 2
        """
        output_type = "email" if task_instance.task_name == "email_writing" else "summary"

        WritingQualityAssessor.llm.clear_chat_history(
            "You are an experienced editor that is evaluating writing samples. "
        )
        prompt = (f"You received the following {output_type}:\n"
                  f"\"\"\"\n"
                  f"{agent_completion}\n"
                  f"\"\"\"\n"
                  f"Does the above {output_type} exhibit the following preference: \"{preference}\"?\n" 
                  f"Identify, analyze, and reason about specific excerpts that show similarities or contradictions "
                  f"of underlying preferences. "
                  f"After reasoning, select one of the following options:\n"
                  f"clearly exhibits, somewhat exhibits, neither exhibits nor contradicts, "
                  f"somewhat contradicts, clearly contradicts\n"
                  f"Your final selection should be on a new line prefaced with \"Verdict:\"")
        response = WritingQualityAssessor.llm.prompt_llm(prompt).lower()

        answer = ""
        for line in response.split("\n"):
            # The llm often outputs either quotation marks or Markdown formatting "*" around the word.
            line = line.strip('"*').lower()
            if line.startswith("verdict:"):
                answer = line.removeprefix("verdict:").strip('"*: ')
                if answer != "":
                    break

        if "clearly exhibits" in answer:
            score = 2
        elif "somewhat exhibits" in answer:
            score = 1
        elif "neither exhibits nor contradicts" in answer:
            score = 0
        elif "somewhat contradicts" in answer:
            # adhere has to be last because 'agrees' will be 'in' most other options
            score = -1
        elif "clearly contradicts" in answer:
            score = -2
        else:
            self.logger.warning(f"Unknown response in LLM Judge (ppm): {response}")
            score = 0

        self.df.loc[len(self.df)] = [prompt, score]
        return score

    def preference_len(self, pred_preference: str) -> int:
        """
        Returns how long a preference is. Shorter is better for the same score
        Args:
            pred_preference: The inferred (predicted) preference
        Returns:
            A integer denoting the length of the preference string
        """
        return len(pred_preference)

    def score_preference_similarity(self, pred_preference: str, true_preference: str) -> int:
        """
        Uses LLM-as-a-Judge to calculate how well an inferred (predicted) preference matches the true preference on a
        5-point likert scale.
        Args:
            pred_preference: The inferred (predicted) preference
            true_preference: The true preference to compare against
        Returns:
            A integer score between -2 and 2
        """
        WritingQualityAssessor.llm.clear_chat_history(
            "You are an experienced editor that is evaluating how similar writing preferences are. "
        )
        prompt = (f"You receive the following description of a user's writing preferences:\n"
                  f"\"\"\"\n"
                  f"Inferred preferences:\n{pred_preference}\n"
                  f"\"\"\"\n"
                  f"How similar are the inferred preferences to the true writing preferences below?\n"
                  f"\"\"\"\n"
                  f"True preferences:\n{true_preference}\n"
                  f"\"\"\"\n" 
                  f"Analyze how the preferences would impact a user's writing. "
                  f"After reasoning, select one of the following options:\n"
                  f"extremely similar, very similar, moderately similar, slightly similar, not at all similar\n"
                  f"Your final selection should be on a new line prefaced with \"Verdict:\"")
        response = WritingQualityAssessor.llm.prompt_llm(prompt).lower()

        answer = ""
        for line in response.split("\n"):
            # The llm often outputs either quotation marks or Markdown formatting "*" around the word.
            line = line.strip('"*').lower()
            if line.startswith("verdict:"):
                answer = line.removeprefix("verdict:").strip('"*: ')
                if answer != "":
                    break

        if "extremely similar" in answer:
            score = 4
        elif "very similar" in answer:
            score = 3
        elif "moderately similar" in answer:
            score = 2
        elif "slightly similar" in answer:
            # adhere has to be last because 'agrees' will be 'in' most other options
            score = 1
        elif "not at all similar" in answer:
            score = 0
        else:
            self.logger.warning(f"Unknown response in LLM Judge (ppm): {response}")
            score = 0

        self.df.loc[len(self.df)] = [prompt, score]
        return score

    def score_adherence_to_compound_preference(self, task_instance: TaskInstance, agent_completion: str,
                                               preference: str) -> int:
        """
        Uses LLM-as-a-Judge to calculate how well a writing sample (agent_completion) exhibits a set of preferences on a
        5-point likert scale.
        NOTE: The difference between this metric and `score_adherence_to_preference_component` is that this metric
              focuses on matching compound preferences rather than a singular preference component
        Args:
            task_instance: The task that the writing sample came from
            agent_completion: The writing sample to evaluate
            preference: The preference to evaluate the writing sample on
        Returns:
            A integer score between -2 and 2
        """
        output_type = "email" if task_instance.task_name == "email_writing" else "summary"
        WritingQualityAssessor.llm.clear_chat_history(
            "You are an experienced editor that is evaluating writing samples. "
        )
        prompt = (f"You received the following {output_type}:\n"
                  f"\"\"\"\n"
                  f"{agent_completion}\n"
                  f"\"\"\"\n"
                  f"To what extent does the above writing exhibit the following preferences: {preference}?\n"  
                  f"Identify, analyze, and reason about specific excerpts that show similarities or contradictions "
                  f"of underlying preferences. "
                  f"After reasoning, select one of the following options:\n"
                  f"exhibits all preferences, exhibits most preferences, exhibits some preferences, "
                  f"exhibits few preferences, exhibits none of the preferences\n"
                  f"Your final selection should be on a new line prefaced with \"Verdict:\"")
        response = WritingQualityAssessor.llm.prompt_llm(prompt).lower()

        answer = ""
        for line in response.split("\n"):
            # The llm often outputs either quotation marks or Markdown formatting "*" around the word.
            line = line.strip('"*').lower()
            if line.startswith("verdict:"):
                answer = line.removeprefix("verdict:").strip('"*: ')
                if answer != "":
                    break

        if "all preferences" in answer:
            score = 2
        elif "most preferences" in answer:
            score = 1
        elif "some preferences" in answer:
            score = 0
        elif "few preferences" in answer:
            score = -1
        elif "none of the preferences" in answer:
            score = -2
        else:
            score = 0
            self.logger.warning(f"Unknown response in LLM Judge (fpm): {response}")
        return score

    def llm_judges_preference_matching(self, task_instance: TaskInstance,
                                       true_preference_set: PreferenceSet, per_preference=True) -> float:
        """
        Calculates the per-preference-component-match (ppcm) or compound-preference-match (cpm) depending on
        the `per_preference` parameter
        Args:
            task_instance: The task that the writing sample came from
            true_preference_set: The true set of preferences
            per_preference: If true, calculate ppcm, otherwise calculate cpm
        Returns:
            A score between -2 and 2
        """
        if WritingQualityAssessor.llm is None:
            WritingQualityAssessor.llm = get_llm(self.config.user.llm_name, "JUDGE", self.config)

        agent_completion = task_instance.agent_completion
        if per_preference:
            scores = [self.score_adherence_to_preference_component(task_instance, agent_completion, preference)
                      for preference in sorted(true_preference_set.as_single_set())]
            score = np.mean(scores)
        else:
            compound_preference = true_preference_set.in_natural_language(mode="string")
            score = self.score_adherence_to_compound_preference(task_instance, agent_completion, compound_preference)
        return score

    @staticmethod
    def get_metrics() -> t.Dict:
        """
        Returns a dictionary of metrics
        """
        if WritingQualityAssessor.llm is not None:
            return WritingQualityAssessor.llm.get_metrics()
        else:
            return {}

    def get_llm_judge_question_and_answer_df(self) -> pd.DataFrame:
        """
        Save LLM as a Judge questions and answer to csv to enable human comparison
        """
        return self.df
