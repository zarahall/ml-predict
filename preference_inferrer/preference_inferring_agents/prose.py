"""
Copyright (C) 2025 Apple Inc. All rights reserved.

The main preference inferrer class.

Defines a wrapper to an LLM to use it to infer preferences.
Includes "reasoning" structure, functions to create full prompts for prompt parts defined in template classes,
and functions for extracting certain kinds of outputs from llm responses.
"""
import copy
import json
import logging
import typing as t
from types import SimpleNamespace

import numpy as np

from preference_inferrer.tasks.trajectory_logger import Trajectory
from preference_inferrer.preference_inferring_agents.abstract_agent import Agent
from preference_inferrer.preference_inferring_agents.example_retriever import ExampleRetriever
from preference_inferrer.preference_sets import (PreferenceSet, WritingPreferenceSet,
                                                 get_empty_preference_set, get_preference_set)
from preference_inferrer.tasks.task_instances import TaskInstance


# PROSE: Preference Reasoning by Observing and Synthesizing Examples
class Prose(Agent):
    def __init__(self, config: SimpleNamespace):
        """
        Args:
            config: Configurations
        """
        super().__init__(config)
        self.example_retriever = ExampleRetriever(config)
        # Score of preferences over time. Note that once a certain average over enough samples has been reached
        # they move to the confirmed preferences and we no longer keep track of their score
        self.preference_scores = {}
        # keys are (preference: str, user_example_idx: int) tuples, values are entailment score output by the LLM
        # Note: items are only added if the LLM outputs a valid score.
        self.previously_validated = {}
        # If preference scores reach a certain threshold, confirm them, which avoid future validation
        self.confirmed_preferences = set()
        # Total number of "inferred preferences". Monotonically increases. Sees whether we are succeeding through actual
        # reasoning of just guessing a lot and then validating it down
        self.number_of_inferred_preferences = []
        # In case we early stop refinement because of thresholding, how many refinement steps are we actually seeing
        self.total_num_performed_refinement_steps = 0
        self.total_seen_examples = 0
        self.breakdown_skips = 0
        self.preferences_filtered_by_validation = 0
        # Number of times the "infer", "breakdown", and "validate" prompts are fed to LLMs
        self.prompt_counts = {"infer": 0, "breakdown": 0, "validate": 0, "attribute_in_example": 0, "coalesce": 0}
        # Number of times the LLM outputs from the "infer", "breakdown", and "validate" prompts are not parseable
        self.prompt_errors = {"infer": 0, "breakdown": 0, "validate": 0, "attribute_in_example": 0, "coalesce": 0}
        # Keeps track of what part of the prompting we are currently in. We start in 'infer' mode.
        self.current_prompting = "infer"
        # These are set in self.complete and then re-used in self.learn.
        # The current context keeps track of the task instance context to ensure that
        # every self.learn is preempted with a self.complete
        self.current_context = None
        self.starting_preferences = None
        self.related_examples_indices = None
        # Mapping between words outputted by LLM and a validation score
        self.validation_word_to_score = {"strongly confirms": 2, "somewhat confirms": 1, "neutral": 0,
                                         "somewhat contradicts": -1, "strongly contradicts": -2}

        # Logger
        self.logger = logging.getLogger("INFERRER")
        self.logger.setLevel(level=self.config.logging_level)

        self.system_prompt = self.environment_prompts.system_prompt

    def complete(self, task_instance: TaskInstance) -> t.Tuple[Trajectory, PreferenceSet]:
        """
        Complete the task instance. First collect relevant examples, then aggregate their preferences.
        Use the aggregated preferences to complete the task.
        Args:
            task_instance: The task instance to complete
        Returns:
            The trajectory that completes the task and the preferences used to generate the trajectory
        """
        self.current_context = task_instance.context
        # Retrieve relevant examples
        related_examples = self.example_retriever.get_most_similar_examples(task_instance.context,
                                                                            top_k=self.config.num_other_sequences,
                                                                            doc_source=task_instance.source)
        # Aggregate relevant preferences
        self.related_examples_indices = []
        self.starting_preferences = get_empty_preference_set(self.config.task.name, self.config)
        self.icl_examples = [] if self.config.use_icl else None
        related_examples.reverse()
        for _, sample_idx, _, (rel_task_instance, user_trajectory, preferences) in related_examples:
            self.starting_preferences.update(preferences)
            self.related_examples_indices.append(sample_idx)
            if self.config.use_icl:
                self.icl_examples.append((rel_task_instance.context, user_trajectory.completed_task))

        # Coalesce relevant preferences
        if len(self.starting_preferences) > 0 and task_instance.task_name in ["email_writing", "summarization"]:
            self.llm.clear_chat_history(self.system_prompt)
            self.starting_preferences = self.coalesce_preferences(self.starting_preferences)

        # Solve task
        completed_task = self.pc_agent.solve_task(task_instance, preferences=self.starting_preferences,
                                                  in_context_examples=self.icl_examples)
        return completed_task, self.starting_preferences

    def learn(self, task_instance: TaskInstance, agent_trajectory: Trajectory, user_trajectory: Trajectory):
        """
        Main function that runs the full inference pipeline on a given user trajectory.
        Args:
             task_instance: The task instance that was solved
             agent_trajectory: The agent's attempt at completing the task
             user_trajectory: The user's correct completion of the task
        """
        self.total_seen_examples += 1
        # Step 1. Get most relevant sequences and preferences from historical data.
        # Make sure a self.complete was run on this task instance before learning from it
        # This ensures that the starting preferences and related examples are correct
        assert task_instance.context == self.current_context
        inferred_preferences: PreferenceSet = copy.deepcopy(self.starting_preferences)
        if self.config.candidate_type.lower() == "none":
            counterfactual_sequence = None
        else:
            counterfactual_sequence = agent_trajectory

        # Step 2. Start refinement process
        # Baselines "one step single candidate (1sc)" and "one step no candidate (1nc)" only runs 1 step
        refinement_step_inferred_preferences = copy.deepcopy(inferred_preferences)
        for refinement_step in range(self.config.num_refinement_steps):
            # If current preferences create sequence identical to user preference, we've reached convergence
            # on this example and stop refinement.
            if counterfactual_sequence == user_trajectory:
                break

            self.logger.info(f"Starting refinement step {refinement_step + 1}")
            self.total_num_performed_refinement_steps += 1

            # Perform a refinement step
            self.llm.clear_chat_history(self.system_prompt)
            raw_preference: str = self.infer_preferences(
                refinement_step_inferred_preferences,
                user_trajectory,
                counterfactual_sequence
            )
            # Step 3. Optional break preferences down
            if self.config.do_breakdown_preferences_each_iteration and not self.config.do_breakdown_preferences_after_refinement:
                # Check if raw preference is already formatted correctly.
                new_preference_set: PreferenceSet = self.parse_json_string_into_preference_set(raw_preference)
                # If it isn't formatted correctly prompt for an explicit json list
                if new_preference_set.is_empty():
                    self.llm.clear_chat_history(self.system_prompt)
                    new_preference_set = self.breakdown_preferences(raw_preference)
                else:
                    self.logger.info(f"Successful infer parse. Skipping breakdown step.")
                    self.breakdown_skips += 1
            else:
                new_preference_set: PreferenceSet = get_preference_set(
                    task_instance.task_name,
                    raw_preference,
                    self.config
                )

            # In case there's a bad parse, don't update preferences
            if not new_preference_set.is_empty():
                refinement_step_inferred_preferences: PreferenceSet = new_preference_set
                inferred_preferences.update(refinement_step_inferred_preferences)

            # Step 4. Iterate on candidate trajectory ---> Core Contribution 1. Iterative refinement
            # Baselines "single candidate (sc)", "one step single candidate (1sc)", and "one step no candidate (1nc)"
            # skip this step
            # If we are using iterative counterfactuals, generate a new counterfactual
            # Skip if we are in the last refinement step
            if self.config.candidate_type == "iterate" and refinement_step != (self.config.num_refinement_steps - 1):
                counterfactual_sequence = self.pc_agent.solve_task(
                    task_instance,
                    preferences=refinement_step_inferred_preferences,
                    in_context_examples=self.icl_examples
                )

        # Step 5. Optional break preferences down -- user for ablations
        if self.config.do_breakdown_preferences_after_refinement:
            # Check if the refinement-inferred preference is already formatted correctly.
            # To break down the preference into components, we need to work with a string and no preference sets object
            new_preference_set = self.parse_json_string_into_preference_set(
                inferred_preferences.in_natural_language(mode="string")
            )
            # If it isn't formatted correctly prompt for an explicit json list
            if new_preference_set.is_empty():
                self.llm.clear_chat_history(self.system_prompt)
                new_preference_set = self.breakdown_preferences(inferred_preferences.in_natural_language(mode="string"))
            # Update the inferred preferences, but in case there's a bad parse, don't update preferences
            if not new_preference_set.is_empty():
                inferred_preferences = new_preference_set

        self.number_of_inferred_preferences.append(len(inferred_preferences.as_single_set()))

        # Collect validation trajectories and their indices. Include the current trajectory in this list
        validation_trajectories = [(len(self.example_retriever), task_instance, user_trajectory)]
        for idx in self.related_examples_indices:
            _, _, _, (ti, ut, _) = self.example_retriever[idx]
            validation_trajectories.append((idx, ti, ut))

        # Step 6. Validate preferences ---> Core Contribution 2. Consistent Preferences
        # Baseline "No Validation" skips this step.
        self.logger.info(f"Preferences after inference: "
                         f"\"{inferred_preferences.in_natural_language(mode='string')}\". "
                         f"Starting Validation.")
        preferences_to_remove = []
        if self.config.do_validate_preferences:
            validation_scores = {}
            # For every trajectory/preference pair
            for idx, v_task_instance, v_user_trajectory in validation_trajectories:
                for preference_str in inferred_preferences.in_natural_language(mode="list"):
                    # If we've validated this preference on this user example before, then use previously
                    # identified entailment
                    if (preference_str, idx) in self.previously_validated:
                        v_score = self.previously_validated[(preference_str, idx)]
                    # Else if this preference has already been strongly confirmed/contradicted, continue
                    elif (preference_str in validation_scores and len(validation_scores[preference_str]) >= 2 and
                          abs(np.mean(validation_scores[preference_str])) > 1.5):
                        continue
                    # Otherwise, probe llm for validation
                    else:
                        # Prompts LLM to validate a preference against a trajectory on a 5-point likert scale
                        self.llm.clear_chat_history(self.system_prompt)
                        v_score = self.validate_preference(preference_str, v_user_trajectory,
                                                           inferred_preferences.in_natural_language(mode="list"))
                        if v_score is None:
                            continue

                        self.previously_validated[(preference_str, idx)] = v_score

                    # Aggregate validation scores
                    if preference_str not in validation_scores:
                        validation_scores[preference_str] = []
                    validation_scores[preference_str].append(v_score)

            for preference, scores in validation_scores.items():
                # Step 7. If preference is good enough that it explains multiple trajectories, save it for future use
                #         If not, then this preference is not useful enough and discard it
                self.logger.info(f"Preference: {preference} -> Score: {np.mean(scores)} "
                                 f"(from {len(scores)} validations)")
                # If we have more than 1 validation and the average score is below the threshold,
                # add preference to removal list
                required_scores = 1
                # check if the scores meet the validation threshold
                if np.mean(scores) <= self.config.validation_threshold and len(scores) >= required_scores:
                    preferences_to_remove.append(preference)

            # Remove preferences
            for preference in preferences_to_remove:
                inferred_preferences.remove(preference)
                self.preferences_filtered_by_validation += 1

        # Add sample to historical user examples for future use
        self.example_retriever.add(task_instance.context, task_instance.source,
                                   (task_instance, user_trajectory, inferred_preferences))

        self.logger.info(f"Preferences after validation: "
                         f"\"{inferred_preferences.in_natural_language(mode='string')}\". "
                         f"End of sample.")

    def add_reasoning_prompts(self, prompt: str, validation: bool = False) -> str:
        """
        Adds reasoning prompts, e.g. chain-of-thought, to the existing prompts.
        Args:
            prompt: the prompt to add reasoning prompts to
            validation: If false (default), adds inferring reasoning prompts
                        If true, adds validation reasoning prompts
        Returns:
            The original prompt with reasoning prompts appended to it
        """
        if validation:
            if self.environment_prompts.other_val_reasoning:
                prompt += "\n\n"
                prompt += "While validating the preference, you should:\n"
                prompt += "\n".join(self.environment_prompts.other_val_reasoning)
        else:
            prompt += "\n\n"
            prompt += "While refining the preference set, you should:\n"
            prompt += "\n".join(self.environment_prompts.other_reasoning)
        return prompt

    def add_formatting_prompts(self, prompt: str, prompt_type: str, line_keyword: str) -> str:
        """
        Adds formatting prompts to the existing prompts. Formatting prompts are prompts that outline what we expect the
        output to look like so that we can reasonably parse it.
        Args:
            prompt: the prompt to add formatting prompts to
            prompt_type: which kind of prompt we are formatting for. Must be one of ["infer", "breakdown", "validate"]
            line_keyword: the keyword that defines the line where we are expecting our answer (e.g. "Verdict:")
        Returns:
            The original prompt with formatting prompts appended to it
        """
        prompt += "\n\n"
        prompt += self.environment_prompts.format_prompts[prompt_type](line_keyword)
        return prompt

    def infer_preferences(self, inferred_preferences: PreferenceSet, user_trajectory: Trajectory,
                          counterfactual: t.Optional[Trajectory] = None) -> str:
        """
        Main function to infer preferences from a user trajectory.
        Args:
            inferred_preferences: The currently inferred preferences
            user_trajectory: the user trajectory we are trying to infer preferences from
            counterfactual: an optional counterfactual trajectory used as comparison
        Returns:
            inferred preferences (as a string) from this generation
        """
        self.prompt_counts["infer"] += 1
        self.current_prompting = "infer"

        line_keyword = "Preferences:"

        inferred_preferences = inferred_preferences.in_natural_language(mode="string")
        infer_prompt = self.environment_prompts.refine_preferences_prompt(inferred_preferences, user_trajectory,
                                                                          counterfactual)

        infer_prompt = self.add_reasoning_prompts(infer_prompt, validation=False)
        infer_prompt = self.add_formatting_prompts(infer_prompt, "infer", line_keyword)

        raw_preference = self.llm.prompt_llm(infer_prompt)
        raw_preference = self.extract_raw_string_from_response(raw_preference, line_keyword)

        return raw_preference

    def breakdown_preferences(self, raw_inferred_preference: str) -> PreferenceSet:
        """
        Main function that breaks down preferences down into a JSON list of preferences, and then converts the list
        to PreferenceSet
        Args:
            raw_inferred_preference: Raw output string from infer step
        Returns:
            A PreferenceSet with the broken down preferences
        """
        self.prompt_counts["breakdown"] += 1
        self.current_prompting = "breakdown"

        line_keyword = "Preferences:"
        # Whether we are removing these preferences. Changes the prompt slightly.
        basis_prompt = self.environment_prompts.basis_instruction_prompt(raw_inferred_preference)
        basis_prompt = self.add_formatting_prompts(basis_prompt, "basis", line_keyword)
        basis_response = self.llm.prompt_llm(basis_prompt)
        # Extract string containing the json list
        preference_str = self.extract_raw_string_from_response(basis_response, line_keyword)
        # Create a preference set from that string
        preference_set = self.parse_json_string_into_preference_set(preference_str)
        return preference_set

    def coalesce_preferences(self, all_preferences: PreferenceSet) -> PreferenceSet:
        """
        Coalesces a set of preferences into a more condensed version of itself.
        Attempts to remove redundancy and overlap between preferences within the set
        Args:
            all_preferences: Preferences to coalesce
        Returns:
            A PreferenceSet with the broken down preferences
        """
        self.prompt_counts["coalesce"] += 1
        self.current_prompting = "coalesce"

        line_keyword = "Preferences:"

        all_preferences = all_preferences.in_natural_language(mode="json")
        coalesce_prompt = self.environment_prompts.coalesce_prompt(all_preferences)
        coalesce_prompt = self.add_formatting_prompts(coalesce_prompt, "coalesce", line_keyword)
        coalesce_response = self.llm.prompt_llm(coalesce_prompt)
        # Extract string containing the json list
        preference_str = self.extract_raw_string_from_response(coalesce_response, line_keyword)

        if self.config.do_breakdown_preferences_each_iteration:
            # Extract string containing the json list
            preference_set = self.parse_json_string_into_preference_set(preference_str)
            # If it isn't prompt for an explicit json list
            if preference_set.is_empty():
                preference_set = self.breakdown_preferences(preference_str)
            else:
                self.logger.info(f"Successful coalesce parse. Skipping breakdown in step.")
                self.breakdown_skips += 1
        else:
            preference_set = get_preference_set(self.config.task.name, preference_str, self.config)
        return preference_set

    def validate_preference(self, preference_to_validate: str, user_trajectory: Trajectory,
                            all_preferences: t.List[str]) -> int:
        """
        Given a preference in string form and a trajectory, ask the llm if the trajectory confirms, contradicts, or
        is neutral to the preference. Returns +1 for confirmation, 0 for neutral, and -1 for contradictions
        Args:
            preference_to_validate: the hypothesized preference
            user_trajectory: the trajectory to test the preference against
            all_preferences: The total set of inferred preferences to date. Optionally used as context (configurable).
        Returns:
             A score (+1/0/-1) based on how well the trajectory matches the preference
        """
        self.prompt_counts["validate"] += 1
        self.current_prompting = "validate"
        user_trajectory = user_trajectory.in_natural_language_for_validation("user")
        other_preferences = [pref for pref in all_preferences if pref != preference_to_validate]
        # Get prompt
        validation_prompt = self.environment_prompts.validation_instruction_prompt(preference_to_validate,
                                                                                   user_trajectory,
                                                                                   other_preferences)

        validation_prompt = self.add_reasoning_prompts(validation_prompt, validation=True)
        validation_prompt = self.add_formatting_prompts(validation_prompt, "validate", "Verdict:")
        # Feed prompt to LLM
        response = self.llm.prompt_llm(validation_prompt)
        # Words we expect to find
        response_keywords = ["strongly confirms", "somewhat confirms", "neutral", "somewhat contradicts",
                             "strongly contradicts"]
        keyword_extraction = self.extract_keywords_from_response(response, "verdict", response_keywords)
        keyword_extraction = keyword_extraction or "neutral"
        # Returns 1 for confirms, -1 for contradicts, 0 for neutral
        return self.validation_word_to_score[keyword_extraction]

    def extract_raw_string_from_response(self, response: str, line_keyword: str) -> str:
        """
        Given a full LLM response, extract the content on the line that is prefixed with "line_keyword".
        For example, given line keyword "Preferences:", and response:
            We reason about XYZ
            Overall we think that X indicates Y.
            Preferences: <Z>
        This would extract <Z>. If line keyword is not found anywhere, returns an empty string
        Args:
            response: the llms output
            line_keyword: the keyword indicating the line to extract
        Returns:
            The string immediately proceeding the line keyword until the end of the line, or an empty string if the line
            keyword is never found.
        """
        output_string = ""
        # Make sure keyword is lower case for comparisons
        line_keyword = line_keyword.lower()
        keyword_found = False
        # Extract line of output which contains the specified "line_keyword"
        for line in response.split("\n"):
            # The llm often outputs either quotation marks or Markdown formatting "*" around the word.
            line = line.strip('"*').lower()
            if line.startswith(line_keyword):
                keyword_found = True
                output_string = line.removeprefix(line_keyword).strip('"*: ') + "\n"
                if output_string.strip() != "":
                    break
            elif keyword_found:
                # In case it prints the preferences on the next lines, which it does sometimes, catch all text until
                # next empty line
                if line.strip() == "":
                    break
                output_string += f"{line}\n"

        # No properly formatted line found, throw warning and return empty string
        if output_string == "":
            self.prompt_errors[self.current_prompting] += 1
            self.logger.warning(f"Malformed LLM output in {self.current_prompting} stage. "
                                f"Did not find any valid line with keyword \"{line_keyword}\". "
                                f"Returning empty string.\n")
            return ""

        return output_string

    def extract_keywords_from_response(self, response: str, line_keyword: t.Union[str, None],
                                       response_keywords: t.List[str]) -> t.Union[t.Dict[str, int], str, None]:
        """
        Returns an extracted response keyword based on the llms response.
        Similar to extract_raw_string_from_response (see that docstring for more detail) except:
            1. The resulting string must match a response keyword. If so, return that response keyword
            2. If no line keyword is found the output does not match a response keyword
               then return the response keyword that occurs most frequently in the response
        Args:
            response: the llms output
            line_keyword: the keyword indicating the line to extract
            response_keywords: list of keywords, of which one must be selected from the response (or empty string).
        Returns:
            The parsed response keyword, or empty string if none can be determined
        """
        # Extract line of output where verdict is output
        lines = response.split("\n")
        keyword_scores = {k: 0 for k in response_keywords}
        # make sure keyword is lower case for comparisons
        line_keyword = line_keyword.lower()
        verdict = ""
        for line in lines:
            # find line stating the verdict, then check if the corresponding keywords exist in that line
            line = line.lower()
            for keyword in response_keywords:
                keyword_lemma = keyword.removesuffix("s")
                if keyword_lemma in line:
                    # Count keyword scores in case we don't get an exact match
                    keyword_scores[keyword] += 1
                    # Exact match found
                    if line_keyword in line:
                        verdict = keyword

        # No properly formatted line found, log warning and try keyword count method
        if verdict == "":
            self.logger.warning(f"Malformed LLM output, no line keyword found during the {self.current_prompting} "
                                f"stage. Using keyword count method instead.\n")

            # find most common keyword
            most_common_keyword, most_common_keyword_count = None, 0
            for keyword in response_keywords:
                if keyword == "neutral":
                    continue
                if keyword_scores[keyword] > most_common_keyword_count:
                    most_common_keyword, most_common_keyword_count = keyword, keyword_scores[keyword]
                # if two keywords are tied for highest count, set most common keyword back to None
                elif keyword_scores[keyword] == most_common_keyword_count:
                    most_common_keyword, most_common_keyword_count = None, keyword_scores[keyword]

            verdict = most_common_keyword

            # i.e. if not a single keyword was found, log warning and return None
            if verdict is None:
                self.prompt_errors[self.current_prompting] += 1
                self.logger.warning(f"Malformed LLM output. Keyword count method failed as well. "
                                    f"Returning empty string\n")
                return ""

        return verdict

    def parse_json_string_into_preference_set(self, json_str: str) -> PreferenceSet:
        """
        Parse string formatted as json list containing strings.
        Empty strings will return empty preference sets.
        WARNING: Parsing errors (from the json string or inner strings) are caught and do not interrupt flow.
                 Instead, warnings are logged
        Args:
            json_str: string to parse into the list
        Returns:
            A preference set containing the preferences held in the json string
        """
        # Special case. If string is empty, return empty preference set
        if json_str == "" or json_str is None:
            return get_empty_preference_set(self.config.task.name, self.config)

        # Load list into python and test format. Expects `broken_down_preferences` to be in a valid json list format.
        # If an invalid json format is found, throw warning and return empty set
        try:
            new_preferences = json.loads(json_str)
        except json.decoder.JSONDecodeError:
            self.prompt_errors[self.current_prompting] += 1
            self.logger.warning(f"Malformed json string output in breakdown_preference_into_basis. "
                                f"Returning empty preference set.\n")
            return get_empty_preference_set(self.config.task.name, self.config)
        
        return WritingPreferenceSet(new_preferences, self.config)

    def get_metrics(self) -> t.Dict[str, t.Union[int, float]]:
        """
        Calculates and returns metrics related to the LLM
        Returns:
            Returns failure counts and total counts for the inference step, breakdown step, and validation step.
        """
        llm_metrics = {
            **self.llm.get_metrics(),
            **self.pc_agent.get_metrics(),
            **self.example_retriever.get_metrics(),
            "avg_refinement_steps": self.total_num_performed_refinement_steps / self.total_seen_examples,
            "total_filtered_by_val": self.preferences_filtered_by_validation,
            "num_inferred_preferences": self.number_of_inferred_preferences[-1],
        }

        return llm_metrics
