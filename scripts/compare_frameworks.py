#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

"""
Uses a single preference agent to compare the PRELUDE and PLUME Frameworks.
Single preference agents all have the same preference accuracy, and should therefore have similar scores.
The framework with the lower variance is considered to be a better evaluation
"""
import typing as t
import itertools as it
import logging
from collections import defaultdict

import pandas as pd
import scipy.stats as stats
import wandb

from preference_inferrer.common.configurations import configure_config
from preference_inferrer.common.personal_keys import WANDB_ENTITY, WANDB_KEY, WANDB_HOST
from preference_inferrer.common.utils import set_seed_everywhere
from preference_inferrer.preference_conditioned_agents import get_conditioned_agent
from preference_inferrer.preference_inferring_agents import SinglePreferenceAgent
from preference_inferrer.preference_sets import WritingPreferenceSet
from preference_inferrer.tasks import get_task

RUN_NOTES = "plume vs prelude testing"


def powerset(iterable) -> t.Iterator[t.Tuple]:
    """powerset([1,2,3]) → () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return it.chain.from_iterable(it.combinations(s, r) for r in range(len(s) + 1))


def compare_prelude_and_plume(config) -> None:
    """
    Script that compares how well prelude or plume evaluate preference inferring agents.
    Specifically, it generates the powerset of from the preference components of true preferences
    and tests how well each does at solving the task. Intuitively, as we increase the number of preference components,
    we should see performance increase. We measure this by calculating the correlation between the preference quality
    and the writing quality on a variety of metrics.
    """

    columns = ["framework", "task_name", "agent_name", "llm_name", "seed",  # Experiment set up
               "task_source", "example_num", "source_example_num",  # flags/labels
               "accuracy", "bertscore", "preference_string_length", "preference_llm_judge_similarity",  # preference metrics
               "ldist", "n_ldist", "fpm", "ppcm"]  # action metrics
    df = pd.DataFrame(columns=columns)

    wandb.login(key=WANDB_KEY, host=WANDB_HOST, verify=True)
    # Setup wandb and parameters we want to log
    wandb.init(project="plume_vs_prelude", config=config, entity=WANDB_ENTITY, reinit=True,
               notes=RUN_NOTES, mode="online", name=config["exp_name"],
               group=config["group_name"])

    # Setup configurations
    config = configure_config(config)

    framework = config.task.framework
    task_name = config.task.name
    agent_name = "single_preference_agent"
    llm_name = config.user.llm_name
    seed = config.seed

    # Set up logger
    logger = logging.getLogger("MAIN")
    logger.setLevel(level=config.logging_level)

    # Set random seed
    set_seed_everywhere(config.seed)

    task = get_task(config.task.name, config)
    user = get_conditioned_agent(config.task.name, "USER", config)

    # Initialize a default dict to accumulate metrics
    preference_metrics = defaultdict(list)
    writing_metrics = defaultdict(list)
    metric_accumulator = defaultdict(list)

    agent_generated_tokens = 0
    agent_prompt_tokens = 0

    example_num = 0
    step = 0

    for task_instance in task.get_task_instances():
        print(f"Task Instance {example_num}. Source: {task_instance.source}")
        example_num += 1

        true_preferences = user.get_true_preferences(task_instance)

        # Solve this task using every subset of preferences
        all_preference_subsets = powerset(true_preferences.as_single_set())
        for preference_subset in all_preference_subsets:
            print(f"Step: {step}. Current Subset: {preference_subset}. "
                  f"True preference: {true_preferences.as_single_set()}")
            step += 1

            agent = SinglePreferenceAgent(WritingPreferenceSet(preference_subset, config), config)

            # Get subset agent to complete task
            agent_trajectory, pred_preferences = agent.complete(task_instance)
            task_instance.agent_completion = agent_trajectory.completed_task
            # Get user to complete task
            user_trajectory = user.solve_task_as_user(task_instance)
            task_instance.user_completion = user_trajectory.completed_task
            # Have the agent learn from user's completion of task
            agent.learn(task_instance, agent_trajectory, user_trajectory)
            # Get metrics on how well the agent performed
            metrics = task.get_metrics(task_instance, pred_preferences, true_preferences)
            metrics.update(user.get_metrics())

            # We accumulate here instead of the more typical agent.get_metrics() because we are creating a
            # new agent frequently, and we want the total across them
            agent_generated_tokens += agent.pc_agent.llm.total_generated_tokens
            agent_prompt_tokens += agent.pc_agent.llm.total_prompt_tokens
            metrics["cf_agent_generated_tokens"] = agent_generated_tokens
            metrics["cf_agent_prompt_tokens"] = agent_prompt_tokens
            metrics["percent_of_matching_components"] = len(preference_subset) / len(true_preferences)

            for metric, value in metrics.items():
                metric_accumulator[metric].append(value)
                if "writing" in metric:
                    writing_metrics[metric].append(value)
                elif "preference" in metric:
                    preference_metrics[metric].append(value)

            wandb.log(metrics)

            row = [
                framework, task_name, "single_preference_agent", llm_name, seed,
                task_instance.source, example_num, -1,
                metrics["preference_acc"], metrics["preference_f1"], metrics["preference_string_length"], metrics["preference_llm_judge_similarity"],
                metrics["writing_ldist"], metrics["writing_n_ldist"], metrics["writing_fpm"], metrics["writing_ppm"]
            ]

            df.loc[len(df)] = row

    correlations = []
    first_loop = True
    for preference_metric, preference_values in preference_metrics.items():
        print(preference_metric, ":", preference_values)
        for writing_metric, writing_values in writing_metrics.items():
            if first_loop:
                print(writing_metric, writing_values)
            # Calculate Pearson correlation
            pearson_r, _ = stats.pearsonr(preference_values, writing_values)
            # Calculate Spearman correlation
            spearman_r, _ = stats.spearmanr(preference_values, writing_values)
            # Calculate Kendall's Tau correlation
            kendall_t, _ = stats.kendalltau(preference_values, writing_values)
            correlations.append(
                (preference_metric, writing_metric, pearson_r, spearman_r, kendall_t)
            )
        first_loop = False

    print("-" * 50)
    print(f"EXP NAME: {config.exp_name}")
    corrs_dict = {}
    # Create a table for the aggregated metrics
    sorted_corrs = sorted(correlations, key=lambda x: abs(x[2]))
    for preference_metric, writing_metric, pearson_r, spearman_r, kendall_t in sorted_corrs:
        print(f"Correlation between {preference_metric:25} and {writing_metric:25} "
              f"(pearson, spearman, kendall): "
              f"{pearson_r:5.3f}, {spearman_r:5.3f}, {kendall_t:5.3f}")

        if "match" in writing_metric and ("f1" in preference_metric or "n_cost" in preference_metric):
            corrs_dict[f"{preference_metric}_{writing_metric}_pearson_r"] = pearson_r

    wandb.log(corrs_dict)

    df.to_csv(f"{framework}-{task_name}-{agent_name}-{llm_name}-{seed}-compare_frameworks.csv")


def main():
    """Main"""
    import yaml

    with open("child_config.yaml", 'r') as file:
        config = yaml.safe_load(file)["parameters"]
    compare_prelude_and_plume(config)


if __name__ == "__main__":
    main()
