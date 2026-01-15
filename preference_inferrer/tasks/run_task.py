#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

"""
Run an experiment
"""

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import wandb

from preference_inferrer.common.configurations import configure_config
from preference_inferrer.common.personal_keys import WANDB_ENTITY, WANDB_KEY, WANDB_HOST
from preference_inferrer.common.utils import set_seed_everywhere
from preference_inferrer.preference_conditioned_agents import get_conditioned_agent
from preference_inferrer.preference_inferring_agents import get_inferring_agent
from preference_inferrer.tasks import get_task

RUN_NOTES = "validate using costs"


def run(config):
    """
    Run an experiment
    Args:
        config: configurations that define experiment parameters
    """
    if not WANDB_KEY == "" and WANDB_KEY is not None:
        # Setup wandb and parameters we want to log
        wandb.login(key=WANDB_KEY, host=WANDB_HOST, verify=True)
        wandb.init(project="preference_inferrer_final", config=config, entity=WANDB_ENTITY, reinit=True,
                   notes=RUN_NOTES, mode="online", name=config["exp_name"], group=config["group_name"])

        # Setup configurations
        wandb.config = config
    config = configure_config(config)
    # Set random seed
    set_seed_everywhere(config.seed)
    # Set up logger
    logger = logging.getLogger("MAIN")
    logger.setLevel(level=config.logging_level)
    # Initialize a default dict to accumulate metrics
    metrics_accumulator = defaultdict(list)
    # Create an empty DataFrame with the specified columns
    _, agent_name, llm_name, variation_name = config.exp_name.split(".")
    agent_name = f"{agent_name}.{variation_name}"
    framework, task_name, seed = config.group_name.split(".")

    columns = ["framework", "task_name", "agent_name", "llm_name", "seed",  # Experiment set up
                "task_source", "example_num", "source_example_num",  # flags/labels
                "accuracy", "bertscore",  # preference metrics
                "ldist", "n_ldist", "fpm", "ppcm",  # action metrics
                "num_refinement_steps",  # number of counterfactual refinement steps
                "validation_threshold",  # average validation score threshold required to keep preference
                "preference_string_length",  # number of characters used to describe the user's preferences
                "preference_llm_judge_similarity",  # similarity between the predicted and the true preference using LLM-as-a-Judge
                ]
    df = pd.DataFrame(columns=columns)

    # Create DataFrame to store PROSE outputs for competitor methods
    prose_output_columns = ["example_num", "task_source", "source_example_num",
                            "task_prompt", "inferred_preferences", "user_demo",
                            "agent_output", "true_preferences"]
    prose_outputs_df = pd.DataFrame(columns=prose_output_columns)

    # Create DataFrame to store data for each turn for steering vector comparison
    turn_data_columns = ["example_num", "turn", "task_source", "source_example_num",
                         "task_prompt", "user_preferences", "user_demo"]
    turn_data_df = pd.DataFrame(columns=turn_data_columns)

    task = get_task(config.task.name, config)
    agent = get_inferring_agent(config.agent.name, config)
    agent.set_task(task)
    user = get_conditioned_agent(config.task.name, "USER", config)

    example_idx = 0
    for task_instance in task.get_task_instances():
        logger.info(f"STARTING EXAMPLE: {example_idx}")
        true_preferences = user.get_true_preferences(task_instance)
        if agent.can_cheat:
            agent.cheat(true_preferences)
        # Get agent to complete task
        agent_trajectory, pred_preferences = agent.complete(task_instance)
        task_instance.agent_completion = agent_trajectory.completed_task
        # Get user to complete task
        user_trajectory = user.solve_task_as_user(task_instance)
        task_instance.user_completion = user_trajectory.completed_task

        # Store data before learning (for competitor methods)
        prose_output_row = {
            "example_num": example_idx,
            "task_source": task_instance.source,
            "source_example_num": example_idx % config.task.number_of_examples_per_preference_set,
            "task_prompt": task_instance.context,
            "inferred_preferences": pred_preferences.in_natural_language(mode="string") if hasattr(pred_preferences, 'in_natural_language') else str(pred_preferences),
            "user_demo": user_trajectory.completed_task,
            "agent_output": agent_trajectory.completed_task,
            "true_preferences": true_preferences.in_natural_language(mode="string") if hasattr(true_preferences, 'in_natural_language') else str(true_preferences)
        }
        prose_outputs_df.loc[len(prose_outputs_df)] = prose_output_row

        # Store turn data for steering vector comparison (each turn within a source)
        turn_num = example_idx % config.task.number_of_examples_per_preference_set
        turn_data_row = {
            "example_num": example_idx,
            "turn": turn_num,
            "task_source": task_instance.source,
            "source_example_num": example_idx % config.task.number_of_examples_per_preference_set,
            "task_prompt": task_instance.context,
            "user_preferences": true_preferences.in_natural_language(mode="string") if hasattr(true_preferences, 'in_natural_language') else str(true_preferences),
            "user_demo": user_trajectory.completed_task
        }
        turn_data_df.loc[len(turn_data_df)] = turn_data_row

        # Have the agent learn from user's completion of task
        agent.learn(task_instance, agent_trajectory, user_trajectory)
        # Get metrics on how well the agent performed
        metrics = task.get_metrics(task_instance, pred_preferences, true_preferences)
        metrics.update(agent.get_metrics())
        metrics.update(user.get_metrics())

        for metric, value in metrics.items():
            metrics_accumulator[metric].append(value)
        metrics["example_idx"] = example_idx
        metrics["source_idx"] = task_instance.source_idx

        row = [
            framework, task_name, agent_name, llm_name, seed,
            task_instance.source, example_idx, example_idx % config.task.number_of_examples_per_preference_set,
            metrics["preference_acc"], metrics["preference_f1"],
            metrics["writing_ldist"], metrics["writing_n_ldist"], metrics["writing_fpm"], metrics["writing_ppm"],
            config.num_refinement_steps, config.validation_threshold,
            metrics["preference_string_length"], metrics["preference_llm_judge_similarity"]
        ]

        df.loc[len(df)] = row

        example_idx += 1

        if not WANDB_KEY == "" and WANDB_KEY is not None:
            wandb.log(metrics)

    # Calculate the average for each metric
    metric_averages = {
        **{f"avg_{metric}": np.mean(values) for metric, values in metrics_accumulator.items()
           if ("total" not in metric or "tokens" not in metric)},
        **{f"std_{metric}": np.std(values) for metric, values in metrics_accumulator.items()
           if metric in ['jaccard', 'f1']}
    }

    # Output the averages
    logger.info(f"Averages:\n")
    for metric, average in metric_averages.items():
        logger.info(f"{metric}: {average}")

    if not WANDB_KEY == "" and WANDB_KEY is not None:
        wandb.log(metric_averages)

    df.to_csv(f"{framework}-{task_name}-{agent_name}-{llm_name}-{seed}-inferring_results.csv")
    # Save PROSE outputs for competitor methods
    prose_outputs_df.to_csv(f"{framework}-{task_name}-{agent_name}-{llm_name}-{seed}-prose_outputs.csv", index=False, quoting=1)
    logger.info(f"Saved PROSE outputs to {framework}-{task_name}-{agent_name}-{llm_name}-{seed}-prose_outputs.csv")
    # Save turn-by-turn data for steering vector comparison
    turn_data_df.to_csv(f"{framework}-{task_name}-{agent_name}-{llm_name}-{seed}-turn_data.csv", index=False, quoting=1)
    logger.info(f"Saved turn data to {framework}-{task_name}-{agent_name}-{llm_name}-{seed}-turn_data.csv")
    if framework == "plume":
        qa_df = task.get_llm_judge_question_and_answer_df()
        qa_df.to_csv(f"{framework}-{task_name}-{agent_name}-{llm_name}-{seed}-llm_judge_qa.csv", quoting=1)
