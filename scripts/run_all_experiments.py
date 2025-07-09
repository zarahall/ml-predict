#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

"""
Script to launch multiple repeats of the same task with different configurations.
"""

import copy
import typing as t
import yaml
from copy import deepcopy

from preference_inferrer.tasks import run_task

# Base config for the experiment
with open("experiment_config.yaml", 'r') as file:
    experiment_config = yaml.safe_load(file)


def _launch_task(config: t.Dict):
    """
    Run an experiment with the given config
    Args:
        config: the specifications of how to run the experiment
    """
    experiment_config = deepcopy(config)
    run_task.run(experiment_config)


def run_main_ablations(
        task_name: str, llm_name: str = None, seed: int = None,
        default_num_refinement_steps: int = 4,
        default_validation_threshold: float = 0.25,
        prose_breakdown_variation_to_hparams: t.Optional[t.Mapping[str, t.Mapping[str, t.Union[float, int]]]] = None,
        ) -> None:
    """
    Runs tasks to evaluate various preference inferring algorithms
    Args:
        task_name: the task to run ablations on
        llm_name: name of the llm to use for the inferring agent (if none, default to child config setting
        seed: random seed to use
        default_num_refinement_steps: the number of refinement steps to use if method specific values aren't provided
        default_validation_threshold: the validation threshold to use if method specific values aren't provided
        prose_breakdown_variation_to_hparams: mapping from method name to number of refinement steps and validation
                                              threshold
    """
    if llm_name is not None:
        experiment_config["parameters"]["agent"]["llm_name"] = llm_name
    if seed is not None:
        experiment_config["parameters"]["seed"] = seed

    framework = "plume"

    agents = ["prose", "cipherN", "cipher1", "icl", "oracle_preference", "no_learning"]

    for agent_name in agents:
        variations = {
            # MAIN PAPER Runs
            # To run a subset, just comment out the runs you do not want
            "1sc_ap_nv": {
                "candidate_type": "single",
                "num_refinement_steps": 1,
                "do_breakdown_preferences_each_iteration": False,
                "do_breakdown_preferences_after_refinement": True,
                "do_validate_preferences": False,
            },
            "1nc": {
                "candidate_type": "none",
                "num_refinement_steps": 1,
                "validation_threshold": (
                    default_num_refinement_steps if prose_breakdown_variation_to_hparams is None
                    else prose_breakdown_variation_to_hparams["full_after_refinement"]["validation_threshold"]
                ),
                "do_breakdown_preferences_each_iteration": False,
                "do_breakdown_preferences_after_refinement": True,
                "do_validate_preferences": True,
            },
            "1sc": {
                "candidate_type": "single",
                "num_refinement_steps": 1,
                "validation_threshold": (
                    default_num_refinement_steps if prose_breakdown_variation_to_hparams is None
                    else prose_breakdown_variation_to_hparams["full_after_refinement"]["validation_threshold"]
                ),
                "do_breakdown_preferences_each_iteration": False,
                "do_breakdown_preferences_after_refinement": True,
                "do_validate_preferences": True,
            },
            "sc": {
                "candidate_type": "single",
                "num_refinement_steps": (
                    default_num_refinement_steps if prose_breakdown_variation_to_hparams is None
                    else prose_breakdown_variation_to_hparams["full_after_refinement"]["iterative_steps"]
                ),
                "validation_threshold": (
                    default_validation_threshold if prose_breakdown_variation_to_hparams is None
                    else prose_breakdown_variation_to_hparams["full_after_refinement"]["validation_threshold"]
                ),
                "do_breakdown_preferences_each_iteration": False,
                "do_breakdown_preferences_after_refinement": True,
                "do_validate_preferences": True,
            },
            "nv": {
                "candidate_type": "iterate",
                "num_refinement_steps": (
                    default_num_refinement_steps if prose_breakdown_variation_to_hparams is None
                    else prose_breakdown_variation_to_hparams["full_after_refinement"]["iterative_steps"]
                ),
                "validation_threshold": (
                    default_validation_threshold if prose_breakdown_variation_to_hparams is None
                    else prose_breakdown_variation_to_hparams["full_after_refinement"]["validation_threshold"]
                ),
                "do_breakdown_preferences_each_iteration": False,
                "do_breakdown_preferences_after_refinement": True,
                "do_validate_preferences": False,
            },
            "full_after_refinement": {
                "candidate_type": "iterate",
                "num_refinement_steps": (
                    default_num_refinement_steps if prose_breakdown_variation_to_hparams is None
                    else prose_breakdown_variation_to_hparams["full_after_refinement"]["iterative_steps"]
                ),
                "validation_threshold": (
                    default_validation_threshold if prose_breakdown_variation_to_hparams is None
                    else prose_breakdown_variation_to_hparams["full_after_refinement"]["validation_threshold"]
                ),
                "do_breakdown_preferences_each_iteration": False,
                "do_breakdown_preferences_after_refinement": True,
                "do_validate_preferences": True,
            },
            "full_after_refinement_icl": {
                "candidate_type": "iterate",
                "num_refinement_steps": (
                    default_num_refinement_steps if prose_breakdown_variation_to_hparams is None
                    else prose_breakdown_variation_to_hparams["full_after_refinement"]["iterative_steps"]
                ),
                "validation_threshold": (
                    default_validation_threshold if prose_breakdown_variation_to_hparams is None
                    else prose_breakdown_variation_to_hparams["full_after_refinement"]["validation_threshold"]
                ),
                "do_breakdown_preferences_each_iteration": False,
                "do_breakdown_preferences_after_refinement": True,
                "do_validate_preferences": True,
                "use_icl": True,
            },

        } if agent_name == "prose" else {"base": {}}

        for variation_name, variation_parameters in variations.items():
            config = copy.deepcopy(experiment_config)

            config["parameters"].update(variation_parameters)
            config["parameters"]["task"]["name"] = task_name
            config["parameters"]["task"]["framework"] = framework
            config["parameters"]["agent"]["name"] = agent_name
            llm_name = config["parameters"]["agent"]["llm_name"]
            config["parameters"]["user"]["llm_name"] = "gpt-4o"

            # WANDB exp & group names
            tn = {"email_writing": "em", "summarization": "sm"}[task_name]
            exp_name = f"{tn}.{agent_name}.{llm_name}.{variation_name}"
            group_name = f"{framework}.{task_name}.{config['parameters']['seed']}"

            config["parameters"]["include_preferences_in_validate"] = False

            config["name"] = exp_name
            config["parameters"]["exp_name"] = exp_name
            config["parameters"]["group_name"] = group_name
            _launch_task(config)


def run_compare_frameworks_metric_correlations(llm_name: str = None, seed: int = None):
    """
    Runs tasks to measure the correlation between preference quality and action quality
    Args:
        llm_name: name of the llm to use for the user and judge agent (if none, default to child config setting
        seed: random seed to use
    """
    if seed is not None:
        experiment_config["parameters"]["seed"] = seed

    if llm_name is not None:
        experiment_config["parameters"]["agent"]["llm_name"] = llm_name
        experiment_config["parameters"]["user"]["llm_name"] = llm_name

    for framework in ["prelude", "prelude.no_edit", "plume"]:
        for task_name in ["summarization", "email_writing"]:
            config = copy.deepcopy(experiment_config)

            config["parameters"]["logging_level"] = "info"
            config["parameters"]["task"]["framework"] = framework
            config["parameters"]["task"]["name"] = task_name
            config["parameters"]["task"]["number_of_examples_per_preference_set"] = 1

            config["parameters"]["group_name"] = "plume_vs_prelude.final"
            config["parameters"]["exp_name"] = f"{framework}.{task_name}.{llm_name}"

            config["name"] = config["parameters"]["exp_name"]
            _launch_task(config)


def run_compare_llms(task_name: str, seed: int) -> None:
    """
    Runs tasks to evaluate how different LLMs perform
    Args:
        task_name: name of task to compare llms on
        seed: random seed to use
    """
    if seed is not None:
        experiment_config["parameters"]["seed"] = seed

    framework = "plume"

    agents = ["prose", "cipher1", "icl", "oracle_preference", "no_learning"]

    for agent_name in agents:
        variations = {
            # MAIN PAPER Runs
            "gpt-4o": {
                "agent": {"llm_name": "gpt-4o", agent_name: agent_name},
            },
            "gpt-4o-mini": {
                "agent": {"llm_name": "gpt-4o-mini", "name": agent_name},
            },
            "qwen-2_5-7b": {
                "agent": {"llm_name": "qwen-2_5-7b", "name": agent_name},
            },
            "qwen-2_5-72b": {
                "agent": {"llm_name": "qwen-2_5-72b", "name": agent_name},
            }
        }

        for variation_name, variation_parameters in variations.items():
            config = copy.deepcopy(experiment_config)

            config["parameters"].update(variation_parameters)
            config["parameters"]["task"]["name"] = task_name
            config["parameters"]["task"]["framework"] = framework
            llm_name = config["parameters"]["agent"]["llm_name"]
            config["parameters"]["user"]["llm_name"] = "gpt-4o"

            # WANDB exp & group names
            tn = {"email_writing": "em", "summarization": "sm"}[task_name]
            exp_name = f"{tn}.{agent_name}.{llm_name}.{variation_name}"
            group_name = f"{framework}.{task_name}.{config['parameters']['seed']}"

            config["parameters"]["include_preferences_in_validate"] = False

            config["name"] = exp_name
            config["parameters"]["exp_name"] = exp_name
            config["parameters"]["group_name"] = group_name
            _launch_task(config)


def run_main_models_on_prelude(task_name: str, llm_name: str = None, seed: int = None) -> None:
    """
    Runs tasks to evaluate various preference inferring algorithms
    Args:
        task_name: the task to run ablations on
        llm_name: name of the llm to use for the inferring agent (if none, default to child config setting
        seed: random seed to use
    """
    if llm_name is not None:
        experiment_config["parameters"]["agent"]["llm_name"] = llm_name
    if seed is not None:
        experiment_config["parameters"]["seed"] = seed

    framework = "prose"

    agents = ["prose", "cipherN", "cipher1", "icl", "oracle_preference", "no_learning"]

    for agent_name in agents:
        config = copy.deepcopy(experiment_config)

        config["parameters"]["task"]["name"] = task_name
        config["parameters"]["task"]["framework"] = framework
        config["parameters"]["agent"]["name"] = agent_name
        llm_name = config["parameters"]["agent"]["llm_name"]
        config["parameters"]["user"]["llm_name"] = "gpt-4o"

        # WANDB exp & group names
        tn = {"email_writing": "em", "summarization": "sm"}[task_name]
        exp_name = f"{tn}.{agent_name}.{llm_name}.base"
        group_name = f"{framework}.{task_name}.{config['parameters']['seed']}"

        config["parameters"]["include_preferences_in_validate"] = False

        config["name"] = exp_name
        config["parameters"]["exp_name"] = exp_name
        config["parameters"]["group_name"] = group_name
        _launch_task(config)


def run_compare_example_retriever_methods(task_name: str, llm_name: str = None, seed: int = None) -> None:
    """
    Runs tasks to evaluate how different example retrieval methods perform
    Args:
        task_name: name of task to run
        llm_name: name of the llm to use for the inferring agent (if none, default to child config setting
        seed: random seed to use
    """
    if llm_name is not None:
        experiment_config["parameters"]["agent"]["llm_name"] = llm_name
    if seed is not None:
        experiment_config["parameters"]["seed"] = seed
    assert task_name in ["email_writing", "summarization"]
    framework = "plume"

    agent_name = "prose"

    variations = {
        # MAIN PAPER Runs
        "plume_em": {
            "retrieval_mode": "exact_match"  ## FOR EXACT MATCH, TAKE RESULTS FROM MAIN ABLATIONS
        },
        "plume_cs": {
            "retrieval_underlying_model": "roberta-large",
            "retrieval_mode": "cosine_sim"
        },
        "plume_bs": {
            "retrieval_underlying_model": "roberta-large",
            "retrieval_mode": "bertscore"
        },
        "plume_cs_mpnet": {
            "retrieval_underlying_model": "mpnet",
            "retrieval_mode": "cosine_sim"
        },
    }

    for variation_name, variation_parameters in variations.items():
        config = copy.deepcopy(experiment_config)

        config["parameters"].update(variation_parameters)
        config["parameters"]["task"]["name"] = task_name
        config["parameters"]["task"]["framework"] = framework
        llm_name = config["parameters"]["agent"]["llm_name"]
        config["parameters"]["user"]["llm_name"] = "gpt-4o"

        # WANDB exp & group names
        tn = {"email_writing": "em", "summarization": "sm"}[task_name]
        exp_name = f"{tn}.{agent_name}.{llm_name}.{variation_name}"
        group_name = f"{framework}.{task_name}.{config['parameters']['seed']}"

        config["parameters"]["include_preferences_in_validate"] = False

        config["name"] = exp_name
        config["parameters"]["exp_name"] = exp_name
        config["parameters"]["group_name"] = group_name
        _launch_task(config)


if __name__ == "__main__":
    EXPERIMENT_NAME = "LLM_SWEEP"
    LLM_NAME = "gpt-4o"
    for seed in [6936, 2045, 23456, 34567, 45678]: 
        if EXPERIMENT_NAME == "ABLATIONS_AND_BASELINES_SWEEP":
            # TABLE 2 - Main result table
            llm_to_iterative_refinement_steps_validation_threshold_by_method: t.Mapping[str, t.Mapping] = {
                "gpt-4o": {
                    "full_after_refinement": {"iterative_steps": 5, "validation_threshold": 0.5}
                },
                "gpt-4o-mini": {
                    "full_after_refinement": {"iterative_steps": 3, "validation_threshold": 0.5}
                },
                "qwen-2_5-7b": {
                    "full_after_refinement": {"iterative_steps": 5, "validation_threshold": 0.25},
                },
                "qwen-2_5-72b": {
                    "full_after_refinement": {"iterative_steps": 5, "validation_threshold": 0.75},
                }
            }
            run_main_ablations("email_writing", LLM_NAME, seed, prose_breakdown_variation_to_hparams=llm_to_iterative_refinement_steps_validation_threshold_by_method[LLM_NAME])
            run_main_ablations("summarization", LLM_NAME, seed, prose_breakdown_variation_to_hparams=llm_to_iterative_refinement_steps_validation_threshold_by_method[LLM_NAME])
        elif EXPERIMENT_NAME == "METRIC_CORRELATION":
            # TABLE 1 - Compare correlation of metrics across prelude, prelude.no_edit, and plume
            run_compare_frameworks_metric_correlations(LLM_NAME, seed)
        elif EXPERIMENT_NAME == "LLM_SWEEP":
            # Compare how good different LLMs are
            run_compare_llms("email_writing", seed)
            run_compare_llms("summarization", seed)
        elif EXPERIMENT_NAME == "PRELUDE":
            # Run main models on the prelude framework
            run_main_models_on_prelude("email_writing", LLM_NAME, seed)
            run_main_models_on_prelude("summarization", LLM_NAME, seed)
        elif EXPERIMENT_NAME == "EXAMPLE_RETRIEVER":
            # Compares to when agents do not have access to ground truth contexts
            run_compare_example_retriever_methods("email_writing", LLM_NAME, seed)
            run_compare_example_retriever_methods("summarization", LLM_NAME, seed)
