#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import yaml

from preference_inferrer.common.configurations import configure_config
from preference_inferrer.preference_inferring_agents.prose import Prose
from preference_inferrer.preference_sets.writing_preference_set import WritingPreferenceSet
from preference_inferrer.tasks import get_task


def main():
    """Main"""
    with open("child_config.yaml", 'r') as file:
        config = yaml.safe_load(file)

    task_name = "summarization"

    config["parameters"]["logging_level"] = "info"
    config["parameters"]["user"]["llm_name"] = "test_llm"
    config["parameters"]["agent"]["llm_name"] = "test_llm"
    config["parameters"]["agent"]["type"] = "prose"
    config["parameters"]["candidate_type"] = "iterate"
    config["parameters"]["include_preferences_in_validate"] = False
    config["parameters"]["task"]["name"] = task_name
    config["parameters"]["task"]["framework"] = "plume"

    config = configure_config(config["parameters"])
    inferrer = Prose(config)

    if task_name == "summarization":
        agent_preferences = WritingPreferenceSet("use emojis", config)
    elif task_name == "email_writing":
        agent_preferences = WritingPreferenceSet("write in bullet points", config)
    else:
        raise ValueError(f"Unknown task {task_name}")

    system_prompt = inferrer.environment_prompts.system_prompt

    inferrer.llm.clear_chat_history(system_prompt)

    new_task = get_task(config.task.name, config)
    for task_instance in new_task.get_task_instances():
        break

    # user_preferences = inferrer.pc_agent.get_true_preferences(task_instance)
    # print("+++++", user_preferences.in_natural_language(single_string=True), "+++++")

    user_example = inferrer.pc_agent.solve_task_as_user(task_instance)
    agent_example = inferrer.pc_agent.solve_task(task_instance, preferences=agent_preferences)

    counterfactual = None if config.candidate_type == "none" else agent_example

    print("-------------------------------------------")
    print("---------------- INFERENCE ----------------")
    print("-------------------------------------------")
    raw_preference_str = inferrer.infer_preferences(agent_preferences, user_example, counterfactual)
    print("-------------------------------------------")
    print("---------------- BREAKDOWN ----------------")
    print("-------------------------------------------")
    inferrer.breakdown_preferences(raw_preference_str)
    print("-------------------------------------------")
    print("---------------- VALIDATE  ----------------")
    print("-------------------------------------------")

    inferrer.validate_preference(agent_preferences.in_natural_language(mode="string"), agent_example, [])


if __name__ == "__main__":
    main()
