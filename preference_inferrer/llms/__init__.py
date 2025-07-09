#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from types import SimpleNamespace

from preference_inferrer.llms.base_llm import BaseLLM
from preference_inferrer.llms.qwen2_5 import Qwen2_5
from preference_inferrer.llms.openai_gpt import OpenAIGPT
from preference_inferrer.llms.test_llm import TestLLM


def get_llm(llm_name: str, logger_name: str, config: SimpleNamespace) -> BaseLLM:
    """
    Creates agent for the provided environment using the config
    Args:
        llm_name: Name of the llm to load
        logger_name: Name to provide the logger. Used to differentiate the output of the user's and agent's LLMs.
        config: Configurations
    Returns:
        The created agent
    """
    if "gpt" in llm_name:
        llm = OpenAIGPT(llm_name, logger_name, config)
    elif "qwen" in llm_name:
        llm = Qwen2_5(llm_name, logger_name, config)
    elif "test" in llm_name:
        llm = TestLLM(llm_name, logger_name, config)
    else:
        raise ValueError(f"Unknown LLM name: {llm_name}")
    return llm
