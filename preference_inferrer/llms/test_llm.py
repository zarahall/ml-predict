#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

"""
A test LLM that doesn't actually use an LLM. Allows for testing other components of the system on local machines
"""
import typing as t
from types import SimpleNamespace

from preference_inferrer.llms.base_llm import BaseLLM


class TestLLM(BaseLLM):
    def __init__(self, model_name: str, logger_name: str, config: SimpleNamespace):
        """
        Args:
            model_name: specific model name. For test llm, this should be anything with the word `test` in it
            logger_name: Name to provide the logger. Used to differentiate the output of the user's and agent's LLMs.
            config: configuration namespace
        """
        super().__init__(model_name, logger_name, config)

    def prompt_llm(self, user_prompt: str, temperature: t.Optional[float] = None) -> str:
        """
        Since this is a static 'llm', just try and reply with something that will match parsing requirements
        which depend on the input.
        Args:
            user_prompt: the user's prompt
            temperature: Unused for test_llm
        Returns:
            A sample response
        """
        self.add_to_chat_history("user", user_prompt)

        self.chat_log.append(
            {"role": "user", "content": user_prompt}
        )
        user_prompt = user_prompt.lower()

        if "json" in user_prompt:
            response = "Preferences: [\"<preference 1>\", \"...\", \"<preference n>\"]"
        elif "verdict:" in user_prompt:
            response = "Verdict: <verdict>"
        elif "refine" in user_prompt:
            response = "Preferences: <preferences>"
        else:
            response = "<user output>"
        # Record chat history
        self.chat_log.append(
            {"role": "assistant", "content": response}
        )

        self.add_to_chat_history("assistant", response)
        return response
