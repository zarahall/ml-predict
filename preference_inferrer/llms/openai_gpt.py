"""
Copyright (C) 2025 Apple Inc. All rights reserved.
"""
import logging
import time
import typing as t
from types import SimpleNamespace

import openai
from openai.types import Completion

from preference_inferrer.common.personal_keys import OPEN_AI_KEY
from preference_inferrer.llms.base_llm import BaseLLM


class OpenAIGPT(BaseLLM):
    """
    Wrapper for OpenAI's GPT models.
    """

    def __init__(self, model_name: str, logger_name: str, config: SimpleNamespace):
        """
        Args:
            model_name: Name of the model. See https://platform.openai.com/docs/models for options
            logger_name: Name to provide the logger. Used to differentiate the output of the user's and agent's LLMs.
            config: Configurations.
        """
        super().__init__(model_name, logger_name, config)
        self.client = openai.OpenAI(api_key=OPEN_AI_KEY)

    def prompt_llm(self, user_prompt: str, temperature: t.Optional[float] = None) -> str:
        """
        Prompt gpt for a response.
        Args:
            user_prompt: The prompt to feed to the GPT model
            temperature: Optionally provide a temperature to use in generation.
                         If not provided, use default temperature specified in the config
        Returns:
            The response string produced by the GPT model.
        """
        self.add_to_chat_history("user", user_prompt)

        oai_args = {
            "model": self.name,
            "messages": self.chat_log,
            "temperature": temperature or self.config.gpt_temperature,
            "max_tokens": 4000,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": None,
            "seed": self.config.seed
        }

        response = OpenAIGPT.retry_openai_func(
            self.client.chat.completions.create,
            expected_finish_reason="stop",
            max_attempt=100,
            **oai_args,
        )

        self.logger.info(f"NEW GENERATED TOKENS: {response.usage.completion_tokens}")
        self.total_prompt_tokens += response.usage.prompt_tokens
        self.total_generated_tokens += response.usage.completion_tokens
        response = response.choices[0].message.content.strip()

        self.add_to_chat_history("assistant", response)
        return response

    @staticmethod
    def retry_openai_func(func, expected_finish_reason="stop", max_attempt=10, **kwargs) -> Completion:
        """
        Continuously retries function `func` if it fails. Stop if max attempts is reached
        Args:
            func: The function to try and run, and retry if it fails
            expected_finish_reason: The expected reason that generation ended.
            max_attempt: Maximum number of retries
            **kwargs:
        Returns:
            The response from func
        """
        import regex as re
        attempts = 0
        while True:
            try:
                response = func(**kwargs)
                if response.choices[0].finish_reason not in ("stop", "length"):
                    logging.warning(
                        f"retry_openai_func: successful openai response but finish_reason is not 'stop' or 'length'. "
                        f"Finish reason given was {response.choices[0].finish_reason}. retrying.."
                    )
                    time.sleep(10)
                else:
                    if response.choices[0].finish_reason == "length" and expected_finish_reason == "stop":
                        logging.warning("retry_openai_func: reached max tokens - consider increasing max_tokens")
                    break
            except Exception as e:
                logging.error(
                    f"retry_openai_func: call {func.__module__}.{func.__name__}: attempt {attempts} failed {e}"
                )
                r = re.search(r"Please retry after\s(\d+)", str(e))
                sleep_time = int(r.group(1)) if r else 10
                attempts += 1
                if attempts == max_attempt:
                    logging.critical(f"retry_openai_func: reached max attempt ({max_attempt}). failing")
                    return ""
                else:
                    time.sleep(sleep_time)
        return response
