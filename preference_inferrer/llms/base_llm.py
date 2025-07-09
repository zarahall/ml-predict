#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

"""
Defines an abstract base class for a wrapper to an LLM.
Includes several helper functions that help create full prompts for prompt parts defined in template classes
and for extracting certain kinds of outputs from llm responses.
"""
import logging
import typing as t
from abc import ABC, abstractmethod
from types import SimpleNamespace

import transformers


class BaseLLM(ABC):
    """
    Wrapper around an LLM to make it act as a preference inferrer
    """
    # Must be defined in inherited classes.
    # Share model and tokenizer across instances to avoid memory issues
    model: transformers.AutoModelForCausalLM = None
    tokenizer:transformers.AutoTokenizer = None

    def __init__(self, model_name_or_path: str, logger_name: str, config: SimpleNamespace):
        """
        Args:
            model_name_or_path: Name of llm model
            logger_name: Name to provide the logger. Used to differentiate the output of the user's and agent's LLMs.
            config: Configurations
        """
        self.name = model_name_or_path
        self.config = config

        # Keeps track of the number of generated tokens
        self.total_prompt_tokens = 0
        self.total_generated_tokens = 0
        # Mapping between words outputted by LLM and a validation score
        self.validation_word_to_score = {"confirms": 1, "neutral": 0, "contradicts": -1}
        # Chat log keeps a history of the chat to date (until reset)
        self.chat_log = []
        # Set up logger
        self.logger_name = logger_name
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(level=self.config.logging_level)

    @abstractmethod
    def prompt_llm(self, user_prompt: str, temperature: t.Optional[float] = None) -> str:
        """
        Prompts are fed to the llm and a response is returned.
        Args:
            user_prompt: Prompt to feed to llm
            temperature: Optionally provide a temperature to use in generation.
                         If not provided, use default temperature specified in the config
        Returns:
            Token ids of generated output
        """

    def clear_chat_history(self, system_prompt: t.Optional[str] = None):
        """
        Clear previous chat history. Reset it with the system prompt.
        Args:
            system_prompt: System prompt to include
        """
        self.chat_log = []
        if system_prompt is not None:
            self.chat_log.append({"role": "system", "content": system_prompt})
        # Log system prompt
        self.logger.info(f">>> system: {system_prompt}\n")

    def add_to_chat_history(self, role: str, content: str):
        """
        Add message to the chat history. Each message consists of a role (who speaks the message) and the message
        content. Logs each message to the log at info level
        Chats have 3 kinds of roles.
            (1) "system": high level prompt describing overall task
            (2) "user": used to prompt the llm for specific outputs
            (3) "assistant": the llms outputs
        Args:
            role: who is speaking the message
            content: the content of the message
        """
        # Record chat history
        self.chat_log.append(
            {"role": role, "content": content}
        )
        # Log to chat
        self.logger.info(f">>> {role}: {content}\n")

    def get_metrics(self) -> t.Dict:
        """
        Returns relevant metrics
        Returns:
            A dictionary of metrics
        """
        return {
            f"{self.logger_name.lower()}_prompt_tokens": self.total_prompt_tokens,
            f"{self.logger_name.lower()}_generated_tokens": self.total_generated_tokens
        }
