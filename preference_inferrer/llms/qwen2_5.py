"""
Copyright (C) 2025 Apple Inc. All rights reserved.

Defines a wrapper for Qwen2.5 Instruct
"""
import torch as th
import typing as t
from types import SimpleNamespace

import numpy as np

from preference_inferrer.common.personal_keys import HF_TOKEN
from transformers import AutoModelForCausalLM, AutoTokenizer

from preference_inferrer.llms.base_llm import BaseLLM


class Qwen2_5(BaseLLM):
    """
    We use a single class model and tokenizer so that we don't have to load separate model's for the agent and user.
    Making them instance variables would 2x GPU requirements in some cases.
    NOTE: Because we use a single class model and tokenizer, we cannot run the 7b and 72b model concurrently.
          i.e. we can't have the agent use the 7b and the user use the 72b. However, we can mix and match other models
          (e.g. 7b qwen2.5 and gpt-4o).
    """
    # Share model and tokenizer across instances to avoid memory issues
    model: None
    tokenizer: None
    hf_mapper: t.Mapping = {
        "qwen-2_5-7b": "Qwen/Qwen2.5-7B-Instruct",
        "qwen-2_5-72b": "Qwen/Qwen2.5-72B-Instruct",
    }

    def __init__(self, model_name: str, logger_name: str, config: SimpleNamespace):
        """
        Args:
            model_name: specific model name.
            logger_name: Name to provide the logger. Used to differentiate the output of the user's and agent's LLMs.
            config: configuration namespace
        """
        super().__init__(model_name, logger_name, config)
        if Qwen2_5.model is None:
            # For release, replace above line with below lines
            if "72b" in model_name:
                Qwen2_5.model = AutoModelForCausalLM.from_pretrained(Qwen2_5.hf_mapper[model_name],
                                                                     device_map="auto",
                                                                     token=HF_TOKEN,
                                                                     cache_dir="/local/nlp/zyh2000/hf_cache")
            else:
                Qwen2_5.model = AutoModelForCausalLM.from_pretrained(Qwen2_5.hf_mapper[model_name],
                                                                     device_map="auto",
                                                                     token=HF_TOKEN,
                                                                     cache_dir="/local/nlp/zyh2000/hf_cache")
            Qwen2_5.tokenizer = AutoTokenizer.from_pretrained(Qwen2_5.hf_mapper[model_name],
                                                              token=HF_TOKEN,
                                                              cache_dir="/local/nlp/zyh2000/hf_cache")

    def prompt_llm(self, user_prompt: str, temperature: t.Optional[float] = None) -> str:
        """
        Format prompt, tokenize prompt, put tokens on device, and feed prompt to llm
        Only returns newly generated token ids. Input ids are trimmed.
        Args:
            user_prompt: prompt to feed to llm
            temperature: Optionally provide a temperature to use in generation.
                         If not provided, use default temperature specified in the config
        Returns:
            The llm's response
        """
        self.add_to_chat_history("user", user_prompt)

        with th.no_grad():
            model_inputs = Qwen2_5.tokenizer.apply_chat_template(self.chat_log, add_generation_prompt=True,
                                                                 tokenize=True, return_tensors="pt")
            model_inputs = model_inputs.to(self.config.device)
            # Generate response
            temperature = temperature or self.config.qwen_temperature

            generated_ids = Qwen2_5.model.generate(model_inputs,
                                                   max_new_tokens=1000,
                                                   do_sample=True,
                                                   temperature=temperature,
                                                   top_p=0.8,
                                                   top_k=20,
                                                   repetition_penalty=1.05)
            response_ids = generated_ids[0, model_inputs.shape[1]:]
            # Keep track of generated tokens
            self.total_prompt_tokens += model_inputs.shape[1]
            self.total_generated_tokens += np.prod(response_ids.shape)
            response = Qwen2_5.tokenizer.decode(response_ids, skip_special_tokens=True)

        self.add_to_chat_history("assistant", response)

        return response
