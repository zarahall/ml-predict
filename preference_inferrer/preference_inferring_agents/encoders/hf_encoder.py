#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import typing as t
from types import SimpleNamespace

import torch as th
from torch.nn.functional import normalize
from transformers import AutoTokenizer, AutoModel

from preference_inferrer.preference_inferring_agents.encoders.abstract_encoder import AbstractEncoder


class HFEncoder(AbstractEncoder):
    def __init__(self, hf_model_name: str, config: SimpleNamespace):
        """
        Args:
            hf_model_name: The hugging face model to use
            config: Configurations. The `encoder_aggregation` specified by `config.encoder_aggregation` must be
                    one of "cls", "mean_pool", "mean_pool2", or "max_pool".
        """
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        self.model = AutoModel.from_pretrained(hf_model_name)
        self.model.to(self.device)
        self.aggregation_method = self.config.encoder_aggregation

    def encode(self, text: t.Union[str, t.List[str]]) -> th.Tensor:
        """
        Encode text using HF Model into a numerical tensor
        Args:
            text: The text to encode
        Returns:
            The numerical tensor
        """
        model_input = self.tokenizer(text, padding="longest", return_tensors="pt", truncation=True,
                                     max_length=512).to(self.device)

        model_output = self.model(**model_input)

        batch_size, seq_len, hidden_dim = model_output.last_hidden_state.shape

        if self.aggregation_method == "cls":
            encoding = model_output.pooler_output
            assert encoding.shape == (batch_size, hidden_dim,)
        elif self.aggregation_method == "mean_pool":
            encoding = model_output.last_hidden_state
            assert encoding.shape == (batch_size, seq_len, hidden_dim)
            encoding = th.mean(encoding, dim=1)
        elif self.aggregation_method == "mean_pool2":
            token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
            attention_mask = model_input["attention_mask"]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            encoding = th.sum(token_embeddings * input_mask_expanded, 1) / th.clamp(input_mask_expanded.sum(1),
                                                                                    min=1e-9)
        elif self.aggregation_method == "max_pool":
            encoding = model_output.last_hidden_state
            assert encoding.shape == (batch_size, seq_len, hidden_dim)
            encoding, _ = th.max(encoding, dim=1)
            # encoding = th.zeros_like(encoding)
        else:
            raise ValueError(f"Unknown encoding type: {self.aggregation_method}")

        encoding = normalize(encoding).squeeze()
        return encoding
