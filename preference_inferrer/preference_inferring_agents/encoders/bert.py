#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import typing as t

import torch as th
from torch.nn.functional import normalize
from transformers import AutoTokenizer
from transformers import BertModel

from preference_inferrer.preference_inferring_agents.encoders.abstract_encoder import AbstractEncoder


class BertEncoding(AbstractEncoder):
    """
    Adapted from https://github.com/gao-g/prelude/blob/main/src/agent/encoders/bert.py
    Uses the mean token vector from bert-base-uncased to encode text
    """

    def __init__(self):
        """Initializes the encoder"""
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.model.to(self.device)

    def encode(self, text: t.Union[str, t.List[str]]) -> th.Tensor:
        """
        Encode text using BERT into a numerical tensor
        Args:
            text: The text to encode
        Returns:
            The numerical tensor
        """
        batch_token_ids = self.tokenizer(text,
                                         padding="longest",
                                         return_tensors="pt",
                                         truncation=True,
                                         max_length=512).to(self.device)

        results = self.model(**batch_token_ids)

        hidden_state = results[0]  # batch x max_len x hidden_dim
        attention_mask = batch_token_ids.attention_mask.unsqueeze(2)  # batch x max_len x 1
        num_tokens = attention_mask.sum(dim=1)  # batch x 1

        masked_hidden_state = th.sum(hidden_state * attention_mask, dim=1)  # batch x hidden_dim
        avg_masked_hidden_state = th.div(masked_hidden_state, num_tokens)  # batch x hidden_dim

        # Normalize embeddings
        avg_masked_hidden_state = normalize(avg_masked_hidden_state, p=2, dim=1)

        # Detach and put on CPU
        avg_masked_hidden_state = avg_masked_hidden_state.detach().cpu()

        return avg_masked_hidden_state


if __name__ == '__main__':
    preferences = ["The preference is short, concise, academic writing", "The preference is bullet point",
                   "The preference is comic writing"]

    encoder = BertEncoding()
    v = encoder.encode(preferences)
