#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import typing as t

import numpy as np
import torch as th
from evaluate import load

from preference_inferrer.preference_inferring_agents.encoders import get_encoder


class ExampleRetriever:
    """
    Retrieval method for retrieval augmented generation
    """

    def __init__(self, config):
        """
        Args:
            config: Configurations
        """
        self.config = config
        if config.retrieval_mode == "bertscore":
            self.bertscore = load("bertscore")
        elif config.retrieval_mode == "cosine_sim":
            self.encoder_model = get_encoder(self.config.retrieval_underlying_model, config)
        self.encodings = {}
        self.items = []
        self.document_source_accuracy = []

        self.matches = []
        self.errors = []
        self.num_related_items = []

    def __getitem__(self, item: int) -> t.Tuple:
        """
        With __setitem__, overwrites the [] operator (e.g. x = y[i]). Gets the specified item.
        Args:
            item: the index of the item
        Returns:
            The item
        """
        return self.items[item]

    def __setitem__(self, key, value):
        """
        With __getitem__, overwrites the [] operator (e.g. x[i] = y). Sets the specified item.
        Args:
            key: the index of the item
            value: the value to set it at
        """
        self.items[key] = value

    def __len__(self) -> int:
        """Overrides the len() operator."""
        return len(self.items)

    def encode(self, context: str) -> th.Tensor:
        """
        Returns the encoding for the context.
        If previously encoded, retrieve the encoding from the dictionary of encodings
        Otherwise, encode the context and add it to the dictionary of encodings.
        Encoding is dependent on the encoder model chosen in __init__
        Args:
            context: The context to encode
        Returns:
            The encoding of the context
        """
        if context not in self.encodings:
            self.encodings[context] = self.encoder_model.encode(context).view(-1)
        return self.encodings[context]

    def add(self, context: t.Union[int, str], doc_source: str, info: t.Any):
        """
        Add an item to the list of items.
        An item is composed of the context (which can be thought of as a kind of key), it's index in the list,
        the document source (this is oracle knowledge required for some metrics), and "info" which can be anything
        that needs to be stored.
        Args:
            context: A sort of key that is used to retrieve relevant examples
            doc_source: Oracle knowledge of where the context originates from. required for some metrics
            info: The primary information that needs to be stored alongside the context.
        """
        example_idx = len(self)
        if self.config.retrieval_mode == "exact_match":
            context = doc_source
        elif self.config.retrieval_mode == "cosine_sim":
            context = self.encode(context)
        self.items.append((context, example_idx, doc_source, info))

    def get_most_similar_examples(self, context: t.Union[str, int], top_k: int = 1,
                                  doc_source: t.Optional[str] = None) -> t.List[t.Tuple]:
        """
        Retrieves the most "relevant" trajectories from list of previously seen trajectories
        Args:
            context: context that should help determine "relevance".
                     If the context is an int, search for examples with an identical context.
                     If the context is a str, encode it using an encoder LLM and find the examples whose encodings
                         having the highest dot product (should this be cosine similarity?)
            top_k: number of samples to retrieve
            doc_source: The source of the document. Requires oracle knowledge, but useful to determine how often we are
                        retrieving "correct" examples (i.e. examples that share a true shared context).
        Returns:
            List of related examples. Each related example is a tuple containing (context, sample_idx, doc_source, info)
            info can be anything that needs to be stored.
        """
        if self.config.retrieval_mode == "exact_match":
            related_items = [item for item in self.items if item[0] == doc_source]
            top_k = min(len(related_items), top_k)
            related_items = related_items[-top_k:]
        elif self.config.retrieval_mode == "most_recent":
            related_items = self.items[-top_k:]
        else:
            top_k = min(top_k, len(self))
            if top_k == 0:
                return []

            if self.config.retrieval_mode == "bertscore":
                other_docs = [d for d, _, _, _ in self.items]
                this_doc = [context] * len(other_docs)
                preference_similarity = self.bertscore.compute(predictions=this_doc, references=other_docs,
                                                               model_type=self.config.retrieval_underlying_model,
                                                               rescale_with_baseline=True,
                                                               lang="en")
                similarity = th.tensor(preference_similarity["f1"], device=self.config.device)
            elif self.config.retrieval_mode == "cosine_sim":
                docs = th.stack([enc for enc, _, _, _ in self.items])
                similarity = th.matmul(docs, self.encode(context))
            else:
                raise ValueError(f"Unknown retrieval mode: {self.config.retrieval_mode}")

            if self.config.retrieval_threshold is not None:
                related_items = [self.items[i] for i in similarity.topk(top_k)[1] if
                                 similarity[i] >= self.config.retrieval_threshold]
            else:
                related_items = [self.items[i] for i in similarity.topk(top_k)[1]]

            for i, item in enumerate(self.items):
                if item[2] == doc_source:
                    self.matches.append(similarity[i].detach())
                else:
                    self.errors.append(similarity[i].detach())

            self.num_related_items.append(len(related_items))

        if doc_source is not None:
            self.document_source_accuracy.extend([int(ds == doc_source) for _, _, ds, _ in related_items])

        return related_items

    def get_metrics(self) -> t.Dict:
        """
        Return metric related to example retrieval. Main metric is how accurate the retrieved examples were.
        Returns:
            A dictionary containing metrics (keys are strings with metric names, values are the numeric metric values).
        """
        return {"document_source_accuracy": np.mean(self.document_source_accuracy)}
