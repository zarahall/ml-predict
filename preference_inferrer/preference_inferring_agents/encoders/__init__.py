#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import typing as t
from types import SimpleNamespace

from preference_inferrer.preference_inferring_agents.encoders.bert import BertEncoding
from preference_inferrer.preference_inferring_agents.encoders.hf_encoder import HFEncoder
from preference_inferrer.preference_inferring_agents.encoders.mpnet_base import MPNetEncoding


def get_encoder(encoder_name: str, config: SimpleNamespace) -> t.Union[BertEncoding, HFEncoder, MPNetEncoding]:
    """
    Creates and returns the encoder specified by encoder_name
    Args:
        encoder_name: Specifies the kind of encoder to create
        config: configurations
    Returns:
        The created encoder
    """
    if encoder_name == "bert":
        encoder = BertEncoding()
    elif encoder_name == "mpnet":
        encoder = MPNetEncoding()
    else:
        encoder = HFEncoder(encoder_name, config)
    return encoder
