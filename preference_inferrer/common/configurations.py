#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

"""
Config utils from the command line.
"""

import logging
import os
import typing as t
from pathlib import Path
from types import SimpleNamespace

import torch as th


def dict_to_namespace(dictionary_to_convert: t.Dict) -> t.Union[SimpleNamespace, t.List[SimpleNamespace], t.Any]:
    """
    Recursively convert (nested) dictionaries into (nested) namespaces.
    If inner item is a list, keep it as a list
    If inner item is neither a list nor a dict, return it as is
    Args:
        dictionary_to_convert: dictionary to convert
    Returns:
        Namespace / List of Namespaces
    """
    if isinstance(dictionary_to_convert, dict):
        # Convert the dictionary into a namespace
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in dictionary_to_convert.items()})
    elif isinstance(dictionary_to_convert, list):
        # If it's a list, apply the conversion to each item in the list
        return [dict_to_namespace(item) for item in dictionary_to_convert]
    else:
        # If it's neither a dictionary nor a list, return the item as is
        return dictionary_to_convert


def configure_config(config: t.Dict) -> SimpleNamespace:
    """
    1. Adds / resolves paths in the configuration file.
    2. Adds which pytorch device to use to configuration
    3. Converts dictionary into a namespace to make using the configuration slightly more pleasant
    Args:
        config: a dictionary version of the configuration file (e.g. loaded using load_config)
    Returns:
        A namespace version of the configuration file with some added fields
    """
    config = dict_to_namespace(config)

    # Set base_dir as Path object and resolve default base_dir
    config.base_dir = Path(config.base_dir if config.base_dir else os.getcwd())
    config.device = th.device("cuda" if th.cuda.is_available() else "cpu")

    # Set up logging levels, basically a switch statement
    config.logging_level = \
        {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR,
         "critical": logging.CRITICAL}[config.logging_level]

    logging.basicConfig(level=config.logging_level)
    return config
