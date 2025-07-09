#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from types import SimpleNamespace

from preference_inferrer.preference_inferring_agents.abstract_agent import Agent


def get_inferring_agent(agent_type: str, config: SimpleNamespace) -> Agent:
    """
    Create and return the specified agent type
    Args:
        agent_type: The type of agent to create
        config: Configurations to pass to the agent
    Returns:
        The created agent
    """
    if agent_type == "cipher1":
        from preference_inferrer.preference_inferring_agents.cipher import Cipher1Agent
        agent = Cipher1Agent(config)
    elif agent_type == "cipherN":
        from preference_inferrer.preference_inferring_agents.cipher import CipherNAgent
        agent = CipherNAgent(config)
    elif agent_type == "icl":
        from preference_inferrer.preference_inferring_agents.in_context_learning import InContextLearningAgent
        agent = InContextLearningAgent(config)
    elif agent_type == "bc":
        from preference_inferrer.preference_inferring_agents.behavioral_cloning import BehavioralCloning
        agent = BehavioralCloning(config)
    elif agent_type == "no_learning":
        from preference_inferrer.preference_inferring_agents.no_learning import NoLearningAgent
        agent = NoLearningAgent(config)
    elif agent_type == "oracle_preference":
        from preference_inferrer.preference_inferring_agents.oracle_preference import OraclePreferenceAgent
        agent = OraclePreferenceAgent(config)
    elif agent_type == "prose":
        from preference_inferrer.preference_inferring_agents.prose import Prose
        agent = Prose(config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    return agent
