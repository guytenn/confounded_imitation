"""Wrap Google's RecSim environment for RLlib

RecSim is a configurable recommender systems simulation platform.
Source: https://github.com/google-research/recsim
"""

from collections import OrderedDict
import gym
from gym import spaces
import numpy as np
from src.envs.recsim_extension import interest_evolution
from typing import List

from ray.rllib.utils.error import UnsupportedSpaceException
from ray.tune.registry import register_env

from ray.rllib.utils.framework import try_import_torch
from ray.rllib.policy.sample_batch import SampleBatch
torch, nn = try_import_torch()
from ray.rllib.models.modelv2 import restore_original_dimensions


class RecSimObservationSpaceWrapper(gym.ObservationWrapper):
    """Fix RecSim environment's observation space

    In RecSim's observation spaces, the "doc" field is a dictionary keyed by
    document IDs. Those IDs are changing every step, thus generating a
    different observation space in each time. This causes issues for RLlib
    because it expects the observation space to remain the same across steps.

    This environment wrapper fixes that by reindexing the documents by their
    positions in the list.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_space = self.env.observation_space
        doc_space = spaces.Dict(
            OrderedDict(
                [(str(k), doc)
                 for k, (_,
                         doc) in enumerate(obs_space["doc"].spaces.items())]))
        self.observation_space = spaces.Dict(
            OrderedDict([
                ("user", obs_space["user"]),
                ("doc", doc_space),
                ("response", obs_space["response"]),
            ]))

    def observation(self, obs):
        new_obs = OrderedDict()
        new_obs["user"] = obs["user"]
        new_obs["doc"] = {
            str(k): v
            for k, (_, v) in enumerate(obs["doc"].items())
        }
        new_obs["response"] = obs["response"]
        return new_obs


class RecSimResetWrapper(gym.Wrapper):
    """Fix RecSim environment's reset() and close() function

    RecSim's reset() function returns an observation without the "response"
    field, breaking RLlib's check. This wrapper fixes that by assigning a
    random "response".

    RecSim's close() function raises NotImplementedError. We change the
    behavior to doing nothing.
    """

    def reset(self):
        obs = super().reset()
        obs["response"] = self.env.observation_space["response"].sample()
        return obs

    def close(self):
        pass


class MultiDiscreteToDiscreteActionWrapper(gym.ActionWrapper):
    """Convert the action space from MultiDiscrete to Discrete

    At this moment, RLlib's DQN algorithms only work on Discrete action space.
    This wrapper allows us to apply DQN algorithms to the RecSim environment.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        if not isinstance(env.action_space, spaces.MultiDiscrete):
            raise UnsupportedSpaceException(
                f"Action space {env.action_space} "
                f"is not supported by {self.__class__.__name__}")
        self.action_space_dimensions = env.action_space.nvec
        self.action_space = spaces.Discrete(
            np.prod(self.action_space_dimensions))

    def action(self, action: int) -> List[int]:
        """Convert a Discrete action to a MultiDiscrete action"""
        multi_action = [None] * len(self.action_space_dimensions)
        for idx, n in enumerate(self.action_space_dimensions):
            action, dim_action = divmod(action, n)
            multi_action[idx] = dim_action
        return multi_action


def restore_samples(obs_samples, action_samples, observation_space):
    obs = restore_original_dimensions(
        obs_samples,
        observation_space,
        tensorlib=np)

    doc = torch.cat([torch.tensor(val).unsqueeze(1) for val in obs["doc"].values()], 1)
    actions = torch.tensor(action_samples)

    # Step 1: Build user choice model loss
    _, _, embedding_size = doc.shape
    # selected_doc.shape: [batch_size, slate_size, embedding_size]
    selected_doc = torch.gather(
        # input.shape: [batch_size, num_docs, embedding_size]
        input=doc,
        dim=1,
        # index.shape: [batch_size, slate_size, embedding_size]
        index=actions.unsqueeze(2).expand(-1, -1, embedding_size))
    selected_doc = selected_doc.numpy().reshape(-1, np.prod(selected_doc.shape[1:]))

    return obs['user'], selected_doc


def make_recsim_env(config):
    DEFAULT_ENV_CONFIG = {
        "num_candidates": 10,
        "slate_size": 2,
        "resample_documents": True,
        "seed": 0,
        "convert_to_discrete_action_space": False,
        "alpha": 0.5,
        "beta": 0.5
    }
    env_config = DEFAULT_ENV_CONFIG.copy()
    env_config.update(config)
    if env_config["seed"] == -1:
        env_config["seed"] = np.random.randint(2 ** 30 - 1)
    env = interest_evolution.create_environment(env_config)
    env = RecSimResetWrapper(env)
    env = RecSimObservationSpaceWrapper(env)
    if env_config and env_config["convert_to_discrete_action_space"]:
        env = MultiDiscreteToDiscreteActionWrapper(env)
    return env


env_name = "RecSim-v2"
register_env(name=env_name, env_creator=lambda config: make_recsim_env({"alpha": config['alpha'],
                                                                        "beta": config['beta'],
                                                                        "seed": -1}))
