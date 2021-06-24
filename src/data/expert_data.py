from stable_baselines3.common.buffers import BaseBuffer
from typing import Dict, Generator, Optional, Union
from stable_baselines3.common.vec_env import VecNormalize
from types import SimpleNamespace

import numpy as np
import torch
from gym import spaces


class ExpertData:
    def __init__(
        self,
        observations, actions, dones, device
    ):
        self.observations = torch.from_numpy(observations).to(device)
        self.actions = torch.from_numpy(actions).to(device)
        self.dones = torch.from_numpy(dones).to(device)
        self.weights = torch.ones_like(self.dones, dtype=torch.float32, device=device)

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None, weights=None):
        if weights is None:
            weights = self.weights
        idx = torch.multinomial(weights, batch_size, replacement=True)

        data = SimpleNamespace(observations=BaseBuffer._normalize_obs(self.observations[idx], env),
                               actions=self.actions[idx],
                               dones=self.dones[idx])

        return data

    def __len__(self):
        return len(self.actions)
