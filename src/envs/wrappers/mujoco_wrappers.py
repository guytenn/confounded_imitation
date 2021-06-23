from gym import Wrapper
from typing import Dict, Callable


class StochasticModel(Wrapper):
    def __init__(self, env, property2model: Dict[str, Callable]):
        super(StochasticModel, self).__init__(env)

        self.property2model = property2model

    def step(self, action):
        self._before_step(action)
        observation, reward, done, info = self.env.step(action)
        done = self._after_step(observation, reward, done, info)

        return observation, reward, done, info

    def reset(self, **kwargs):
        self._before_reset()
        observation = self.env.reset(**kwargs)
        self._after_reset(observation)

        return observation