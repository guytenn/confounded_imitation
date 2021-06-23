import os
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback
from abc import abstractmethod
from src.data.data_generator import DataGenerator
from tqdm import tqdm
import numpy as np


class EvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_steps=10000, wandb_logger=None, prefix="", desc="Evaluating Policy"):
        super(EvalCallback, self).__init__(verbose=0)
        self.eval_env = eval_env
        self.eval_steps = eval_steps
        self.wandb_logger = wandb_logger
        if prefix:
            prefix += '_'
        self.prefix = prefix
        self.desc = desc

    def _eval_policy(self, policy):
        obs = self.eval_env.reset()
        traj_rewards = [0]
        for _ in tqdm(range(self.eval_steps), desc=self.desc, leave=True):
            policy_input = obs

            action, _state = policy(policy_input, deterministic=False)
            obs, reward, done, info = self.eval_env.step(action)
            traj_rewards[-1] += reward[0]

            if done[0]:
                obs = self.eval_env.reset()
                traj_rewards.append(0)
        return np.mean(traj_rewards[:-1])

    def _on_step(self) -> bool:
        reward = self._eval_policy(self.model.predict)
        for _ in range(5):
            print('-' * 20)
        print(f'Result = {reward}')
        for _ in range(5):
            print('-' * 20)

        if self.wandb_logger is not None:
            self.wandb_logger.log({self.prefix + 'reward': reward},
                                  step=self.num_timesteps)

        return True


class SaveCallback(BaseCallback):
    def __init__(self, run_name, verbose=0):
        super(SaveCallback, self).__init__(verbose)
        self.run_name = run_name
        self.save_path = None
        self.counter = 1

    def _init_callback(self) -> None:
        env_name = self.training_env.get_attr('spec')[0].id
        self.save_path = os.path.join(os.path.expanduser('~/.datasets'), env_name, self.run_name)
        Path(self.save_path).mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def _on_step(self) -> bool:
        return True


class SaveModelCallback(SaveCallback):
    def __init__(self, run_name, verbose=0):
        super(SaveModelCallback, self).__init__(run_name=run_name, verbose=verbose)

    def _on_step(self) -> bool:
        print(f'Saving model {self.counter}')

        model_save_path = os.path.join(self.save_path, f'model_{self.counter}')
        self.model.save(model_save_path)
        self.counter += 1
        return True


class SaveDataCallback(SaveCallback):
    def __init__(self, run_name, data_size=1000000, random=False, verbose=0):
        super(SaveDataCallback, self).__init__(run_name=run_name, verbose=verbose)
        self.data_size = data_size
        self.random = random

    def _on_step(self) -> bool:
        print(f'Generating data {self.counter}')
        data_generator = DataGenerator(env=self.training_env, agent=self.model)

        for _ in tqdm(range(self.data_size // self.training_env.num_envs)):
            data_generator.step(deterministic=False, random=self.random)

        data_generator.save(self.save_path, self.counter)
        self.counter += 1
        return True
