import numpy as np
import h5py
import os
from stable_baselines3.common.vec_env import VecEnv


class DataGenerator:
    def __init__(self, env: VecEnv, agent):
        if not isinstance(env, VecEnv):
            raise TypeError('Must input VecEnv environment, use DummyVecEnv for single environment')
        self.env = env
        self.agent = agent

        self.data = dict(states=[], actions=[], rewards=[], dones=[])
        self.current_state = self.env.reset()
        self.data['dones'].append(np.zeros(len(self.current_state)).astype(np.bool))
        # self.data['states'].append(np.copy(self.current_state).astype(np.int8))
        self._final = False
        self.hash_function = None

    def update(self, env, agent):
        self.env = env
        self.agent = agent

    def step(self, deterministic=False, random=False):
        if self._final:
            print("Dataset finalized, cannot call step")
            return

        if random:
            action = self.env.action_space.sample()
            if isinstance(action, int):
                action = np.array([action])
        else:
            action = self.agent.predict(self.current_state, deterministic=deterministic)
            if type(action) == tuple:
                action = action[0]
            # action = action.astype('int')

        self.data['states'].append(self.current_state)
        self.data['actions'].append(action)

        next_state, reward, done, info = self.env.step(action)
        # No need to reset on done, as env is of type VecEnv

        self.data['rewards'].append(reward.astype('float32'))
        self.data['dones'].append(done)
        self.current_state = np.copy(next_state)

    def save(self, path, save_idx=None):
        self._finalize()

        if save_idx is None:
            file_path = os.path.join(path, 'data.h5')
        else:
            file_path = os.path.join(path, f'data_{save_idx}.h5')
        hf = h5py.File(file_path, 'w')
        for k, v in self.data.items():
            v_numpy = np.array(v)
            data_to_save = np.reshape(v, (np.prod(v_numpy.shape[0:2]),) + v_numpy.shape[2:], order='F')
            hf.create_dataset(k, data=data_to_save)
        hf.close()

    def _finalize(self):
        self.data['dones'] = self.data['dones'][1:]
        self._final = True