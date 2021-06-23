from collections import defaultdict
import gym
import src.envs
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import h5py
import os
from pathlib import Path


matplotlib.use('TkAgg')

input2command = defaultdict(int, {'w': 0, 's': 1, 'a': 2, 'd': 3})  # default value is 0

if __name__ == '__main__':
    env_name = 'rooms-v0'
    run_name = 'human'

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.show()

    env = gym.make(env_name, rows=32, cols=32, fixed_reset=True)

    data = dict(states=[], actions=[], rewards=[], dones=[])
    n_traj = 0
    prev_command = None

    while True:
        command = input("Input command ")

        if command == 'r':
            s = env.reset()
            n_traj += 1

            if n_traj > 1:
                data['states'].pop()
                data['dones'][-1] = np.array([True])
            data['states'].append(np.array([s]))
        elif command == 'f':
            data['states'].pop()
            break
        else:
            if prev_command is not None and command == '':
                command = prev_command

            for c in command:
                if c in ['w', 's', 'a', 'd']:
                    a = input2command[c]

                s, r, d, i = env.step(a)

                data['states'].append(np.array([s]))
                data['actions'].append(np.array([a]).astype('<f4'))
                data['rewards'].append(np.array([r]).astype('<f4'))
                data['dones'].append(np.array([d]))

        prev_command = command

        s = np.moveaxis(s, 0, -1).astype('<f4')
        s[:, :, -1] *= 0
        ax.imshow(s)
        plt.draw()
        plt.pause(.001)

    save_path = os.path.join(os.path.expanduser('~/.datasets'), env_name, run_name)
    Path(save_path).mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(save_path, 'data.h5')
    hf = h5py.File(file_path, 'w')
    for k, v in data.items():
        v_numpy = np.array(v)
        data_to_save = np.reshape(v, (np.prod(v_numpy.shape[0:2]),) + v_numpy.shape[2:], order='F')
        hf.create_dataset(k, data=data_to_save)
    hf.close()