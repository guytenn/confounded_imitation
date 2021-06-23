import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
#print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import numpy as np
import torch

import time
import h5py
from pathlib import Path

import gym
import src.envs

from pynput import keyboard
import cv2

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--seed', type=int, default=-1, help='random seed (default: -1)')
parser.add_argument(
    '--env-name',
    default='rooms-v0',
    help='environment to train on (default: rooms-v0)')
parser.add_argument(
    '--save-dir',
    default='./data/',
    help='directory to save agent logs (default: ./data/)')
parser.add_argument('--spatial', action='store_true')
args = parser.parse_args()

if args.seed == -1:
    args.seed = np.random.randint(2 ** 30 - 1)

num_frame_stack = 1 # MAKE 4 FOR ATARI

env = gym.make(args.env_name, px=(0.7, 0.3), spatial=args.spatial)

obs = env.reset()

states = []
actions = []
rewards = []

traj_id = 0

Dx1 = np.zeros_like(env.unwrapped._obs_from_state(spatial=True)[0])
Dx2 = np.zeros_like(env.unwrapped._obs_from_state(spatial=True)[0])
x_count = np.zeros(2)
while True:
    obs_spatial = env.unwrapped._obs_from_state(spatial=True)
    if env.unwrapped.x == 0:
        Dx1 += obs_spatial[0]
        x_count[0] += 1
    else:
        Dx2 += obs_spatial[0]
        x_count[1] += 1
    print(f"Percentage of context views: {100 * x_count / x_count.sum()}%")

    img = obs_spatial.mean(axis=0) + obs_spatial[0]*3 + obs_spatial[2] / 3
    img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_AREA)
    cv2.imshow('game', img)
    cv2.waitKey(0)
    with keyboard.Events() as events:
        # Block for as much as possible
        event = events.get(1e6)
        if event.key == keyboard.KeyCode.from_char('s'):
            action = 1
        elif event.key == keyboard.KeyCode.from_char('w'):
            action = 0
        elif event.key == keyboard.KeyCode.from_char('a'):
            action = 2
        elif event.key == keyboard.KeyCode.from_char('d'):
            action = 3
        elif event.key == keyboard.KeyCode.from_char('q'):
            break
        elif event.key == keyboard.KeyCode.from_char('r'):
            obs = env.reset()
            continue
        else:
            continue

    # Obser reward and next obs
    obs_old = obs.copy()
    obs, reward, done, info = env.step(action)

    if done:
        rewards.append(info['episode']['r'])
        obs = env.reset()
        traj_id += 1
        # print(f'{100 * traj_id / args.n_traj}% Complete, avg reward = {np.mean(rewards)}', end="\r")
        # print(f'{100 * traj_id / args.n_traj}% Complete, avg reward = {np.mean(rewards)}')
    else:
        states.append(obs_old)
        actions.append(float(info['a']))



fig = plt.figure()
plt.imshow(Dx1)
plt.title("Context 1")
plt.savefig("context1.png")
fig = plt.figure()
plt.imshow(Dx2)
plt.title("Context 2")
plt.savefig("context2.png")
Dx = Dx1 + Dx2
fig = plt.figure()
plt.imshow(Dx)
plt.title("Marginalized Context")
plt.savefig("marginalized_context.png")

data = dict(states=np.array(states), actions=np.array(actions), rewards=np.array(rewards))
Path(args.save_dir).mkdir(parents=True, exist_ok=True)
file_path = os.path.join(args.save_dir, f'expert_data.h5')
hf = h5py.File(file_path, 'w')
for k, v in data.items():
    data_to_save = np.array(v)
    hf.create_dataset(k, data=data_to_save)
hf.close()
