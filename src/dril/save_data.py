import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
#print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
# workaround to unpickle olf model files

import numpy as np
import torch

from src.dril.a2c_ppo_acktr.envs import make_vec_envs
from src.dril.a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

import time
import h5py
from pathlib import Path

import src.envs

sys.path.append('a2c_ppo_acktr')

parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--seed', type=int, default=-1, help='random seed (default: -1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    help='log interval, one log per n updates (default: 10)')
parser.add_argument(
    '--env-name',
    default='rooms-v0',
    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument(
    '--save-dir',
    default='./data/',
    help='directory to save agent logs (default: ./data/)')
parser.add_argument(
    '--load-dir',
    default='./trained_models/wind_visible_fixed_env/',
    help='directory to save agent logs (default: ./trained_models/wind_visible_fixed_env/)')
parser.add_argument(
    '--n-traj',
    type=int,
    default=1000,
    help='number of trajectories to save (default: 1000)')
parser.add_argument(
    '--render-every',
    type=int,
    default=10,
    help='render every this many trajectories (default: 100)')
parser.add_argument(
    '--non-det',
    action='store_true',
    default=False,
    help='whether to use a non-deterministic policy')

parser.add_argument(
    '--rules',
    action='store_true',
    default=False,
    help='used a fixed rule-based policy')
args = parser.parse_args()

if args.seed == -1:
    args.seed = np.random.randint(2 ** 30 - 1)

args.det = not args.non_det

num_frame_stack = 1 # MAKE 4 FOR ATARI

env = make_vec_envs(
    args.env_name,
    args.seed + 1000,
    1,
    1,
    device='cpu',
    num_frame_stack=num_frame_stack)

# Get a render function
render_func = get_render_func(env)

if not args.rules:
    # We need to use the same statistics for normalization as used in training
    actor_critic, obs_rms = torch.load(os.path.join(args.load_dir, args.env_name + ".pt"),
                                       map_location='cpu')

vec_norm = get_vec_normalize(env)

if not args.rules:
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms

    recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
    masks = torch.zeros(1, 1)

obs = env.reset()

if args.render_every > 0 and render_func is not None:
    render_func('human')

if args.env_name.find('Bullet') > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i

states = []
actions = []
rewards = []
lengths = np.zeros(args.n_traj)

traj_id = 0
while True:
    if args.rules:
        action = torch.from_numpy(np.array([[-1]]))
    else:
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                obs, recurrent_hidden_states, masks, deterministic=args.det)

    # Obser reward and next obs
    obs_old = obs.detach().numpy().copy()
    obs, reward, done, info = env.step(action)

    if args.render_every > 0 and traj_id % args.render_every == 0 and render_func is not None:
        render_func('human')
        time.sleep(0.1)

    if done:
        rewards.append(info[0]['episode']['r'])
        obs = env.reset()
        traj_id += 1
        # print(f'{100 * traj_id / args.n_traj}% Complete, avg reward = {np.mean(rewards)}', end="\r")
        print(f'{100 * traj_id / args.n_traj}% Complete, avg reward = {np.mean(rewards)}')
        if traj_id >= args.n_traj:
            break
    else:
        states.append(obs_old[0])
        actions.append(info[0]['a'])
        lengths[traj_id] += 1

data = dict(states=np.array(states), actions=np.array(actions), rewards=np.array(rewards), lengths=lengths)
Path(args.save_dir).mkdir(parents=True, exist_ok=True)
file_path = os.path.join(args.save_dir, f'expert_data.h5')
hf = h5py.File(file_path, 'w')
for k, v in data.items():
    data_to_save = np.array(v)
    hf.create_dataset(k, data=data_to_save)
hf.close()
