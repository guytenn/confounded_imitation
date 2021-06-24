import os
import gym, src.envs
import pybullet as p
import numpy as np
import h5py
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description='Assistive Data Generation')
parser.add_argument(
    '--seed', type=int, default=-1, help='random seed (default: -1)')
parser.add_argument(
    '--env-name',
    default='FeedingSawyer-v1',
    help='environment to train on (default: FeedingSawyer-v1')
parser.add_argument(
    '--save-dir',
    default='./data/assistive/',
    help='root directory to save data (default: ./data/assistive/)')
args = parser.parse_args()

if args.seed == -1:
    args.seed = np.random.randint(2 ** 30 - 1)

env = gym.make(args.env_name, seed=args.seed)
env.render()
obs = env.reset()

# Map keys to position and orientation end effector movements
pos_keys_actions = {ord('s'): np.array([-0.01, 0, 0]), ord('f'): np.array([0.01, 0, 0]),
                    ord('d'): np.array([0, -0.01, 0]), ord('e'): np.array([0, 0.01, 0]),
                    ord('k'): np.array([0, 0, -0.01]), ord('i'): np.array([0, 0, 0.01])}
rpy_keys_actions = {ord('k'): np.array([-0.05, 0, 0]), ord('i'): np.array([0.05, 0, 0]),
                    ord('d'): np.array([0, -0.05, 0]), ord('e'): np.array([0, 0.05, 0]),
                    ord('s'): np.array([0, 0, -0.05]), ord('f'): np.array([0, 0, 0.05])}

start_pos, orient = env.robot.get_pos_orient(env.robot.right_end_effector)
start_rpy = env.get_euler(orient)
target_pos_offset = np.zeros(3)
target_rpy_offset = np.zeros(3)

states = []
actions = []
rewards = []
dones = []
while True:
    key_pressed = False
    keys = p.getKeyboardEvents()
    # Process position movement keys ('u', 'i', 'o', 'j', 'k', 'l')
    for key, action in pos_keys_actions.items():
        if p.B3G_SHIFT not in keys and key in keys and keys[key] & p.KEY_IS_DOWN:
            target_pos_offset += action
            key_pressed = True
    # Process rpy movement keys (shift + movement keys)
    for key, action in rpy_keys_actions.items():
        if p.B3G_SHIFT in keys and keys[p.B3G_SHIFT] & p.KEY_IS_DOWN and (key in keys and keys[key] & p.KEY_IS_DOWN):
            target_rpy_offset += action
            key_pressed = True

    if ord('q') in keys and keys[ord('q')] & p.KEY_IS_DOWN:
        break

    if ord('r') in keys and keys[ord('r')] & p.KEY_IS_DOWN:
        obs = env.reset()
        continue

    if not key_pressed:
        continue

    # print('Target position offset:', target_pos_offset, 'Target rpy offset:', target_rpy_offset)
    target_pos = start_pos + target_pos_offset
    target_rpy = start_rpy + target_rpy_offset

    # Use inverse kinematics to compute the joint angles for the robot's arm
    # so that its end effector moves to the target position.
    target_joint_angles = env.robot.ik(env.robot.right_end_effector, target_pos, env.get_quaternion(target_rpy), env.robot.right_arm_ik_indices, max_iterations=200, use_current_as_rest=True)
    # Get current joint angles of the robot's arm
    current_joint_angles = env.robot.get_joint_angles(env.robot.right_arm_joint_indices)
    # Compute the action as the difference between target and current joinlllt angles.
    action = (target_joint_angles - current_joint_angles) * 10
    # Step the simulation forward
    prev_obs = obs.copy()
    obs, reward, done, info = env.step(action)

    if done:
        obs = env.reset()
    else:
        states.append(prev_obs)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)

data = dict(states=np.array(states), actions=np.array(actions), rewards=np.array(rewards))

save_dir = os.path.join(args.save_dir, args.env_name)
Path(save_dir).mkdir(parents=True, exist_ok=True)
file_path = os.path.join(save_dir, f'data_1.h5')
hf = h5py.File(file_path, 'w')
for k, v in data.items():
    data_to_save = np.array(v)
    hf.create_dataset(k, data=data_to_save)
hf.close()