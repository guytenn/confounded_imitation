import os, sys, multiprocessing, gym, ray, shutil, argparse, importlib, glob
import numpy as np
# from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from src.rllib_extensions.imppo import PPOTrainer
from ray.rllib.agents import ppo, sac
from ray.tune.logger import pretty_print
import ray.rllib.utils.exploration.curiosity
try:
    from numpngw import write_apng
except:
    write_apng = None
import gym
import src.envs
from tqdm import tqdm
import h5py
from pathlib import Path
from src.data.utils import get_largest_suffix
from src.rllib_extensions.dice import DICE


def setup_config(env, algo, dice_coef=0, no_context=False, covariate_shift=False, num_processes=None, coop=False, seed=0, extra_configs={}):
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    config = dict()
    if algo == 'ppo':
        config = ppo.DEFAULT_CONFIG.copy()
        config['train_batch_size'] = 19200
        config['num_sgd_iter'] = 50
        config['sgd_minibatch_size'] = 128
        config['lambda'] = 0.95
        config['model']['fcnet_hiddens'] = [100, 100]
    elif algo == 'sac':
        # NOTE: pip3 install tensorflow_probability
        config = sac.DEFAULT_CONFIG.copy()
        config['timesteps_per_iteration'] = 400
        config['learning_starts'] = 1000
        config['Q_model']['fcnet_hiddens'] = [100, 100]
        config['policy_model']['fcnet_hiddens'] = [100, 100]
        # config['normalize_actions'] = False

    if covariate_shift:
        config['env_config'] = \
            {
                'context_params': \
                    {
                        "gender": [0.2, 0.8],
                        "mass_delta": 10,
                        "mass_std": 20,
                        "radius_delta": -0.1,
                        "radius_std": 0.2,
                        "height_delta": 0.1,
                        "height_std": 0.2,
                        "velocity_deltas": [0.1, 0.2],
                        "force_nontarget_deltas": [0.002, 0.005],
                        "high_forces_deltas": [0.001, 0.03],
                        "food_hit_deltas": [-0.5, 1.],
                        "food_velocities_deltas": [-0.5, 1.],
                        "dressing_force_deltas": [0, 0.1],
                        "high_pressures_deltas": [0, 0.1],
                        "impairment": [0.1, 0.1, 0.1, 0.7]
                    }
            }
    else:
        config['env_config'] = {'context_params': None}
    config['num_workers'] = num_processes
    config['num_cpus_per_worker'] = 0
    config['seed'] = seed
    config['log_level'] = 'ERROR'
    config['framework'] = 'torch'
    if dice_coef > 0:
        expert_data_path = os.path.join(os.path.expanduser('~/.datasets'), env.spec.id, 'data_1.h5')
        config["dice_config"] = {
            "lr": 0.0001,
            "gamma": config['gamma'],
            "features_to_remove": env.unwrapped.context_features if no_context else [],
            "expert_path": expert_data_path,
            "hidden_dim": 100,
            "dice_coef": dice_coef,
            "action_space": env.action_space,
            "state_dim": env.observation_space.shape[0],
            'standardize': True
            }
    # if algo == 'sac':
    #     config['num_workers'] = 1
    if coop:
        obs = env.reset()
        policies = {'robot': (None, env.observation_space_robot, env.action_space_robot, {}), 'human': (None, env.observation_space_human, env.action_space_human, {})}
        config['multiagent'] = {'policies': policies, 'policy_mapping_fn': lambda a: a}
        config['env_config'] = {'num_agents': 2}
    return {**config, **extra_configs}


def load_policy(env, algo, env_name, policy_path=None, dice_coef=0, no_context=False, covariate_shift=False, num_processes=None, coop=False, seed=0, extra_configs={}):
    if algo == 'ppo':
        agent = PPOTrainer(setup_config(env, algo, dice_coef, no_context, covariate_shift, num_processes, coop, seed, extra_configs), 'assistive_gym:'+env_name)
    elif algo == 'sac':
        agent = sac.SACTrainer(setup_config(env, algo, dice_coef, no_context, covariate_shift, num_processes, coop, seed, extra_configs), 'assistive_gym:'+env_name)
    if policy_path != '':
        if 'checkpoint' in policy_path:
            agent.restore(policy_path)
        else:
            # Find the most recent policy in the directory
            directory = os.path.join(policy_path, algo, env_name)
            files = [f.split('_')[-1] for f in glob.glob(os.path.join(directory, 'checkpoint_*'))]
            files_ints = [int(f) for f in files]
            if files:
                checkpoint_max = max(files_ints)
                checkpoint_num = files_ints.index(checkpoint_max)
                checkpoint_path = os.path.join(directory, 'checkpoint_%s' % files[checkpoint_num], 'checkpoint-%d' % checkpoint_max)
                agent.restore(checkpoint_path)
                # return agent, checkpoint_path
            return agent, None
    return agent, None

def make_env(env_name, coop=False, seed=1001):
    if not coop:
        env = gym.make(env_name)
    else:
        module = importlib.import_module('assistive_gym.envs')
        env_class = getattr(module, env_name.split('-')[0] + 'Env')
        env = env_class()
    env.seed(seed)
    return env


def train(env_name, algo, timesteps_total=1000000, save_dir='./trained_models/', load_policy_path='', dice_coef=0, coop=False, load=False, no_context=False, covariate_shift=False, num_processes=None, seed=0, extra_configs={}):
    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
    env = make_env(env_name, coop)
    agent, checkpoint_path = load_policy(env, algo, env_name, load_policy_path, dice_coef, no_context, num_processes, covariate_shift, coop, seed, extra_configs)
    env.disconnect()

    timesteps = 0
    while timesteps < timesteps_total:
        result = agent.train()
        timesteps = result['timesteps_total']
        if coop:
            # Rewards are added in multi agent envs, so we divide by 2 since agents share the same reward in coop
            result['episode_reward_mean'] /= 2
            result['episode_reward_min'] /= 2
            result['episode_reward_max'] /= 2
        print(f"Iteration: {result['training_iteration']}, total timesteps: {result['timesteps_total']}, total time: {result['time_total_s']:.1f}, FPS: {result['timesteps_total']/result['time_total_s']:.1f}, mean reward: {result['episode_reward_mean']:.1f}, min/max reward: {result['episode_reward_min']:.1f}/{result['episode_reward_max']:.1f}")
        sys.stdout.flush()

        # Delete the old saved policy
        if checkpoint_path is not None:
            shutil.rmtree(os.path.dirname(checkpoint_path), ignore_errors=True)
        # Save the recently trained policy
        checkpoint_path = agent.save(os.path.join(save_dir, algo, env_name))
    return checkpoint_path


def render_policy(env, env_name, algo, policy_path, coop=False, colab=False, seed=0, n_episodes=1, extra_configs={}):
    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
    if env is None:
        env = make_env(env_name, coop, seed=seed)
        if colab:
            env.setup_camera(camera_eye=[0.5, -0.75, 1.5], camera_target=[-0.2, 0, 0.75], fov=60, camera_width=1920//4, camera_height=1080//4)
    test_agent, _ = load_policy(env, algo, env_name, policy_path, 0, False, 1, coop, seed, extra_configs)

    if not colab:
        env.render()
    frames = []
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            if coop:
                # Compute the next action for the robot/human using the trained policies
                action_robot = test_agent.compute_action(obs['robot'], policy_id='robot')
                action_human = test_agent.compute_action(obs['human'], policy_id='human')
                # Step the simulation forward using the actions from our trained policies
                obs, reward, done, info = env.step({'robot': action_robot, 'human': action_human})
                done = done['__all__']
            else:
                # Compute the next action using the trained policy
                action = test_agent.compute_action(obs)
                # Step the simulation forward using the action from our trained policy
                obs, reward, done, info = env.step(action)
            if colab:
                # Capture (render) an image from the camera
                img, depth = env.get_camera_image_depth()
                frames.append(img)
    env.disconnect()
    if colab:
        filename = 'output_%s.png' % env_name
        write_apng(filename, frames, delay=100)
        return filename


def evaluate_policy(env_name, algo, policy_path, n_episodes=1001, covariate_shift=False, coop=False, seed=0, verbose=False, save_data=False, min_reward_to_save=100,extra_configs={}):
    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
    env = make_env(env_name, coop, seed=seed)
    test_agent, _ = load_policy(env, algo, env_name, covariate_shift=covariate_shift, policy_path=policy_path, coop=coop, seed=seed, extra_configs=extra_configs)

    data = dict(states=[], actions=[], rewards=[], dones=[])
    rewards = []
    forces = []
    task_successes = []
    lengths = np.zeros(n_episodes)
    for episode in tqdm(range(n_episodes)):
        obs = env.reset()
        done = False
        reward_total = 0.0
        force_list = []
        task_success = 0.0
        episode_data = dict(states=[], actions=[], rewards=[], dones=[])
        while not done:
            lengths[episode] += 1
            if coop:
                # Compute the next action for the robot/human using the trained policies
                action_robot = test_agent.compute_action(obs['robot'], policy_id='robot')
                action_human = test_agent.compute_action(obs['human'], policy_id='human')
                # Step the simulation forward using the actions from our trained policies
                obs, reward, done, info = env.step({'robot': action_robot, 'human': action_human})
                reward = reward['robot']
                done = done['__all__']
                info = info['robot']
            else:
                action = test_agent.compute_action(obs)
                prev_obs = obs.copy()
                obs, reward, done, info = env.step(action)
                if save_data:
                    episode_data['states'].append(prev_obs)
                    episode_data['actions'].append(action)
                    episode_data['rewards'].append(reward)
                    episode_data['dones'].append(done)
            reward_total += reward

            force_list.append(info['total_force_on_human'])
            task_success = info['task_success']

        if reward_total < min_reward_to_save:
            for key, val in episode_data.items():
                data[key] += episode_data[key]

        rewards.append(reward_total)
        forces.append(np.mean(force_list))
        task_successes.append(task_success)
        if verbose:
            print('Reward total: %.2f, mean force: %.2f, task success: %r' % (reward_total, np.mean(force_list), task_success))
        sys.stdout.flush()
    env.disconnect()

    print('\n', '-'*50, '\n')
    # print('Rewards:', rewards)
    print('Reward Mean:', np.mean(rewards))
    print('Reward Std:', np.std(rewards))

    # print('Forces:', forces)
    print('Force Mean:', np.mean(forces))
    print('Force Std:', np.std(forces))

    # print('Task Successes:', task_successes)
    print('Task Success Mean:', np.mean(task_successes))
    print('Task Success Std:', np.std(task_successes))

    print('Task Length Mean:', np.mean(lengths))
    print('Task Length Std:', np.std(lengths))
    sys.stdout.flush()

    if save_data:
        for key in data.keys():
            data[key] = np.array(data[key])
        save_dir = os.path.join('data', 'assistive', env_name)
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        suffix = get_largest_suffix(save_dir, 'data_')
        file_path = os.path.join(save_dir, f'data_{suffix}.h5')
        hf = h5py.File(file_path, 'w')
        for k, v in data.items():
            data_to_save = np.array(v)
            hf.create_dataset(k, data=data_to_save)
        hf.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL for Assistive Gym')
    parser.add_argument('--env', default='FeedingSawyer-v1',
                        help='Environment to train on (default: ScratchItchJaco-v0)')
    parser.add_argument('--algo', default='ppo',
                        help='Reinforcement learning algorithm')
    parser.add_argument('--seed', type=int, default=-1,
                        help='Random seed (default: -1)')
    parser.add_argument('--train', action='store_true', default=False,
                        help='Whether to train a new policy')
    parser.add_argument('--no_context', action='store_true', default=False,
                        help='Remove context for imitation')
    parser.add_argument('--covariate_shift', action='store_true', default=False,
                        help='Add covariate shift to environment')
    parser.add_argument('--num-processes', type=int, default=-1,
                        help='Number of workers during training (default = -1, use all cpus)')
    parser.add_argument('--load', action='store_true', default=False,
                        help='Whether to load from checkpoint')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Whether to render a single rollout of a trained policy')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='Whether to evaluate a trained policy over n_episodes')
    parser.add_argument('--min_reward_to_save', type=float, default=100,
                        help='Minimum total reward to save while creating data')
    parser.add_argument('--save-data', action='store_true', default=False,
                        help='Whether to save data of policy over n_episodes')
    parser.add_argument('--train-timesteps', type=int, default=1000000,
                        help='Number of simulation timesteps to train a policy (default: 1000000)')
    parser.add_argument('--dice_coef', type=float, default=0,
                        help='Dice coefficient, between 0 and 1')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='Directory to save trained policy in (default ./trained_models/)')
    parser.add_argument('--load-policy-path', default='./trained_models/',
                        help='Path name to saved policy checkpoint (NOTE: Use this to continue training an existing policy, or to evaluate a trained policy)')
    parser.add_argument('--render-episodes', type=int, default=1,
                        help='Number of rendering episodes (default: 1)')
    parser.add_argument('--eval-episodes', type=int, default=100,
                        help='Number of evaluation episodes (default: 100)')
    parser.add_argument('--colab', action='store_true', default=False,
                        help='Whether rendering should generate an animated png rather than open a window (e.g. when using Google Colab)')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Whether to output more verbose prints')
    args = parser.parse_args()

    coop = ('Human' in args.env)
    checkpoint_path = None

    if args.dice_coef < 0 or args.dice_coef > 1:
        raise ValueError("dice_coeff must be a value in [0,1]")

    if args.seed == -1:
        args.seed = np.random.randint(2 ** 30 - 1)

    if args.num_processes == -1:
        args.num_processes = None

    if args.train:
        checkpoint_path = train(args.env, args.algo, timesteps_total=args.train_timesteps, save_dir=args.save_dir, load_policy_path=args.load_policy_path, dice_coef=args.dice_coef, coop=coop, load=args.load, seed=args.seed, no_context=args.no_context, covariate_shift=args.covariate_shift, num_processes=args.num_processes)
    if args.render:
        render_policy(None, args.env, args.algo, checkpoint_path if checkpoint_path is not None else args.load_policy_path, coop=coop, colab=args.colab, seed=args.seed, n_episodes=args.render_episodes)
    if args.evaluate or args.save_data:
        evaluate_policy(args.env, args.algo, checkpoint_path if checkpoint_path is not None else args.load_policy_path, n_episodes=args.eval_episodes, min_reward_to_save=args.min_reward_to_save, coop=coop, seed=args.seed, verbose=args.verbose, save_data=args.save_data, covariate_shift=args.covariate_shift)

