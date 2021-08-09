import os, sys, multiprocessing, gym, ray, shutil, argparse, importlib, glob
from src.rllib_extensions.recsim_wrapper import make_recsim_env
import numpy as np
from src.rllib_extensions.imppo import PPOTrainer
import src.rllib_extensions.imppo as ppo
from ray.rllib.agents import sac
from src.rllib_extensions import slateq
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
import recsim_expert

trainer_selector = dict(ppo=PPOTrainer, sac=sac.SACTrainer, slateq=slateq.SlateQTrainer)


def setup_config(env, args):
    env_name = 'RecSim-v2' if env.spec is None else env.spec.id

    if args.num_processes is None:
        num_processes = multiprocessing.cpu_count()
    else:
        num_processes = args.num_processes

    config = dict()
    if args.algo == 'ppo':
        config = ppo.DEFAULT_CONFIG.copy()
        config['train_batch_size'] = 19200
        config['num_sgd_iter'] = 50
        config['sgd_minibatch_size'] = 128
        config['lambda'] = 0.95
        config['model']['fcnet_hiddens'] = [100, 100]
    elif args.algo == 'sac':
        # NOTE: pip3 install tensorflow_probability
        config = sac.DEFAULT_CONFIG.copy()
        config['timesteps_per_iteration'] = 400
        config['learning_starts'] = 1000
        config['Q_model']['fcnet_hiddens'] = [100, 100]
        config['policy_model']['fcnet_hiddens'] = [100, 100]
        # config['normalize_actions'] = False
    elif args.algo == 'slateq':
        config = slateq.DEFAULT_CONFIG.copy()
        config["hiddens"] = [256, 256]
        config["train_batch_size"] = 128

    if args.wandb:
        config['wandb_logger'] = dict(project=args.project_name, name=args.run_name, config=args.__dict__)
    else:
        config['wandb_logger'] = None

    if env_name == 'RecSim-v2':
        airl = False
        state_dim = 10# 20  # config["recsim_embedding_size"] * 2
        context_features = range(state_dim)
        hidden_dim = 256
    else:
        state_dim = env.observation_space.shape[0]
        airl = True
        context_features = env.unwrapped.context_features
        hidden_dim = 100
    if args.n_confounders == -1:
        n_confounders = len(context_features)
    else:
        n_confounders = args.n_confounders

    if args.covariate_shift:
        if env_name == 'RecSim-v2':
            config['env_config'] = \
                {
                    'alpha': [10, 1.5],
                    'beta': [4, 4],
                    'n_confounders': n_confounders,
                    'confounding_strength': 1
                }
        else:
            config['env_config'] = \
                {
                    'sparse_reward': args.sparse,
                    'context_params': \
                        {
                            "confounding_strength": args.confounding_strength,
                            "gender": [0.8, 0.2],
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
        if env_name == 'RecSim-v2':
            config['env_config'] = \
                {
                    'alpha': None,
                    'beta': None,
                    'n_confounders': 0,  # even if n_confounders > 0 we set this to zero to not ruin the distribution
                    'confounding_strength': 0
                }
        else:
            config['env_config'] = {'sparse_reward': args.sparse, 'context_params': None}
    config['num_workers'] = num_processes
    config['num_cpus_per_worker'] = 0
    config['seed'] = args.seed
    config['log_level'] = 'ERROR'
    config['framework'] = 'torch'
    if args.dice_coef > 0:
        load_dir = os.path.join(os.path.expanduser(f'{args.data_root_dir}/.datasets'), env_name)
        if args.data_suffix == '':
            data_suffix = get_largest_suffix(load_dir, 'data_')
        else:
            data_suffix = args.data_suffix
        expert_data_path = os.path.join(load_dir, f'data_{data_suffix}.h5')


        config["dice_config"] = {
            "env_name": env_name,
            "imitation_method": args.imitation_method,
            "resampling_coef": args.resampling_coef,
            "lr": 0.0001,
            "gamma": config['gamma'],
            "features_to_remove": context_features[:n_confounders] if args.no_context else [],
            "expert_path": expert_data_path,
            "hidden_dim": hidden_dim,
            "dice_coef": args.dice_coef,
            "observation_space": env.observation_space,
            "action_space": env.action_space,
            "state_dim": state_dim,
            'standardize': True,  # This seems quite important (normalize reward according to batch)
            "airl": airl,
            }

    return config


def load_agent(env, args):
    if args.env != "RecSim-v2":
        rllib_env_name = 'confounded_imitation:'+args.env
    else:
        rllib_env_name = args.env

    config = setup_config(env, args)
    agent = trainer_selector[args.algo](config, rllib_env_name)

    if args.load_model and args.load_policy_path != '':
        if 'checkpoint' in args.load_policy_path:
            print(f'Restoring checkpoint in {args.load_policy_path}')
            agent.restore(args.load_policy_path)
        else:
            # Find the most recent policy in the directory
            directory = os.path.join(args.load_policy_path, args.algo, args.env)
            files = [f.split('_')[-1] for f in glob.glob(os.path.join(directory, 'checkpoint_*'))]
            files_ints = [int(f) for f in files]
            if files:
                checkpoint_max = max(files_ints)
                checkpoint_num = files_ints.index(checkpoint_max)
                checkpoint_path = os.path.join(directory, 'checkpoint_%s' % files[checkpoint_num], 'checkpoint-%d' % checkpoint_max)
                print(f'Restoring checkpoint in {checkpoint_path}')
                agent.restore(checkpoint_path)
                # return agent, checkpoint_path
            return agent, None
    return agent, None


def make_env(env_name, coop=False, env_config={}, seed=1001):
    if not coop:
        if env_name == 'RecSim-v2':
            env = make_recsim_env(env_config)
        else:
            env = gym.make(env_name)
    else:
        module = importlib.import_module('assistive_gym.envs')
        env_class = getattr(module, env_name.split('-')[0] + 'Env')
        env = env_class()
    env.seed(seed)
    return env


def train(args):
    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
    if args.env == 'RecSim-v2':
        # env = None
        env = make_recsim_env({})
    else:
        env = make_env(args.env, coop)
    agent, checkpoint_path = load_agent(env, args)
    if args.env != 'RecSim-v2':
        env.disconnect()

    timesteps = 0
    while timesteps < args.train_timesteps:
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
        checkpoint_path = agent.save(os.path.join(args.save_dir, args.algo, args.env))
    return checkpoint_path


def render_policy(env, env_name, algo, policy_path, coop=False, colab=False, seed=0, no_context=False, covariate_shift=False, n_episodes=1, extra_configs={}):
    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
    if env is None:
        env = make_env(env_name, coop, seed=seed)
        if colab:
            env.setup_camera(camera_eye=[0.5, -0.75, 1.5], camera_target=[-0.2, 0, 0.75], fov=60, camera_width=1920//4, camera_height=1080//4)
    test_agent, _ = load_agent(env, args, env_name, policy_path, True, 0, no_context, -1, covariate_shift, None, None, coop, seed, extra_configs)

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


def evaluate_policy(args, extra_configs={}):
    # env_name, algo, policy_path, n_episodes = 1001, data_suffix = '', covariate_shift = False, coop = False, seed = 0, verbose = False, save_data = False, min_reward_to_save = 100,
    env = make_env(args.env, coop, extra_configs, seed=args.seed)

    if args.env == 'RecSim-v2':
        test_agent = None
    else:
        ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
        test_agent, _ = load_agent(env, args)

    data = dict(states=[], actions=[], rewards=[], dones=[])
    rewards = []
    forces = []
    task_successes = []
    lengths = np.zeros(args.eval_episodes)
    for episode in tqdm(range(args.eval_episodes)):
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
                if args.env == 'RecSim-v2':
                    action = recsim_expert.compute_action(env)
                else:
                    action = test_agent.compute_action(obs)
                prev_obs = obs.copy()
                obs, reward, done, info = env.step(action)
                if args.save_data:
                    if args.env == 'RecSim-v2':
                        doc = np.concatenate([val[np.newaxis, :] for val in obs["doc"].values()], 0)
                        # episode_data['states'].append(np.concatenate((prev_obs['user'], doc.sum(0).clip(0, 1)), axis=-1))
                        episode_data['states'].append(prev_obs['user'])
                        episode_data['actions'].append(np.array(list(prev_obs['doc'].values()))[action].flatten())
                    else:
                        episode_data['states'].append(prev_obs)
                        episode_data['actions'].append(action)
                    episode_data['rewards'].append(reward)
                    episode_data['dones'].append(done)
            reward_total += reward

            # force_list.append(info['total_force_on_human'])
            # task_success = info['task_success']

        if reward_total > args.min_reward_to_save:
            for key, val in episode_data.items():
                data[key] += episode_data[key]

        rewards.append(reward_total)
        # forces.append(np.mean(force_list))
        # task_successes.append(task_success)
        if args.verbose:
            # print('Reward total: %.2f, mean force: %.2f, task success: %r' % (reward_total, np.mean(force_list), task_success))
            print('Reward total: %.2f,vtask success: %r' % (reward_total, task_success))

        sys.stdout.flush()
    # env.disconnect()

    if args.env == 'RecSim-v2':
        a = np.array(data['actions'])
        b = np.argmax(a, axis=1)
        h = [np.sum(b == i) for i in range(np.max(b))]
        print(f'Action Histogram: {h}')

    print('\n', '-'*50, '\n')
    # print('Rewards:', rewards)
    print('Reward Mean:', np.mean(rewards))
    print('Reward Std:', np.std(rewards))

    # # print('Forces:', forces)
    # print('Force Mean:', np.mean(forces))
    # print('Force Std:', np.std(forces))

    # print('Task Successes:', task_successes)
    print('Task Success Mean:', np.mean(task_successes))
    print('Task Success Std:', np.std(task_successes))

    print('Task Length Mean:', np.mean(lengths))
    print('Task Length Std:', np.std(lengths))
    sys.stdout.flush()

    if args.save_data:
        for key in data.keys():
            data[key] = np.array(data[key])
        save_dir = os.path.join(os.path.expanduser('~/.datasets'), args.env)
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        if args.data_suffix == '':
            suffix = get_largest_suffix(save_dir, 'data_')
            file_path = os.path.join(save_dir, f'data_{suffix + 1}.h5')
        else:
            file_path = os.path.join(save_dir, f'data_{args.data_suffix}.h5')
        hf = h5py.File(file_path, 'w')
        for k, v in data.items():
            data_to_save = np.array(v)
            hf.create_dataset(k, data=data_to_save)
        hf.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL for Assistive Gym')
    parser.add_argument('--env', default='FeedingSawyer-v1',
                        help='Environment to train on (default: FeedingSawyer-v1)')
    parser.add_argument('--algo', default='ppo', choices=['ppo', 'sac', 'slateq'],
                        help='Reinforcement learning algorithm')
    parser.add_argument('--project_name', default='Confounded Imitation RL',
                        help='run name for wandb logging')
    parser.add_argument('--run_name', default='',
                        help='run name for wandb logging')
    parser.add_argument('--seed', type=int, default=-1,
                        help='Random seed (default: -1)')
    parser.add_argument('--train', action='store_true', default=False,
                        help='Whether to train a new policy')
    parser.add_argument('--imitation_method', default='gail', choices=['gail', 'chi', 'kl'],
                        help='imitation method to use')
    parser.add_argument('--no_context', action='store_true', default=False,
                        help='Remove context for imitation')
    parser.add_argument('--n_confounders', type=int, default=-1,
                        help='Number of confounders when no context is on (default: -1)')
    parser.add_argument('--resampling_coef', type=float, default=0.2,
                        help='Number between 0 and 1. Higher means will attempt larger covariate shifts sampling')
    parser.add_argument('--confounding_strength', type=float, default=10,
                        help='Interpolate between covariate shift distribution when confounding is present, number between 0 and 10')
    parser.add_argument('--covariate_shift', action='store_true', default=False,
                        help='Add covariate shift to environment')
    parser.add_argument('--sparse', action='store_true', default=False,
                        help='Use sparse reward signal')
    parser.add_argument('--num-processes', type=int, default=-1,
                        help='Number of workers during training (default = -1, use all cpus)')
    parser.add_argument('--load', action='store_true', default=False,
                        help='Whether to load from checkpoint')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Whether to render a single rollout of a trained policy')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='Whether to evaluate a trained policy over n_episodes')
    parser.add_argument('--min_reward_to_save', type=float, default=0,
                        help='Minimum total reward to save while creating data')
    parser.add_argument('--save-data', action='store_true', default=False,
                        help='Whether to save data of policy over n_episodes')
    parser.add_argument('--data_suffix', default='',
                        help='Use special suffix for data (saving and loading)')
    parser.add_argument('--train-timesteps', type=int, default=1000000,
                        help='Number of simulation timesteps to train a policy (default: 1000000)')
    parser.add_argument('--dice_coef', type=float, default=0,
                        help='Dice coefficient, between 0 and 1')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='Directory to save trained policy in (default ./trained_models/)')
    parser.add_argument('--data_root_dir', default='~',
                        help='Root directory for loading data')
    parser.add_argument('--load_policy_path', default='./trained_models/',
                        help='Path name to saved policy checkpoint (NOTE: Use this to continue training an existing policy, or to evaluate a trained policy)')
    parser.add_argument('--load-model', action='store_true', default=False,
                        help='Whether to load checkpoint from load-policy-path')
    parser.add_argument('--render-episodes', type=int, default=1,
                        help='Number of rendering episodes (default: 1)')
    parser.add_argument('--eval-episodes', type=int, default=100,
                        help='Number of evaluation episodes (default: 100)')
    parser.add_argument('--colab', action='store_true', default=False,
                        help='Whether rendering should generate an animated png rather than open a window (e.g. when using Google Colab)')
    parser.add_argument('--wandb', action='store_true', default=False,
                        help='Log to wandb')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Whether to output more verbose prints')
    args = parser.parse_args()

    coop = ('Human' in args.env)
    checkpoint_path = None

    if args.dice_coef < 0 or args.dice_coef > 1:
        raise ValueError("dice_coeff must be a value in [0,1]")

    if args.dice.resampling_coef < 0 or args.resampling_coef > 1:
        raise ValueError("resampling_coef must be a value in [0,1]")

    if args.seed == -1:
        args.seed = np.random.randint(2 ** 30 - 1)

    if args.num_processes == -1:
        args.num_processes = None

    if args.train:
        checkpoint_path = train(args)
    if args.render:
        render_policy(None, args.env, args.algo, checkpoint_path if checkpoint_path is not None else args.load_policy_path, coop=coop, colab=args.colab, seed=args.seed, no_context=False, covariate_shift=False, n_episodes=args.render_episodes)
    if args.evaluate or args.save_data:
        evaluate_policy(args, extra_configs={'alpha': [10, 1.5], 'beta': [4, 4], 'n_confounders': args.n_confounders, 'confounding_strength': args.confounding_strength / 10})

