from stable_baselines3 import SAC, TD3, DQN
from src.sb3_extensions.im_sac import ImSAC
from src.sb3_extensions.im_td3 import ImTD3
from stable_baselines3.sac import MlpPolicy as SacMlpPolicy
from stable_baselines3.td3 import MlpPolicy as Td3MlpPolicy
from src.sb3_extensions.cnn_policy import CustomActorCriticCnnPolicy, CustomDqnCnnPolicy
from src.sb3_extensions.im_ppo import ImPPO
from src.sb3_extensions.im_dqn import ImDQN
from src.sb3_extensions.env_util import make_vec_env
import gym
import src.envs
import argparse
import GPUtil
import torch
from stable_baselines3.common.callbacks import CallbackList, EveryNTimesteps
from src.sb3_extensions.callbacks import SaveModelCallback, SaveDataCallback, EvalCallback
from src.im_module.dice_trainer import DICETrainer
from src.data.utils import load_expert_data
try:
    import wandb
except:
    wandb = None

features_to_remove_dict = {"SparseHopper-v0": [12, 13, 14], "rooms-v0": [2, 3, 4, 5]}
def get_env_args(env_name, args) -> dict:
    if env_name == "SparseHopper-v0":
        return dict(max_x=args.max_force, max_y=args.max_force, max_z=args.max_force, sparse=args.sparse)
    elif env_name == 'rooms-v0':
        return dict()
    else:
        return dict()

def run(args):
    try:
        deviceIds = GPUtil.getFirstAvailable(order='memory', maxLoad=0.95, maxMemory=0.95)
        device = torch.device(f'cuda:{deviceIds[0]}')
        print(f'Using cuda device {device}')
    except:
        device = torch.device('cpu')

    env = make_vec_env(args.env_name, n_envs=args.n_envs, env_kwargs=get_env_args(args.env_name, args))
    eval_env = make_vec_env(args.env_name, n_envs=1, env_kwargs=get_env_args(args.env_name, args))

    if args.wandb:
        wandb.init(project="Confounded Imitation RL", name=args.run_name, config=args.__dict__)

    algo_params = dict(gamma=args.gamma,
                       batch_size=args.batch_size,
                       policy_kwargs=dict(net_arch=[100, 100]),
                       learning_rate=args.learning_rate,
                       device=device,
                       verbose=1)

    if args.algo == 'sac' or args.algo == 'td3':
        # for sac (currently not used)
        target_update_interval = -1
        use_sde = True
        algo_params.update(dict(learning_starts=args.learning_starts,
                                buffer_size=args.buffer_size,
                                train_freq=args.train_freq,
                                gradient_steps=args.gradient_steps))
    elif args.algo == 'ppo':
        algo_params.update(dict(n_steps=args.n_steps, n_epochs=args.n_epochs))
    elif args.algo == "dqn":
        algo_params.update(dict(buffer_size=args.buffer_size))

    if args.dice_n_epochs > 0:
        if args.partial_data:
            features_to_remove = features_to_remove_dict[args.env_name]
        else:
            features_to_remove = []
        expert_data = load_expert_data(env, args.expert_load_name, device)
        if len(expert_data) // args.batch_size < 1:
            raise ValueError("Expert_data too small, use smaller batch size.")
        dice_trainer = DICETrainer(expert_data, env.action_space, args.batch_size, 400, args.gamma, device, features_to_remove=features_to_remove)
        dice_params = dict(dice_trainer=dice_trainer, dice_coeff=args.dice_coeff,
                           dice_n_epochs=args.dice_n_epochs, dice_train_every=args.dice_train_every)
        algo_params.update(dice_params)

    if args.algo == 'sac':
        model = ImSAC(SacMlpPolicy, env, **algo_params)
    elif args.algo == 'td3':
        model = ImTD3(Td3MlpPolicy, env, **algo_params)
    elif args.algo == 'ppo':
        model = ImPPO("MlpPolicy", env, **algo_params)
        # model = ImPPO(CustomActorCriticCnnPolicy, env, **algo_params)
    elif args.algo == 'dqn':
        model = ImDQN(CustomDqnCnnPolicy, env, **algo_params)

    #############
    # Callbacks #
    #############
    eval_callback = EvalCallback(eval_env=eval_env,
                                 eval_steps=args.eval_steps,
                                 desc='MID TRAINING: Evaluating policy.',
                                 wandb_logger=wandb if args.wandb else None,
                                 prefix=args.run_name)
    eval_every_callback = EveryNTimesteps(n_steps=args.eval_every, callback=eval_callback)

    save_model_callback = SaveModelCallback(run_name=args.run_name)
    tmp_callbacks = CallbackList([save_model_callback])
    if args.save_data:
        save_data_callback = SaveDataCallback(run_name=args.run_name, data_size=args.data_size, random=False)
        tmp_callbacks.callbacks.append(save_data_callback)
    end_of_training_callbacks = EveryNTimesteps(n_steps=args.total_timesteps, callback=tmp_callbacks)

    callbacks = CallbackList([eval_every_callback, end_of_training_callbacks])

    # Train Model
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', required=True, type=str)
    parser.add_argument('--env_name', default='rooms-v0', type=str)
    parser.add_argument('--partial_data', action='store_true')
    parser.add_argument('--algo', default='ppo', choices=['dqn', 'ppo', 'sac', 'td3'], type=str)
    parser.add_argument('--total_timesteps', default=1000000, type=int)
    parser.add_argument('--buffer_size', default=100000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--learning_rate', default=float(5e-5), type=float)

    # sparse mujoco params
    parser.add_argument('--max_force', default=1.0, type=float)
    parser.add_argument('--sparse', action='store_true')

    parser.add_argument('--eval_every', default=10000, type=int)
    parser.add_argument('--eval_steps', default=1000, type=int)

    parser.add_argument('--save_data',action='store_true')
    parser.add_argument('--data_size', default=1000000, type=int)

    parser.add_argument('--dice_coeff', default=1, type=float)
    parser.add_argument('--dice_n_epochs', default=0, type=int)
    parser.add_argument('--dice_train_every', default=10, type=int)
    parser.add_argument('--expert_load_name', default='', type=str)

    parser.add_argument('--wandb', action='store_true')

    # PPO parameters
    parser.add_argument('--n_envs', default=1, type=int)
    parser.add_argument('--n_steps', default=2048, type=int)
    parser.add_argument('--n_epochs', default=10, type=int)

    # SAC and TD3 parameters
    parser.add_argument('--learning_starts', default=1000, type=int)
    parser.add_argument('--train_freq', default=1000, type=int)
    parser.add_argument('--gradient_steps', default=1000, type=int)

    parsed_args = parser.parse_args()

    if parsed_args.dice_coeff < 0 or parsed_args.dice_coeff > 1:
        raise ValueError("dice_coeff must be a value in [0,1]")
    if parsed_args.total_timesteps < 1000:
        raise ValueError("total_times steps must at least 1000")

    run(parsed_args)