import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import copy
import glob
import os
import time
from collections import deque
import sys
import warnings

import gym
import pybulletgym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from src.dril.a2c_ppo_acktr import algo
from src.dril.a2c_ppo_acktr import utils
from src.dril.a2c_ppo_acktr.algo import gail
from src.dril.a2c_ppo_acktr.algo.behavior_cloning import BehaviorCloning
from src.dril.a2c_ppo_acktr.algo.ensemble import Ensemble
from src.dril.a2c_ppo_acktr.algo.dril import DRIL
from src.dril.a2c_ppo_acktr.arguments import get_args
from src.dril.a2c_ppo_acktr.envs import make_vec_envs
from src.dril.a2c_ppo_acktr.model import Policy
from src.dril.a2c_ppo_acktr.algo.gail import ExpertDataset
from src.dril.a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
import pandas as pd

import src.envs
from tqdm import tqdm
try:
    import wandb
except:
    wandb = None


def main():
    args = get_args()

    if args.debug:
        print('WARNING: DEBUG MODE')

    if args.wandb and not args.debug:
        if args.run_name:
            wandb.init(project="Transition Dependent Context", name=args.run_name, config=args)
        else:
            wandb.init(project="Transition Dependent Context", config=args)

    if args.seed == -1:
        args.seed = np.random.randint(2 ** 30 - 1)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.partial_data and args.env_name == 'rooms-v0':
        channels_to_remove = [3]  # this is only for rooms-v0
        print(f"Using partial context in data, removing channels {channels_to_remove}")
    else:
        channels_to_remove = []

    if args.system == 'philly':
        args.demo_data_dir = os.getenv('PT_OUTPUT_DIR') + '/demo_data/'
        args.save_model_dir = os.getenv('PT_OUTPUT_DIR') + '/trained_models/'
        args.save_results_dir = os.getenv('PT_OUTPUT_DIR') + '/trained_results/'


    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")


    # assistive = make_vec_envs(args.env_name, args.seed, args.num_processes, args.gamma,
    #                      args.log_dir, device, False, use_obs_norm=args.use_obs_norm,
    #                      max_steps=args.atari_max_steps)

    num_frame_stack = 1  # MAKE 4 FOR ATARI

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, device, num_frame_stack=num_frame_stack)

    dril_observation_shape = (envs.observation_space.shape[0] - len(channels_to_remove), *envs.observation_space.shape[1:])
    dril_actor_critic = Policy(
        dril_observation_shape,  # this is only for images
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    dril_actor_critic.to(device)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    dril_actor_critic = actor_critic

    # stores results
    main_results = []

    if args.behavior_cloning or args.dril or args.warm_start:
        file_name = os.path.join(args.experts_dir, "expert_data.h5")
        expert_dataset = ExpertDataset(file_name, args.ensemble_shuffle_type, args.debug)
        # expert_dataset = ExpertDataset(args.demo_data_dir, args.env_name,\
        #                    args.num_trajs, args.seed, args.ensemble_shuffle_type)
        bc_model_save_path = os.path.join(args.save_model_dir, 'bc')
        bc_file_name = f'bc_{args.env_name}_policy_ntrajs={args.num_trajs}_seed={args.seed}'
        #bc_file_name = f'{args.env_name}_bc_policy_ntraj={args.num_trajs}_seed={args.seed}'
        bc_model_path = os.path.join(bc_model_save_path, f'{bc_file_name}.model.pth')
        bc_results_save_path = os.path.join(args.save_results_dir, 'bc', f'{bc_file_name}.perf')

        bc_model = BehaviorCloning(dril_actor_critic, device, batch_size=args.bc_batch_size,
                                   lr=args.bc_lr, training_data_split=args.training_data_split,
                                   expert_dataset=expert_dataset, envs=envs, channels_to_remove=channels_to_remove)

        # Check if model already exist
        test_reward = None
        if os.path.exists(bc_model_path):
            best_test_params = torch.load(bc_model_path, map_location=device)
            print(f'*** Loading behavior cloning policy: {bc_model_path} ***')
        else:
            bc_results = []
            best_test_loss, best_test_model = np.float('inf'), None
            for bc_epoch in range(args.bc_train_epoch):
                train_loss = bc_model.update(update=True, data_loader_type='train')
                with torch.no_grad():
                    test_loss = bc_model.update(update=False, data_loader_type='test')
                #if test_loss < best_test_loss:
                #    best_test_loss = test_loss
                #    best_test_params = copy.deepcopy(dril_actor_critic.state_dict())
                if test_loss < best_test_loss:
                    print('model has improved')
                    best_test_loss = test_loss
                    best_test_params = copy.deepcopy(dril_actor_critic.state_dict())
                    patience = 20
                else:
                    patience -= 1
                    print('model has not improved')
                    if patience == 0:
                        print('model has not improved in 20 epochs, breaking')
                        break

                print(f'bc-epoch {bc_epoch}/{args.bc_train_epoch} | train loss: {train_loss:.4f}, test loss: {test_loss:.4f}')
            # Save the Behavior Cloning model and training results
            test_reward = evaluate(dril_actor_critic, None, envs, args.num_processes, device, num_episodes=10, channels_to_remove=channels_to_remove)
            bc_results.append({'epoch': bc_epoch, 'trloss':train_loss, 'teloss': test_loss,\
                        'test_reward': test_reward})

            torch.save(best_test_params, bc_model_path)
            df = pd.DataFrame(bc_results, columns=np.hstack(['epoch', 'trloss', 'teloss', 'test_reward']))
            df.to_csv(bc_results_save_path)

        # Load Behavior cloning model
        dril_actor_critic.load_state_dict(best_test_params)
        if test_reward is None:
            bc_model_reward = evaluate(dril_actor_critic, None, envs, args.num_processes, device, num_episodes=10, channels_to_remove=channels_to_remove)
        else:
            bc_model_reward = test_reward
        print(f'Behavior cloning model performance: {bc_model_reward}')
# If behavior cloning terminate the script early
        if args.behavior_cloning:
             sys.exit()
        # Reset the behavior cloning optimizer
        bc_model.reset()

    if args.dril:
        file_name = os.path.join(args.experts_dir, "expert_data.h5")
        expert_dataset = ExpertDataset(file_name, args.ensemble_shuffle_type, args.debug)

        # Train or load ensemble policy
        ensemble_policy = Ensemble(device=device, envs=envs,
                                   expert_dataset=expert_dataset,
                                   uncertainty_reward=args.dril_uncertainty_reward,
                                   ensemble_hidden_size=args.ensemble_hidden_size,
                                   ensemble_drop_rate=args.ensemble_drop_rate,
                                   ensemble_size=args.ensemble_size,
                                   ensemble_batch_size=args.ensemble_batch_size,
                                   ensemble_lr=args.ensemble_lr,
                                   num_ensemble_train_epoch=args.num_ensemble_train_epoch,
                                   num_trajs=args.num_trajs,
                                   seed=args.seed,
                                   env_name=args.env_name,
                                   training_data_split=args.training_data_split,
                                   save_model_dir=args.save_model_dir,
                                   save_results_dir=args.save_results_dir,
                                   channels_to_remove=channels_to_remove)

        # If only training ensemble
        if args.pretrain_ensemble_only:
            sys.exit()

        # Train or load behavior cloning policy
        dril_bc_model = bc_model

        dril = DRIL(device=device,envs=envs,ensemble_policy=ensemble_policy,
                    dril_bc_model=dril_bc_model, expert_dataset=expert_dataset,
                    ensemble_quantile_threshold=args.ensemble_quantile_threshold,
                    ensemble_size=args.ensemble_size, dril_cost_clip=args.dril_cost_clip,
                    env_name=args.env_name, num_dril_bc_train_epoch=args.num_dril_bc_train_epoch,
                    training_data_split=args.training_data_split, channels_to_remove=channels_to_remove)
    else:
        dril = None


    if args.algo == 'a2c':
        #TODO: Not sure why this is needed
        from src.dril.a2c_ppo_acktr import algo
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm,
            dril=dril)
    elif args.algo == 'ppo':
        #TODO: Not sure why this is needed
        from src.dril.a2c_ppo_acktr import algo
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
            dril=dril)
    elif args.algo == 'acktr':
        from src.dril.a2c_ppo_acktr import algo
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        # assert len(assistive.observation_space.shape) == 1
        discr = gail.DiscTrainer(state_dims=envs.observation_space.shape,
                                 num_actions=envs.action_space.n,
                                 hidden_dim=512,
                                 device=device,
                                 channels_to_remove=channels_to_remove)

        file_name = os.path.join(args.experts_dir, "expert_data.h5")
        expert_dataset = gail.ExpertDataset(file_name, "norm_shuffle", args.debug)
        gail_train_loader = expert_dataset.preprocess(1.0, args.gail_batch_size, None, channels_to_remove)['trdata']

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    channels_to_keep = [i for i in range(obs.shape[1]) if i not in channels_to_remove]

    episode_rewards = deque(maxlen=10)
    episode_uncertainty_rewards = deque(maxlen=10)
    running_uncertainty_reward = np.zeros(args.num_processes)
    gail_mean_reward = 0

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

    previous_action = None
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])
                if args.dril:
                    # NOTE: currently doesn't work with recurrent state
                    _, bc_action, _, _, bc_prob_dist = dril_actor_critic.act(rollouts.obs[step][:, channels_to_keep, :, :], None, None)
                    action_features = actor_critic.get_action(rollouts.obs[step])
                    # bc_action_features = dril_actor_critic.get_action(rollouts.obs[step])

            # Obser reward and next obs
            if isinstance(envs.action_space, gym.spaces.Box):
                clip_action = torch.clamp(action, float(envs.action_space.low[0]), float(envs.action_space.high[0]))
            else:
                clip_action = action

            if args.dril:
                # bc_reward = (prob_dist * (bc_prob_dist.log() - prob_dist.log())).sum(-1, keepdim=True)
                # bc_reward = (action == bc_action).float()
                # bc_reward = -F.cross_entropy(action_features, bc_action.flatten().long())
                # bc_reward = -torch.norm(prob_dist-bc_prob_dist,p=1,dim=1,keepdim=True)
                var_reward = dril.predict_reward(clip_action, obs[:, channels_to_keep, :, :], envs.action_space)
                # dril_reward = bc_reward.detach().cpu() + var_reward
                dril_reward = var_reward
                running_uncertainty_reward += dril_reward.view(-1).numpy()

            obs, env_reward, done, infos = envs.step(clip_action)

            if args.dril:
                reward = env_reward * args.add_env_reward + args.dril_coef * dril_reward
            else:
                reward = env_reward

            #for info in infos:
            for i, info in enumerate(infos):
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    episode_uncertainty_rewards.append(running_uncertainty_reward[i] / info['episode']['l'])
                    running_uncertainty_reward[i] = 0

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.dril and args.algo == 'ppo':
            # Normalize the rewards for ppo
            # (Implementation Matters in Deep RL: A Case Study on PPO and TRPO)
            # (https://openreview.net/forum?id=r1etN1rtPB)
            for step in range(args.num_steps):
                rollouts.rewards[step] = dril.normalize_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step], rollouts.rewards[step])

        if args.gail:
            #if j >= 10:
            #    assistive.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 10  # Warm up
            for _ in range(gail_epoch):
                try:
                    # Continous control task have obfilt
                    obfilt = utils.get_vec_normalize(envs)._obfilt
                except:
                    # CNN doesnt have obfilt
                    obfilt = None
                discr.update(gail_train_loader, rollouts, obfilt)

            gail_rewards = torch.zeros_like(rollouts.rewards)
            for step in range(args.num_steps):
                gail_rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma, rollouts.masks[step])
                if args.add_env_reward:
                    rollouts.rewards[step] += args.dril_coef * gail_rewards[step]
                else:
                    rollouts.rewards[step] = gail_rewards[step]
            gail_mean_reward = gail_rewards.mean().item()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_model_dir != "":
            save_path = os.path.join(args.save_model_dir, args.algo)
            model_file_name = f'{args.env_name}_policy_ntrajs={args.num_trajs}_seed={args.seed}'
            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, f'{model_file_name}.pt'))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Entropy {} \n "
                "Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f} "
                "mean/median U reward {:.4f}/{:.4f}\n"
                "mean gail reward {:.4f} \n\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        dist_entropy,
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), np.mean(episode_uncertainty_rewards),
                        np.median(episode_uncertainty_rewards),
                        gail_mean_reward))
            if args.wandb and not args.debug:
                wandb.log({"reward": np.mean(episode_rewards),
                           "dril reward": np.mean(episode_uncertainty_rewards),
                           "gail reward": gail_mean_reward},
                          step=total_num_steps)

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            if args.dril:
                ob_rms = None
            else:
                try:
                    ob_rms = utils.get_vec_normalize(envs).ob_rms
                except:
                    ob_rms = None

            print(f'ob_rms: {ob_rms}')
            test_reward = evaluate(actor_critic, None, envs, args.num_processes, device, num_episodes=10, channels_to_remove=[])
            main_results.append({'total_num_steps': total_num_steps, 'train_loss': 0,
                'test_loss': 0, 'test_reward':test_reward, 'num_trajs': args.num_trajs,
                'train_reward': np.mean(episode_rewards),
                'u_reward': np.mean(episode_uncertainty_rewards)})
            save_results(args, main_results, algo, args.dril, args.gail)


            if dril: algo ='dril'
            elif gail: algo ='gail'
            else: algo = args.algo
            save_path = os.path.join(args.save_model_dir, algo)
            file_name = f'{algo}_{args.env_name}_policy_ntrajs={args.num_trajs}_seed={args.seed}'

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, f"{file_name}.pt"))

    # # Final evaluation
    # try:
    #     ob_rms = utils.get_vec_normalize(assistive).ob_rms
    # except:
    #     ob_rms = None
    # test_reward = evaluate(actor_critic, ob_rms, args.env_name, args.seed,
    #          args.num_processes, eval_log_dir, device, num_episodes=10, atari_max_steps=args.atari_max_steps)
    # main_results.append({'total_num_steps': total_num_steps, 'train_loss': 0, 'test_loss': 0,\
    #                  'num_trajs': args.num_trajs, 'test_reward':test_reward,\
    #                  'train_reward': np.mean(episode_rewards),\
    #                  'u_reward': np.mean(episode_uncertainty_rewards)})
    # save_results(args, main_results, algo, args.dril, args.gail)


def save_results(args, main_results, algo, dril, gail):
    if dril: algo ='dril'
    elif gail: algo ='gail'
    else: algo = args.algo

    if dril:
        exp_name  = f'{algo}_{args.env_name}_ntraj={args.num_trajs}_'
        exp_name += f'ensemble_lr={args.ensemble_lr}_'
        exp_name += f'lr={args.bc_lr}_bcep={args.bc_train_epoch}_shuffle={args.ensemble_shuffle_type}_'
        exp_name += f'quantile={args.ensemble_quantile_threshold}_'
        exp_name += f'cost_{args.dril_cost_clip}_seed={args.seed}.perf'
    elif gail:
        exp_name  = f'{algo}_{args.env_name}_ntraj={args.num_trajs}_'
        exp_name += f'gail_lr={args.gail_disc_lr}_lr={args.bc_lr}_bcep={args.bc_train_epoch}_'
        exp_name += f'gail_reward_type={args.gail_reward_type}_seed={args.seed}.perf'
    else:
        exp_name  = f'{algo}_{args.env_name}.pef'

    results_save_path = os.path.join(args.save_results_dir, f'{algo}', f'{exp_name}')
    df = pd.DataFrame(main_results, columns=np.hstack(['x', 'total_num_steps', 'train_loss', 'test_loss', 'train_reward', 'test_reward', 'num_trajs', 'u_reward']))
    df.to_csv(results_save_path)

if __name__ == "__main__":
    main()
