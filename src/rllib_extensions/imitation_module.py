from src.rllib.models.torch.misc import SlimFC
from src.rllib.utils.framework import try_import_torch
from src.rllib.policy.sample_batch import SampleBatch
torch, nn = try_import_torch()
F = None
if nn is not None:
    F = nn.functional

from src.data.expert_data import ExpertData
from src.data.utils import load_h5_dataset
from src.sb3_extensions.buffers import CustomReplayBuffer
import numpy as np
import GPUtil

from src.rllib_extensions.recsim_wrapper import restore_samples
import gym.spaces as spaces
from src.common.utils import to_onehot

from src.rllib.evaluation.postprocessing import compute_gae_for_sample_batch
import nevergrad as ng
import copy


class ImitationModule:
    def __init__(self, workers, dice_config):
        self.workers = workers
        self.imitation_method = dice_config['imitation_method']
        self.is_recsim = dice_config['env_name'] == 'RecSim-v2'
        self.is_rooms = dice_config['env_name'] == 'rooms-v0'
        self.expert_path = dice_config['expert_path']
        self.gamma = dice_config['gamma']
        self.features_to_remove = dice_config['features_to_remove']
        self.adaptive_coef = dice_config['adaptive_coef']
        self.dice_coef = dice_config['dice_coef']
        self.lr = dice_config['lr']
        self.state_dim = dice_config['state_dim']
        self.observation_space = dice_config['observation_space']
        self.action_space = dice_config['action_space']
        self.hidden_dim = dice_config['hidden_dim']
        self.standardize = dice_config["standardize"]
        self.resampling_coef = dice_config['resampling_coef']
        self.airl = dice_config["airl"]
        self.decaying_coef = dice_config["decaying_coef"]
        self.n_samples = 0

        self.features_to_keep = [i for i in range(self.state_dim) if i not in self.features_to_remove]

        try:
            deviceIds = GPUtil.getFirstAvailable(order='memory', maxLoad=0.95, maxMemory=0.95)
            self.device = torch.device(f'cuda:{deviceIds[0]}')
        except:
            self.device = torch.device('cpu')

        self.expert_buffer = None

        data = load_h5_dataset(self.expert_path)
        if 'dones' not in data.keys():
            data['dones'] = np.zeros(len(data['actions']))
        self.expert_buffer = ExpertData(data['states'].astype('float32'), data['actions'].astype('float32'),
                                        data['dones'], device=self.device)

        if self.is_recsim:
            # action_shape = self.action_space.n
            # action_shape = self.action_space.nvec[0]
            action_shape = self.state_dim
        elif self.is_rooms:
            action_shape = self.action_space.n
        else:
            action_shape = self.action_space.shape[0]

        input_shape = len(self.features_to_keep) + action_shape + 1
        self.g = self._create_fc_net((input_shape, self.hidden_dim, self.hidden_dim, 1), "relu", name="g_net")
        opt_params = list(self.g.parameters())
        self.g = self.g.to(self.device)
        self.g_clone = copy.deepcopy(self.g)
        clone_params = list(self.g_clone.parameters())
        if self.airl:
            self.h = self._create_fc_net((len(self.features_to_keep), self.hidden_dim, self.hidden_dim, 1), "relu", name="h_net")
            self.h = self.h.to(self.device)
            self.h_clone = copy.deepcopy(self.h)
            opt_params += list(self.h.parameters())
            clone_params += list(self.h_clone.parameters())
        else:
            self.h = None
            self.h_clone = None

        self.optimizer = torch.optim.Adam(opt_params, lr=self.lr)
        self.optimizer_clone = torch.optim.Adam(clone_params, lr=self.lr)

        self.mean = None
        self.var = None
        self.count = None
        self.returns = None

    def __call__(self, samples: SampleBatch) -> SampleBatch:
        policy = self.workers.local_worker().policy_map['default_policy']

        if self.is_recsim:
            samples_batch = samples
            user, selected_doc = restore_samples(samples_batch[SampleBatch.OBS],
                                                 samples_batch[SampleBatch.ACTIONS],
                                                 self.observation_space)
            samples_input = {SampleBatch.OBS: user,
                             SampleBatch.ACTIONS: selected_doc,
                             SampleBatch.DONES: samples_batch[SampleBatch.DONES],
                             SampleBatch.NEXT_OBS: user}
        else:
            samples_batch = samples_input = samples
        # ESTIMATE REWARD BONUS
        reward_bonus = self._predict_reward(samples_input)

        # ESTIMATE TRAJECTORY SAMPLE MINIMIZER
        reweighted = 0
        if self.dice_coef < 1 and self.resampling_coef > 0:
            for param, target_param in zip(self.g.parameters(), self.g_clone.parameters()):
                target_param.data.copy_(param.data)
            n_traj = self.expert_buffer.dones.sum()
            cov_sensitivity = self.resampling_coef  # number between 0 and 1. Higher means will attempt larger covariate shifts sampling
            instrum = ng.p.Instrumentation(ng.p.Array(shape=(n_traj.item(),)).set_bounds(lower=-10, upper=10))
            optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=100, num_workers=1)
            try:
                weights = optimizer.minimize(lambda w: self._sampler_trainer(samples_input, cov_sensitivity, w)).value[0][0]
                projected_weights = weights / 10. + 1. / cov_sensitivity
                sample_weights = torch.repeat_interleave(torch.from_numpy(projected_weights).to(self.device),
                                                         self.expert_buffer.traj_lengths)
                reweighted = 1
            except:
                sample_weights = None
        else:
            sample_weights = None


        # TRAIN DICE
        self._train(samples_input, sample_weights)

        # UPDATE DICE COEF IF NEEDED
        if self.adaptive_coef:
            r_mean = np.mean(samples_batch[SampleBatch.REWARDS])
            r_bonus_mean = np.mean(reward_bonus)
            dice_coef = r_mean / (r_mean + r_bonus_mean)
        else:
            dice_coef = self.dice_coef

        if self.decaying_coef > 0:
            self.n_samples += len(reward_bonus)
            dice_coef *= self.decaying_coef / np.sqrt(self.n_samples)

        # UPDATE REWARD AND RECALCULATE ADVANTAGE
        rollouts = samples_batch.split_by_episode()
        start_idx = 0
        for i in range(len(rollouts)):
            rollouts[i][SampleBatch.REWARDS] = \
                (1 - dice_coef) * rollouts[i][SampleBatch.REWARDS] + \
                dice_coef * reward_bonus[start_idx:start_idx+len(rollouts[i])]
            # rollouts[i] = compute_advantages(rollouts[i], 0, 0.99, 0.95, True, True)
            rollouts[i] = compute_gae_for_sample_batch(policy, rollouts[i])
            start_idx += len(rollouts[i])
        samples_batch = SampleBatch.concat_samples(rollouts)

        # Send back extra info for metrics
        policy.config['extra_info'] = {"imitation_reward": reward_bonus.mean(),
                                       "augmented_total_reward": samples_batch[SampleBatch.REWARDS].mean(),
                                       "resamp_weights": reweighted}

        # Return the postprocessed sample batch (with the corrected rewards).
        return samples_batch

    def _forward_model(self, obs, actions, next_obs, dones, use_clone=False):
        if use_clone:
            g = self.g_clone
            h = self.h_clone
        else:
            g = self.g
            h = self.h
        if isinstance(self.action_space, spaces.Discrete):
            actions = to_onehot(actions.flatten(), self.action_space.n).clone()

        rs = g(torch.cat((obs, dones.unsqueeze(-1).float(), actions), dim=1)).flatten()
        if self.airl:
            vs = h(obs).flatten()
            next_vs = h(next_obs).flatten()
            res = rs + self.gamma * (1 - dones.float()) * next_vs - vs
        else:
            res = rs

        if self.imitation_method == 'gail':
            return torch.sigmoid(res)
        elif self.imitation_method == 'tv':
            return 0.5 * torch.tanh(res)
        else:
            return res

    def _sampler_trainer(self, samples, cov_sensitivity, weights):
        weights = weights / 10. + 1. / cov_sensitivity
        sample_weights = torch.repeat_interleave(torch.from_numpy(weights).to(self.device),
                                                 self.expert_buffer.traj_lengths)
        return self._train(samples, sample_weights, use_clone=True).item()

    def _train(self, samples, sample_weights=None, use_clone=False):
        alpha = 0.9

        if use_clone:
            batch_size = 512
            dice_epochs = 1
            n_samples = 10
            optimizer = self.optimizer_clone
        else:
            batch_size = 128
            dice_epochs = 50
            n_samples = len(samples[SampleBatch.OBS]) // batch_size
            optimizer = self.optimizer

        for _ in range(dice_epochs):
            for _ in range(n_samples):
                expert_data = self.expert_buffer.sample(batch_size, weights=sample_weights)

                expert_d = self._forward_model(expert_data.observations[:, self.features_to_keep],
                                               expert_data.actions,
                                               expert_data.next_observations[:, self.features_to_keep],
                                               expert_data.dones,
                                               use_clone=use_clone)

                # if isinstance(self.action_space, spaces.Discrete):
                #     policy_actions = to_onehot(policy_actions.flatten(), self.model.action_dim)
                idx = np.random.choice(len(samples), batch_size)
                policy_d = self._forward_model(torch.from_numpy(samples[SampleBatch.OBS][idx][:, self.features_to_keep]).to(self.device),
                                               torch.from_numpy(samples[SampleBatch.ACTIONS][idx]).to(self.device),
                                               torch.from_numpy(samples[SampleBatch.NEXT_OBS][idx][:, self.features_to_keep]).to(self.device),
                                               torch.from_numpy(samples[SampleBatch.DONES][idx]).to(self.device),
                                               use_clone=use_clone)

                if self.imitation_method == 'gail':
                    loss = -torch.log(1 - policy_d + float(1e-6)).mean() - torch.log(expert_d + float(1e-6)).mean()
                elif self.imitation_method == 'kl':
                    loss = torch.log(0.9 * torch.exp(expert_d).mean() + 0.1 * torch.exp(policy_d).mean()) - policy_d.mean()
                elif self.imitation_method == 'chi':
                    loss = alpha * torch.pow(expert_d, 2).mean() + (1 - alpha) * torch.pow(policy_d, 2).mean() - 2 * policy_d.mean()
                elif self.imitation_method == 'tv':
                    loss = alpha * expert_d.mean() + (1 - alpha) * policy_d.mean() - policy_d.mean()
                else:
                    raise ValueError(f'Unknown imitation method {self.imitation_method}')

                # loss = -F.logsigmoid(-policy_d).mean() - F.logsigmoid(expert_d).mean()
                # Perform an optimizer step.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return loss

    def _predict_reward(self, samples):
        policy_obs = torch.from_numpy(samples[SampleBatch.OBS][:, self.features_to_keep]).to(self.device)
        policy_actions = torch.from_numpy(samples[SampleBatch.ACTIONS]).to(self.device)
        policy_next_obs = torch.from_numpy(samples[SampleBatch.NEXT_OBS][:, self.features_to_keep]).to(self.device)
        policy_dones = torch.from_numpy(samples[SampleBatch.DONES]).float().to(self.device)

        policy_d = self._forward_model(policy_obs, policy_actions, policy_next_obs, policy_dones)

        if self.imitation_method == 'gail':
            reward_bonus = -torch.log(1.0 - policy_d * (1.0 - float(1e-6)))
        else:
            # Comment: It might work better to use sigmoid and use same log reward as in gail
            reward_bonus = -policy_d

        reward_bonus = reward_bonus.detach().cpu().numpy()
        if self.standardize:
            reward_bonus = (reward_bonus - reward_bonus.mean()) / max(1e-4, reward_bonus.std())

        return reward_bonus

    def _create_fc_net(self, layer_dims, activation, name=None):
        """Given a list of layer dimensions (incl. input-dim), creates FC-net.

        Args:
            layer_dims (Tuple[int]): Tuple of layer dims, including the input
                dimension.
            activation (str): An activation specifier string (e.g. "relu").

        Examples:
            If layer_dims is [4,8,6] we'll have a two layer net: 4->8 (8 nodes)
            and 8->6 (6 nodes), where the second layer (6 nodes) does not have
            an activation anymore. 4 is the input dimension.
        """
        layers = []
        for i in range(len(layer_dims) - 1):
            act = activation if i < len(layer_dims) - 2 else None
            layers.append(
                SlimFC(in_size=layer_dims[i],
                       out_size=layer_dims[i + 1],
                       initializer=torch.nn.init.xavier_uniform_,
                       activation_fn=act))
        return nn.Sequential(*layers)
