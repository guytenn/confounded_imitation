from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.policy.sample_batch import SampleBatch
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

from ray.rllib.evaluation.postprocessing import compute_advantages


class ImitationModule:
    def __init__(self, dice_config):
        self.is_recsim = dice_config['env_name'] == 'RecSim-v2'
        self.expert_path = dice_config['expert_path']
        self.gamma = dice_config['gamma']
        self.features_to_remove = dice_config['features_to_remove']
        self.dice_coef = dice_config['dice_coef']
        self.lr = dice_config['lr']
        self.state_dim = dice_config['state_dim']
        self.observation_space = dice_config['observation_space']
        self.action_space = dice_config['action_space']
        self.hidden_dim = dice_config['hidden_dim']
        self.standardize = dice_config["standardize"]
        self.airl = dice_config["airl"]

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
            action_shape = self.state_dim
        else:
            action_shape = self.action_space.shape[0]

        if self.is_recsim:
            input_shape = len(self.features_to_keep) + action_shape + 1
        else:
            input_shape = len(self.features_to_keep) + action_shape + 1
        self.g = self._create_fc_net((input_shape, self.hidden_dim, self.hidden_dim, 1), "relu", name="g_net")
        opt_params = list(self.g.parameters())
        self.g = self.g.to(self.device)
        if self.airl:
            self.h = self._create_fc_net((len(self.features_to_keep), self.hidden_dim, self.hidden_dim, 1), "relu", name="h_net")
            self.h = self.h.to(self.device)
            opt_params += list(self.h.parameters())

        self.optimizer = torch.optim.Adam(opt_params, lr=self.lr)

        self.mean = None
        self.var = None
        self.count = None
        self.returns = None


    def __call__(self, samples: SampleBatch) -> SampleBatch:
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

        # TRAIN DICE
        self._train(samples_input)

        # UPDATE REWARD AND RECALCULATE ADVANTAGE
        rollouts = samples_batch.split_by_episode()
        start_idx = 0
        for i in range(len(rollouts)):
            rollouts[i][SampleBatch.REWARDS][:-1] = \
                (1 - self.dice_coef) * rollouts[i][SampleBatch.REWARDS][:-1] + \
                self.dice_coef * reward_bonus[start_idx:start_idx+len(rollouts[i])-1]
            rollouts[i] = compute_advantages(rollouts[i],
                                             rollouts[i][SampleBatch.REWARDS][-1],
                                             0.99, 0.95, True, True)
            start_idx += len(rollouts[i])
        samples_batch = SampleBatch.concat_samples(rollouts)

        for i, info in enumerate(samples_batch[SampleBatch.INFOS]):
            info.update({'imitation_reward': reward_bonus[i]})

        # Return the postprocessed sample batch (with the corrected rewards).
        return samples


    def _forward_model(self, obs, actions, next_obs, dones):
        if isinstance(self.action_space, spaces.Discrete):
            actions = to_onehot(actions.flatten(), self.action_space.n).clone()
        rs = self.g(torch.cat((obs, dones.unsqueeze(-1).float(), actions), dim=1)).flatten()
        if self.airl:
            vs = self.h(obs).flatten()
            next_vs = self.h(next_obs).flatten()
            res = rs + self.gamma * (1 - dones.float()) * next_vs - vs
        else:
            res = rs
        return torch.sigmoid(res)

    def _train(self, samples):
        if self.is_recsim:
            batch_size = len(samples)
            dice_epochs = 1
            alpha = 0.9
        else:
            batch_size = 128
            dice_epochs = 50
            alpha = 0.9

        for _ in range(dice_epochs):
            for _ in range(len(samples) // batch_size):
                expert_data = self.expert_buffer.sample(batch_size)

                expert_d = self._forward_model(expert_data.observations[:, self.features_to_keep],
                                               expert_data.actions,
                                               expert_data.next_observations[:, self.features_to_keep],
                                               expert_data.dones)

                # if isinstance(self.action_space, spaces.Discrete):
                #     policy_actions = to_onehot(policy_actions.flatten(), self.model.action_dim)
                idx = np.random.choice(len(samples), batch_size)
                policy_d = self._forward_model(torch.from_numpy(samples[SampleBatch.OBS][idx][:, self.features_to_keep]).to(self.device),
                                               torch.from_numpy(samples[SampleBatch.ACTIONS][idx]).to(self.device),
                                               torch.from_numpy(samples[SampleBatch.NEXT_OBS][idx][:, self.features_to_keep]).to(self.device),
                                               torch.from_numpy(samples[SampleBatch.DONES][idx]).to(self.device))

                # loss = alpha * torch.pow(expert_d, 2).mean() + (1 - alpha) * torch.pow(policy_d, 2).mean() - 2 * policy_d.mean()
                # kl divergence
                # loss = torch.log(0.9 * torch.exp(expert_d).mean() + 0.1 * torch.exp(policy_d).mean()) - policy_d.mean()
                # GAIL loss
                loss = -torch.log(1 - policy_d + float(1e-6)).mean() - torch.log(expert_d + float(1e-6)).mean()
                # loss = -F.logsigmoid(-policy_d).mean() - F.logsigmoid(expert_d).mean()
                # Perform an optimizer step.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def _predict_reward(self, samples):
        policy_obs = torch.from_numpy(samples[SampleBatch.OBS][:, self.features_to_keep]).to(self.device)
        policy_actions = torch.from_numpy(samples[SampleBatch.ACTIONS]).to(self.device)
        policy_next_obs = torch.from_numpy(samples[SampleBatch.NEXT_OBS][:, self.features_to_keep]).to(self.device)
        policy_dones = torch.from_numpy(samples[SampleBatch.DONES]).float().to(self.device)

        policy_d = self._forward_model(policy_obs, policy_actions, policy_next_obs, policy_dones)
        # policy_d = torch.sigmoid(policy_d)
        # reward_bonus = -policy_d
        reward_bonus = -torch.log(1.0 - policy_d * (1.0 - float(1e-6)))

        reward_bonus = reward_bonus.detach().cpu().numpy()
        if self.standardize:
            reward_bonus = reward_bonus / max(1e-4, reward_bonus.std())

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
