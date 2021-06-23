import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.running_mean_std import RunningMeanStd
from src.common.utils import to_onehot
import gym.spaces as spaces


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class DICE(nn.Module):
    def __init__(self, state_dims, action_dim, hidden_dim, gamma, device):
        super(DICE, self).__init__()

        self.gamma = gamma
        self.device = device
        self.action_dim = action_dim

        if len(state_dims) == 3:
            init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                   constant_(x, 0), nn.init.calculate_gain('relu'))

            cnn_base = nn.Sequential(
                init_(nn.Conv2d(state_dims[0], 32, 8, stride=4)), nn.ReLU(),
                init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
                init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            ).to(device)

            # Compute shape by doing one forward pass
            with torch.no_grad():
                n_flatten = cnn_base(torch.zeros(1, *state_dims).to(device)).shape[1]

            g_mlp = nn.Sequential(
                nn.Linear(n_flatten, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1)).to(device)

            h_mlp = nn.Sequential(
                nn.Linear(n_flatten, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1)).to(device)

            self.g = nn.Sequential(cnn_base, g_mlp)
            self.h = nn.Sequential(cnn_base, h_mlp)
        else:
            n_flatten = state_dims[0]

            self.g = nn.Sequential(
                nn.Linear(n_flatten, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1)).to(device)

            self.h = nn.Sequential(
                nn.Linear(n_flatten, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1)).to(device)

    def forward(self, state, next_state, dones):
        # return self.g(state).flatten()

        rs = self.g(state)
        vs = self.h(state)
        next_vs = self.h(next_state)
        return rs.flatten() + self.gamma * (1 - dones.float()) * next_vs.flatten() - vs.flatten()

        # if self.cnn_base is not None:
        #     state = self.cnn_base(state)
        # return self.trunk(torch.cat([state, action], dim=1))


class DICETrainer(nn.Module):
    def __init__(self, expert_buffer, action_space, batch_size, hidden_dim, gamma, device, features_to_remove=[]):
        super(DICETrainer, self).__init__()

        self.gamma = gamma
        self.device = device

        self.expert_buffer = expert_buffer
        sample = expert_buffer.sample(1, env=None)

        if len(sample.observations.shape) == 2:  # (b, x)
            state_dims = (sample.observations.shape[1] - len(features_to_remove),)
        elif len(sample.observations.shape) == 4:  # (b, c, w, h)
            channel_dim = sample.observations.shape[1] - len(features_to_remove)
            state_dims = (channel_dim, sample.observations.shape[2], sample.observations.shape[3])
        else:
            raise ValueError(f"Unsupported observation shape {sample.observations.shape}.")
        if isinstance(action_space, spaces.Discrete):  # discrete
            self.action_dim = action_space.n
        else:  # continuous
            self.action_dim = sample.actions.shape[1]
        self.action_space = action_space

        self.model = DICE(state_dims, self.action_dim, hidden_dim, gamma, device)

        self.batch_size = batch_size

        self.optimizer = torch.optim.Adam(self.model.parameters())

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

        self.features_to_keep = [i for i in range(sample.observations.shape[1]) if i not in features_to_remove]

    def train_step(self, replay_buffer, dice_n_epochs=10, normalize_env=None):
        self.train()

        new_weights = torch.rand(len(self.expert_buffer.weights), device=self.expert_buffer.weights.device)
        for _ in range(dice_n_epochs):
            for policy_data in replay_buffer.get(self.batch_size):
                expert_data_aug = self.expert_buffer.sample(self.batch_size, env=normalize_env, weights=new_weights)
                expert_data_orig = self.expert_buffer.sample(self.batch_size, env=normalize_env)

                policy_states, policy_actions = policy_data.observations[:, self.features_to_keep], policy_data.actions
                # if isinstance(self.action_space, spaces.Discrete):
                #     policy_actions = to_onehot(policy_actions.flatten(), self.model.action_dim)
                policy_d = self.model(policy_states[:-1], policy_states[1:], policy_data.dones[:-1])

                loss_tmp = torch.zeros(2, device=self.device)
                for i, expert_data in enumerate([expert_data_orig, expert_data_aug]):
                    expert_states, expert_actions = expert_data.observations[:, self.features_to_keep], expert_data.actions
                    # if isinstance(self.action_space, spaces.Discrete):
                    #     expert_actions = to_onehot(expert_actions.flatten(), self.model.action_dim)
                    expert_d = self.model(expert_states[:-1], expert_states[1:], expert_data.dones[:-1])

                    # chi divergence
                    loss_tmp[i] = 0.9 * torch.pow(expert_d, 2).mean() + 0.1 * torch.pow(policy_d, 2).mean() - 2*policy_d.mean()

                loss_idx = torch.argmin(torch.abs(loss_tmp))
                # loss_idx = torch.tensor(1)
                loss = loss_tmp[loss_idx]
                # kl divergence
                # loss = torch.log(0.9 * torch.exp(expert_d).mean() + 0.1 * torch.exp(policy_d).mean()) - policy_d.mean()
                # GAIL loss
                # loss = -F.logsigmoid(-policy_d).mean() - F.logsigmoid(expert_d).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        if loss_idx.item() == 1:
            expert_data.weights = new_weights

    def predict_reward(self, states, dones, next_states, log_pi=None, update_rms=True):
        with torch.no_grad():
            self.eval()
            # if isinstance(self.action_space, spaces.Discrete):
            #     action = to_onehot(actions.flatten(), self.model.action_dim)
            d = self.model(states[:, self.features_to_keep], next_states[:, self.features_to_keep], dones)
            # reward = -F.logsigmoid(-(d - log_pi))
            #
            # return reward

            reward = -torch.tanh(d)

            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * self.gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            return reward / np.sqrt(self.ret_rms.var + 1e-8)
