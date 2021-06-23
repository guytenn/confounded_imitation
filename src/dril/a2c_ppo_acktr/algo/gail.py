import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import autograd

from stable_baselines3.common.running_mean_std import RunningMeanStd

from src.common.utils import get_keys
from src.dril.a2c_ppo_acktr.utils import init
from src.dril.a2c_ppo_acktr.model import Flatten
from src.common.utils import to_onehot

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class Discriminator(nn.Module):
    def __init__(self, state_dims, num_actions, hidden_dim, device):
        super(Discriminator, self).__init__()

        self.device = device
        self.num_actions = num_actions

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.cnn_base = nn.Sequential(
            init_(nn.Conv2d(state_dims[0], 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
        ).to(device)

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn_base(torch.zeros(1, *state_dims).to(device)).shape[1]

        # self.trunk = nn.Sequential(
        #     nn.Linear(num_actions + n_flatten, hidden_dim), nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        #     nn.Linear(hidden_dim, 1)).to(device)

        self.trunk = nn.Linear(n_flatten, 1).to(device)

    def forward(self, state):
        state_features = self.cnn_base(state)
        return self.trunk(state_features)
        # state_action = torch.cat([state_features, one_hot_action], dim=1)
        # output = self.trunk(state_action)
        # return output


class DiscTrainer(nn.Module):
    def __init__(self, state_dims, num_actions, hidden_dim, device, channels_to_remove=[]):
        super(DiscTrainer, self).__init__()

        self.device = device

        partial_state_dims = (state_dims[0] - len(channels_to_remove), *state_dims[1:])
        self.model = Discriminator(partial_state_dims, num_actions, hidden_dim, device)

        self.optimizer = torch.optim.Adam(self.model.parameters())

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

        self.channels_to_keep = [i for i in range(state_dims[0]) if i not in channels_to_remove]

    def compute_grad_pen(self,
                         expert_state,
                         expert_one_hot_action,
                         policy_state,
                         policy_one_hot_action,
                         lambda_=10):
        # alpha = torch.rand(expert_state.size(0), 1)
        # alpha_state = alpha.expand_as(expert_state).to(expert_state.device)
        # alpha_action = alpha.expand_as(expert_one_hot_action).to(expert_one_hot_action.device)
        alpha_state = torch.rand_like(expert_state).to(expert_state.device)
        alpha_action = torch.rand_like(expert_one_hot_action.float()).to(expert_state.device)

        mixup_state = alpha_state * expert_state + (1 - alpha_state) * policy_state
        mixup_action = alpha_action * expert_one_hot_action + (1 - alpha_action) * policy_one_hot_action
        mixup_state.requires_grad = True
        mixup_action.requires_grad = True

        # mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        # mixup_data.requires_grad = True

        # disc = self.model(mixup_state, mixup_action)
        disc = self.model(mixup_state)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            # inputs=[mixup_state, mixup_action],
            inputs=[mixup_state],
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update(self, expert_loader, rollouts, obsfilt=None):
        self.train()

        policy_data_generator = rollouts.feed_forward_generator(
            None, mini_batch_size=expert_loader.batch_size)

        loss = 0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader,
                                              policy_data_generator):
            policy_state, policy_action = policy_batch[0][:, self.channels_to_keep, :, :], policy_batch[2]
            policy_one_hot_action = to_onehot(policy_action.flatten(), self.model.num_actions)
            # policy_d = self.model(policy_state, policy_one_hot_action)
            policy_d = self.model(policy_state)

            expert_state, expert_action = expert_batch
            if obsfilt is not None:
                expert_state = obsfilt(expert_state.numpy(), update=False)
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)
            expert_one_hot_action = to_onehot(expert_action.flatten(), self.model.num_actions)
            # expert_d = self.model(expert_state, expert_one_hot_action)
            expert_d = self.model(expert_state)

            # expert_loss = F.binary_cross_entropy_with_logits(
            #     expert_d,
            #     torch.ones(expert_d.size()).to(self.device))
            # policy_loss = F.binary_cross_entropy_with_logits(
            #     policy_d,
            #     torch.zeros(policy_d.size()).to(self.device))
            #
            # gail_loss = expert_loss + policy_loss
            # grad_pen = self.compute_grad_pen(expert_state, expert_one_hot_action,
            #                                  policy_state, policy_one_hot_action)
            #
            # loss += (gail_loss + grad_pen).item()
            loss = 0.9 * torch.pow(expert_d, 2).mean() + 0.1 * torch.pow(policy_d, 2).mean() - 2*policy_d.mean()
            n += 1

            self.optimizer.zero_grad()
            # (gail_loss + grad_pen).backward()
            loss.backward()
            self.optimizer.step()
        return loss / n

    def predict_reward(self, state, action, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.eval()
            one_hot_action = to_onehot(action.flatten(), self.model.num_actions)
            # d = self.model(state[:, self.channels_to_keep, :, :], one_hot_action)
            d = self.model(state[:, self.channels_to_keep, :, :])
            # s = torch.clamp(torch.sigmoid(d), 0.01, 0.99)
            # reward = -(1 - s).log()
            # reward = s
            # reward = s.log() - (1 - s).log()
            # reward = (s+1e-8).log() - (1 - s + 1e-8).log()
            reward = -d
            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)


class DiscriminatorCNN(nn.Module):
    def __init__(self, obs_shape, hidden_dim, num_actions, device, disc_lr,\
                 gail_reward_type=None, envs=None):
        super(DiscriminatorCNN, self).__init__()

        self.device = device

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.num_actions = num_actions
        self.action_emb = nn.Embedding(num_actions, num_actions).cuda()
        num_inputs = obs_shape.shape[0] + num_actions


        self.cnn = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_dim)), nn.ReLU()).to(device)


        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)).to(device)

        self.cnn.train()
        self.trunk.train()

        self.optimizer = torch.optim.Adam(list(self.trunk.parameters()) + list(self.cnn.parameters()), lr=disc_lr)

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

        self.reward_type = gail_reward_type

    def compute_grad_pen(self,
                         expert_state,
                         expert_action,
                         policy_state,
                         policy_action,
                         lambda_=10):
        alpha = torch.rand(expert_state.size(0), 1)

        '''
        expert_data = torch.cat([expert_state, expert_action], dim=1)
        policy_data = torch.cat([policy_state, policy_action], dim=1)
        '''

        expert_data = self.combine_states_actions(expert_state, expert_action, detach=True)
        policy_data = self.combine_states_actions(policy_state, policy_action, detach=True)

        alpha = alpha.view(-1, 1, 1, 1).expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True

        disc = self.trunk(self.cnn(mixup_data))
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def combine_states_actions(self, states, actions, detach=False):
        batch_size, height, width = states.shape[0], states.shape[2], states.shape[3]
        action_emb = self.action_emb(actions).squeeze()
        action_emb = action_emb.view(batch_size, self.num_actions, 1, 1).expand(batch_size, self.num_actions, height, width)
        if detach:
            action_emb = action_emb.detach()
        state_actions = torch.cat((states / 255.0, action_emb), dim=1)
        return state_actions

    def update(self, expert_loader, rollouts, obsfilt=None):
        self.train()

        policy_data_generator = rollouts.feed_forward_generator(
            None, mini_batch_size=expert_loader.batch_size)

        loss = 0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader,
                                              policy_data_generator):
            policy_state, policy_action = policy_batch[0], policy_batch[2]
            policy_data = self.combine_states_actions(policy_state, policy_action)
            policy_d = self.trunk(self.cnn(policy_data))

            expert_state, expert_action = expert_batch

            if obsfilt is not None:
                expert_state = obsfilt(expert_state.numpy(), update=False)
                expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)
            expert_state = expert_state.to(self.device)

            expert_data = self.combine_states_actions(expert_state, expert_action)

            expert_d = self.trunk(self.cnn(expert_data))

            expert_loss = F.binary_cross_entropy_with_logits(
                expert_d,
                torch.ones(expert_d.size()).to(self.device))
            policy_loss = F.binary_cross_entropy_with_logits(
                policy_d,
                torch.zeros(policy_d.size()).to(self.device))

            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen(expert_state, expert_action,
                                             policy_state, policy_action)

            loss += (gail_loss + grad_pen).item()
            n += 1
            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()
        return loss / n

    def predict_reward(self, state, action, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.eval()
            policy_data = self.combine_states_actions(state, action)
            d = self.trunk(self.cnn(policy_data))
            s = torch.sigmoid(d)

            if self.reward_type == 'unbias':
                reward = s.log() - (1 - s).log()
            elif self.reward_type == 'favor_zero_reward':
                reward = reward = s.log()
            elif self.reward_type == 'favor_non_zero_reward':
                reward = - (1 - s).log()

            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)


class ExpertDataset:
    def __init__(self, file_path, ensemble_shuffle_type, debug=False):
        try:
            dataset_file = h5py.File(file_path, 'r')
        except:
            raise ValueError(f"No such file {file_path}")
        self.dataset = {k: dataset_file[k][:] for k in get_keys(dataset_file)}
        dataset_file.close()

        self.length = len(self.dataset['actions'])
        self.ensemble_shuffle_type = ensemble_shuffle_type
        self.debug = debug

    def preprocess(self, training_data_split, batch_size, ensemble_size, channels_to_remove=[]):
        states = torch.from_numpy(np.delete(self.dataset['states'], channels_to_remove, axis=1))
        actions = torch.from_numpy(self.dataset['actions'])

        perm = torch.randperm(self.length)
        states = states[perm]
        actions = actions[perm]
        if self.debug:
            states = states[0:1000]
            actions = actions[0:1000]
            self.length = len(actions)

        n_train = int(self.length*training_data_split)
        obs_train = states[:n_train]
        acs_train = actions[:n_train]
        obs_test = states[n_train:]
        acs_test = actions[n_train:]

        if self.ensemble_shuffle_type == 'norm_shuffle' or ensemble_size is None:
            shuffle = True
        elif self.ensemble_shuffle_type == 'no_shuffle' and ensemble_size is not None:
            shuffle = False
        elif self.ensemble_shuffle_type == 'sample_w_replace' and ensemble_size is not None:
            print('***** sample_w_replace (this make take a while) *****')
            # sample with replacement
            obs_train_resamp, acs_train_resamp = [], []
            for _ in tqdm(range(n_train * ensemble_size), desc='Resampling Dataset'):
                indx = np.random.randint(0, n_train - 1)
                obs_train_resamp.append(obs_train[indx])
                acs_train_resamp.append(acs_train[indx])
            obs_train = torch.stack(obs_train_resamp)
            acs_train = torch.stack(acs_train_resamp)
            shuffle = False

        tr_batch_size = min(batch_size, len(obs_train))
        # If Droplast is False, insure that that dataset is divisible by
        # the number of polices in the ensemble
        tr_drop_last = (tr_batch_size!=len(obs_train))
        if not tr_drop_last and ensemble_size is not None:
            tr_batch_size = int(ensemble_size * np.floor(tr_batch_size/ensemble_size))
            obs_train = obs_train[:tr_batch_size]
            acs_train = acs_train[:tr_batch_size]
        trdata = DataLoader(TensorDataset(obs_train, acs_train),
                            batch_size = tr_batch_size, shuffle=shuffle, drop_last=tr_drop_last)

        if len(obs_test) == 0:
            tedata = None
        else:
            te_batch_size = min(batch_size, len(obs_test))
            # If Droplast is False, insure that that dataset is divisible by
            # the number of polices in the ensemble
            te_drop_last = (te_batch_size!=len(obs_test))
            if not te_drop_last and ensemble_size is not None:
                te_batch_size = int(ensemble_size * np.floor(te_batch_size/ensemble_size))
                obs_test = obs_test[:te_batch_size]
                acs_test = acs_test[:te_batch_size]
            tedata = DataLoader(TensorDataset(obs_test, acs_test),
                                batch_size=te_batch_size, shuffle=shuffle, drop_last=te_drop_last)
        return {'trdata': trdata, 'tedata': tedata}


# class ExpertDataset(torch.utils.data.Dataset):
#     def __init__(self, file_path):
#         try:
#             dataset_file = h5py.File(file_path, 'r')
#         except:
#             raise ValueError(f"No such file {file_path}")
#         self.dataset = {k: dataset_file[k][:] for k in get_keys(dataset_file)}
#         dataset_file.close()
#
#         self.length = len(self.dataset['actions'])
#
#     def __len__(self):
#         return self.length
#
#     def __getitem__(self, i):
#         return self.dataset['states'][i], self.dataset['actions'][i]


class ExpertDatasetOld(torch.utils.data.Dataset):
    def __init__(self, file_name, num_trajectories=4, subsample_frequency=1):
        all_trajectories = torch.load(file_name)
        
        perm = torch.randperm(all_trajectories['states'].size(0))
        idx = perm[:num_trajectories]

        self.trajectories = {}
        
        # See https://github.com/pytorch/pytorch/issues/14886
        # .long() for fixing bug in torch v0.4.1
        start_idx = torch.randint(
            0, subsample_frequency, size=(num_trajectories, )).long()

        for k, v in all_trajectories.items():
            data = v[idx]

            if k != 'lengths':
                samples = []
                for i in range(num_trajectories):
                    samples.append(data[i][start_idx[i]::subsample_frequency])
                self.trajectories[k] = torch.stack(samples)
            else:
                self.trajectories[k] = data // subsample_frequency

        self.i2traj_idx = {}
        self.i2i = {}
        
        self.length = self.trajectories['lengths'].sum().item()

        traj_idx = 0
        i = 0

        self.get_idx = []
        
        for j in range(self.length):
            
            while self.trajectories['lengths'][traj_idx].item() <= i:
                i -= self.trajectories['lengths'][traj_idx].item()
                traj_idx += 1

            self.get_idx.append((traj_idx, i))

            i += 1
            
            
    def __len__(self):
        return self.length

    def __getitem__(self, i):
        traj_idx, i = self.get_idx[i]

        return self.trajectories['states'][traj_idx][i], \
               self.trajectories['actions'][traj_idx][i]
