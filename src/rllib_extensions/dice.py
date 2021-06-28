from gym.spaces import Discrete, MultiDiscrete, Space
import numpy as np
from typing import Optional, Tuple, Union

from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import Categorical, MultiCategorical
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_action_dist import TorchCategorical, \
    TorchMultiCategorical
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import NullContextManager
from ray.rllib.utils.annotations import override
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.utils.framework import try_import_tf, \
    try_import_torch
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.tf_ops import get_placeholder, one_hot as tf_one_hot
from ray.rllib.utils.torch_ops import one_hot
from ray.rllib.utils.typing import FromConfigSpec, ModelConfigDict, TensorType

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()
F = None
if nn is not None:
    F = nn.functional

from src.data.expert_data import ExpertData
from src.data.utils import load_h5_dataset
from src.sb3_extensions.buffers import CustomReplayBuffer


class DICE(Exploration):
    def __init__(self,
                 action_space: Space,
                 *,
                 framework: str,
                 model: ModelV2,
                 lr: float = 1e-3,
                 sub_exploration: Optional[FromConfigSpec] = None,
                 expert_path=None,
                 hidden_dim=400,
                 gamma=0.99,
                 features_to_remove=[],
                 state_dim=None,
                 dice_coef=0.5,
                 **kwargs):

        super().__init__(
            action_space, model=model, framework=framework, **kwargs)

        # if self.policy_config["num_workers"] != 0:
        #     raise ValueError(
        #         "Curiosity exploration currently does not support parallelism."
        #         " `num_workers` must be 0!")

        self.expert_path = expert_path
        self.gamma = gamma
        self.features_to_remove = features_to_remove
        self.dice_coef = dice_coef
        self.lr = lr
        # TODO: (sven) if sub_exploration is None, use Trainer's default
        #  Exploration config.
        if sub_exploration is None:
            raise NotImplementedError
        self.sub_exploration = sub_exploration

        self.expert_buffer = None

        self.g = self._create_fc_net((state_dim + action_space.shape[0], hidden_dim, hidden_dim, 1), "relu", name="g_net")
        self.h = self._create_fc_net((state_dim, hidden_dim, hidden_dim, 1), "relu", name="h_net")

        self.mean = None
        self.var = None
        self.count = None
        self.returns = None

        self.replay_buffer = None

        self.count_i = 0

        # This is only used to select the correct action
        self.exploration_submodule = from_config(
            cls=Exploration,
            config=self.sub_exploration,
            action_space=self.action_space,
            framework=self.framework,
            policy_config=self.policy_config,
            model=self.model,
            num_workers=self.num_workers,
            worker_index=self.worker_index,
        )

    @override(Exploration)
    def get_exploration_action(self,
                               *,
                               action_distribution: ActionDistribution,
                               timestep: Union[int, TensorType],
                               explore: bool = True):
        # Simply delegate to sub-Exploration module.
        return self.exploration_submodule.get_exploration_action(
            action_distribution=action_distribution,
            timestep=timestep,
            explore=explore)

    @override(Exploration)
    def get_exploration_optimizer(self, optimizers):
        data = load_h5_dataset(self.expert_path)
        if 'dones' not in data.keys():
            data['dones'] = np.zeros(len(data['actions']))
        self.expert_buffer = ExpertData(data['states'].astype('float32'), data['actions'].astype('float32'), data['dones'], device=self.device)

        self.replay_buffer = CustomReplayBuffer(
            self.policy_config['train_batch_size'], #// self.policy_config['num_workers'],
            self.model.obs_space,
            self.action_space,
            self.device,
            optimize_memory_usage=False,
        )

        if self.framework == "torch":
            g_params = list(self.g.parameters())
            h_params = list(self.h.parameters())

            # Now that the Policy's own optimizer(s) have been created (from
            # the Model parameters (IMPORTANT: w/o(!) the curiosity params),
            # we can add our curiosity sub-modules to the Policy's Model.
            self.model.g = self.g.to(self.device)
            self.model.h = self.h.to(self.device)
            self._optimizer = torch.optim.Adam(g_params + h_params, lr=self.lr)
        else:
            raise NotImplementedError

        return optimizers

    @override(Exploration)
    def postprocess_trajectory(self, policy, sample_batch, tf_sess=None):
        if self.framework != "torch":
            raise NotImplementedError
        else:
            self._postprocess_torch(policy, sample_batch)

    def _forward_model(self, obs, actions, next_obs, dones):
        rs = torch.sigmoid(self.model.g(torch.cat((obs, actions), dim=1)))
        vs = self.model.h(obs)
        next_vs = self.model.h(next_obs)
        return rs.flatten() + self.gamma * (1 - dones.float()) * next_vs.flatten() - vs.flatten()

    def _train_step(self):
        batch_size = 200
        dice_epochs = 50
        alpha = 0.9
        batch_generator = self.replay_buffer.get(len(self.expert_buffer) // batch_size, batch_size)

        for _ in range(dice_epochs):
            for policy_data in batch_generator:
                expert_data = self.expert_buffer.sample(batch_size+1)

                expert_d = self._forward_model(expert_data.observations[:-1],
                                               expert_data.actions[:-1],
                                               expert_data.observations[1:],
                                               expert_data.dones[:-1])

                # if isinstance(self.action_space, spaces.Discrete):
                #     policy_actions = to_onehot(policy_actions.flatten(), self.model.action_dim)
                policy_d = self._forward_model(policy_data.observations[:-1],
                                               policy_data.actions[:-1],
                                               policy_data.observations[1:],
                                               policy_data.dones[:-1])

                loss = alpha * torch.pow(expert_d, 2).mean() + (1 - alpha) * torch.pow(policy_d, 2).mean() - 2 * policy_d.mean()
                # kl divergence
                # loss = torch.log(0.9 * torch.exp(expert_d).mean() + 0.1 * torch.exp(policy_d).mean()) - policy_d.mean()
                # GAIL loss
                # loss = -F.logsigmoid(-policy_d).mean() - F.logsigmoid(expert_d).mean()
                # Perform an optimizer step.
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

    def _predict_reward(self, policy, samples):
        policy_obs = torch.from_numpy(samples[SampleBatch.OBS]).to(policy.device)
        policy_actions = torch.from_numpy(samples[SampleBatch.ACTIONS]).to(policy.device)
        policy_next_obs = torch.from_numpy(samples[SampleBatch.NEXT_OBS]).to(policy.device)
        policy_dones = torch.from_numpy(samples[SampleBatch.DONES]).float().to(policy.device)

        policy_d = self._forward_model(policy_obs, policy_actions, policy_next_obs, policy_dones)
        # policy_d = torch.sigmoid(policy_d)
        # reward_bonus = -torch.log(1.0 - policy_d * (1.0 - float(1e-8)))
        reward_bonus = -policy_d

        # if self.returns is None or (self.returns is not None and self.returns.shape != reward_bonus.shape):
        #     self.mean = None
        #     self.returns = reward_bonus.clone()
        #
        # if True:  # update_rms:
        #     self.returns = self.returns * self.gamma + reward_bonus
        #     self.update_running_avg(self.returns)
        #
        # reward_bonus_std = np.nan_to_num(np.sqrt(self.var.detach().cpu().numpy() + 1e-8), nan=1.0)
        # reward_bonus = reward_bonus.detach().cpu().numpy() / reward_bonus_std

        return reward_bonus.detach().cpu().numpy()

    def _postprocess_torch(self, policy, sample_batch):
        # ADD SAMPLES TO REPLAY
        for i in range(len(sample_batch)):
            self.replay_buffer.add(sample_batch[SampleBatch.OBS][i:i + 1],
                                   sample_batch[SampleBatch.NEXT_OBS][i:i + 1],
                                   sample_batch[SampleBatch.ACTIONS][i:i + 1],
                                   sample_batch[SampleBatch.REWARDS][i:i + 1],
                                   sample_batch[SampleBatch.DONES][i:i + 1])


        # ESTIMATE REWARD BONUS
        reward_bonus = self._predict_reward(policy, sample_batch)

        sample_batch[SampleBatch.REWARDS] = \
            (1-self.dice_coef) * sample_batch[SampleBatch.REWARDS] + self.dice_coef * reward_bonus

        # self.count_i += len(sample_batch)
        # with open('file2.txt', 'a') as f:
        #     print(self.count_i, file=f)

        # TRAIN DICE
        if self.replay_buffer.full or self.replay_buffer.pos > 1000:
            self._train_step()

        # expert_data = self.expert_buffer.sample(len(policy_obs) + 1)
        # expert_obs, expert_next_obs, expert_dones = \
        #     expert_data.observations[:-1], expert_data.observations[1:], expert_data.dones[:-1]
        #
        # expert_d = self._forward_model(expert_obs, expert_next_obs, expert_dones)
        #
        # alpha = 0.9
        # loss = alpha * torch.pow(expert_d, 2).mean() + (1-alpha) * torch.pow(policy_d, 2).mean() - 2 * policy_d.mean()
        # # kl divergence
        # # loss = torch.log(0.9 * torch.exp(expert_d).mean() + 0.1 * torch.exp(policy_d).mean()) - policy_d.mean()
        # # GAIL loss
        # # loss = -F.logsigmoid(-policy_d).mean() - F.logsigmoid(expert_d).mean()
        # # Perform an optimizer step.
        # self._optimizer.zero_grad()
        # loss.backward()
        # self._optimizer.step()

        # Return the postprocessed sample batch (with the corrected rewards).
        return sample_batch

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
        layers = [
            tf.keras.layers.Input(
                shape=(layer_dims[0], ), name="{}_in".format(name))
        ] if self.framework != "torch" else []

        for i in range(len(layer_dims) - 1):
            act = activation if i < len(layer_dims) - 2 else None
            if self.framework == "torch":
                layers.append(
                    SlimFC(
                        in_size=layer_dims[i],
                        out_size=layer_dims[i + 1],
                        initializer=torch.nn.init.xavier_uniform_,
                        activation_fn=act))
            else:
                layers.append(
                    tf.keras.layers.Dense(
                        units=layer_dims[i + 1],
                        activation=get_activation_fn(act),
                        name="{}_{}".format(name, i)))

        if self.framework == "torch":
            return nn.Sequential(*layers)
        else:
            return tf.keras.Sequential(layers)

    def update_running_avg(self, batch):
        if self.mean is None:
            self.mean = torch.zeros(batch.shape)
            self.var = torch.ones(batch.shape)
            self.count = float(1e-4)
        batch_mean = torch.mean(batch, dim=0)
        batch_var = torch.var(batch, dim=0)
        batch_count = batch.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count
