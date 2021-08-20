"""
TensorFlow policy class used for PG.
"""

from typing import List, Type, Union

import ray
from src.rllib.agents.pg.utils import post_process_advantages
from src.rllib.evaluation.postprocessing import Postprocessing
from src.rllib.models.action_dist import ActionDistribution
from src.rllib.models.modelv2 import ModelV2
from src.rllib.policy import Policy
from src.rllib.policy.tf_policy_template import build_tf_policy
from src.rllib.policy.sample_batch import SampleBatch
from src.rllib.utils.framework import try_import_tf
from src.rllib.utils.typing import TensorType

tf1, tf, tfv = try_import_tf()


def pg_tf_loss(
        policy: Policy, model: ModelV2, dist_class: Type[ActionDistribution],
        train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
    """The basic policy gradients loss function.

    Args:
        policy (Policy): The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        dist_class (Type[ActionDistribution]: The action distr. class.
        train_batch (SampleBatch): The training data.

    Returns:
        Union[TensorType, List[TensorType]]: A single loss tensor or a list
            of loss tensors.
    """
    # Pass the training data through our model to get distribution parameters.
    dist_inputs, _ = model.from_batch(train_batch)

    # Create an action distribution object.
    action_dist = dist_class(dist_inputs, model)

    # Calculate the vanilla PG loss based on:
    # L = -E[ log(pi(a|s)) * A]
    return -tf.reduce_mean(
        action_dist.logp(train_batch[SampleBatch.ACTIONS]) * tf.cast(
            train_batch[Postprocessing.ADVANTAGES], dtype=tf.float32))


# Build a child class of `DynamicTFPolicy`, given the extra options:
# - trajectory post-processing function (to calculate advantages)
# - PG loss function
PGTFPolicy = build_tf_policy(
    name="PGTFPolicy",
    get_default_config=lambda: src.rllib.agents.pg.pg.DEFAULT_CONFIG,
    postprocess_fn=post_process_advantages,
    loss_fn=pg_tf_loss)
