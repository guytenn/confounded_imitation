from typing import Any, List, Dict
import time

from ray.actor import ActorHandle
from ray.util.iter import LocalIterator
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
from ray.rllib.execution.common import AGENT_STEPS_SAMPLED_COUNTER, \
    STEPS_SAMPLED_COUNTER, _get_shared_metrics
from ray.rllib.evaluation.worker_set import WorkerSet

from ray.rllib.execution.metric_ops import OncePerTimeInterval, OncePerTimestepsElapsed
import numpy as np
import wandb


def StandardMetricsReporting(
        train_op: LocalIterator[Any],
        workers: WorkerSet,
        config: dict,
        selected_workers: List[ActorHandle] = None) -> LocalIterator[dict]:
    """Operator to periodically collect and report metrics.

    Args:
        train_op (LocalIterator): Operator for executing training steps.
            We ignore the output values.
        workers (WorkerSet): Rollout workers to collect metrics from.
        config (dict): Trainer configuration, used to determine the frequency
            of stats reporting.
        selected_workers (list): Override the list of remote workers
            to collect metrics from.

    Returns:
        LocalIterator[dict]: A local iterator over training results.

    Examples:
        >>> train_op = ParallelRollouts(...).for_each(TrainOneStep(...))
        >>> metrics_op = StandardMetricsReporting(train_op, workers, config)
        >>> next(metrics_op)
        {"episode_reward_max": ..., "episode_reward_mean": ..., ...}
    """

    output_op = train_op \
        .filter(OncePerTimestepsElapsed(config["timesteps_per_iteration"])) \
        .filter(OncePerTimeInterval(config["min_iter_time_s"])) \
        .for_each(CollectMetrics(
            workers, min_history=config["metrics_smoothing_episodes"],
            timeout_seconds=config["collect_metrics_timeout"],
            selected_workers=selected_workers,
            wandb_config=config['wandb_logger']))
    return output_op


class CollectMetrics:
    """Callable that collects metrics from workers.

    The metrics are smoothed over a given history window.

    This should be used with the .for_each() operator. For a higher level
    API, consider using StandardMetricsReporting instead.

    Examples:
        >>> output_op = train_op.for_each(CollectMetrics(workers))
        >>> print(next(output_op))
        {"episode_reward_max": ..., "episode_reward_mean": ..., ...}
    """

    def __init__(self,
                 workers: WorkerSet,
                 min_history: int = 100,
                 timeout_seconds: int = 180,
                 selected_workers: List[ActorHandle] = None,
                 wandb_config=None):
        self.workers = workers
        self.episode_history = []
        self.to_be_collected = []
        self.min_history = min_history
        self.timeout_seconds = timeout_seconds
        self.selected_workers = selected_workers
        if wandb_config is not None:
            self.wandb_logger = wandb.init(**wandb_config)
        else:
            self.wandb_logger = None

    def __call__(self, _: Any) -> Dict:
        # Collect worker metrics.
        episodes, self.to_be_collected = collect_episodes(
            self.workers.local_worker(),
            self.selected_workers or self.workers.remote_workers(),
            self.to_be_collected,
            timeout_seconds=self.timeout_seconds)
        orig_episodes = list(episodes)
        missing = self.min_history - len(episodes)
        if missing > 0:
            episodes = self.episode_history[-missing:] + episodes
            assert len(episodes) <= self.min_history
        self.episode_history.extend(orig_episodes)
        self.episode_history = self.episode_history[-self.min_history:]
        res = summarize_episodes(episodes, orig_episodes)

        # Add in iterator metrics.
        metrics = _get_shared_metrics()
        custom_metrics_from_info = metrics.info.pop("custom_metrics", {})
        timers = {}
        counters = {}
        info = {}
        info.update(metrics.info)
        for k, counter in metrics.counters.items():
            counters[k] = counter
        for k, timer in metrics.timers.items():
            timers["{}_time_ms".format(k)] = round(timer.mean * 1000, 3)
            if timer.has_units_processed():
                timers["{}_throughput".format(k)] = round(
                    timer.mean_throughput, 3)
        res.update({
            "num_healthy_workers": len(self.workers.remote_workers()),
            "timesteps_total": metrics.counters[STEPS_SAMPLED_COUNTER],
            "agent_timesteps_total": metrics.counters.get(
                AGENT_STEPS_SAMPLED_COUNTER, 0),
        })
        res["timers"] = timers
        res["info"] = info
        res["info"].update(counters)
        res["custom_metrics"] = res.get("custom_metrics", {})
        res["episode_media"] = res.get("episode_media", {})
        res["custom_metrics"].update(custom_metrics_from_info)

        if self.wandb_logger is not None:
            policy = self.workers.local_worker().policy_map['default_policy']
            log_dict = {'reward_min': res['episode_reward_min'],
                        'reward_max': res['episode_reward_max'],
                        'reward_mean': res['episode_reward_mean'],
                        'reward_std': np.std(res['hist_stats']['episode_reward']),
                        'policy_loss_mean': policy._mean_policy_loss.item(),
                        'vf_loss_mean': policy._mean_vf_loss.item(),
                        'total_loss': policy._total_loss.item()
                        }
            if 'extra_info' in policy.config:
                log_dict.update(policy.config['extra_info'])

            self.wandb_logger.log(log_dict,
                                  step=res['timesteps_total'])

        return res