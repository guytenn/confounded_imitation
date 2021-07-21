import ray
from ray import tune
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLogger
from src.rllib_extensions.slateq import SlateQTrainer
from ray.rllib.env.wrappers.recsim_wrapper import make_recsim_env  # this is important as it imports the recsim env

import ray.rllib.agents.slateq
ray.init()
tune.run(
    SlateQTrainer,
    checkpoint_freq=1,
    config={
        "framework": "torch",
        "num_workers": 1,
        "num_gpus": 0,
        "env": "RecSim-v2",
        "slateq_strategy": "MYOP",
        "logger_config" : {
            "wandb": {
                "project": "RecSim",
                "api_key_file": "./wandb_api_key_file",
            }
        }
    },
    stop={
        "training_iteration": 1000000
    },
    loggers=DEFAULT_LOGGERS + (WandbLogger, )
)

