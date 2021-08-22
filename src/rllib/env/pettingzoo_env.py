from src.rllib.env.wrappers.pettingzoo_env import PettingZooEnv as PE
from src.rllib.utils.deprecation import deprecation_warning

deprecation_warning(
    old="src.rllib.env.pettingzoo_env.PettingZooEnv",
    new="src.rllib.env.wrappers.pettingzoo_env.PettingZooEnv",
    error=False,
)

PettingZooEnv = PE
