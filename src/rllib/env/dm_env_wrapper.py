from src.rllib.env.wrappers.dm_env_wrapper import DMEnv as DE
from src.rllib.utils.deprecation import deprecation_warning

deprecation_warning(
    old="src.rllib.env.dm_env_wrapper.DMEnv",
    new="src.rllib.env.wrappers.dm_env_wrapper.DMEnv",
    error=False,
)

DMEnv = DE
