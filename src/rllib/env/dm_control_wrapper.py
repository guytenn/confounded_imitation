from src.rllib.env.wrappers.dm_control_wrapper import DMCEnv as DCE
from src.rllib.utils.deprecation import deprecation_warning

deprecation_warning(
    old="src.rllib.env.dm_control_wrapper.DMCEnv",
    new="src.rllib.env.wrappers.dm_control_wrapper.DMCEnv",
    error=False,
)

DMCEnv = DCE
