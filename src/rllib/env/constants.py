from src.rllib.env.wrappers.group_agents_wrapper import GROUP_REWARDS as GR, \
    GROUP_INFO as GI
from src.rllib.utils.deprecation import deprecation_warning

deprecation_warning(
    old="src.rllib.env.constants.GROUP_[REWARDS|INFO]",
    new="src.rllib.env.wrappers.group_agents_wrapper.GROUP_[REWARDS|INFO]",
    error=False,
)

GROUP_REWARDS = GR
GROUP_INFO = GI
