from src.rllib.env.wrappers.group_agents_wrapper import GroupAgentsWrapper as \
    GAW
from src.rllib.utils.deprecation import deprecation_warning

deprecation_warning(
    old="src.rllib.env.group_agents_wrapper._GroupAgentsWrapper",
    new="src.rllib.env.wrappers.group_agents_wrapper.GroupAgentsWrapper",
    error=False,
)

_GroupAgentsWrapper = GAW
