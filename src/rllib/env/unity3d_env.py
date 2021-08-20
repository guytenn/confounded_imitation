from src.rllib.env.wrappers.unity3d_env import Unity3DEnv as UE
from src.rllib.utils.deprecation import deprecation_warning

deprecation_warning(
    old="src.rllib.env.unity3d_env.Unity3DEnv",
    new="src.rllib.env.wrappers.unity3d_env.Unity3DEnv",
    error=False,
)

Unity3DEnv = UE
