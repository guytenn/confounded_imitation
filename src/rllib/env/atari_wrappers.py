from src.rllib.env.wrappers.atari_wrappers import is_atari, \
    get_wrapper_by_cls, MonitorEnv, NoopResetEnv, ClipRewardEnv, \
    FireResetEnv, EpisodicLifeEnv, MaxAndSkipEnv, WarpFrame, FrameStack, \
    FrameStackTrajectoryView, ScaledFloatFrame, wrap_deepmind
from src.rllib.utils.deprecation import deprecation_warning

deprecation_warning(
    old="src.rllib.env.atari_wrappers....",
    new="src.rllib.env.wrappers.atari_wrappers....",
    error=False,
)

is_atari = is_atari
get_wrapper_by_cls = get_wrapper_by_cls
MonitorEnv = MonitorEnv
NoopResetEnv = NoopResetEnv
ClipRewardEnv = ClipRewardEnv
FireResetEnv = FireResetEnv
EpisodicLifeEnv = EpisodicLifeEnv
MaxAndSkipEnv = MaxAndSkipEnv
WarpFrame = WarpFrame
FrameStack = FrameStack
FrameStackTrajectoryView = FrameStackTrajectoryView
ScaledFloatFrame = ScaledFloatFrame
wrap_deepmind = wrap_deepmind
