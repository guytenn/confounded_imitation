from .bed_bathing import BedBathingEnv
from .agents import pr2, baxter, sawyer, jaco, stretch, panda, human
from .agents.pr2 import PR2
from .agents.baxter import Baxter
from .agents.sawyer import Sawyer
from .agents.jaco import Jaco
from .agents.stretch import Stretch
from .agents.panda import Panda
from .agents.human import Human
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
import numpy as np

robot_arm = 'left'
human_controllable_joint_indices = human.right_arm_joints
class BedBathingPR2Env(BedBathingEnv):
    def __init__(self, context_params=None, seed=-1):
        if seed == -1:
            seed = np.random.randint(2 ** 30 - 1)
        super(BedBathingPR2Env, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=False), context_params=context_params, seed=seed)
register_env('confounded_imitation:BedBathingPR2-v1', lambda config: BedBathingPR2Env(context_params=config['context_params'], seed=-1))

class BedBathingBaxterEnv(BedBathingEnv):
    def __init__(self, context_params=None, seed=-1):
        if seed == -1:
            seed = np.random.randint(2 ** 30 - 1)
        super(BedBathingBaxterEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=False), context_params=context_params, seed=seed)
register_env('confounded_imitation:BedBathingBaxter-v1', lambda config: BedBathingBaxterEnv(context_params=config['context_params'], seed=-1))

class BedBathingSawyerEnv(BedBathingEnv):
    def __init__(self, context_params=None, seed=-1):
        if seed == -1:
            seed = np.random.randint(2 ** 30 - 1)
        super(BedBathingSawyerEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=False), context_params=context_params, seed=seed)
register_env('confounded_imitation:BedBathingSawyer-v1', lambda config: BedBathingSawyerEnv(context_params=config['context_params'], seed=-1))

class BedBathingJacoEnv(BedBathingEnv):
    def __init__(self, context_params=None, seed=-1):
        if seed == -1:
            seed = np.random.randint(2 ** 30 - 1)
        super(BedBathingJacoEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=False), context_params=context_params, seed=seed)
register_env('confounded_imitation:BedBathingJaco-v1', lambda config: BedBathingJacoEnv(context_params=config['context_params'], seed=-1))

class BedBathingStretchEnv(BedBathingEnv):
    def __init__(self):
        super(BedBathingStretchEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class BedBathingPandaEnv(BedBathingEnv):
    def __init__(self):
        super(BedBathingPandaEnv, self).__init__(robot=Panda(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class BedBathingPR2HumanEnv(BedBathingEnv, MultiAgentEnv):
    def __init__(self):
        super(BedBathingPR2HumanEnv, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('confounded_imitation:BedBathingPR2Human-v1', lambda config: BedBathingPR2HumanEnv())

class BedBathingBaxterHumanEnv(BedBathingEnv, MultiAgentEnv):
    def __init__(self):
        super(BedBathingBaxterHumanEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('confounded_imitation:BedBathingBaxterHuman-v1', lambda config: BedBathingBaxterHumanEnv())

class BedBathingSawyerHumanEnv(BedBathingEnv, MultiAgentEnv):
    def __init__(self):
        super(BedBathingSawyerHumanEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('confounded_imitation:BedBathingSawyerHuman-v1', lambda config: BedBathingSawyerHumanEnv())

class BedBathingJacoHumanEnv(BedBathingEnv, MultiAgentEnv):
    def __init__(self):
        super(BedBathingJacoHumanEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('confounded_imitation:BedBathingJacoHuman-v1', lambda config: BedBathingJacoHumanEnv())

class BedBathingStretchHumanEnv(BedBathingEnv, MultiAgentEnv):
    def __init__(self):
        super(BedBathingStretchHumanEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('confounded_imitation:BedBathingStretchHuman-v1', lambda config: BedBathingStretchHumanEnv())

class BedBathingPandaHumanEnv(BedBathingEnv, MultiAgentEnv):
    def __init__(self):
        super(BedBathingPandaHumanEnv, self).__init__(robot=Panda(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('confounded_imitation:BedBathingPandaHuman-v1', lambda config: BedBathingPandaHumanEnv())

