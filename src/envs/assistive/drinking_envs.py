from .drinking import DrinkingEnv
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

robot_arm = 'right'
human_controllable_joint_indices = human.head_joints
class DrinkingPR2Env(DrinkingEnv):
    def __init__(self, sparse_reward=False, context_params=None, seed=-1):
        if seed == -1:
            seed = np.random.randint(2 ** 30 - 1)
        super(DrinkingPR2Env, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=False), context_params=context_params, sparse_reward=sparse_reward, seed=seed)
register_env('confounded_imitation:DrinkingPR2-v1', lambda config: DrinkingPR2Env(sparse_reward=config['sparse_reward'], context_params=config['context_params'], seed=-1))

class DrinkingBaxterEnv(DrinkingEnv):
    def __init__(self, sparse_reward=False, context_params=None, seed=-1):
        if seed == -1:
            seed = np.random.randint(2 ** 30 - 1)
        super(DrinkingBaxterEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=False), context_params=context_params, sparse_reward=sparse_reward, seed=seed)
register_env('confounded_imitation:DrinkingBaxter-v1', lambda config: DrinkingBaxterEnv(sparse_reward=config['sparse_reward'], context_params=config['context_params'], seed=-1))

class DrinkingSawyerEnv(DrinkingEnv):
    def __init__(self, sparse_reward=False, context_params=None, seed=-1):
        if seed == -1:
            seed = np.random.randint(2 ** 30 - 1)
        super(DrinkingSawyerEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=False), context_params=context_params, sparse_reward=sparse_reward, seed=seed)
register_env('confounded_imitation:DrinkingSawyer-v1', lambda config: DrinkingSawyerEnv(sparse_reward=config['sparse_reward'], context_params=config['context_params'], seed=-1))

class DrinkingJacoEnv(DrinkingEnv):
    def __init__(self, sparse_reward=False, context_params=None, seed=-1):
        if seed == -1:
            seed = np.random.randint(2 ** 30 - 1)
        super(DrinkingJacoEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=False), context_params=context_params, sparse_reward=sparse_reward, seed=seed)
register_env('confounded_imitation:DrinkingJaco-v1', lambda config: DrinkingJacoEnv(sparse_reward=config['sparse_reward'], context_params=config['context_params'], seed=-1))

class DrinkingStretchEnv(DrinkingEnv):
    def __init__(self):
        super(DrinkingStretchEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class DrinkingPandaEnv(DrinkingEnv):
    def __init__(self):
        super(DrinkingPandaEnv, self).__init__(robot=Panda(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class DrinkingPR2HumanEnv(DrinkingEnv, MultiAgentEnv):
    def __init__(self):
        super(DrinkingPR2HumanEnv, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('confounded_imitation:DrinkingPR2Human-v1', lambda config: DrinkingPR2HumanEnv())

class DrinkingBaxterHumanEnv(DrinkingEnv, MultiAgentEnv):
    def __init__(self):
        super(DrinkingBaxterHumanEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('confounded_imitation:DrinkingBaxterHuman-v1', lambda config: DrinkingBaxterHumanEnv())

class DrinkingSawyerHumanEnv(DrinkingEnv, MultiAgentEnv):
    def __init__(self):
        super(DrinkingSawyerHumanEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('confounded_imitation:DrinkingSawyerHuman-v1', lambda config: DrinkingSawyerHumanEnv())

class DrinkingJacoHumanEnv(DrinkingEnv, MultiAgentEnv):
    def __init__(self):
        super(DrinkingJacoHumanEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('confounded_imitation:DrinkingJacoHuman-v1', lambda config: DrinkingJacoHumanEnv())

class DrinkingStretchHumanEnv(DrinkingEnv, MultiAgentEnv):
    def __init__(self):
        super(DrinkingStretchHumanEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('confounded_imitation:DrinkingStretchHuman-v1', lambda config: DrinkingStretchHumanEnv())

class DrinkingPandaHumanEnv(DrinkingEnv, MultiAgentEnv):
    def __init__(self):
        super(DrinkingPandaHumanEnv, self).__init__(robot=Panda(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('confounded_imitation:DrinkingPandaHuman-v1', lambda config: DrinkingPandaHumanEnv())

