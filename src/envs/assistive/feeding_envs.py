from .feeding import FeedingEnv
from .feeding_mesh import FeedingMeshEnv
from .agents import pr2, baxter, sawyer, jaco, stretch, panda, human, human_mesh
from .agents.pr2 import PR2
from .agents.baxter import Baxter
from .agents.sawyer import Sawyer
from .agents.jaco import Jaco
from .agents.stretch import Stretch
from .agents.panda import Panda
from .agents.human import Human
from .agents.human_mesh import HumanMesh
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
import numpy as np

robot_arm = 'right'
human_controllable_joint_indices = human.head_joints
class FeedingPR2Env(FeedingEnv):
    def __init__(self, seed=1001):
        super(FeedingPR2Env, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=False), seed=seed)

class FeedingBaxterEnv(FeedingEnv):
    def __init__(self, seed=1001):
        super(FeedingBaxterEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=False), seed=seed)

class FeedingSawyerEnv(FeedingEnv):
    def __init__(self, context_params=None, seed=-1):
        if seed == -1:
            seed = np.random.randint(2 ** 30 - 1)
        super(FeedingSawyerEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=False), context_params=context_params, seed=seed)
register_env('assistive_gym:FeedingSawyer-v1', lambda config: FeedingSawyerEnv(context_params=config['context_params'], seed=-1))

class FeedingJacoEnv(FeedingEnv):
    def __init__(self, seed=1001):
        super(FeedingJacoEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=False), seed=seed)

class FeedingStretchEnv(FeedingEnv):
    def __init__(self, seed=1001):
        super(FeedingStretchEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=False), seed=seed)

class FeedingPandaEnv(FeedingEnv):
    def __init__(self, seed=1001):
        super(FeedingPandaEnv, self).__init__(robot=Panda(robot_arm), human=Human(human_controllable_joint_indices, controllable=False), seed=seed)

class FeedingPR2HumanEnv(FeedingEnv, MultiAgentEnv):
    def __init__(self, seed=1001):
        super(FeedingPR2HumanEnv, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=True), seed=seed)
register_env('assistive_gym:FeedingPR2Human-v1', lambda config: FeedingPR2HumanEnv())

class FeedingBaxterHumanEnv(FeedingEnv, MultiAgentEnv):
    def __init__(self, seed=1001):
        super(FeedingBaxterHumanEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=True), seed=seed)
register_env('assistive_gym:FeedingBaxterHuman-v1', lambda config: FeedingBaxterHumanEnv())

class FeedingSawyerHumanEnv(FeedingEnv, MultiAgentEnv):
    def __init__(self, seed=1001):
        super(FeedingSawyerHumanEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=True), seed=seed)
register_env('assistive_gym:FeedingSawyerHuman-v1', lambda config: FeedingSawyerHumanEnv())

class FeedingJacoHumanEnv(FeedingEnv, MultiAgentEnv):
    def __init__(self, seed=1001):
        super(FeedingJacoHumanEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=True), seed=seed)
register_env('assistive_gym:FeedingJacoHuman-v1', lambda config: FeedingJacoHumanEnv())

class FeedingStretchHumanEnv(FeedingEnv, MultiAgentEnv):
    def __init__(self, seed=1001):
        super(FeedingStretchHumanEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=True), seed=seed)
register_env('assistive_gym:FeedingStretchHuman-v1', lambda config: FeedingStretchHumanEnv())

class FeedingPandaHumanEnv(FeedingEnv, MultiAgentEnv):
    def __init__(self, seed=1001):
        super(FeedingPandaHumanEnv, self).__init__(robot=Panda(robot_arm), human=Human(human_controllable_joint_indices, controllable=True), seed=seed)
register_env('assistive_gym:FeedingPandaHuman-v1', lambda config: FeedingPandaHumanEnv())

class FeedingPR2MeshEnv(FeedingMeshEnv):
    def __init__(self, seed=1001):
        super(FeedingPR2MeshEnv, self).__init__(robot=PR2(robot_arm), human=HumanMesh(), seed=seed)

class FeedingBaxterMeshEnv(FeedingMeshEnv):
    def __init__(self, seed=1001):
        super(FeedingBaxterMeshEnv, self).__init__(robot=Baxter(robot_arm), human=HumanMesh(), seed=seed)

class FeedingSawyerMeshEnv(FeedingMeshEnv):
    def __init__(self, seed=1001):
        super(FeedingSawyerMeshEnv, self).__init__(robot=Sawyer(robot_arm), human=HumanMesh(), seed=seed)

class FeedingJacoMeshEnv(FeedingMeshEnv):
    def __init__(self, seed=1001):
        super(FeedingJacoMeshEnv, self).__init__(robot=Jaco(robot_arm), human=HumanMesh(), seed=seed)

class FeedingStretchMeshEnv(FeedingMeshEnv):
    def __init__(self, seed=1001):
        super(FeedingStretchMeshEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=HumanMesh(), seed=seed)

class FeedingPandaMeshEnv(FeedingMeshEnv):
    def __init__(self, seed=1001):
        super(FeedingPandaMeshEnv, self).__init__(robot=Panda(robot_arm), human=HumanMesh(), seed=seed)

