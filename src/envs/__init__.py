from gym.envs.registration import register
from ray.tune.registry import register_env

from src.envs.rooms.rooms import RoomsEnv

register(
    id='rooms-v0',
    entry_point='src.envs.rooms.rooms:RoomsEnv',
    max_episode_steps=100000,
    kwargs={}
)
register_env('confounded_imitation:rooms-v0', lambda config: RoomsEnv(seed=-1))

register(
    id='SparseHopper-v0',
    entry_point='src.envs.sparse.hopper:SparseHopper',
    max_episode_steps=1000,
    reward_threshold=2500.0
)


tasks = ['ScratchItch', 'BedBathing', 'Feeding', 'Drinking', 'Dressing', 'ArmManipulation']
robots = ['PR2', 'Jaco', 'Baxter', 'Sawyer', 'Stretch', 'Panda']

for task in tasks:
    for robot in robots:
        register(
            id='%s%s-v1' % (task, robot),
            entry_point='src.envs.assistive:%s%sEnv' % (task, robot),
            max_episode_steps=200,
            kwargs={}
        )

for task in ['ScratchItch', 'Feeding']:
    for robot in robots:
        register(
            id='%s%sMesh-v1' % (task, robot),
            entry_point='src.envs.assistive:%s%sMeshEnv' % (task, robot),
            max_episode_steps=200,
            kwargs={}
        )

register(
    id='HumanTesting-v1',
    entry_point='src.envs.assistive:HumanTestingEnv',
    max_episode_steps=200,
    kwargs={}
)

register(
    id='SMPLXTesting-v1',
    entry_point='src.envs.assistive:SMPLXTestingEnv',
    max_episode_steps=200,
    kwargs={}
)
