from pybulletgym.envs.mujoco.envs.locomotion.walker_base_env import WalkerBaseMuJoCoEnv
from pybulletgym.envs.mujoco.robots.locomotors.hopper import Hopper
import numpy as np
import gym
from recsim.agents import full_slate_q_agent


class SparseHopper(WalkerBaseMuJoCoEnv):
    def __init__(self, max_x=1, max_y=1, max_z=1, sparse=False):
        self.robot = Hopper()
        WalkerBaseMuJoCoEnv.__init__(self, self.robot)
        self.observation_space.shape = (self.observation_space.shape[0] + 3,)
        self.max_force = np.array([max_x, max_y, max_z])
        self.sparse = sparse
        self.force = None
        self.reset()
        self.p = self.robot._p

    def reset(self):
        self.force = np.random.rand(3) * self.max_force
        s = super(WalkerBaseMuJoCoEnv, self).reset()
        return np.concatenate([s, self.force])

    def step(self, a):

        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.robot.apply_action(a)
            self.p.applyExternalForce(1, 0, (self.force[0], self.force[1], self.force[2]), (0, 0, 0), self.p.WORLD_FRAME)
            self.scene.global_step()

        alive_bonus = 1.0
        power_cost = -1e-3 * np.square(a).sum()
        if self.sparse:
            # self.rewards = [int(abs(self.robot.robot_body.get_pose()[0]) >= 15.)]
            self.rewards = [0]
        else:
            potential = self.robot.calc_potential()
            self.rewards = [potential, alive_bonus, power_cost]

        self.reward += sum(self.rewards)

        state = self.robot.calc_state()

        height, ang = state[0], state[1]

        done = not (np.isfinite(state).all() and
                    (np.abs(state[2:]) < 100).all() and
                    (height > -0.3) and # height starts at 0 in pybullet
                    (abs(ang) < .2))

        self.HUD(state, a, done)

        return np.concatenate([state, self.force]), sum(self.rewards), bool(done), {}