import numpy as np
from gym import core, spaces
import matplotlib.pyplot as plt
import cv2
from typing import Union, Tuple, List, Iterable
import scipy.ndimage
import random


c2rgb = [[0, 0, 255],
         [255, 255, 255],
         [0, 255, 0],
         [255, 0, 0]]

# APPLE_CELLS = [(9, 10), (9, 8), (12, 10), (1, 7), (5, 4), (5, 14), (1, 13), (7, 2), (4, 2), (14, 1), (5, 1), (3, 4), (9, 3), (9, 14), (9, 2), (2, 13), (6, 4), (10, 4), (4, 10), (7, 5)]
MINE_CELLS = [(4, 6), (2, 12), (8, 10), (4, 1), (14, 6), (4, 8), (4, 7), (10, 12), (5, 13), (5, 8), (7, 10), (9, 9), (2, 7), (11, 14), (4, 13), (5, 11), (13, 3), (8, 1), (4, 12), (10, 2)]
START_CELL = [(1, 1), (13, 1)]
APPLE_CELLS = [(1, 13), (13, 13)]
GOAL_CELLS = [(1, 13), (13, 13)]


class RoomsEnv(core.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, rows=15, cols=15, empty=False, random_walls=False,
                 obstacles: Iterable[Union[Tuple, List]] = None, spatial=False,
                 n_apples=0, n_mines=0,
                 action_repeats=1, max_steps=None, seed=None, fixed_reset=True,
                 mask_size=0, px=(0.2, 0.8),
                 wind_in_state=False, random_wind=False, vert_wind=(0.2, 0.2), horz_wind=(0.2, 0.2)):
        '''
        vert_wind = (up, down)
        horz_wind = (right, left)
        '''
        if seed == -1:
            seed = np.random.randint(2 ** 30 - 1)

        self.context_features = [2, 3, 4, 5]

        self.rows, self.cols = rows, cols
        if max_steps is None:
            self.max_steps = 1 * (rows + cols)
        else:
            self.max_steps = max_steps

        self.px = px

        self.n_apples = n_apples
        self.n_mines = n_mines

        self.random_wind = random_wind
        self.wind_in_state = wind_in_state
        self.vert_wind = np.array(vert_wind)
        self.horz_wind = np.array(horz_wind)
        self.obstacles = obstacles

        self.action_space = spaces.Discrete(4)
        self.spatial = spatial
        self.scale = np.maximum(rows, cols)
        if spatial:
            # n_channels = 4 + wind_in_state * 4 + 2
            n_channels = 4
            self.observation_space = spaces.Box(low=0, high=1, shape=(n_channels, 80, 80),
                                                dtype=np.float32)
        else:
            n_channels = 6 # 2 + (n_apples + n_mines) * 2 + wind_in_state * 4
            self.observation_space = spaces.Box(low=-1, high=1, shape=(n_channels,), dtype=np.float32)

        self.directions = [np.array((-1, 0)), np.array((1, 0)), np.array((0, -1))] + [np.array((0, 1))]
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()

        self.random_walls= random_walls
        self.empty = empty

        self.map, self.cell_seed = self._randomize_walls(random=random_walls, empty=empty)
        self.mask = np.ones_like(self.map)
        if mask_size > 0:
            self.mask[1:1+mask_size, 1:1+mask_size] = 0

        self.room_with_mines = None
        self.fixed_reset = fixed_reset
        self.taken_positions = np.copy(self.map)
        self.state_cell, self.state = self._random_from_map(1, START_CELL, force_random=False)
        self.apples_cells, self.apples_map, self.mines_cells, self.mines_map = None, None, None, None
        # self.apples_cells, self.apples_map = self._random_from_map(n_apples, APPLE_CELLS)
        self.mines_cells, self.mines_map = self._random_from_map(n_mines, MINE_CELLS, force_random=False)
        self.x = None
        self._random_goal()

        self.wind_state = self._generate_wind_state()

        if fixed_reset:
            self.reset_state_cell, self.reset_state = self.state_cell, self.state.copy()
            self.reset_apples_cells, self.reset_apples_map = self.apples_cells, self.apples_map.copy()
            # self.reset_mines_cells, self.reset_mines_map = self.mines_cells, self.mines_map.copy()
        else:
            self.reset_state_cell, self.reset_state = None, None

        self.n_resets = 0
        self.tot_reward = 0
        self.viewer = None

        self.action_repeats = action_repeats

        print(f'Initializing {rows}x{cols} Rooms Environment with {n_apples} apples and {n_mines} mines. (seed = {self.rng.seed})')

    def reset(self):
        self.wind_state = self._generate_wind_state()
        # print(f'Horz wind = {self.horz_wind}, Vert wind = {self.vert_wind}')
        self.map, self.cell_seed = self._randomize_walls(random=self.random_walls, empty=self.empty)
        self.taken_positions = np.copy(self.map)

        # if self.fixed_reset:
        #     self.state_cell, self.state = self.reset_state_cell, self.reset_state.copy()
        #     self.apples_cells, self.apples_map = self.reset_apples_cells, self.reset_apples_map.copy()
        #     self.mines_cells, self.mines_map = self.reset_mines_cells, self.reset_mines_map.copy()
        # else:
        self.state_cell, self.state = self._random_from_map(1, START_CELL, force_random=False)
        # self.apples_cells, self.apples_map = self._random_from_map(self.n_apples, APPLE_CELLS)
        # self.mines_cells, self.mines_map = self._random_from_map(self.n_mines, MINE_CELLS, force_random=True, px=self.px)
        self.apples_cells, self.apples_map, self.mines_cells, self.mines_map = None, None, None, None
        self.mines_cells, self.mines_map = self._random_from_map(self.n_mines, MINE_CELLS, force_random=False)
        self._random_goal()

        self.nsteps = 0
        self.tot_reward = 0
        self.n_resets += 1

        obs = self._obs_from_state(self.spatial)

        return obs

    def step(self, action: int):
        if action == -1:
            return self._step_rule()
        # actions: 0 = up, 1 = down, 2 = left, 3:end = right

        # n_repeats = np.random.choice(range(1, self.action_repeats+1))
        obs = r = done = None
        for _ in range(self.action_repeats):
            self._move(action)
            wind_up = np.random.choice([-1, 0, 1], p=[1 - self.vert_wind.sum(), self.vert_wind[0], self.vert_wind[1]])
            wind_right = np.random.choice([-1, 2, 3], p=[1 - self.horz_wind.sum(), self.horz_wind[1], self.horz_wind[0]])
            if wind_up >= 0:
                self._move(wind_up)
            if wind_right >= 0:
                self._move(wind_right)

            # TODO: apples and mines currently not supporting non spatial obs
            apple_collected_location = self.apples_map * self.state
            mine_collected_location = self.mines_map * self.state
            # if self.n_resets > 1000:
            r = 1 * np.sum(apple_collected_location) - 1 * np.sum(mine_collected_location)
            # r -= 0.1 * np.linalg.norm(np.array(self.state_cell) - np.array(self.apples_cells)) / np.sqrt(self.rows * self.cols)
            # else:
            #     r = np.sum(apple_collected_location)
            # self.apples_map -= apple_collected_location
            # self.mines_map -= mine_collected_location
            # if np.sum(mine_collected_location) > 0:
            #     self.state_cell, self.state = self._random_from_map(1, START_CELL)

            # done = np.sum(self.apples_map) == 0 or np.sum(self.mines_map) == 0 or self.nsteps >= self.max_steps
            done = self.nsteps >= self.max_steps

            obs = self._obs_from_state(self.spatial)

            self.tot_reward += r
            self.nsteps += 1
            info = dict()

            if done:
                break

        info['a'] = action
        if done:
            info['episode'] = {'r': np.copy(self.tot_reward), 'l': self.nsteps}

        return obs, r, done, info

    def _move(self, action: int):
        action = int(action)
        next_cell = self.state_cell + self.directions[action]
        if self.map[next_cell[0], next_cell[1]] == 0:
            self.state_cell = next_cell
            self.state = np.zeros_like(self.map)
            self.state[self.state_cell[0], self.state_cell[1]] = 1

    def _random_goal(self):
        self.apples_map = np.zeros_like(self.map)
        self.mines_map = np.zeros_like(self.map)
        idx = np.random.choice(2, p=self.px)
        self.x = idx
        goal_cell = GOAL_CELLS[idx]
        mine_cell = GOAL_CELLS[1-idx]
        self.apples_map[goal_cell[0], goal_cell[1]] = 1
        self.mines_map[mine_cell[0], mine_cell[1]] = 1
        self.apples_cells = goal_cell
        self.mines_cells = mine_cell

    def _random_from_map(self, n=1, array=None, force_random=False, px=None):
        map = np.zeros_like(self.map)
        cells = []
        room_num = 0
        if px is not None:
            room_num = self.rng.choice(4, p=px)
            if room_num == 1:
                self.room_with_mines = [np.zeros_like(self.map), np.ones_like(self.map)]
            elif room_num == 2:
                self.room_with_mines = [np.ones_like(self.map), np.zeros_like(self.map)]
        for i in range(n):
            if self.fixed_reset and not force_random and array is not None:
                cell = random.choice(array)
            else:
                cell = self.rng.choice(self.rows), self.rng.choice(self.cols)
                while self.taken_positions[cell[0], cell[1]] == 1 or \
                        (px is not None and self._which_room(cell) != room_num):
                    cell = (self.rng.choice(self.rows), self.rng.choice(self.cols))

            cells.append(cell)
            map[cell[0], cell[1]] = 1
            self.taken_positions[cell[0], cell[1]] = 1

        if n == 1:
            cells = cells[0]

        return cells, map

    def _generate_wind_state(self):
        if self.random_wind:
            vw1 = max(self.rng.rand() - 0.7, 0)
            vw2 = max(self.rng.rand() - 0.7, 0)
            hw1 = max(self.rng.rand() - 0.7, 0)
            hw2 = max(self.rng.rand() - 0.7, 0)
            self.vert_wind = np.array([vw1, vw2])
            self.horz_wind = np.array([hw1, hw2])
        wind_state = [np.ones((self.rows, self.cols)) for _ in range(4)]
        wind_state[0] *= self.vert_wind[0]
        wind_state[1] *= self.vert_wind[1]
        wind_state[2] *= self.horz_wind[0]
        wind_state[3] *= self.horz_wind[1]
        return wind_state

    def _obs_from_state(self, spatial):
        if spatial:
            # im_list = [self.state, self.map, self.apples_map, self.mines_map]
            # if self.room_with_mines is not None:
            #     im_list.extend(self.room_with_mines)
            # if self.wind_in_state:
            #     im_list.extend(self.wind_state)
            im_list = [self.state, self.map, self.apples_map, self.mines_map] # FOR DEBUG

            obs = np.stack(im_list, axis=0)
            obs = scipy.ndimage.zoom(obs, (1, 80./obs.shape[1], 80./obs.shape[2]), order=0)
            return obs.astype('float32')
        else:
            obs = list(self.state_cell)
            obs = np.concatenate([obs, self.apples_cells, self.mines_cells])
            obs = 2 * (np.array(obs) / self.scale - 0.5)
            if self.wind_in_state:
                obs = np.concatenate([obs, [*self.vert_wind, *self.horz_wind]])
            return obs

    def _color_obs(self, obs):
        rgb_obs = np.zeros((3, *obs.shape[1:]))
        for i in range(4):
            rgb_obs += obs[i:i+1] * np.array(c2rgb[i])[:, np.newaxis, np.newaxis]
        res = dict(img=np.moveaxis(rgb_obs, 0, 2), vert_wind=self.vert_wind, horz_wind=self.horz_wind)
        return res

    def _which_room(self, cell):
        if cell[0] <= self.cell_seed[0] and cell[1] <= self.cell_seed[1]:
            return 0
        elif cell[0] <= self.cell_seed[0] and cell[1] > self.cell_seed[1]:
            return 1
        elif cell[0] > self.cell_seed[0] and cell[1] <= self.cell_seed[1]:
            return 2
        else:
            return 3

    def _step_rule(self):
        if self.state_cell[1] + 1 <= self.apples_cells[1]:
            right_free = 1 - self.mines_map[self.state_cell[0], self.state_cell[1] + 1]
        else:
            right_free = 0
        if self.state_cell[0] + 1 <= self.apples_cells[0]:
            down_free = 1 - self.mines_map[self.state_cell[0] + 1, self.state_cell[1]]
        else:
            down_free = 0
        if right_free + down_free == 0:
            action_probs = np.array([0.5, 0, 0.5, 0])
        else:
            action_weights = np.array([0, down_free, 0, right_free])
            action_probs = action_weights / action_weights.sum()
        action = np.random.choice(range(4), p=action_probs)
        return self.step(action)

    # def _make_plan(self):
    #     if self._which_room(self.state) == 0:
    #         plan1 = [self.doors[0], self.doors[3], self.apples_cells]
    #         dist1 = self.doors[0]
    #         plan2 = [self.doors[1], self.doors[2], self.apples_cells]
    #
    #     elif self._which_room(self.state) == 1:
    #         plan = [self.doors[3], self.apples_cells]
    #     elif self._which_room(self.state) == 2:
    #         plan = [self.doors[2], self.apples_cells]
    #     else:
    #         plan = [self.apples_cells]
    #
    #     last_state = self.state
    #     for plan in [plan1, plan2]:
    #         for goal in plan:
    #             if
    #
    # def _diff(self, cell1, cell2):
    #     return cell2[0] - cell1[0], cell2[1] - cell1[1]
    #
    # def _diff2action(self, cell1, cell2):
    #     if self._diff(cell1, cell2)



    def _randomize_walls(self, random=False, empty=False):
        map = np.zeros((self.rows, self.cols))

        map[0, :] = 1
        map[:, 0] = 1
        map[-1:, :] = 1
        map[:, -1:] = 1

        if self.obstacles:
            for obstacle in self.obstacles:
                map[obstacle[0] - 1:obstacle[0] + 2, obstacle[1] - 1:obstacle[1] + 2] = 1

        if random:
            seed = (self.rng.randint(3, self.rows - 3), self.rng.randint(3, self.cols - 3))
            doors = (self.rng.randint(1, seed[0]),
                     self.rng.randint(seed[0] + 1, self.rows - 1),
                     self.rng.randint(1, seed[1]),
                     self.rng.randint(seed[1] + 1, self.cols - 1))
        else:
            seed = (self.rows // 2, self.cols // 2)
            doors = (self.rows // 4, 3 * self.rows // 4, self.cols // 4, 3 * self.cols // 4)

        if empty:
            return map, seed

        map[seed[0]:seed[0] + 1, :] = 1
        map[:, seed[1]:(seed[1] + 1)] = 1
        map[doors[0]:(doors[0]+1), seed[1]:(seed[1] + 1)] = 0
        map[doors[1]:(doors[1]+1), seed[1]:(seed[1] + 1)] = 0
        map[seed[0]:(seed[0] + 1), doors[2]:(doors[2]+1)] = 0
        map[seed[0]:(seed[0] + 1), doors[3]:(doors[3]+1)] = 0

        # [0 -> 1, 2 -> 3, 0 -> 2, 1 -> 3]
        self.doors = [(doors[0], seed[1]), (doors[1], seed[1]), (seed[0], doors[2]), (seed[0], doors[3])]

        return map, seed

    def render(self, mode='human'):
        im = self._obs_from_state(True)
        res = self._color_obs(im)

        img = cv2.resize(res['img'].astype(np.uint8), dsize=(256, 256), interpolation=cv2.INTER_AREA)
        if mode == 'rgb_array':
            return res['img']
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def seed(self, seed=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()
        return self.rng.seed

    def disconnect(self):
        pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    env = RoomsEnv()
    obs = env.reset()
    env.step(env.action_space.sample())
    print(env.state_cell)
    print(env.apples_cells)
    print(env.mines_cells)
    img = env.render('rgb_array')
    plt.imshow(img)
    # plt.title(f'Horz wind = {res["horz_wind"]}, Vert wind = {res["vert_wind"]}')
    plt.show()
