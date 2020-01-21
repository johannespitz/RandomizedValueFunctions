import gym
from gym import spaces
import numpy as np

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Circle(gym.Env):
    """Circle environment
    """
    def __init__(self, radius, expand_observation=True):
        self.radius = int(radius)
        self.expand_observation = expand_observation
        self.edge = self.radius + 1
        self.action_space = spaces.Discrete(4)
        if expand_observation:
            self.observation_space = spaces.Discrete((2 * self.edge + 1) ** 2)
        else:
            self.observation_space = spaces.Box(low=np.array([-self.edge, -self.edge]),
                                                high=np.array([self.edge, self.edge]), dtype=np.int32)
        self.max_nsteps = 4 * 2 * radius + 8
        self.startpos = np.array([0, self.radius])

        self.state = None
        self.prev_state = None
        self.multiply_reward = None
        self.nsteps = None
        self.furthest_point = None
        self.reset()

    def reset(self):
        self.state = self.startpos
        self.prev_state = self.startpos
        self.multiply_reward = 0.0
        self.nsteps = 0
        self.furthest_point = self.startpos
        return self._expand_observation() if self.expand_observation else self.state

    def _expand_observation(self):
        x, y = self.state[0], self.state[1]
        pos = (x + self.edge) *                  1 + \
              (y + self.edge) * (2 * self.edge + 1)
        arr = np.zeros(self.observation_space.n)
        arr[pos] = 1
        return arr

    def _vaild_cell(self, x, y):
        corners = np.asarray([np.array([x - .5, y - .5]),
                              np.array([x - .5, y + .5]),
                              np.array([x + .5, y - .5]),
                              np.array([x + .5, y + .5])])
        dists = np.linalg.norm(corners, axis=1)
        return not (np.all(dists < self.radius) or np.all(dists > self.radius))

    def render(self, mode='human'):
        for y in range(self.edge, -self.edge - 1, -1):
            line = ''
            for x in range(-self.edge, self.edge + 1):
                if x == self.state[0] and y == self.state[1]:
                    line += f'{bcolors.WARNING}X{bcolors.ENDC}'
                elif x == self.prev_state[0] and y == self.prev_state[1]:
                    line += f'{bcolors.OKGREEN}x{bcolors.ENDC}'
                else:
                    line += f'{bcolors.OKBLUE}+{bcolors.ENDC}' if self._vaild_cell(x, y) else f'{bcolors.FAIL}-{bcolors.ENDC}'
            print(line)


    def step(self, action):
        assert self.action_space.contains(action)
        self.prev_state = self.state

        if action == 0:
            self.state = self.state + np.array([0, -1])
        elif action == 1:
            self.state = self.state + np.array([0,  1])
        elif action == 2:
            self.state = self.state + np.array([-1, 0])
        elif action == 3:
            self.state = self.state + np.array([ 1, 0])

        if np.all(self.state == np.array([-1, self.radius])):
            reward = 1 * self.multiply_reward
            is_done = True
        elif self._vaild_cell(self.state[0], self.state[1]):
            if np.linalg.norm(self.furthest_point - np.array([0, self.radius])) < np.linalg.norm(self.state - np.array([0, self.radius])):
                self.furthest_point = self.state
                reward = 0.001
            else:
                reward = 0.0
            is_done = self.nsteps >= self.max_nsteps
        else:
            reward = 0.0 # TODO: try 0
            is_done = True

        observation = self._expand_observation() if self.expand_observation else self.state

        self.nsteps += 1
        if self.state[0] < 0:
            self.multiply_reward = 1.0

        return observation, reward, is_done, None




if __name__ == '__main__':

    test = Circle(6)
    test.render()
    print(test.step(3))
    test.render()
    print(test.step(0))
    test.render()
    print(test.step(0))
    test.render()
    print(test.step(2))
    test.render()
    print(test.step(2))
    test.render()
    print(test.step(1))
    test.render()
    print(test.step(1))
    test.render()