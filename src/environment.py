import gym
import numpy as np
import PIL

import constants

class Environment(object):
    def __init__(self, env_name):
        self.gym_env = gym.make(env_name)

        self.is_state_image = env_name in constants.ENVS_WITH_IMAGE_STATES
        if self.is_state_image:
            self.exp_length = self.state_shape[1]
            self.frame_size = self.state_shape[-2:]
        else:
            self.exp_length = 1
            self.frame_size = None

        self.num_actions = self.gym_env.action_space.n
        if type(self.gym_env.observation_space) is gym.spaces.box.Box:
            self.isEnvStateDiscrete = False
            self.state_shape = list(self.gym_env.observation_space.shape)
            self.state = np.zeros(self.state_shape, dtype=np.float32)
            self._discretizeContinuousSpace()
        else:
            self.isEnvStateDiscrete = True
            self.state_shape = self.gym_env.observation_space.n
            self.state = np.zeros(1, dtype=np.uint8)

    def render(self):
        self.gym_env.render()

    def getState(self, getDiscreteState=False):
        if self.isEnvStateDiscrete:
            return self.state[0]
        elif getDiscreteState:
            return self._getDiscreteState()
        else:
            return self.state

    # Get the discrete state from the current state
    def _getDiscreteState(self):
        discrete_state = 0
        for i, (discrete_space, s) in enumerate(zip(self.discrete_spaces, self.state)):
            discrete_state += 2**i * self._findNearest(discrete_space, s)
        return discrete_state

    # Take a random action in the env
    def takeRandomActions(self, steps):
        self._newGame()
        for _ in range(steps):
            self.takeAction(self.gym_env.action_space.sample())

            if self.done:
                self._newGame()

    # Take action in the env
    def takeAction(self, action):
        new_state, self.reward, self.done, self.info = self.gym_env.step(action)
        new_state = self._processState(new_state)

        np.copyto(self.state, new_state)

    # Reset the env state to a random starting state
    def newRandomGame(self):
        num_steps = np.random.randint(self.exp_length, constants.NEW_GAME_MAX_RANDOM_STEPS)
        self.takeRandomActions(num_steps)

    # Reset the env state to a starting state
    def _newGame(self):
        self.state *= 0
        self.done = False
        self.reward = None
        self.info = None

        new_state = self._processState(self.gym_env.reset())
        self.state[:] = new_state

    # Preprocess state if state is image
    def _processState(self, state):
        if self.is_state_image:
            # Using Pillow-SIMD for fast image processing
            im = PIL.Image.fromarray(state)
            g_im = im.convert('L')
            r_im = g_im.resize(self.frame_size)
            state = np.asarray(r_im, dtype=np.uint8)
        else:
            state = state

        return state

    # Discretizes a continuous gym env observation space
    def _discretizeContinuousSpace(self):
        low = self.gym_env.observation_space.low
        high = self.gym_env.observation_space.high

        self.discrete_spaces = list()
        self.num_discrete_states = list()
        for l, h in zip(low, high):
            states = np.linspace(l, h, 50)
            self.discrete_spaces.append(states)
            self.num_discrete_states.append(len(states))

    # Find the index for the element in a array that is closest to the given value
    def _findNearest(self, array, value):
        idx, val = min(enumerate(array), key=lambda x: abs(x[1]-value))
        return idx
