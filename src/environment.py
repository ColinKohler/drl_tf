import gym
import numpy as np
import PIL

import constants

class Environment(object):
    def __init__(self, conf):
        self.name = conf.env_name
        self.gym_env = gym.make(self.name)
        self.max_eps_steps = self.gym_env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
        self.history_length = conf.history_length

        # Set state variables for image/non-images
        self.is_state_image = self.name in constants.ENVS_WITH_IMAGE_STATES
        if self.is_state_image:
            self.frame_size = constants.FRAME_SIZE
            self.state_shape = [self.history_length, self.frame_size, self.frame_size]
        else:
            self.frame_size = None
            self.state_shape = [self.history_length] + list(self.gym_env.observation_space.shape)

        # Set discrete or continuous state
        self.num_actions = self.gym_env.action_space.n
        if type(self.gym_env.observation_space) is gym.spaces.box.Box:
            self.is_env_state_discrete = False
            self.state = np.zeros(self.state_shape, dtype=np.float32)
            self._discretizeContinuousSpace()
        else:
            self.is_env_state_discrete = True
            self.num_discrete_states = [self.gym_env.observation_space.n]
            self.state = np.zeros(self.state_shape, dtype=np.uint8)

    # Render the gym env
    def render(self):
        self.gym_env.render()

    # Get the current state
    def getState(self, getDiscreteState=False):
        if getDiscreteState:
            return np.array(self._getDiscreteState())
        else:
            return self.state

    # Get the state shape
    def getStateShape(self, getDiscreteState=False):
        if getDiscreteState:
            return [self.history_length, 1]
        else:
            return self.state_shape

    # Get the discrete state from the current state
    def _getDiscreteState(self):
        #discrete_state = 0
        discrete_state = list()
        for i, (discrete_bin, s) in enumerate(zip(self.discrete_bins, self.state.flatten())):
            #discrete_state += 2**i * self._findNearest(discrete_space, s)
            discrete_state.append(self._findNearest(discrete_bin, s))
        return np.array(discrete_state)

    # Take action in the env
    def takeAction(self, action):
        new_state, self.reward, self.done, self.info = self.gym_env.step(action)
        self.reward = max(-1.0, min(1.0, self.reward))
        new_state = self._processState(new_state)
        self.eps_steps += 1

        if self.history_length == 1:
            np.copyto(self.state, new_state)
        else:
            self.state[:-1] = self.state[1:]
            self.state[-1] = new_state

        # Reset the env if past max number of steps per episode
        if self.eps_steps >= self.max_eps_steps:
            self.done = True

    # Reset the env state to a random starting state
    def newRandomGame(self):
        num_steps = np.random.randint(self.history_length, constants.NEW_GAME_MAX_RANDOM_STEPS)

        self.newGame()
        for _ in range(num_steps):
            self.takeAction(self.gym_env.action_space.sample())

            if self.done:
                self.newGame()

    # Reset the env state to a starting state
    def newGame(self):
        self.state *= 0
        self.done = False
        self.reward = None
        self.info = None
        self.eps_steps = 0

        new_state = self._processState(self.gym_env.reset())
        self.state[:] = new_state

    # Preprocess state if state is image
    def _processState(self, state):
        if self.is_state_image:
            # Using Pillow-SIMD for fast image processing
            im = PIL.Image.fromarray(state)
            g_im = im.convert('L')
            r_im = g_im.resize(self.state_shape[-2:])
            state = np.asarray(r_im, dtype=np.uint8)
        elif self.is_env_state_discrete:
            tmp_state = np.zeros(self.state_shape, dtype=np.uint8)
            tmp_state[state] = 1
            state = tmp_state
        else:
            state = state

        return state

    # Discretizes a continuous gym env observation space
    def _discretizeContinuousSpace(self):
        low = self.gym_env.observation_space.low
        high = self.gym_env.observation_space.high

        self.discrete_bins = list()
        self.num_discrete_states = list()
        for l, h in zip(low, high):
            states = np.linspace(l, h, 100)
            self.discrete_bins.append(states)
            self.num_discrete_states.append(len(states))

        self.discrete_bins = np.array(self.discrete_bins)
        #self.discrete_bins = [self.discrete_bins for i in range(self.history_length)]

    # Find the index for the element in a array that is closest to the given value
    def _findNearest(self, array, value):
        idx = np.digitize(value, array)
        return idx
