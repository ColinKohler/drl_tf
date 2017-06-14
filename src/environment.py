import gym
import numpy as np
import PIL

import constants

class Environment(object):
    def __init__(self, env_name, render=False):
        self.gym_env = gym.make(env_name)
        self.render = render

        self.state_shape = list(self.gym_env.observation_space.shape)
        self.num_actions = self.gym_env.action_space.n
        self.is_state_image = env_name in constants.ENVS_WITH_IMAGE_STATES
        self.state = np.zeros(self.state_shape, dtype=np.float32)
        self.state_ = np.zeros(self.state_shape, dtype=np.float32)

        if self.is_state_image:
            self.exp_length = self.state_shape[1]
            self.frame_size = self.state_shape[-2:]
        else:
            self.exp_length = 1
            self.frame_size = None

    # Take a random action in the env
    def takeRandomAction(self):
        self.takeAction(self.gym_env.action_space.sample())

    # Take action in the env
    def takeAction(self, action):
        if self.render: self.gym_env.render()
        new_state, self.reward, self.done, self.info = self.gym_env.step(action)
        new_state = self._processState(new_state)

        np.copyto(self.state, self.state_)
        np.copyto(self.state_, new_state)

    # Reset the env state to a random starting state
    def newRandomGame(self):
        self._newGame()
        for _ in range(np.random.randint(self.exp_length, constants.NEW_GAME_MAX_RANDOM_STEPS)):
            action = self.gym_env.action_space.sample()
            self.takeAction(action)

            if self.done:
                self._newGame()

    # Reset the env state to a starting state
    def _newGame(self):
        self.state *= 0
        self.state_ *= 0

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
