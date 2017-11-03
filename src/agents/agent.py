import numpy as np
import tensorflow as tf

from exp_replay import ExpReplay

class Agent(object):
    def __init__(self, env, conf):
        self.env = env
        self.lr = conf.lr
        self.discount = conf.discount
        self.batch_size = conf.batch_size

        self.exp_replay = ExpReplay(self.env.getStateShape()[-1:], self.batch_size, self.env.history_length, capacity=conf.exp_replay_size)

        self.test_epsilon = 0.0
        self.train_epsilon = conf.s_eps
        self.train_end_epsilon = conf.e_eps
        self.epsilon_decay = (conf.s_eps - conf.e_eps) / float(conf.eps_decay_steps)
        self.newGame = self.env.newRandomGame if conf.random_start else self.env.newGame

        self.callback = None

    # Randomly take the given number of steps and store experiences
    def randomExplore(self, num_steps):
        step = 0;
        self.newGame()
        while step < num_steps:
            action = self.env.gym_env.action_space.sample()
            self.env.takeAction(action)
            self._storeExperience(action)

            step += 1
            if self.env.done or step >= num_steps:
                self.newGame()

    # Run the agent for the desired number of steps either training or testing
    def train(self, num_eps):
        for i in range(num_eps):
            self.newGame()

            while not self.env.done:
                state = self.env.getState(getDiscreteState=self.discrete)
                action = self._selectAction(state, self.train_epsilon)
                self.env.takeAction(action)
                self._storeExperience(action)
                self._decayEps()

                self.callback.onStep(action, self.env.reward, self.env.done, self.train_epsilon)

                # Call agent specific train method
                self._train()

    # Test the agent for the desired number of steps
    def test(self, num_eps, render=False):
        for i in range(num_eps):
            reward_sum = 0.0;
            self.newGame()

            while not self.env.done:
                if render: self.env.render()
                state = self.env.getState(getDiscreteState=self.discrete)
                action = self._selectAction(state, self.test_epsilon)
                self.env.takeAction(action)

                reward_sum += self.env.reward
                self.callback.onStep(action, self.env.reward, self.env.done, self.test_epsilon)

    # Store the transition into memory
    def _storeExperience(self, action):
        with self.replay_lock:
            state = self.env.getState(getDiscreteState=self.discrete)
            if self.env.history_length == 1:
                self.exp_replay.storeExperience(state, action, self.env.reward, self.env.done)
            else:
                self.exp_replay.storeExperience(state[-1], action, self.env.reward, self.env.done)

    # Decay the random action chance
    def _decayEps(self):
        if self.train_epsilon > self.train_end_eps:
            self.train_epsilon -= self.epsilon_decay
        if self.train_epsilon < self.train_end_eps:
            self.train_epsilon = self.train_end_eps
