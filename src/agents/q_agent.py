import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from agent import Agent
import constants

class Q_Agent(Agent):
    def __init__(self, env, conf):
        super(Q_Agent, self).__init__(env, conf)
        self.discrete = True

        num_obs_space = self.env.history_length * reduce(lambda x,y : x*y, self.env.num_discrete_states)
        self.q_table = np.zeros([num_obs_space, self.env.num_actions])

    # Run the agent for the desired number of steps either training or testing
    def train(self, num_steps):
        step = 0; episode_num = 0
        eps = self.train_eps
        while step < num_steps:
            reward_sum = 0.0
            self.env.newGame()

            while not self.env.done:
                d_state = self.env.getState(getDiscreteState=True)

                # Take action greedly with eps proability
                if np.random.rand(1) < eps:
                    action = np.random.randint(self.env.num_actions)
                else:
                    action = self._selectAction(d_state)
                self.env.takeAction(action)
                self._updateQTable(d_state, action)
                eps = self._decayEps()

                reward_sum += self.env.reward
                step += 1

                # Handle episode termination
                if self.env.done:
                    episode_num += 1
                    self.callback.onStep(action, self.env.reward, True, eps)
                    break
                else:
                    self.callback.onStep(action, self.env.reward, self.env.done, eps)

    # Update the state, action Q Value in the Q table
    def _updateQTable(self, s, a):
        new_state = self.env.getState(getDiscreteState=True)
        self.q_table[s, a] += self.lr * (self.env.reward + self.discount * np.max(self.q_table[new_state,:]) - self.q_table[s,a])

    def _selectAction(self, state, a=0.0):
        return np.argmax(self.q_table[state,:])

    def saveModel(self, name):
        print '[*] Saving checkpoint...'
        np.save('{}{}/{}.npy'.format(constants.TF_MODELS_PATH, self.env.name, name), self.q_table)

    def loadModel(self, model):
        self.q_table = np.load('{}{}/{}.npy'.format(constants.TF_MODELS_PATH, self.env.name, model))
        print '[*] Load SUCCESS: {}{}/{}'.format(constants.TF_MODELS_PATH, self.env.name, model)
