import sys
import numpy as np

from agent import Agent
import constants

class Q_Agent(Agent):
    def __init__(self, env, conf):
        Agent.__init__(env, conf)

        self.discrete = True
        num_obs_space = self.env.history_length * reduce(lambda x,y : x*y, self.env.num_discrete_states)
        self.q_table = np.zeros([num_obs_space, self.env.num_actions])

    # Train the agent by updating the Q Table
    def _train(self):
        update = (self.env.reward + self.discount *
                np.max(self.q_table[self.env.state,:]) - self.q_table[self.env.prev_state, self.env.action])
        self.q_table[self.env.prev_state, self.env.action] += self.lr * update

    # Take action greedly with eps proability
    def _selectAction(self, state, eps):
        if np.random.rand(1) < eps:
            action = np.random.randint(self.env.num_actions)
        else:
            action = self._selectAction(d_state)

        return np.argmax(self.q_table[state,:])

    def saveModel(self, name):
        print '[*] Saving checkpoint...'
        np.save('{}{}/{}.npy'.format(constants.TF_MODELS_PATH, self.env.name, name), self.q_table)

    def loadModel(self, model):
        self.q_table = np.load('{}{}/{}.npy'.format(constants.TF_MODELS_PATH, self.env.name, model))
        print '[*] Load SUCCESS: {}{}/{}'.format(constants.TF_MODELS_PATH, self.env.name, model)
