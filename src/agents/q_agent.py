import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from agent import Agent
import constants

class Q_Agent(Agent):
    def __init__(self, env, lr, discount, s_eps, e_eps, eps_decay_steps, test_eps):
        super(Q_Agent, self).__init__(env, lr, discount, s_eps, e_eps, eps_decay_steps, test_eps)

        num_obs_space = reduce(lambda x,y : x*y, self.env.num_discrete_states)
        self.q_table = np.zeros([num_obs_space, self.env.num_actions])

    # Run the agent for the desired number of steps either training or testing
    def run(self, num_steps, train=False, render=False):
        step = 0; episode_num = 0
        eps = self.train_eps if train else self.test_eps
        while step < num_steps:
            reward_sum = 0.0
            self.env.newGame()

            while not self.env.done:
                if render: self.env.render()
                d_state = self.env.getState(getDiscreteState=True)

                # Take action greedly with eps proability
                if np.random.rand(1) < eps:
                    action = np.random.randint(self.env.num_actions)
                else:
                    action = np.argmax(self.q_table[d_state,:])
                self.env.takeAction(action)
                self.callback.onStep(action, self.env.reward, self.env.done, eps)

                # Handle various vairable updates
                if train:
                    self._updateQTable(d_state, action)
                    self._decayEps()
                reward_sum += self.env.reward
                step += 1

                # Handle episode termination
                if self.env.done or step >= num_steps:
                    episode_num += 1
                    break

    # Update the state, action Q Value in the Q table
    def _updateQTable(self, s, a):
        new_state = self.env.getState(getDiscreteState=True)
        self.q_table[s, a] += self.lr * (self.env.reward + self.discount * np.max(self.q_table[new_state,:]) - self.q_table[s,a])
