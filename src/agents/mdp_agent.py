import numpy as np

from agent import Agent
import constants

class MLP_Agent(Agent):
    def __init__(self, env, discount, s_eps, e_eps, eps_decay_steps, test_eps):
        super(DDQN_Agent, self).__init__(env, discount, s_eps, e_eps, eps_decay_steps, test_eps)

    # Run the agent for the desired number of steps either training or testing
    def run(self, num_steps, train=False):
        step = 0; episode_num = 0
        eps = self.train_eps if train else self.test_eps
        while step < num_steps:
            reward_sum = 0.0; done = False
            self.env.newRandomGame()

            while not done:
                # Take action greedly with eps proability
                if np.random.rand(1) < eps:
                    action = np.random.randint(self.env.num_actions)
                else:
                    action = self._selectAction(self.env.state)
                self.env.takeAction(action)

                # TODO: Update MDP, etc, etc

                # Handle various vairable updates
                if train: self._decayEps()
                reward_sum += self.env.reward
                step += 1

                # Handle episode termination
                if done or step >= num_steps:
                    episode_num += 1
                    break

    # Choose action greedly from MDP
    def _selectAction(self, state):
        pass
