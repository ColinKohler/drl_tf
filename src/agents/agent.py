import numpy as np

class Agent(object):
    def __init__(self, env, conf):
        self.env = env
        self.lr = conf.lr
        self.discount = conf.discount

        self.test_eps = 0.0
        self.train_eps = conf.s_eps
        self.train_end_eps = conf.e_eps
        self.decay_eps = (conf.s_eps - conf.e_eps) / float(conf.eps_decay_steps)
        self.newGame = self.env.newRandomGame if conf.random_start else self.env.newGame

        self.callback = None

    # Decay the random action chance
    def _decayEps(self):
        if self.train_eps > self.train_end_eps:
            self.train_eps -= self.decay_eps
        if self.train_eps < self.train_end_eps:
            self.train_eps = self.train_end_eps
        return self.train_eps

    # Run the agent for the desired number of steps either training or testing
    def train(self, num_steps):
        raise NotImplementedError("All agents must have a run method implemented...")

    # Test the agent for the desired number of steps
    def test(self, num_eps, render=False):
        for i in range(num_eps):
            reward_sum = 0.0;
            self.newGame()

            while not self.env.done:
                if render: self.env.render()
                state = self.env.getState(getDiscreteState=self.discrete)
                action = self._selectAction(state, 0.1)
                self.env.takeAction(action)
                reward_sum += self.env.reward

                # Handle episode termination
                if self.env.done:
                    self.callback.onStep(action, self.env.reward, True, self.test_eps)
                    break
                else:
                    self.callback.onStep(action, self.env.reward, self.env.done, self.test_eps)
