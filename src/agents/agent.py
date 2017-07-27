import numpy as np

class Agent(object):
    def __init__(self, sess, env, conf):
        self.sess = sess
        self.env = env
        self.lr = conf.lr
        self.discount = conf.discount

        self.test_eps = conf.t_eps
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

    # Run the agent for the desired number of steps either training or testing
    def train(self, num_steps):
        raise NotImplementedError("All agents must have a run method implemented...")

    # Test the agent for the desired number of steps
    def test(self, num_steps, render=False):
        step = 0; episode_num = 0
        while step < num_steps:
            reward_sum = 0.0;
            self.newGame()

            while not self.env.done:
                if render: self.env.render()

                # Take action greedly with eps proability
                if np.random.rand(1) < self.test_eps:
                    action = np.random.randint(self.env.num_actions)
                else:
                    action = self._selectAction(self.env.state)
                self.env.takeAction(action)

                reward_sum += self.env.reward
                step += 1

                # Handle episode termination
                if self.env.done or step >= num_steps:
                    episode_num += 1
                    self.callback.onStep(action, self.env.reward, True, self.test_eps)
                    break
                else:
                    self.callback.onStep(action, self.env.reward, self.env.done, self.test_eps)
