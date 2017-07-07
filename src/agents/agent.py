
class Agent(object):
    def __init__(self, env, lr,  discount, s_eps, e_eps, eps_decay_steps, test_eps):
        self.env = env
        self.lr = lr
        self.discount = discount

        self.test_eps = test_eps
        self.train_eps = s_eps
        self.train_end_eps = e_eps
        self.decay_eps = (s_eps - e_eps) / float(eps_decay_steps)

        self.callback = None

    # Decay the random action chance
    def _decayEps(self):
        if self.train_eps > self.train_end_eps:
            self.train_eps -= self.decay_eps
        if self.train_eps < self.train_end_eps:
            self.train_eps = self.train_end_eps

    # Run the agent for the desired number of steps either training or testing
    def run(self, num_steps, train=False):
        raise NotImplementedError("All agents must have a run method implemented...")
