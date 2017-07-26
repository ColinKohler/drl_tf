
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

        self.callback = None

    # Decay the random action chance
    def _decayEps(self):
        if self.train_eps > self.train_end_eps:
            self.train_eps -= self.decay_eps
        if self.train_eps < self.train_end_eps:
            self.train_eps = self.train_end_eps

    # Run the agent for the desired number of steps either training or testing
    def run(self, num_steps, train=False, render=False):
        raise NotImplementedError("All agents must have a run method implemented...")
