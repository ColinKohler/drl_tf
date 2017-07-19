import tensorflow as tf
import numpy as np
import threading
import time

from agent import Agent
from exp_replay import ExpReplay
from network import Network
import constants

class DDQN_Agent(Agent):
    def __init__(self, env, lr, discount, s_eps, e_eps, eps_decay_steps, test_eps, net_config,
                       train_freq, update_target_freq, batch_size, exp_replay_size, saved_model=None, use_tensorboard=True):
        super(DDQN_Agent, self).__init__(env, lr, discount, s_eps, e_eps, eps_decay_steps, test_eps)

        self.net_config = net_config
        self.batch_size = batch_size
        self.queue_size = self.batch_size * 4
        self.use_tensorboard = use_tensorboard
        if self.env.exp_length == 1:
            self.exp_replay = ExpReplay(self.env.state_shape, batch_size, self.env.exp_length, capacity=exp_replay_size)
        else:
            self.exp_replay = ExpReplay(self.env.state_shape[-2:], batch_size, self.env.exp_length, capacity=exp_replay_size)
        self.train_freq = train_freq
        self.update_target_freq = update_target_freq

        self.train_iterations = 0
        self.coord = tf.train.Coordinator()
        self.replay_lock = threading.Lock()

        self._initModel(lr)
        if saved_model is not None:
            self.saver.restore(self.sess, saved_model)

    # Init tensorflow network model
    def _initModel(self, lr):
        self.q_model = Network('q_network', self.env.state_shape, self.env.num_actions,
                               self.net_config, lr, self.batch_size, self.queue_size, log=True)
        self.t_model = Network('t_network', self.env.state_shape, self.env.num_actions,
                               self.net_config, lr, self.batch_size, self.queue_size, log=False)
        self.update_ops = self._setupTargetUpdates()

        self.sess = tf.Session()
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(constants.TF_LOG_PATH + '/train', self.sess.graph)
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        self.sess.graph.finalize()

        print 'Initialized new model...'

    # Save the network model
    def _saveModel(self, loc):
        save_path = self.saver.save(self.sess, loc)
        print 'Model saved in file: %s' % save_path

    # Create ops to copy weights from online net to target net
    def _setupTargetUpdates(self):
        update_ops = list()
        for key in self.q_model.weights.keys():
            update_ops.append(self.t_model.weights[key].assign(self.q_model.weights[key]))

        return update_ops

    # Run the online->target update ops
    def _updateTargetModel(self):
        [self.sess.run(op) for op in self.update_ops]

    # Randomly take the given number of steps and store experiences
    def randomExplore(self, num_steps):
        step = 0;
        self.env.newGame()
        while step < num_steps:
            action = self.env.gym_env.action_space.sample()
            self.env.takeAction(action)
            self._storeExperience(action)

            step += 1
            if self.env.done or step >= num_steps:
                self.env.newGame()

    # Run the agent for the desired number of steps either training or testing
    def run(self, num_steps, train=False, render=False):
        step = 0; episode_num = 0
        eps = self.train_eps if train else self.test_eps
        while step < num_steps:
            reward_sum = 0.0;
            self.env.newGame()

            while not self.env.done:
                if render: self.env.render()

                # Take action greedly with eps proability
                if np.random.rand(1) < eps:
                    action = np.random.randint(self.env.num_actions)
                else:
                    action = self._selectAction(self.env.state)
                self.env.takeAction(action)
                self._storeExperience(action)

                # Train and update networks as neccessary
                if train and step % self.train_freq == 0:
                    self._trainNetwork()
                if train and step % self.update_target_freq == 0:
                    self._updateTargetModel()

                # Handle various vairable updates
                if train: self._decayEps()
                reward_sum += self.env.reward
                step += 1;

                # Handle episode termination
                if self.env.done or step >= num_steps:
                    episode_num += 1
                    self.callback.onStep(action, self.env.reward, True, eps)
                    break
                else:
                    self.callback.onStep(action, self.env.reward, self.env.done, eps)

    # Choose action greedly from network
    def _selectAction(self, state):
        state = state.reshape([1] + self.env.state_shape)
        return self.sess.run(self.q_model.predict_op, feed_dict={self.q_model.batch_input : state})[0]

    # Store the transition into memory
    def _storeExperience(self, action):
        with self.replay_lock:
            self.exp_replay.storeExperience(self.env.state[-1], action, self.env.reward, self.env.done)

    # Get Q values based off predicted max future reward
    def _getTargetQValues(self, states, actions, rewards, states_, done_flags):
        q_values = self.sess.run(self.q_model.q_values, feed_dict={self.q_model.batch_input: states})
        future_actions = self.sess.run(self.q_model.predict_op, feed_dict={self.q_model.batch_input: states_})
        target_q_values_with_idxs = self.sess.run(self.t_model.q_values_with_idxs,
                feed_dict={self.t_model.batch_input: states_,
                           self.t_model.q_value_idxs:[[idx, future_a] for idx, future_a in enumerate(future_actions)]})
        pred_q_values = (1.0 - done_flags) * self.discount * target_q_values_with_idxs + rewards
        errors = np.abs(q_values[:, actions] - pred_q_values)
        return errors, pred_q_values

    # Run the train ops
    def _trainNetwork(self):
        # Wait until the queue has been filled up with experiences
        while self.sess.run(self.q_model.queue_size_op) < self.batch_size: continue
        s, _ = self.sess.run([self.merged, self.q_model.train_op])

        if self.use_tensorboard and self.train_iterations % 100 == 0:
            self.writer.add_summary(s, self.train_iterations)

        self.train_iterations += 1
        #if self.callback:
        #    self.callback.onTrain(self.q_model.loss)

    # Start threads to load training data into the network queue
    def startEnqueueThreads(self):
        threads = list()
        for i in range(constants.NUM_QUEUE_THREADS):
            t = threading.Thread(target=self._enqueueThread)
            t.setDaemon(True)
            t.start()

            threads.append(t)
            self.coord.register_thread(t)
            time.sleep(0.1)

    # Enqueue training data inot the network queue
    def _enqueueThread(self):
        while not self.coord.should_stop():
            # Make sure we only keep recent experiences in the batch
            if self.exp_replay.size < self.batch_size or \
                    self.sess.run(self.q_model.queue_size_op) == self.queue_size:
                continue

            with self.replay_lock:
                states, actions, rewards, states_, done_flags = self.exp_replay.getBatch()
            errors, pred_q_values = self._getTargetQValues(states, actions, rewards, states_, done_flags)

            feed_dict = {
                    self.q_model.queue_input : states,
                    self.q_model.queue_action : actions,
                    self.q_model.queue_label : pred_q_values
            }
            try:
                self.sess.run(self.q_model.enqueue_op, feed_dict=feed_dict)
            except tf.errors.CancelledError:
                return
