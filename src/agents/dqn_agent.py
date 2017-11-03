import sys
import tensorflow as tf
import numpy as np
import threading
import time

from agent import Agent
from network import Network
import constants

class DQN_Agent(Agent):
    def __init__(self, sess, env, conf):
        Agent.__init__(env, conf)

        if conf.network_config == 'MLP':
            self.net_config = constants.MLP
        elif conf.network_config == 'CNN':
            self.net_config = constants.CNN
        else:
            print 'Bad network config given! Exiting....'
            sys.exit(-1)

        self.sess = sess
        self.discrete = False
        self.queue_size = self.batch_size * 4
        self.train_freq = conf.train_freq
        self.update_target_freq = conf.update_target_freq
        self.train_iterations = 0

        self.coord = tf.train.Coordinator()
        self.replay_lock = threading.Lock()

        self._initModel(conf.lr, conf.lr_minimum, conf.lr_decay_step, conf.lr_decay)
        self._updateTargetModel()

    # Init tensorflow network model
    def _initModel(self, lr, lr_min, lr_decay_step, lr_decay):
        state_shape = [self.env.history_length, self.env.getStateShape()[-1]]
        self.q_model = Network('q_network', state_shape, self.env.num_actions,
                               self.net_config, lr, lr_min, lr_decay_step, lr_decay,
                               self.batch_size, self.queue_size)
        self.t_model = Network('t_network', state_shape, self.env.num_actions,
                               self.net_config, lr, lr_min, lr_decay_step, lr_decay,
                               self.batch_size, self.queue_size)
        self.sess.run(tf.global_variables_initializer())
        self.copy_op = self._setupTargetUpdates()

    # Create ops to copy weights from online net to target net
    def _setupTargetUpdates(self):
        copy_ops = list()
        for key in self.q_model.weights.keys():
            copy_ops.append(self.t_model.weights[key].assign(self.q_model.weights[key]))

        return tf.group(*copy_ops, name='copy_op')

    # Run the online->target update ops
    def _updateTargetModel(self):
        self.sess.run(self.copy_op)

    # Choose action greedly from network
    def _selectAction(self, state, eps):
        state = state.reshape([1] + self.env.getStateShape())
        q_probs = self.sess.run(self.q_model.q_dist, feed_dict={self.q_model.batch_input : state, self.q_model.keep_prob : eps})
        action_value = np.random.choice(q_probs[0], p=q_probs[0])
        return np.argmax(q_probs[0] == action_value)

    # Train the Q network and update the target network as necessary
    def _train(self):
        if self.env.eps_steps % self.train_freq == self.train_freq - 1:
            self._trainNetwork()
        if self.env.eps_steps % self.update_target_freq == self.update_target_freq - 1:
            self._updateTargetModel()

    # Run the train ops
    def _trainNetwork(self):
        # Wait until the queue has been filled up with experiences
        for i in range(self.sgd_steps):
            while self.sess.run(self.q_model.queue_size_op) < self.batch_size: continue
            q_values, loss, _ = self.sess.run([self.q_model.q_values, self.q_model.loss, self.q_model.train_op])

            self.train_iterations += 1
            self.callback.onTrain(q_values, loss)

    # Get Q values based off predicted max future reward
    def _getTargetQValues(self, states, actions, rewards, states_, done_flags):
        future_actions = self.sess.run(self.q_model.predict_op, feed_dict={self.q_model.batch_input: states_})
        target_q_values_with_idxs = self.sess.run(self.t_model.q_values_with_idxs,
                    feed_dict={self.t_model.batch_input: states_,
                               self.t_model.q_value_idxs:[[idx, future_a] for idx, future_a in enumerate(future_actions)]})
        pred_q_values = (1.0 - done_flags) * self.discount * target_q_values_with_idxs + rewards

        return pred_q_values

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

    # Stop threads thats load training data
    def stopEnqueueThreads(self):
        self.coord.request_stop()

    # Enqueue training data inot the network queue
    def _enqueueThread(self):
        while not self.coord.should_stop():
            # Make sure we only keep recent experiences in the batch
            if self.exp_replay.size < self.batch_size or \
                    self.sess.run(self.q_model.queue_size_op) == self.queue_size:
                continue

            with self.replay_lock:
                states, actions, rewards, states_, done_flags = self.exp_replay.getBatch()
            pred_q_values = self._getTargetQValues(states, actions, rewards, states_, done_flags)

            feed_dict = {
                    self.q_model.queue_input : states,
                    self.q_model.queue_action : actions,
                    self.q_model.queue_label : pred_q_values
            }
            try:
                self.sess.run(self.q_model.enqueue_op, feed_dict=feed_dict)
            except tf.errors.CancelledError:
                return
