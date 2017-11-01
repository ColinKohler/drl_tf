import sys
import tensorflow as tf
import numpy as np
import threading
import time

from agent import Agent
from exp_replay import ExpReplay
from recurrent_network import RecurrentNetwork
import constants

class RDQN_Agent(Agent):
    def __init__(self, sess, env, conf):
        super(RDQN_Agent, self).__init__(env, conf)
        self.num_lstm_units = 32
        self.lstm = True

        if conf.network_config == 'R_MLP':
            self.net_config = constants.R_MLP
            print 'Init new Recurrant MLP network...'
        else:
            print 'Bad network config given! Exiting....'
            sys.exit(-1)

        self.sess = sess
        self.discrete = False
        self.batch_size = conf.batch_size
        self.queue_size = self.batch_size * 4
        self.unroll = conf.unroll
        self.exp_replay = ExpReplay(self.env.state_shape[-1:], self.batch_size, self.unroll, capacity=conf.exp_replay_size)
        self.train_freq = conf.train_freq
        self.update_target_freq = conf.update_target_freq
        self.double_q = conf.double_q

        self.train_iterations = 0
        self.coord = tf.train.Coordinator()
        self.replay_lock = threading.Lock()

        self._initModel(conf.lr, conf.lr_minimum, conf.lr_decay_step, conf.lr_decay)
        self._updateTargetModel()

    # Init tensorflow network model
    def _initModel(self, lr, lr_min, lr_decay_step, lr_decay):
        state_shape = [self.unroll, self.env.state_shape[-1]]
        self.q_model = RecurrentNetwork('q_network', state_shape, self.env.num_actions,
                                        self.net_config, lr, lr_min, lr_decay_step, lr_decay,
                                        self.batch_size, self.queue_size, self.unroll)
        self.t_model = RecurrentNetwork('t_network', state_shape, self.env.num_actions,
                                        self.net_config, lr, lr_min, lr_decay_step, lr_decay,
                                        self.batch_size, self.queue_size, self.unroll)
        self.sess.run(tf.global_variables_initializer())
        self.copy_op = self._setupTargetUpdates()
        self.zero_state = tf.contrib.rnn.LSTMStateTuple(np.zeros((self.batch_size, self.num_lstm_units), dtype=np.float32),
                                                        np.zeros((self.batch_size, self.num_lstm_units), dtype=np.float32))

    # Create ops to copy weights from online net to target net
    def _setupTargetUpdates(self):
        copy_ops = list()
        for key in self.q_model.weights.keys():
            copy_ops.append(self.t_model.weights[key].assign(self.q_model.weights[key]))

        # Setup copy ops for lstm (requires more work as vars are not accessable until var initialization)
        if self.net_config == constants.R_MLP:
            for layer_config in self.net_config['layers']:
                if layer_config['type'] != 'lstm':
                    continue

                q_scope = 'q_network_' + layer_config['name']
                t_scope = 't_network_' + layer_config['name']
                q_lstm_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=q_scope)
                t_lstm_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=t_scope)

                for qv, tv in zip(q_lstm_vars, t_lstm_vars):
                    copy_ops.append(tv.assign(qv))

        return tf.group(*copy_ops, name='copy_op')

    # Run the online->target update ops
    def _updateTargetModel(self):
        self.sess.run(self.copy_op)

    # Randomly take the given number of steps and store experiences
    def randomExplore(self, num_steps):
        step = 0;
        self.newGame()
        while step < num_steps:
            action = self.env.gym_env.action_space.sample()
            self.env.takeAction(action)
            self._storeExperience(action)

            step += 1
            if self.env.done or step >= num_steps:
                self.newGame()

    # Run the agent for the desired number of steps either training
    def train(self, num_steps):
        step = 0; episode_num = 0
        eps = self.train_eps
        while step < num_steps:
            reward_sum = 0.0;
            self.newGame()
            self.lstm_state = tf.contrib.rnn.LSTMStateTuple(np.zeros((1, self.num_lstm_units), dtype=np.float32),
                                                            np.zeros((1, self.num_lstm_units), dtype=np.float32))

            while not self.env.done:
                # Take action greedly with eps proability
                if np.random.rand() < eps:
                    action = np.random.randint(self.env.num_actions)
                else:
                    action = self._selectAction(self.env.state, 0.1)
                #action = self._selectAction(self.env.state, eps)

                self.env.takeAction(action)
                self._storeExperience(action)
                eps = self._decayEps()

                reward_sum += self.env.reward
                step += 1

                # Train and update networks as neccessary
                if step % self.train_freq == 0:
                    self._trainNetwork()
                if step % self.update_target_freq == self.update_target_freq - 1:
                    self._updateTargetModel()

                # Handle episode termination
                if self.env.done or step >= num_steps:
                    episode_num += 1
                    self.callback.onStep(action, self.env.reward, True, eps)
                    break
                else:
                    self.callback.onStep(action, self.env.reward, self.env.done, eps)

    # Choose action greedly from network
    def _selectAction(self, state, eps):
        pad_state = np.zeros((self.unroll, self.env.state_shape[-1]))
        pad_state[:state.shape[0], :state.shape[1]] = state
        pad_state = pad_state.reshape([1] + list(pad_state.shape))
        #state = state.reshape([1] + self.env.state_shape)
        # return self.sess.run(self.q_model.predict_op, feed_dict={self.q_model.batch_input : state})[0]
        feed_dict={self.q_model.batch_input : pad_state,
                   self.q_model.seq_length : self.env.history_length,
                   self.q_model.init_state : self.lstm_state,
                   self.q_model.keep_prob : (1-eps)+0.1,}

        actions, self.lstm_state = self.sess.run([self.q_model.predict_op, self.q_model.lstm_state],
                                                 feed_dict=feed_dict)
        return actions[0]

    # Store the transition into memory
    def _storeExperience(self, action):
        with self.replay_lock:
            if self.env.history_length == 1:
                self.exp_replay.storeExperience(self.env.state, action, self.env.reward, self.env.done)
            else:
                self.exp_replay.storeExperience(self.env.state[-1], action, self.env.reward, self.env.done)

    # Get Q values based off predicted max future reward
    def _getTargetQValues(self, states, actions, rewards, states_, done_flags):
        if self.double_q:
            #q_values = self.sess.run(self.q_model.q_values, feed_dict={self.q_model.batch_input: states})
            future_actions = self.sess.run(self.q_model.predict_op, feed_dict={self.q_model.batch_input: states_,
                                                self.q_model.init_state : self.zero_state})
            target_q_values_with_idxs = self.sess.run(self.t_model.q_values_with_idxs,
                    feed_dict={self.t_model.batch_input: states_,
                               self.t_model.init_state : self.zero_state,
                               self.t_model.q_value_idxs:[[idx, future_a] for idx, future_a in enumerate(future_actions)]})
            pred_q_values = (1.0 - done_flags) * self.discount * target_q_values_with_idxs + rewards
            #errors = np.abs(q_values[:, actions] - pred_q_values)
        else:
            max_future_q_values = self.sess.run(self.t_model.max_q_values, feed_dict={self.t_model.batch_input: states_})
            pred_q_values = (1.0 - done_flags) * self.discount * max_future_q_values + rewards

        errors = None
        return errors, pred_q_values

    # Run the train ops
    def _trainNetwork(self):
        # Wait until the queue has been filled up with experiences
        while self.sess.run(self.q_model.queue_size_op) < self.batch_size: continue
        q_values, loss, _ = self.sess.run([self.q_model.q_values, self.q_model.loss, self.q_model.train_op],
                feed_dict={self.q_model.init_state : self.zero_state})

        self.train_iterations += 1
        if self.callback:
            self.callback.onTrain(q_values, loss)

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
            errors, pred_q_values = self._getTargetQValues(states, actions, rewards, states_, done_flags)

            feed_dict = {
                    self.q_model.queue_input : states,
                    self.q_model.queue_action : actions,
                    self.q_model.queue_label : pred_q_values,
                    self.q_model.init_state : self.zero_state
            }
            try:
                self.sess.run(self.q_model.enqueue_op, feed_dict=feed_dict)
            except tf.errors.CancelledError:
                return
