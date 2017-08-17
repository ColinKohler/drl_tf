import tensorflow as tf
import numpy as np
import constants

class Network(object):
    def __init__(self, name, in_shape, out_shape, network_config, lr,
                       lr_min, lr_decay_step, lr_decay, batch_size, queue_size):
        self.name = name
        self.lr = lr
        self.lr_minimum = lr_min
        self.lr_decay_step = lr_decay_step
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.queue_size = queue_size

        self._createNetwork(in_shape, out_shape, network_config)

    # Create network detailed in config dict
    def _createNetwork(self, in_shape, out_shape, network_config):
        with tf.name_scope(self.name):
            # FIFO Queue for training input
            self.queue_input = tf.placeholder(tf.float32, [None] + in_shape, 'queue_input')
            self.queue_action = tf.placeholder(tf.int32, [None], 'queue_action')
            self.queue_label = tf.placeholder(tf.float32, [None], 'queue_label')

            queue_types = [tf.float32, tf.int32, tf.float32]
            queue_shape = [in_shape, [], []]
            queue = tf.FIFOQueue(capacity=self.queue_size, dtypes=queue_types, shapes=queue_shape)
            self.enqueue_op = queue.enqueue_many([self.queue_input,
                                                  self.queue_action,
                                                  self.queue_label])
            self.queue_size_op = queue.size()
            self.queue_close_op = queue.close(cancel_pending_enqueues=True)

            # Read input from queue if no direct input
            batch_q = queue.dequeue_many(self.batch_size)
            batch_input_q, batch_action_q, batch_label_q = batch_q

            self.batch_input = tf.placeholder_with_default(batch_input_q, [None] + in_shape, 'input')
            self.batch_action = tf.placeholder_with_default(batch_action_q, [None], 'action')
            self.batch_label = tf.placeholder_with_default(batch_label_q, [None], 'label')
            self.keep_prob = tf.placeholder_with_default(1.0, None)

            # Convert images from uint8 to float32
            if network_config['is_input_img']:
                with tf.device('/gpu:0'):
                    self.batch_input = tf.div(self.batch_input, 255.0)

            # Setup layers
            self.weights = dict(); prev_layer = self.batch_input
            for layer_config in network_config['layers']:
                layer_weights, layer_biases, layer = self.createLayer(prev_layer, layer_config, out_shape)

                prev_layer = layer
                layer_name = layer_config['name']
                if layer_weights is not None:
                    self.weights[layer_name+'_w'] = layer_weights
                    self.weights[layer_name+'_b'] = layer_biases

            # Setup training and predict ops
            self.global_step = tf.Variable(0, trainable=False)
            self.q_values = prev_layer
            self.max_q_values = tf.reduce_max(self.q_values, reduction_indices=1)
            self.q_value_idxs = tf.placeholder('int32', [None, None], 'q_idxs')
            self.q_values_with_idxs = tf.gather_nd(self.q_values, self.q_value_idxs)

            action_onehot = tf.one_hot(self.batch_action, out_shape, 1.0, 0.0, dtype=tf.float32)
            q_values_acted = tf.reduce_sum(self.q_values * action_onehot, reduction_indices=1)

            with tf.name_scope('loss'):
                self.err = self.batch_label - q_values_acted
                self.clipped_err = tf.where(tf.abs(self.err) < 1.0,
                                            0.5 * tf.square(self.err),
                                            tf.abs(self.err) - 0.5)
                self.loss = tf.reduce_mean(self.clipped_err)

            with tf.name_scope('train'):
                self.lr_op = tf.maximum(self.lr_minimum,
                    tf.train.exponential_decay(
                        self.lr,
                        self.global_step,
                        self.lr_decay_step,
                        self.lr_decay,
                        staircase=True))

                #self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
                optimizer = tf.train.RMSPropOptimizer(self.lr_op, momentum=0.95, epsilon=0.01)
                self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

            with tf.name_scope('predict'):
                self.predict_op = tf.argmax(self.q_values, 1)

    # Create layer detailed in config dict
    def createLayer(self, prev_layer, config, out_shape):
        name = self.name + '_' + config['name']
        if config['type'] == 'conv':
            return self.convLayer(prev_layer, config['filter'], config['stride'], name)
        elif config['type'] == 'fc' and not config['last_layer']:
            return self.fcLayer(prev_layer, config['num_neurons'], name, act=config['act'])
        elif config['type'] == 'fc' and config['last_layer']:
            return self.fcLayer(prev_layer, out_shape, name, act=tf.identity)
        elif config['type'] == 'lstm':
            return self.lstmLayer(prev_layer, config['num_neurons'], name)
        elif config['type'] == 'pool':
            return self.maxPoolLayer(self, prev_layer, config['filter'], config['stride'], name)
        elif config['type'] == 'flatten':
            return None, None, tf.contrib.layers.flatten(prev_layer)
        elif config['type'] == 'dropout':
            return None, None, tf.nn.dropout(prev_layer, self.keep_prob)

    ###############################################################################################
    #                              Layer Construction Utils                                       #
    ###############################################################################################

    # Create convolutional layer
    def convLayer(self, inp, filter_shape, stride_shape, name, padding='SAME', act=tf.nn.relu):
        with tf.variable_scope(name) as scope:
            weights = self.initWeightVariable('weights', filter_shape)
            biases = self.initBiasVariable('bias', filter_shape[-1])
            conv = tf.nn.conv2d(inp, weights, strides=[1,1]+stride_shape,
                                              data_format='NCHW', padding=padding)
            preactivate = tf.nn.bias_add(conv, biases, data_format='NCHW')
            activations = act(preactivate, name='activation')

        return weights, biases, activations

    # Create fully-connected layer
    def fcLayer(self, inp, out_shape, name, act=tf.nn.relu):
        with tf.variable_scope(name) as scope:
            weights = self.initWeightVariable('weights', [inp.get_shape()[-1], out_shape])
            biases = self.initBiasVariable('bias', out_shape)

            preactivations = tf.nn.bias_add(tf.matmul(inp, weights), biases)
            activations = act(preactivations, name='activation')

        return weights, biases, activations

    # Create RNN LSTM Layer
    def lstmLayer(self, inp, num_neurons, seq_length, name):
        with tf.variable_scope(name) as scope:
            cell = tf.contrib.rnn.BasicLSTMCell(num_neurons, forget_bias=1.0, state_is_tuple=True)
            inp = tf.reshape(inp, [-1, self.batch_size, inp.shape[-1]])
            activations, activation_states = tf.nn.dynamic_rnn(cell, inp, seq_length,
                                                               time_major=False, dtype=tf.float32)
            activations = tf.reshape()

    # Create max pooling layer
    def maxPoolLayer(self, inp, filter_shape, stride_shape, name, padding='SAME'):
        return tf.nn.max_pool(inp, ksize=[1, 1]+filter_shape,
                                   strides=[1, 1]+stride_shape,
                                   data_format='NCHW', padding=padding)

    def initWeightVariable(self, name, shape):
        return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

    def initBiasVariable(self, name, shape):
        return tf.get_variable(name, shape=shape, initializer=tf.zeros_initializer())
