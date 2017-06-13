import tensorflow as tf
import numpy as np
import threading

from network import Network
import constants

class Agent(object):
    def __init__(self, ):
        self.num_actions = num_actions
        self.state_shape = list(state_shape)
        self.net_config = net_config
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.queue_size = self.batch_size * 4
        self.use_tensorboard = use_tensorboard

        self.train_iterations = 0
        self.callback = None
        self.coord = tf.train.Coordinator()
        self.replay_loc = threading.Lock()

        if saved_model is not None:
            self.sess = self._loadModel(saved_model)
        else:
            self.sess = self._initModel(lr)

    # Init tensorflow network model
    def _initModel(self, lr):
        self.q_model = Network('q_network', self.state_shape, self.num_actions,
                               constants.MLP, lr, self.batch_size, self.queue_size)
        self.t_model = Network('t_network', self.state_shape, self.num_actions,
                               constants.MLP, lr, self.batch_size, self.queue_size)
        self.update_ops = self._setupTargetUpdates()

        sess = tf.Session()
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(constants.TF_LOG_PATH + '/train', sess.graph)
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

        print 'Initialized new model...'
        return sess

    # Save the network model
    def _saveModel(self, loc):
        save_path = self.saver.save(self.sess, loc)
        print 'Model saved in file: %s' % save_path

    # Load a pre-trained network
    def _loadModel(self, ):
        pass

    # Create ops to copy weights from online net to target net
    def _setupTargetUpdates(self):
        update_ops = list()
        for key in self.q_model.weights.keys():
            update_ops.append(self.t_model.weights[key].assign(self.q_model.weights[key]))

        return update_ops

    # Run teh online->target update ops
    def _updateTargetModel(self):
        [self.sess.run(op) for op in self.update_ops]

    # Train the agent for the desired number of steps
    def trainAgent(self, ):
        pass

    # Test the agent
    def testAgent(self, ):
        pass

    # Choose action greedly from network
    def _selectAction(self, ):
        pass

    # Get Q values based off predicted max future reward
    def _getTargetQValues(self, ):
        pass

    # Run the train ops
    def _trainNetwork(self, ):
        pass

    # Start threads to load training data into the network queue
    def startEnqueueThreads(self):
        pass

    # Enqueue training data inot the network queue
    def _enqueueThread(self):
        pass
