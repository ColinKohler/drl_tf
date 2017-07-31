import tensorflow as tf

# Important Filepaths
TF_LOG_PATH = '/home/colin/workspace/machine_learning_experiments/tf_logs/'
TF_MODELS_PATH = '/home/colin/workspace/machine_learning_experiments/drl_experiments/models/'
STATS_PATH = '/home/colin/workspace/machine_learning_experiments/drl_experiments/stats/'
PLOTS_PATH = '/home/colin/workspace/machine_learning_experiments/drl_experiments/plots/'

# Various important settings
ENVS_WITH_IMAGE_STATES = ['Seaquest-v0']
FRAMES_PER_STATE = 4
FRAME_SIZE = 84

NUM_QUEUE_THREADS = 1
NEW_GAME_MAX_RANDOM_STEPS = 30

# Network layer configs
MLP = {'is_input_img' : False,
       'layers' : [
           {'name' : 'fc_1',
            'type' : 'fc',
            'num_neurons' : 64,
            'act' : tf.sigmoid,
            'last_layer' : False},
           {'name' : 'fc_2',
            'type' : 'fc',
            'num_neurons' : 8,
            'act' : tf.sigmoid,
            'last_layer' : False},
           {'name' : 'output',
            'type' : 'fc',
            'last_layer' : True}]}

CNN = {'is_input_img' : True,
        'layers' : [
            {'name' : 'conv_1',
             'type' : 'conv',
             'filter' : [8, 8, 4, 32],
             'stride' : [4, 4]},
            {'name' : 'conv_2',
             'type' : 'conv',
             'filter' : [4, 4, 32, 64],
             'stride' : [2, 2]},
            {'name' : 'conv_3',
             'type' : 'conv',
             'filter' : [3, 3, 64, 64],
             'stride' : [1, 1]},
            {'name' : 'conv_3_flat',
             'type' : 'flatten'},
            {'name' : 'fc_1',
             'type' : 'fc',
             'num_neurons' : 512,
             'act' : tf.nn.relu,
             'last_layer' : False},
            {'name' : 'output',
             'type' : 'fc',
             'last_layer' : True}]}
