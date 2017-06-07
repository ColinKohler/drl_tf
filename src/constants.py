# Important Filepaths
TF_LOG_PATH = '/home/colin/tf_logs/'
TF_MODELS_PATH = '/home/colin/workspace/machine_learning_toolbox/saved_models/dqn/'
STATS_PATH = '/home/colin/workspace/rl_experiments/dqn/stats'
PLOTS_PATH = '/home/colin/workspace/rl_experiments/dqn/plots'

# Various important settings
NUM_PROC_THREADS = 1
NEW_GAME_MAX_RANDOM_STEPS = 30

# Network layer configs
MLP = {'is_input_img' : False,
       'layers' : [
           {'name' : 'fc_1',
            'type' : 'fc',
            'num_neurons' : 64,
            'last_layer' : False},
           {'name' : 'fc_2',
            'type' : 'fc',
            'num_neurons' : 64,
            'last_layer' : False},
           {'name' : 'output',
            'type' : 'fc',
            'last_layer' : True}]}
