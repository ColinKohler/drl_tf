#!/usr/bin/python

import os
import gym
import gym.spaces
import numpy as np
import matplotlib.pyplot as plt
import argparse

from agents.ddqn_agent import DDQN_Agent
from environment import Environment
import constants
from statistics import Statistics

# Custom Envs
import helping_hands

# Gets the actions for the current env from the constants
def getActionsForEnv(env_name):
    if env_name in constants.ACTIONS:
        return constants.ACTIONS[env_name]
    else:
        print 'Actions not defined for {}'.format(env_name)
        sys.exit()

def train(args):
    # Setup discrete actions for env if they are continuous
    #if type(env.action_space) is not gym.spaces.Discrete:
    #    actions = getActionsForEnv(env_name)
    #    num_actions = len(actions)
    #else:
    #    actions = env.action_space
    #    num_actions = env.action_space.n

    # Setup save path for run
    if args.load_model and args.saved_model is None:
        saved_model_path = constants.TF_MODELS_PATH + env_name
    elif args.saved_model is not None:
        saved_model_path = '{}/{}'.format(constants.TF_MODELS_PATH + env_name, args.saved_model)
    else:
        saved_model_path = None

    # Create ckpt dir if it does not exist
    if args.save_model:
        save_dir = constants.TF_MODELS_PATH + '{}/'.format(args.env_name)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    # Init environment and agent
    env = Environment(args.env_name)
    agent = DDQN_Agent(env, args.lr, args.discount, args.s_eps, args.e_eps, args.eps_decay_steps, args.t_eps,
                      constants.MLP, args.train_freq, args.update_target_freq, args.batch_size, args.exp_replay_size, saved_model=saved_model_path)
    stats = Statistics(agent, env, args.epochs, args.env_name, args.job_name)

    # Train agent
    print 'Populating experience replay memory with %d random moves' % args.random_steps
    stats.reset()
    agent.randomExplore(args.random_steps)
    stats.write(0, 'random')

    agent.startEnqueueThreads()
    for epoch in range(args.epochs):
        print 'Epoch #%d' % (epoch+1)

        print 'Training for %d steps' % args.train_steps
        stats.reset()
        agent.run(args.train_steps, train=True)
        stats.write(epoch+1, 'train')

        if args.save_model:
            env.agent.saveModel(save_dir + '{}_{}'.format(env_name, epoch+1))

        print 'Testing for %d steps' % args.test_steps
        stats.reset()
        agent.run(args.test_steps)
        stats.write(epoch+1, 'test')

    stats.close()
    print 'Done'

# Parse command line input to strucutre the RL problem
def main():
    parser = argparse.ArgumentParser(description='Train a DQN model on the specified OpenAI Gym environment.')

    envarg = parser.add_argument_group('Environment')
    envarg.add_argument('job_name', type=str,
                        help='The name of the job for this training run')
    envarg.add_argument('env_name', type=str,
                        help='The OpenAI Gym environment to train on')
    envarg.add_argument('--epochs', dest='epochs', type=int, default=200,
                        help='How many epochs to run')
    envarg.add_argument('--random_steps', dest='random_steps', type=int, default=50000,
                        help='Number of random steps to populate memory with before learning.')
    envarg.add_argument('--train_steps', dest='train_steps', type=int, default=250000,
                        help='Number of training steps per epoch.')
    envarg.add_argument('--test_steps', dest='test_steps', type=int, default=125000,
                        help='Number of testing steps per epoch.')
    envarg.add_argument('--frame_size', dest='frame_size', type=int, default=84,
                        help='The NxN size of the frame after resizing')
    envarg.add_argument('--frames_per_state', dest='frames_per_state', type=int, default=4,
                        help='How many frames form a state.')
    envarg.add_argument('--log_freq', dest='log_freq', type=int, default=1,
                        help='The number of episodes to log training info')
    envarg.add_argument('--render', dest='render', action='store_true',
                        help='Should we render the environment during the specified mode')
    envarg.add_argument('--load_model', dest='load_model', action='store_true',
                        help='Should we attempt to load a previously trained model')
    envarg.add_argument('--saved_model', dest='saved_model', type=str, default=None,
                        help='Specify the saved model to be loaded')
    envarg.add_argument('--summarize', dest='summarize', action='store_true',
                        help='Should we summarize the training using TensorBoard')
    envarg.add_argument('--save_model', dest='save_model', action='store_true',
                        help='Save the training weights after each epoch')

    memarg = parser.add_argument_group('Replay Memory')
    memarg.add_argument('--exp_replay_size', dest='exp_replay_size', type=int, default=1e6,
                        help='The size of the experience buffer')

    agentarg = parser.add_argument_group('DDQN Agent')
    agentarg.add_argument('--s_eps', dest='s_eps', type=float, default=1.0,
                          help='The starting epsilon value')
    agentarg.add_argument('--e_eps', dest='e_eps', type=float, default=0.1,
                          help='The ending epsilon value')
    agentarg.add_argument('--t_eps', dest='t_eps', type=float, default=0.05,
                          help='Exploration rate used in testing.')
    agentarg.add_argument('--eps_decay_steps', dest='eps_decay_steps', type=float, default=1000000,
                          help='Number of steps to decay the exploration rate.')
    agentarg.add_argument('--update_target_freq', dest='update_target_freq', type=int, default=10000,
                          help='Update the target network every nth step')
    agentarg.add_argument('--train_freq', dest='train_freq', type=int, default=4,
                          help='Train the DQN every nth step.')

    netarg = parser.add_argument_group('DDQN Network')
    netarg.add_argument('--lr', dest='lr', type=float, default=0.00025,
                        help='The learning rate')
    netarg.add_argument('--discount', dest='discount', type=float, default=0.99,
                        help='The future reward discount')
    netarg.add_argument('--batch', dest='batch_size', type=int, default=32,
                        help='The size of the minibatch')

    parser.set_defaults(render=False, load_model=False, summarize=False, save_model=False)

    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
