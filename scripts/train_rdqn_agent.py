#!/usr/bin/python

import os
import sys
import gym
import gym.spaces
import numpy as np
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf

from agents.rdqn_agent import RDQN_Agent
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
    with tf.Session() as sess:
        # Init environment and agent
        env = Environment(args)
        agent = RDQN_Agent(sess, env, args)
        stats = Statistics(sess, agent, env, args)
        if args.load_model: stats.loadModel()
        sess.graph.finalize()

        try:
            print 'Taking %d random actions before training' % args.steps_pre_train
            agent.randomExplore(args.steps_pre_train)

            agent.startEnqueueThreads()
            for epoch in range(args.epochs):
                print 'Epoch #%d' % (epoch+1)

                if args.train_steps > 0:
                    print 'Training for %d steps' % args.train_steps
                    agent.train(args.train_steps)
                    stats.write(epoch+1, 'train', tensorboard=True)

                if args.test_eps > 0:
                    print 'Testing for %d episodes' % args.test_eps
                    agent.test(args.test_eps, render=True)
                    stats.write(epoch+1, 'test')

            agent.stopEnqueueThreads()
            stats.plot()
            stats.close()
            print 'Done'
        except KeyboardInterrupt:
            agent.test(1, render=True)
            stats.plot()
            stats.close()
            agent.stopEnqueueThreads()
            print 'Caught keyboard interrupt, stopping run...'

# Parse command line input to strucutre the RL problem
def main():
    parser = argparse.ArgumentParser(description='Train a DQN model on the specified OpenAI Gym environment.')

    envarg = parser.add_argument_group('Environment')
    envarg.add_argument('env_name', type=str,
                        help='The OpenAI Gym environment to train on')
    envarg.add_argument('--job_name', dest='job_name', type=str,
                        help='The name of the job for this training run')
    envarg.add_argument('--epochs', dest='epochs', type=int, default=100,
                        help='How many epochs to run')
    envarg.add_argument('--train_steps', dest='train_steps', type=int, default=10000,
                        help='Number of training steps per epoch.')
    envarg.add_argument('--test_eps', dest='test_eps', type=int, default=0,
                        help='Number of testing episodes per epoch.')
    envarg.add_argument('--steps_pre_train', dest='steps_pre_train', type=int, default=0,
                        help='Number of steps to take before starting to train')
    envarg.add_argument('--history_length', dest='history_length', type=int, default=1,
                        help='The length of history of observations to use as state input')
    envarg.add_argument('--random_start', dest='random_start', default=False, action='store_true',
                        help='Start each episode with a random state.')
    envarg.add_argument('--load_model', dest='load_model', default=False, action='store_true',
                        help='Load the latest model')

    memarg = parser.add_argument_group('Replay Memory')
    memarg.add_argument('--exp_replay_size', dest='exp_replay_size', type=int, default=100000,
                        help='The size of the experience buffer')

    agentarg = parser.add_argument_group('DDQN Agent')
    agentarg.add_argument('--double_q', dest='double_q', default=False, action='store_true',
                          help='Wether to use double Q-learning')
    agentarg.add_argument('--s_eps', dest='s_eps', type=float, default=1.0,
                          help='The starting epsilon value')
    agentarg.add_argument('--e_eps', dest='e_eps', type=float, default=0.01,
                          help='The ending epsilon value')
    agentarg.add_argument('--eps_decay_steps', dest='eps_decay_steps', type=float, default=100000,
                          help='Number of steps to decay the exploration rate.')
    agentarg.add_argument('--update_target_freq', dest='update_target_freq', type=int, default=10000,
                          help='Update the target network every nth step')
    agentarg.add_argument('--train_freq', dest='train_freq', type=int, default=4,
                          help='Train the DQN every nth step.')

    netarg = parser.add_argument_group('DDQN Network')
    netarg.add_argument('--network_config', dest='network_config', type=str, default='R_MLP',
                        help='The type of network [R_MLP]')
    netarg.add_argument('--lr', dest='lr', type=float, default=0.00025,
                        help='The learning rate')
    netarg.add_argument('--lr_minimum', dest='lr_minimum', type=float, default=0.00025,
                        help='The minimum learning rate during training')
    netarg.add_argument('--lr_decay_step', dest='lr_decay_step', type=float, default=1000,
                        help='The learning rate of training')
    netarg.add_argument('--lr_decay', dest='lr_decay', type=float, default=0.96,
                        help='The decay of the learning rate')
    netarg.add_argument('--discount', dest='discount', type=float, default=0.99,
                        help='The future reward discount')
    netarg.add_argument('--batch', dest='batch_size', type=int, default=32,
                        help='The size of the minibatch')
    netarg.add_argument('--unroll', dest='unroll', type=int, default=1,
                        help='The number of unroll steps to train LSTM on')

    args, unknown = parser.parse_known_args()
    train(args)

if __name__ == '__main__':
    main()
