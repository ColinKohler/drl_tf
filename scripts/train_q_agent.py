#!/usr/bin/python

import os
import gym
import gym.spaces
import numpy as np
import matplotlib.pyplot as plt
import argparse

from agents.q_agent import Q_Agent
from environment import Environment
import constants
from statistics import Statistics

# Custom Env
import helping_hands

# Gets the actions for the current env from the constants
def getActionsForEnv(env_name):
    if env_name in constants.ACTIONS:
        return constants.ACTIONS[env_name]
    else:
        print 'Actions not defined for {}'.format(env_name)
        sys.exit()

def train(args):
    # Init gym env
    env = Environment(args)
    agent = Q_Agent(env, args)
    stats = Statistics(None, agent, env, args)
    if args.load_model is not None: agent.loadModel(args.load_model)

    # Train agent
    try:
        for epoch in range(args.epochs):
            print 'Epoch #%d' % (epoch+1)

            if args.train_steps > 0:
                print 'Training for %d steps' % args.train_steps
                agent.train(args.train_steps)
                stats.write(epoch+1, 'train', tensorboard=False)

            if args.test_steps > 0:
                print 'Testing for %d steps' % args.test_steps
                agent.test(args.test_steps, render=args.render)
                stats.write(epoch+1, 'test', tensorboard=False)

        agent.saveModel(args.job_name)
        stats.plot()
        stats.close()
        agent.test(1000, render=True)
        print 'Done'
    except KeyboardInterrupt:
        agent.saveModel(args.job_name)
        stats.plot()
        stats.close()
        agent.test(1000, render=True)
        print 'Caught keyboard interrupt, stopping run...'

# Parse command line input to strucutre the RL problem
def main():
    parser = argparse.ArgumentParser(description='Train a Q model on the specified OpenAI Gym environment.')

    envarg = parser.add_argument_group('Environment')
    envarg.add_argument('job_name', type=str,
                        help='The name of the job for this training run')
    envarg.add_argument('env_name', type=str,
                        help='The OpenAI Gym environment to train on')
    envarg.add_argument('--epochs', dest='epochs', type=int, default=200,
                        help='How many epochs to run')
    envarg.add_argument('--train_steps', dest='train_steps', type=int, default=250000,
                        help='Number of training steps per epoch.')
    envarg.add_argument('--test_steps', dest='test_steps', type=int, default=125000,
                        help='Number of testing steps per epoch.')
    envarg.add_argument('--random_start', dest='random_start', default=False, action='store_true',
                        help='Start each episode with a random state.')
    envarg.add_argument('--history_length', dest='history_length', type=int, default=1,
                        help='The length of history of observations to use as state input')
    envarg.add_argument('--render', dest='render', action='store_true',
                        help='Should we render the environment during the specified mode')

    agentarg = parser.add_argument_group('Agent')
    agentarg.add_argument('--s_eps', dest='s_eps', type=float, default=1.0,
                          help='The starting epsilon value')
    agentarg.add_argument('--e_eps', dest='e_eps', type=float, default=0.1,
                          help='The ending epsilon value')
    agentarg.add_argument('--t_eps', dest='t_eps', type=float, default=0.0,
                          help='Exploration rate used in testing.')
    agentarg.add_argument('--eps_decay_steps', dest='eps_decay_steps', type=float, default=1000000,
                          help='Number of steps to decay the exploration rate.')
    agentarg.add_argument('--load_model', dest='load_model', default=None, type=str,
                          help='Load the desired model')

    netarg = parser.add_argument_group('Model')
    netarg.add_argument('--lr', dest='lr', type=float, default=0.5,
                        help='The learning rate')
    netarg.add_argument('--discount', dest='discount', type=float, default=0.99,
                        help='The future reward discount')

    parser.set_defaults(render=False)

    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
