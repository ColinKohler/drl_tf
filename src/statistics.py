import sys
import os
import csv
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf

import constants

class Statistics(object):
    def __init__(self, sess, agent, env, conf):
        self.sess = sess
        self.agent = agent
        self.env = env
        self.env.callback = self
        self.agent.callback = self

        self.job_name = conf.job_name
        self.train_steps = conf.train_steps
        self.e_eps = conf.e_eps

        self.max_avg_eps_reward = -sys.maxint - 1
        self.train_epoch_rewards = list()
        self.test_epoch_rewards = list()
        self.train_eps_rewards = list()
        self.test_eps_rewards = list()

        # Setup tensorboard summaries
        if self.sess is not None:
            self.writer = tf.summary.FileWriter(constants.TF_LOG_PATH + '/train', self.sess.graph)
            self.saver = tf.train.Saver()
            with tf.variable_scope('summary'):
                scalar_summary_tags = ['average/reward', 'average/loss', 'average/q_value',
                                       'episode/max_reward', 'episode/min_reward', 'episode/avg_reward',
                                       'episode/num_episodes', 'training/learning_rate', 'training/epsilon']
                histogram_summary_tags = ['episode/rewards', 'episode/actions']

                self.summary_placeholders = dict()
                self.summary_ops = dict()
                for tag in scalar_summary_tags:
                    self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag)
                    self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])
                for tag in histogram_summary_tags:
                    self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag)
                    self.summary_ops[tag] = tf.summary.histogram(tag, self.summary_placeholders[tag])

        # Create data collection repo
        self.csv_dir = constants.STATS_PATH + '{}/'.format(self.env.name)
        self.plots_dir = constants.PLOTS_PATH + '{}/'.format(self.env.name)
        self.model_dir = constants.TF_MODELS_PATH + '{}/'.format(self.env.name)

        if not os.path.isdir(self.csv_dir):
            os.makedirs(self.csv_dir)
        if not os.path.isdir(self.plots_dir):
            os.makedirs(self.plots_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self._initCSV(conf.job_name)
        self.plot_name = self.plots_dir + conf.job_name + '.jpg'

        # Start stats
        self.start_time = time.time()
        self.validation_states = None
        self.reset()

    # Setup CSV file
    def _initCSV(self, job_name):
        self.csv_name = self.csv_dir + job_name + '.csv'
        self.csv_file = open(self.csv_name, 'wb')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow((
            "epoch",
            "phase",
            "steps",
            "nr_games",
            "average_reward",
            "min_game_reward",
            "max_game_reward",
            "last_exploration_rate",
            "replay_memory_count",
            "mean_q",
            "mean_loss",
            "weight_updates",
            "total_time",
            "epoch_time",
            "steps_per_second"))
        self.csv_file.flush()

    # Resets the logger to the starting state
    def reset(self):
        self.epoch_start_time = time.time()
        self.num_steps = 0
        self.num_eps = 0
        self.eps_reward = 0
        self.avg_reward = 0
        self.min_reward = sys.maxint
        self.max_reward = -sys.maxint - 1
        self.epsilon = 1
        self.avg_loss = 0
        self.q_values = list()
        self.actions = list()
        self.eps_rewards = list()

    # Callback for env
    def onStep(self, action, reward, done, epsilon):
        self.eps_reward += reward
        self.num_steps += 1
        self.epsilon = epsilon
        self.actions.append(action)

        if done:
            self.num_eps += 1
            self.avg_reward += float(self.eps_reward - self.avg_reward) / self.num_eps
            self.min_reward = min(self.min_reward, self.eps_reward)
            self.max_reward = max(self.max_reward, self.eps_reward)
            self.eps_rewards.append(self.eps_reward)
            self.eps_reward = 0

    # Callback for agent
    def onTrain(self, q_values, loss):
        self.avg_loss += (loss - self.avg_loss) / self.agent.train_iterations
        self.q_values.extend(q_values)

    # Handle ending of epoch by writing to tensorbaord and csv file
    def write(self, epoch, phase, tensorboard=False):
        current_time = time.time()
        total_time = current_time - self.start_time
        epoch_time = current_time - self.epoch_start_time
        steps_per_second = self.num_steps / epoch_time

        if self.num_eps == 0:
            self.num_eps = 1
            self.avg_reward = self.eps_reward

        if tensorboard and self.validation_states is None and self.agent.exp_replay.size > self.agent.batch_size:
            self.validation_states, _, _, _, _ = self.agent.exp_replay.getBatch()

        if tensorboard and self.validation_states is not None:
            state_shape = [-1, self.agent.unroll, self.env.getStateShape()[-1]]
            q_values = self.agent.sess.run(self.agent.q_model.q_values,
                    feed_dict={self.agent.q_model.batch_input : self.validation_states.reshape(state_shape)})
            max_q_values = np.max(q_values, axis=1)
            mean_q_value = np.mean(max_q_values)
        else:
            mean_q_value = 0

        if tensorboard:
            self.csv_writer.writerow((
                epoch,
                phase,
                self.num_steps,
                self.num_eps,
                self.avg_reward,
                self.min_reward,
                self.max_reward,
                self.epsilon,
                self.agent.exp_replay.size,
                mean_q_value,
                self.avg_loss,
                self.agent.train_iterations,
                total_time,
                epoch_time,
                steps_per_second))
            self.csv_file.flush()

            # Write data to tensorflow
            self.injectSummary({
                'average/q_value' : np.mean(self.q_values),
                'average/loss' : self.avg_loss,
                'average/reward' : self.avg_reward,
                'episode/max_reward' : self.max_reward,
                'episode/min_reward' : self.min_reward,
                'episode/num_episodes' : self.num_eps,
                'episode/actions' : self.actions,
                'episode/rewards' : self.eps_rewards,
                'training/learning_rate' : self.agent.q_model.lr_op.eval(session=self.sess),
                'training/epsilon' : self.epsilon
            }, epoch)

        # Save episode rewards for plotting later
        if phase == 'train':
            self.train_epoch_rewards.append(self.avg_reward)
            self.train_eps_rewards.extend(self.eps_rewards)
        elif phase == 'test':
            self.test_epoch_rewards.append(self.avg_reward)
            self.test_eps_rewards.extend(self.eps_rewards)

        # Print log for current epoch
        self.log()

        # Save model if it is good enough
        if tensorboard and self.epsilon == self.e_eps and self.max_avg_eps_reward <= self.avg_reward:
            self.saveModel(epoch)
            self.agent.test(1, render=True)
            self.max_avg_eps_reward = max(self.max_avg_eps_reward, self.avg_reward)

        # Reset stats for next epoch
        self.reset()

    # Update plot with current data
    def plot(self):
        gs = gridspec.GridSpec(2,1)
        gs.update(hspace=0.75)

        fig = plt.figure()
        fig.suptitle(self.env.name, fontsize=14, fontweight='bold')
        fig.subplots_adjust(top=0.85)

        # Avg. reward per epoch
        epoch_ax = fig.add_subplot(gs[0])
        epoch_ax.set_title('Avg. Reward per Epoch', fontweight='bold')
        epoch_ax.set_xlabel('Epoch ({} Steps)'.format(self.train_steps))
        epoch_ax.set_ylabel('Reward')
        l1 = epoch_ax.plot(self.train_epoch_rewards, color='b', label='Train')
        l2 = epoch_ax.plot(self.test_epoch_rewards, color='r', label='Test')

        # Reward per episode
        eps_ax = fig.add_subplot(gs[1])
        eps_ax.set_title('Reward per Episode', fontweight='bold')
        eps_ax.set_xlabel('Episode')
        eps_ax.set_ylabel('Reward')
        eps_ax.plot(self.train_eps_rewards, color='b', label='Train')
        eps_ax.plot(self.test_eps_rewards, color='r', label='Test')

        fig.savefig(self.plot_name)

    # Inject summary data into tensorboard
    def injectSummary(self, tag_dict, t):
        summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
            self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
        })
        for summary_str in summary_str_lists:
            self.writer.add_summary(summary_str, t)

    # og statistics during run
    def log(self):
        print 'Num Episodes: %d | Epsilon: %f | Avg Rewards: %f | Min Reward %f | Max Reward %f' % \
                (self.num_eps, self.epsilon, self.avg_reward, self.min_reward, self.max_reward)

    # Close the csv file
    def close(self):
        filepath = '{}{}/{}.pkl'.format(constants.STATS_PATH, self.env.name, self.job_name)
        with open(filepath, 'wb') as fd:
            pickled_data = {'train_eps' : self.train_eps_rewards, 'test_eps' : self.test_eps_rewards,
                            'train_epoch' : self.train_epoch_rewards, 'test_epoch' : self.test_epoch_rewards}
            pickle.dump(pickled_data, fd)
        self.csv_file.close()

    # Save the model to the model directory
    def saveModel(self, epoch):
        print '[*] Saving checkpoint...'
        model_name = type(self).__name__
        self.saver.save(self.sess, self.model_dir+'{}_{}'.format(self.env.name, epoch))

    # Load the most recent saved model in the model directory
    def loadModel(self):
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            fname = os.path.join(self.model_dir, ckpt_name)
            self.saver.restore(self.sess, fname)
            print '[*] Load SUCCESS: %s' % fname
            return True
        else:
            print '[*] Load FAILED: %s' % self.model_dir
            return False
