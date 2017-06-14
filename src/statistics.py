import sys
import os
import csv
import time
import numpy as np
import matplotlib.pyplot as plt

import constants

class Statistics(object):
    def __init__(self, agent, env, num_epochs, env_name, job_name):
        self.agent = agent
        self.env = env
        self.num_epochs = num_epochs

        self.train_epoch_rewards = [0.0]
        self.test_epoch_rewards = [0.0]
        self.env.callback = self
        self.agent.callback = self

        # Create data collection repo
        csv_dir = constants.STATS_PATH + '{}/'.format(env_name)
        plot_dir = constants.PLOTS_PATH + '{}/'.format(env_name)

        if not os.path.isdir(csv_dir):
            os.makedirs(csv_dir)
        if not os.path.isdir(plot_dir):
            os.makedirs(plots_dir)

        self._initCSV()
        self._initPlot()

        self.start_time = time.time()
        self.validation_states = None

    # Setup CSV file
    def _initCSV(self, csv_dir, job_name):
        self.csv_name = csv_dir + job_name + '.csv'
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
            "total_train_steps",
            "replay_memory_count",
            "mean_q",
            "mean_loss",
            "weight_updates",
            "total_time",
            "epoch_time",
            "steps_per_second"))
        self.csv_file.flush()

    # Setup plot
    def _initPlot(self, plot_dir, job_name):
        self.plot_name = plot_dir + job_name + '.csv'

        plt.ion()
        self.epoch_fig = plt.figure(figsizel=(10, 7.5))
        self.epoch_ax = self.epoch_fig.add_subplot(111)
        box = self.epoch_ax.get_position()
        self.epoch_ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        self.epoch_ax.set_xlim(0, self.num_epochs)
        self.y_min = -1
        self.y_max = 1
        self.epoch_ax.set_ylim(self.y_min, self.y_max)
        self.train_line, = self.epoch_ax.plot([], [], linestyle='-', marker='^', color='b')
        self.test_line, = self.epoch_ax.plot([], [], linestyle='-', marker='^', color='r')
        self.legend = self.epoch_ax.legend(['Train', 'Test'], loc='upper left', bbox_to_anchor=(1,1))

    # Resets the logger to the starting state
    def reset(self, train=True):
        self.epoch_start_time = time.time()
        self.num_steps = 0
        self.num_eps = 0
        self.eps_reward = 0
        self.avg_reward = 0
        self.min_reward = 0
        self.max_reward = -sys.maxint - 1
        self.eps = 1
        self.avg_loss = 0

    # Callback for env
    def onStep(self, action, reward, done, frame, eps):
        self.eps_reward += reward
        self.num_steps += 1
        self.eps = eps

        if done:
            self.num_eps += 1
            self.avg_reward += float(self.eps_reward - self.avg_reward) / self.num_eps
            self.min_reward = min(self.min_reward, self.eps_reward)
            self.max_reward = max(self.max_reward, self.eps_reward)
            self.eps_reward = 0

    # Callback for agent
    def onTrain(self, loss):
        self.avg_loss += (loss - self.avg_loss) / self.agent.train_iterations

    # Log statistics during run
    def log(self, epoch_time):
        print '%f Num Episodes: %d | Avg Rewards: %f | Min Reward %d | Max Reward %d' % \
                (epoch_time, self.num_eps, self.avg_rewards, self.min_reward, self.max_rewards)

    # Write data to csv
    def write(self, epoch, phase):
        current_time = time.time()
        total_time = current_time - self.start_time
        epoch_time = current_time - self.epoch_start_time
        steps_per_second = self.num_steps / epoch_time

        if self.num_eps == 0:
            self.num_eps = 1
            self.avg_reward = self.eps_reward

        if self.validation_states is None and self.agent.exp_replay.size > self.agent.minibatch_size:
            self.validation_states, _ = self.agent.exp_replay.getBatch()

        if self.validation_states is not None:
            state_shape = [-1] + self.agent.state_shape
            qs = self.agent.sess.run(self.agent.q_model.qs,
                    feed_dict={self.agent.q_model.inp : np.array(self.validation_states).reshape(state_shape)})
            max_qs = np.max(qs, axis=1)
            mean_q = np.mean(max_qs)
        else:
            mean_q = 0

        self.csv_write.writerow((
            epoch,
            phase,
            self.num_steps,
            self.num_eps,
            self.avg_reward,
            self.min_reward,
            self.max_reward,
            self.eps,
            self.agent.total_train_steps,
            self.agent.exp_replay.size,
            mean_q,
            self.avg_loss,
            self.agent.train_iterations,
            total_time,
            epoch_time,
            steps_per_second))
        self.csv_file.flush()

        if phase == 'train':
            self.train_epoch_rewards.append(self.avg_reward)
        elif phase == 'test':
            self.test_epoch_rewards.append(self.avg_reward)

        self.plot()
        self.log(epoch_time)

    # Update plot with current data
    def plot(self):
        # Check axis limits and adjust if needed
        min_reward = min(min(self.train_epoch_rewards), min(self.test_epoch rewards))
        max_reward = max(max(self.train_epoch_rewards), max(self.test_epoch rewards))

        if min_reward < self.y_min:
            self.y_min = min_reward - (min_reward / 10.0)
        if max_reward > self.y_max:
            self.y_max = max_reward + (max_reward / 10.0)
        self.epoch_ax.set_ylim(self.y_min, self.y_max)

        # Plot data
        x1 = np.arange(0, len(self.train_epoch_rewards))
        x2 = np.arange(0, len(self.test_epoch_rewards))
        self.train_line.set_xdata(x1)
        self.train_line.set_ydata(self.train_epoch_rewards)
        self.train_line.set_xdata(x2)
        self.train_line.set_ydata(self.test_epoch_rewards)
        self.legend.draggable(True)

        self.epoch_fig.canvas.draw()

    def close(self):
        self.epoch_fig.savefig(self.plot_name)
        self.csv_file.close()
