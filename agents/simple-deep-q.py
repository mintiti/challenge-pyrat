from .buffer import PrioritizedReplayBuffer, ExperienceReplayBuffer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

PYTHON = 1
RAT = -1


class QLearningAlgorithm:
    def __init__(self, env, agent1, agent2, total_time_steps=20000, gamma=0.99, epsilon0=1, epsilon_final=0.1,
                 buffer_size=50000, buffer_creator=ExperienceReplayBuffer, learning_starts=500, replay_period=4,
                 minibatch_size=32, learning_rate=0.00025):
        # learning hyperparameters
        self.gamma = gamma
        self.epsilon0 = epsilon0
        self.final_epsilon = epsilon_final
        self.d_epsilon = (epsilon0 - self.final_epsilon) / (total_time_steps - learning_starts)
        self.total_time_steps = total_time_steps
        self.learning_starts = learning_starts
        self.replay_period = replay_period
        self.buffer_conf = {
            'size': buffer_size,
            'replace_flag': True,
            'alpha': 0.7,
            'beta_zero': 0.5,
            'batch_size': self.minibatch_size,
            'learn_start': self.learning_starts,
            'steps': self.total_time_steps,
            'partition_num': self.minibatch_size
        }

        # Learning agents
        self.agent1 = agent1
        self.agent2 = agent2
        # Environment
        self.env = env

        # Replay Buffer
        self.buffer = buffer_creator(self.buffer_conf)

        # Book keeping
        self.current_time_steps = 0
        self.player1_scores = []
        self.player2_scores = []

    def rollout(self):
        done = False
        obs = self.env.reset()
        while not done:
            with torch.no_grad():
                agent1_action = self.agent1.act(obs, self.epsilon)
                agent2_action = self.agent2.act(obs, self.epsilon)
                action = (agent1_action, agent2_action)

                next_obs, reward, done, _ = self.env.step(action)
                transition = (obs, action, reward, next_obs, done)
                # todo : generate symmetries of the trasition
                # 7 more symmetries, generated by horizontal flip and 90 degree rotation
                # see http://pzacad.pitzer.edu/~jhoste/HosteWebPages/Courses/Math05/HW6v2.pdf
                # Augments data considerably so maybe adjust the batch size accordingly
                self.buffer.store(transition)
                self.current_time_steps += 1
                obs = next_obs


            if self.current_time_steps > self.learning_starts:
                self.train()

    def train(self):
        batch, w, _ = self.buffer.sample(self.current_time_steps)
        self.agent1.learn(batch,w)
        self.agent2.learn(batch,w)

    @property
    def epsilon(self):
        return self.epsilon0 + self.current_time_steps * self.d_epsilon

class SimpleDeepQLearning:
    """Simple Deep Q  learning agent , uses a CNN"""

    def __init__(self, env, player, nn, gamma=0.99, device = 'auto'):
        self.neural_net = nn
        self.env = env
        self.player = player  # player  1 or player 2, corresponding to being either the rat or the python

        # Precompute
        self.RAT_matrix = np.full((9, 21, 15), RAT)
        self.PYTHON_matrix = np.full((9, 21, 15), PYTHON)

        # CUDA

        if (device == 'cuda' or device == 'auto') and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.neural_net.to(device = self.device)

        # Pytorch learning
        self.criterion = None # TODo
        self.optimizer = optim.Adam(lr=time_step)
        self.gamma = gamma

    def add_player_info(self, obs):
        "Adds the player who the nn is playing to the obs"
        if self.player == PYTHON:
            return np.append(obs, self.PYTHON_matrix, axis=0)
        elif self.player == RAT:
            return np.append(obs, self.RAT_matrix, axis=0)

    def act(self, obs, epsilon):
        obs = self.add_player_info(obs)
        if epsilon < random.random():
            values = self.neural_net(obs)
            action = torch.argmax(values, dim= 1)[0]
        else:
            action = random.randrange(4)
        return action

        # CAREFUL : Add who the agent is playing
        pass

    def learn(self, batch):
        states = []
        next_states = []
        actions = []
        rewards = []
        dones = []
        for s1, a, r,s2, t in batch:
            states.append(s1)
            next_states.append(s2)
            actions.append(a)
            rewards.append(r)
            dones.append(t)
        states = torch.Tensor(states)
        next_states = torch.Tensor(next_states)
        actions = torch.Tensor(actions)
        rewards = torch.Tensor(rewards)
        dones = torch.Tensor(dones)
        self.optimizer.zero_grad()
        q_values = self.neural_net(states)
        # TODo : select the q value equal = the action
        with torch.no_grad():
            # We wanna fit q_values to next_q_values
            next_q_values = self.neural_net(next_states)
            maxes = next_q_values.max(dim = 1)[0]
            q_targets = q_values.clone().detach()
            for expected_prediction, action, reward, max , done in zip(q_targets,actions,rewards, maxes, dones):
                expected_prediction[action] = reward + self.gamma * max * (1- done)



class QlearningLoss(nn.Module):
    def __init__(self):
        super(QlearningLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, q_values, next_q_values, weights):
