from collections import deque
import torch
import torch.optim as optim
class SimpleDeepQLearning:
    """Simple Deep Q  learning agent , uses a CNN"""
    def __init__(self,env,player, model_creator, gamma = 0.9, epsilon =0.99, buffer_size = 50000, training_steps = 100, batch = 32,):
        self.neural_net = model_creator()
        self.env = env
        self.replay_buffer = deque(buffer_size)
        self.player = player # player  1 or player 2, corresponding to being either the rat or the python

