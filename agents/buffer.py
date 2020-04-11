from abc import ABC, abstractmethod
from collections import  deque
from .prioritized_replay.rank_based import Experience
from random import sample
class BaseBuffer(ABC):
    """Wrapper abstract class for buffer containers"""
    @abstractmethod
    def store(self,transition):
        """Stores transition of form (s1, a, r, s2, t)
        """
        raise NotImplementedError

    def sample(self,global_step):
        """
        sample a mini batch from experience replay buffer
        :param global_step: now training step
        :return: experience, list, samples
        :return: w, list, weights
        :return: rank_e_id, list, samples id, used for update priority
        """

        raise NotImplementedError


class ExperienceReplayBuffer(BaseBuffer):
    """Simple Experience replay container"""

    def __init__(self, conf):
        self.buffer = deque(max_len = conf['size'])
        self.batch_size = conf['batch_size']

    def store(self,transition):
        self.buffer.append(transition)

    def sample(self,global_step):
        batch = sample(self.buffer, self.batch_size)
        w = [1]*self.batch_size # could be adapted to giveback the stepsize
        return batch, w, None # Doesn't use rank_e_id and weights


class PrioritizedReplayBuffer(BaseBuffer):
    def __init__(self, conf):
        self.buffer = Experience(conf)

    def store(self,transition):
        return self.buffer.store(transition)

    def sample(self,global_step):
        return self.buffer.sample(global_step)




