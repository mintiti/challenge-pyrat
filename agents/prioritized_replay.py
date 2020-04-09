from abc import ABC
from collections import deque
from .prioritized_replay import rank_based

class PrioritizedExperienceReplay():
    def __init__(self, max_size = 50000):
        self.data = deque(max_size)
        self.priorities = deque(max_size)

    def store(self,transition):
        self.data.appendleft(transition)
        self.priorities.appendleft(1)