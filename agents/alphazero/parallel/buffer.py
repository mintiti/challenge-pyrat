from collections import deque
import os
from pickle import Pickler, Unpickler
import sys
import random
import ray
class ReplayBuffer :
    def __init__(self, folder, maxlen):
        self.storage = deque(maxlen= maxlen)
        self.path = folder
    
    def store(self,examples):
        self.storage.extend(examples)
        return len(self.storage)
    
    def __len__(self):
        return len(self.storage)

    def shuffle(self):
        random.shuffle(self.storage)

    def save(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        new_name = self.get_name()
        file = os.path.join(self.path,new_name)
        with open(file, "wb+") as f:
            Pickler(f).dump(self.storage)
        f.closed

    def get_name(self):
        n = sum(1 for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f)))
        new_name = f"iter{n+1}.examples"
        return new_name

    def get_n_iters(self):
        n = sum(1 for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f)))
        return n

    def load(self):
        n = sum(1 for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f)))
        name = f"iter{n}.examples"
        examplesFile = os.path.join(self.path, name)
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with train examples found. Read it.")
            with open(examplesFile, "rb") as f:
                self.storage = Unpickler(f).load()
            f.closed

@ray.remote
class RemoteReplayBuffer:
    def __init__(self, folder, maxlen):
        self.storage = deque(maxlen=maxlen)
        self.path = folder

    def store(self, examples):
        self.storage.extend(examples)
        return len(self.storage)

    def __len__(self):
        return len(self.storage)

    def shuffle(self):
        random.shuffle(self.storage)

    def save(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        new_name = self.get_name()
        file = os.path.join(self.path, new_name)
        with open(file, "wb+") as f:
            Pickler(f).dump(self.storage)
        f.closed

    def get_storage(self):
        return self.storage

    def get_name(self):
        n = sum(1 for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f)))
        new_name = f"iter{n + 1}.examples"
        return new_name

    def get_n_iters(self):
        n = sum(1 for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f)))
        return n

    def load(self):
        n = sum(1 for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f)))
        name = f"iter{n}.examples"
        examplesFile = os.path.join(self.path, name)
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with train examples found. Read it.")
            with open(examplesFile, "rb") as f:
                self.storage = Unpickler(f).load()
            f.closed