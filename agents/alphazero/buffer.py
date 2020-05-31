from collections import deque
import os
from pickle import Pickler, Unpickler
import pickle
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

    def get_shuffled_buffer(self):
        buffer = self.storage.copy()
        return random.shuffle(buffer)

    def save(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        new_name = self.get_name()
        file = os.path.join(self.path,new_name)
        with open(file, "wb+") as f:
            Pickler(f).dump(self.storage)
        f.closed

    def temp_save(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        name = "temp.examples"
        file = os.path.join(self.path,name)
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
        max_len = self.storage.maxlen

        n = sum(1 for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f)))
        name = f"iter{n}.examples"
        examplesFile = os.path.join(self.path, name)
        if not os.path.isfile(examplesFile):
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with train examples found. Read it.")
            with open(examplesFile, "rb") as f:
                old_storage = Unpickler(f).load()
            f.closed
            for example in old_storage:
                self.storage.append(example)
    def add_pkl_file(self,path):
        with open(path,"rb") as f :
            games = pickle.load(f)
        print(f"Adding {len(games)} games tp the buffer")

        self.store(games)
    def _save(self):
        n = self.get_n_iters()
        name = f"iter{n}.examples"
        file = os.path.join(self.path, name)
        with open(file, "wb+") as f:
            Pickler(f).dump(self.storage)
        f.closed
        print(f"Saved buffer to {file}.")


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
        print(examplesFile)
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