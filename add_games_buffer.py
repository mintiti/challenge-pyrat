from train_parallel_alphazero import args
from agents.alphazero.buffer import ReplayBuffer

if __name__ == '__main__':
    buffer = ReplayBuffer(args['checkpoint'] + 'examples/', maxlen=args['buffer_size'])
    buffer.load()
    print(len(buffer))
    print(buffer.storage.maxlen)
