import agents.prioritized_replay.rank_based as ranked_based
if __name__ == '__main__':

    conf = {'size': 50,
            'learn_start': 10,
            'partition_num': 5,
            'total_step': 100,
            'batch_size': 4}
    buffer = ranked_based.Experience(conf)
    print(buffer)

    for k in range(50):
        buffer.store(('st', 'k', k, 'st+1', k))
    print(buffer.sample(50))


