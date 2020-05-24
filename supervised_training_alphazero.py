from agents.alphazero.neural_net import ResidualNet, AlphaZeroNetwork
from agents.alphazero.alphazerogeneral.utils import *
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from tqdm import tqdm, trange

args = dotdict({
    'numIters': 1000,
    'numEps': 10,  # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 40,  #
    'updateThreshold': 0.5790,
    # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 50000,  # Number of game examples to train the neural networks.
    'numMCTSSims': 600,  # Number of games moves for MCTS to simulate.
    'arenaCompare': 20,  # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 2,

    'checkpoint': './temp/3x64/',
    'load_model': False,
    'load_folder_file': ('/dev/models/3x64/', 'best.pth.tar'),

    # NN config
    'residual_blocks': 3,
    'filters': 64,

})


def loss_pi(targets, outputs):
    return -torch.sum(targets * outputs) / targets.size()[0]


def loss_v(targets, outputs):
    return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    load_previous = True

    # Tensorboard
    writer = SummaryWriter(log_dir= "runs/t0-3x64-test-deepmind-config")

    # Load the data
    file = "alphazero_dataset_10k.npz"
    loaded = np.load(file)
    x, y, z = loaded['x'], loaded['y'], loaded['z']
    x = torch.FloatTensor(x)

    y = y.reshape((-1, 4))
    y = torch.FloatTensor(y)

    z = torch.FloatTensor(z)

    supervised_dataset = TensorDataset(x, y, z)

    n = int(len(supervised_dataset) * 0.8)  # train/test split
    train_set, test_set = torch.utils.data.random_split(supervised_dataset, [n, len(supervised_dataset) - n])

    # data loaders
    train_loader = DataLoader(train_set, batch_size=2048, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=2048, pin_memory=True)

    # create the net
    net = AlphaZeroNetwork(args.filters, args.residual_blocks)
    #weights = torch.load("weights-Supervised-3x64-epoch20.pt")
    #net.load_state_dict(weights['state_dict'])
    net.to(device)


    lr = 0.01

    optimizer = optim.SGD(net.parameters(),lr = lr,momentum=0.9, weight_decay= 0.00001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4,8,12,14], gamma=0.1)
    epochs = 100
    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}")
        net.train()
        pi_loss = 0
        v_loss = 0

        # statistics
        correct_train = 0
        correct_test =0
        len_train = float(n)
        len_test = float(len(supervised_dataset) - n)

        for x_batch, y_batch, z_batch in tqdm(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            z_batch = z_batch.to(device)

            p_vector, v = net(x_batch)
            policy_loss = loss_pi(y_batch, p_vector)
            value_loss = 0.01 * loss_v(z_batch, v)
            total_loss = policy_loss + value_loss

            # record losses
            pi_loss += policy_loss.item()
            v_loss += value_loss.item()

            # compute gradient and do SGD step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # statistics
            predicted_move = torch.argmax(p_vector, dim =1)
            correct_moves = torch.argmax(y_batch, dim =1 )
            correct_train += (predicted_move == correct_moves).sum()

        # evaluate on test set
        with torch.no_grad():
            net.eval()
            validation_pi_loss = 0
            validation_v_loss = 0
            for x_batch, y_batch, z_batch in tqdm(test_loader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                z_batch = z_batch.to(device)

                p_vector, v = net(x_batch)
                policy_loss = loss_pi(y_batch, p_vector)
                value_loss = 0.01 * loss_v(z_batch, v)
                total_loss = policy_loss + value_loss

                # record losses
                validation_pi_loss += policy_loss.item()
                validation_v_loss += value_loss.item()

                #statistics
                predicted_move = torch.argmax(p_vector, dim=1)
                correct_moves = torch.argmax(y_batch, dim=1)
                correct_test += (predicted_move == correct_moves).sum()
                
        scheduler.step()

        # Record to tensorboard
        writer.add_scalars("Training losses", {"Value loss": v_loss,
                                               "Policy loss": pi_loss,
                                               "Total_loss": v_loss + pi_loss}, epoch)
        writer.add_scalars("Test losses", {"Value loss": validation_v_loss,
                                               "Policy loss": validation_pi_loss,
                                               "Total_loss": validation_v_loss + validation_pi_loss}, epoch)

        writer.add_scalars("Precisions", {"Validation move prediction accuracy":correct_test/len_test ,
                                          "Training move prediction accuracy" : correct_train / len_train}, epoch)


        print(f"\nOn epoch {epoch + 1} : value loss {v_loss} and policy loss {pi_loss}\nMove prediction accuracy train {correct_train / len_train} and test {correct_test/len_test}")

        if (epoch + 1) % 5 == 0:
            torch.save({'state_dict': net.state_dict()},
                       f"weights-Supervised-{args.residual_blocks}x{args.filters}--Deepmind-epoch{epoch + 1}.pt")
        