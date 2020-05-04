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
    'numItersForTrainExamplesHistory': 20,

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

    # Tensorboard
    writer = SummaryWriter()

    # Load the data
    file = "D:\\IMT\\A2\\UE_AI\\PyRat\\npz\\alphazero_dataset_10k.npz"
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
    net.to(device)
    lr = 0.00001

    optimizer = optim.Adam(net.parameters(), lr=lr)
    epochs = 100
    for epoch in trange(epochs):
        net.train()
        pi_loss = 0
        v_loss = 0

        for x_batch, y_batch, z_batch in tqdm(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            z_batch =  z_batch.to(device)

            p_vector, v = net(x_batch)
            policy_loss = loss_pi(y_batch, p_vector)
            value_loss = loss_v(z_batch, v)
            total_loss = policy_loss + value_loss

            # record losses
            pi_loss += policy_loss.item()
            v_loss += value_loss.item()

            # compute gradient and do SGD step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

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
                value_loss = loss_v(z_batch, v)
                total_loss = policy_loss + value_loss

                # record losses
                validation_pi_loss += policy_loss.item()
                validation_v_loss += value_loss.item()


        # Record to tensorboard
        len_test = len(supervised_dataset) - n
        writer.add_scalar('Test set/value loss',validation_v_loss, epoch )
        writer.add_scalar('Test set/policy loss', validation_pi_loss, epoch)
        writer.add_scalar('Test set/total loss', validation_pi_loss + validation_v_loss, epoch)
        writer.add_scalar('Test set/normalized value loss', validation_v_loss/len_test, epoch)
        writer.add_scalar('Test set/normalized policy loss', validation_pi_loss/len_test, epoch)
        writer.add_scalar('Test set/normalized total loss', (validation_pi_loss + validation_v_loss) / len_test, epoch)

        writer.add_scalar('Train set/value loss',v_loss, epoch )
        writer.add_scalar('Train set/policy loss', pi_loss, epoch)
        writer.add_scalar('Train set/total loss', pi_loss + v_loss, epoch)
        writer.add_scalar('Train set/normalized value loss',v_loss/n, epoch )
        writer.add_scalar('Train set/normalized policy loss', pi_loss/n, epoch)
        writer.add_scalar('Train set/normalized total loss', pi_loss + v_loss/n, epoch)

        print(f"On epoch {epoch + 1} : value loss {v_loss} and policy loss {pi_loss}")

        if epoch %10 == 0:
            torch.save(net.state_dict(), f"weights-Supervised-{args.residual_blocks}x{args.filters}-epoch{epoch+1}.pt")

