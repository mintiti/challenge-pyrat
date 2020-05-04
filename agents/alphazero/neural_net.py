import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from progress.bar import Bar
from .alphazerogeneral.pytorch_classification.utils import AverageMeter
from .alphazerogeneral.utils import dotdict
from .alphazerogeneral.NeuralNet import NeuralNet
import numpy as np
import os
import time
from cachetools import LRUCache, cachedmethod

args = dotdict({
    'lr': 0.00001,
    'dropout': 0.3,
    'epochs': 2,
    'batch_size': 2048,
    'cuda': torch.cuda.is_available(),
    'half_precision': True
})


class BasicResidualBlock(nn.Module):
    def __init__(self, nb_filters):
        super(BasicResidualBlock, self).__init__()
        self.conv_module = nn.Sequential(nn.Conv2d(nb_filters, nb_filters, (3, 3), padding=1),
                                         nn.BatchNorm2d(nb_filters),
                                         nn.ReLU(),
                                         nn.Conv2d(nb_filters, nb_filters, (3, 3), padding=1),
                                         nn.BatchNorm2d(nb_filters))

    def forward(self, x):
        conv_out = self.conv_module(x)
        residual_connection = conv_out + x
        output = F.relu(residual_connection)

        return output


class ValueHead(nn.Module):
    def __init__(self, nb_filters):
        super(ValueHead, self).__init__()
        self.conv_block = nn.Sequential(nn.Conv2d(nb_filters, 32, (1, 1)),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU())
        self.fc_block = nn.Sequential(nn.Linear(15 * 21 * 32, 256),
                                      nn.ReLU(),
                                      nn.Linear(256, 1))

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view((x.shape[0], -1))
        x = self.fc_block(x)
        return torch.tanh(x)


class PolicyHead(nn.Module):
    def __init__(self, nb_filters):
        super(PolicyHead, self).__init__()
        self.main_block = nn.Sequential(nn.Conv2d(nb_filters, 2, (1, 1)),
                                        nn.BatchNorm2d(2),
                                        nn.ReLU())
        self.fc_block = nn.Linear(2 * 21 * 15, 4)

    def forward(self, x):
        x = self.main_block(x)
        x = x.view(x.shape[0], -1)
        x = self.fc_block(x)
        return F.log_softmax(x, dim=1)


class AlphaZeroNetwork(nn.Module):
    def __init__(self, nb_filters, nb_residual_blocks):
        super(AlphaZeroNetwork, self).__init__()
        self.conv_layer = nn.Sequential(nn.Conv2d(10, nb_filters, (3, 3), padding=1),
                                        nn.BatchNorm2d(nb_filters),
                                        nn.ReLU())
        self.residual_layers = nn.ModuleList([BasicResidualBlock(nb_filters) for _ in range(nb_residual_blocks)])
        self.value_head = ValueHead(nb_filters)
        self.policy_head = PolicyHead(nb_filters)

    def forward(self, x):
        # Residual tower
        x = self.conv_layer(x)
        for res_layer in self.residual_layers:
            x = res_layer(x)

        # Policy head
        p_vector = self.policy_head(x)

        # Value head
        value = self.value_head(x)

        return p_vector, value


class ResidualNet(NeuralNet):
    def __init__(self, nb_filters, nb_res_blocks):
        self.nn = AlphaZeroNetwork(nb_filters, nb_res_blocks)
        if args.cuda:
            self.nn.cuda()
        # if args.half_precision:
        #     self.nn.half()
        #     for module in self.nn.modules():
        #         if isinstance(module, nn.BatchNorm2d):
        #             module.float()
        self.cache = LRUCache(maxsize=50000)

    def clear_cache(self):
        self.cache.clear()
        print("position evaluation cache cleared")

    def train(self, examples):
        print(len(examples))
        optimizer = optim.Adam(self.nn.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nn.train()
            data_time = AverageMeter()
            batch_time = AverageMeter()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            end = time.time()

            bar = Bar('Training Net', max=int(len(examples) / args.batch_size))
            batch_idx = 0

            while batch_idx < int(len(examples) / args.batch_size):
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.Tensor(boards)
                target_pis = torch.Tensor(pis)
                target_vs = torch.Tensor(vs)

                # predict
                if args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # measure data loading time
                data_time.update(time.time() - end)

                # Compute output
                p_vector, v = self.nn(boards)
                policy_loss = self.loss_pi(target_pis, p_vector)
                value_loss = self.loss_v(target_vs, v)
                total_loss = policy_loss + value_loss

                # record losses
                pi_losses.update(policy_loss.item())
                v_losses.update(total_loss.item())

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                batch_idx += 1

                # plot progress
                bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f}'.format(
                    batch=batch_idx,
                    size=int(len(examples) / args.batch_size),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    lpi=pi_losses.avg,
                    lv=v_losses.avg,
                )
                print(
                    '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f}'.format(
                        batch=batch_idx,
                        size=int(len(examples) / args.batch_size),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        lpi=pi_losses.avg,
                        lv=v_losses.avg,
                    ))
                bar.next()
            bar.finish()
        self.clear_cache()

    @cachedmethod(lambda self: self.cache, key=lambda board: board.tostring())
    def predict(self, board):
        start = time.time()
        with torch.no_grad():
            board = torch.Tensor([board])
            if args.cuda:
                board = board.cuda()
            self.nn.eval()
            p, v = self.nn(board)
        # print(f'PREDICTION TIME TAKEN : {time.time() - start}')

        return torch.exp(p).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nn.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nn.load_state_dict(checkpoint['state_dict'])


if __name__ == '__main__':
    mock_state = torch.rand((1, 10, 21, 15))

    net = AlphaZeroNetwork(128, 3)
    out = net(mock_state)
    print(out)
