import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchviz import make_dot

class BasicResidualBlock(nn.Module):
    def __init__(self, nb_filters):
        super(BasicResidualBlock, self).__init__()
        self.conv_module = nn.Sequential(nn.Conv2d(nb_filters,nb_filters,(3,3),padding= 1),
                                         nn.BatchNorm2d(nb_filters),
                                         nn.ReLU(),
                                         nn.Conv2d(nb_filters,nb_filters,(3,3),padding= 1),
                                         nn.BatchNorm2d(nb_filters))

    def forward(self, x):
        conv_out = self.conv_module(x)
        residual_connection = conv_out + x
        output = F.relu(residual_connection)

        return output

class ValueHead(nn.Module):
    def __init__(self, nb_filters):
        super(ValueHead, self).__init__()
        self.conv_block = nn.Sequential(nn.Conv2d(nb_filters, 1,(1,1)),
                                        nn.BatchNorm2d(1),
                                        nn.ReLU())
        self.fc_block = nn.Sequential(nn.Linear(15*21,256),
                                      nn.ReLU(),
                                      nn.Linear(256,1))

    def forward(self,x):
        x = self.conv_block(x)
        x = x.view((x.shape[0],-1))
        x = self.fc_block(x)
        return torch.tanh(x)

class PolicyHead(nn.Module):
    def __init__(self,nb_filters):
        super(PolicyHead,self).__init__()
        self.main_block = nn.Sequential(nn.Conv2d(nb_filters, 2, (1,1)),
                                        nn.BatchNorm2d(2),
                                        nn.ReLU())
        self.fc_block = nn.Linear(2*21*15,4)

    def forward(self,x):
        x = self.main_block(x)
        x = x.view(x.shape[0], -1)
        x = self.fc_block(x)
        return x

class AlphaZeroNetwork(nn.Module):
    def __init__(self,nb_filters, nb_conv_blocks):
        super(AlphaZeroNetwork,self).__init__()
        self.conv_layer = nn.Sequential(nn.Conv2d(9,nb_filters,(3,3), padding= 1),
                                        nn.BatchNorm2d(nb_filters),
                                        nn.ReLU())
        self.residual_layers = nn.ModuleList([BasicResidualBlock(nb_filters) for _ in range(nb_conv_blocks)])
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


