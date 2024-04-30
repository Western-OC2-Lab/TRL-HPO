import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from calflops import calculate_flops
import torch

class SingleModel(nn.Module):

    def __init__(self, nb_inputs):
        super(SingleModel, self).__init__()
        self.linear1 = nn.Linear(nb_inputs, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 32)
        self.linear4 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))

        return torch.sigmoid(x)
    
nb_hps = 6
input_size = (1, nb_hps)
s = SingleModel(nb_hps)
flops, macs, params = calculate_flops(model=s, 
                                      input_shape=input_size,
                                      output_as_string=True,
                                      output_precision=4)
print("Alexnet FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))


