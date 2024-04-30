import torch.nn as nn
import torch
import torch.nn.functional as F

class StateEncoding(nn.Module):
    
    def __init__(self, action_space, perf_space, output_layer):
        super(StateEncoding, self).__init__()
        self.action_encoding_layer = nn.Linear(action_space, output_layer)
        self.perf_encoding_layer = nn.Linear(perf_space, output_layer)
        
    def forward(self, action, perf):
        action_output = self.action_encoding_layer(action)
        perf_output = self.perf_encoding_layer(perf)
        
        out = action_output + perf_output
        out = torch.tanh(out)

        return out


class IntermediateClassifier(nn.Module):
    
    def __init__(self, prev_layers, type_last_layer, classification_layer):
        super(IntermediateClassifier, self).__init__()
        self.all_layers = nn.ModuleList()
        self.output_layer = nn.ModuleList()
        for layer in prev_layers:
            self.all_layers.append(layer)
        self.type_last_layer = type_last_layer
        self.output_layer.append(classification_layer)
        
    def forward(self, x):
        batch_size = x.size(0)

        for layer in self.all_layers:
            classname = layer.__class__.__name__
            if classname.find("Linear") != -1 and len(x.size()) > 2:
                x = x.view(batch_size, x.size(1) * x.size(2)* x.size(3))
            x = layer(x)

        output_layer = x

        if self.type_last_layer != "FCL":
            output_layer = x.view(batch_size, x.size(1) * x.size(2)* x.size(3))

        for layer in self.output_layer:
            output = layer(output_layer)
        return output
    

class CustomLoss(nn.Module):

    def __init__(self, actor, critic):
        super(CustomLoss, self).__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, state):
        loss = torch.mul(-1, self.critic(state))
        loss = loss.mean()

        return loss
        