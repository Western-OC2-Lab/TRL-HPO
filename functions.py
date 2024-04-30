from hidden_layers_config import *
import torch.nn as nn
import torch
import torch.nn.functional as F

def defining_activation_functions(fcn):
    if fcn == 'sigmoid':
        return nn.Sigmoid()
    elif fcn == 'tanh':
        return nn.Tanh()
    elif fcn == 'relu':
        return nn.ReLU()
    elif fcn == 'leakyrelu':
        return nn.LeakyReLU()
    elif fcn == 'elu':
        return nn.ELU()
    elif fcn == 'gelu':
        return nn.GELU()
    else:
        return nn.Identity()
    
def return_name(name_layer, arg1, arg2, arg3):
    final_name = ""
    if name_layer == 'Conv2d':
        final_name = f"{name_layer}_F-{arg1}_K-{arg2}_S-{arg3}"
    elif name_layer == 'FCL':
        arg2 = int(arg2)
        final_name = f"{name_layer}_N-{arg1}_B-{arg2}_AF-{arg3}"
    else:
        final_name = f"{name_layer}_K-{arg1}_S-{arg2}_P-{arg3}"

    return final_name
    

def action_space_to_layer(input_layer, action_output, idx, prev_layer, default_output_layer = 10):

    layer = None
    name_layer = ""
    nb_neurons = 0
    full_layer_name = ""
    curr_filters_neurons = input_layer[0]

    if prev_layer == None:
        curr_idx = np.argwhere(ACTION_LIMITS['Conv2d']['output_filters'] == curr_filters_neurons)
        lower_limit = 0
        if len(curr_idx) > 0:
            lower_limit = curr_idx
        o_idx = min(int(action_output[1] * (len(ACTION_LIMITS['Conv2d']['output_filters'])-1 - lower_limit) + lower_limit+1), len(ACTION_LIMITS['Conv2d']['output_filters'])-1)
        

        k_idx = int(action_output[2] * (len(ACTION_LIMITS['Conv2d']['kernel_size'])-1))
        s_idx = int(action_output[3] * (len(ACTION_LIMITS['Conv2d']['strides'])-1))
        layer = [nn.Conv2d(input_layer[0], ACTION_LIMITS['Conv2d']['output_filters'][o_idx],
                        kernel_size = ACTION_LIMITS['Conv2d']['kernel_size'][k_idx],
                        stride = ACTION_LIMITS['Conv2d']['strides'][s_idx])]
        name_layer = 'Conv2d'
        full_layer_name = return_name(name_layer, ACTION_LIMITS['Conv2d']['output_filters'][o_idx],
                                      ACTION_LIMITS['Conv2d']['kernel_size'][k_idx], 
                                      ACTION_LIMITS['Conv2d']['strides'][s_idx])

    elif prev_layer == "MaxPool" or prev_layer == 'Conv2d':
        if action_output[0] < 0.33:
            curr_idx = np.argwhere(ACTION_LIMITS['Conv2d']['output_filters'] == curr_filters_neurons)
            lower_limit = 0
            if len(curr_idx) > 0:
                lower_limit = curr_idx
            
            o_idx = min(int(action_output[1] * (len(ACTION_LIMITS['Conv2d']['output_filters'])-1 - lower_limit) + lower_limit+1), len(ACTION_LIMITS['Conv2d']['output_filters'])-1)


            k_idx = int(action_output[2] * (len(ACTION_LIMITS['Conv2d']['kernel_size'])-1))
            s_idx = int(action_output[3] * (len(ACTION_LIMITS['Conv2d']['strides'])-1))
            layer = [nn.Conv2d(input_layer[0], ACTION_LIMITS['Conv2d']['output_filters'][o_idx],
                            kernel_size = ACTION_LIMITS['Conv2d']['kernel_size'][k_idx],
                            stride = ACTION_LIMITS['Conv2d']['strides'][s_idx])]
            name_layer = 'Conv2d'
            full_layer_name = return_name(name_layer, ACTION_LIMITS['Conv2d']['output_filters'][o_idx],
                                      ACTION_LIMITS['Conv2d']['kernel_size'][k_idx], 
                                      ACTION_LIMITS['Conv2d']['strides'][s_idx])
            
        elif action_output[0] >= 0.33 and action_output[0] < 0.67:
            o_idx = int(action_output[1] * (len(ACTION_LIMITS['FCL']['output_neurons'])-1))
            b_idx = int(action_output[2] * (len(ACTION_LIMITS['FCL']['bias'])-1))
            a_idx = int(action_output[3] * (len(ACTION_LIMITS['FCL']['activation_functions'])-1))

            activation_function = defining_activation_functions(ACTION_LIMITS['FCL']['activation_functions'][a_idx])
            
            layer = [nn.Linear(in_features = int(input_layer[0]) * int(input_layer[1]) * int(input_layer[2]), 
                                        out_features = ACTION_LIMITS['FCL']['output_neurons'][o_idx],
                                        bias = ACTION_LIMITS['FCL']['bias'][b_idx]),
                            activation_function]
                            
            name_layer = 'FCL'
            nb_neurons = ACTION_LIMITS['FCL']['output_neurons'][o_idx]

            full_layer_name = return_name(name_layer, ACTION_LIMITS['FCL']['output_neurons'][o_idx],
                                      ACTION_LIMITS['FCL']['bias'][b_idx], 
                                      ACTION_LIMITS['FCL']['activation_functions'][a_idx])

        else:
            k_idx = int(action_output[1] * (len(ACTION_LIMITS['MaxPool']['kernel_size']))-1)
            s_idx = int(action_output[2] * (len(ACTION_LIMITS['MaxPool']['stride']))-1)
            p_idx = int(action_output[3] * (len(ACTION_LIMITS['MaxPool']['padding']))-1)
            layer = [nn.MaxPool2d(
                kernel_size = ACTION_LIMITS['MaxPool']['kernel_size'][k_idx],
                stride = ACTION_LIMITS['MaxPool']['stride'][s_idx],
                padding = ACTION_LIMITS['MaxPool']['padding'][p_idx],
            )]
            name_layer = 'MaxPool'

            full_layer_name = return_name(name_layer, ACTION_LIMITS['MaxPool']['kernel_size'][k_idx],
                                      ACTION_LIMITS['MaxPool']['stride'][s_idx], 
                                      ACTION_LIMITS['MaxPool']['padding'][p_idx])
    else:
        curr_idx = np.argwhere(ACTION_LIMITS['FCL']['output_neurons'] == curr_filters_neurons)
        upper_limit = curr_idx
        o_idx = min(int(action_output[1] * (upper_limit + 1)), len(ACTION_LIMITS['FCL']['output_neurons'])-1)

        b_idx = int(action_output[2] * (len(ACTION_LIMITS['FCL']['bias'])-1))
        a_idx = int(action_output[3] * (len(ACTION_LIMITS['FCL']['activation_functions'])-1))

        activation_function = defining_activation_functions(ACTION_LIMITS['FCL']['activation_functions'][a_idx])
        
        layer = [nn.Linear(in_features = int(input_layer[0]) * int(input_layer[1]) * int(input_layer[2]), 
                                    out_features = ACTION_LIMITS['FCL']['output_neurons'][o_idx],
                                    bias = ACTION_LIMITS['FCL']['bias'][b_idx]),
                        activation_function]
                        
        name_layer = 'FCL'
        nb_neurons = ACTION_LIMITS['FCL']['output_neurons'][o_idx]

        full_layer_name = return_name(name_layer, ACTION_LIMITS['FCL']['output_neurons'][o_idx],
                                      ACTION_LIMITS['FCL']['bias'][b_idx], 
                                      ACTION_LIMITS['FCL']['activation_functions'][a_idx])
    
    # print(input_layer, action_output, idx, prev_layer, layer, name_layer, nb_neurons)

    return layer, name_layer, nb_neurons, full_layer_name

def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


def calculate_accuracy(outputs, labels):
    pred_prob = F.softmax(outputs, dim = 1)
    pred_labels = torch.argmax(pred_prob, dim = 1)
    
    binary_class = [1  if pred_labels[idx].item() == labels[idx].item() else 0 for idx, lbl in enumerate(labels)]
    
    return np.sum(binary_class) / labels.size(0)

