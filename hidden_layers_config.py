import numpy as np

ACTION_LIMITS = {
    'Conv2d': {
        'output_filters': np.arange(8, 128, step = 8),
        'kernel_size': [3, 5, 7],
        'strides': [1, 2, 3],
    },
    'FCL': {
        'output_neurons': np.arange(16, 512, step = 8),
        'bias': [True, False],
        'activation_functions': ['sigmoid', 'tanh', 'relu', 'leakyrelu', 'elu', 'gelu', None],
    },
    'MaxPool': {
        'kernel_size': np.arange(2, 8, step  =1),
        'stride': [1, 2, 3],
        'padding': [0, 1, 2, 3]
    },
    'OutputLayer': {}
}