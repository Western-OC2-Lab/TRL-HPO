#register buffer in Pytorch
# if you have parameters in your model, which should be saved and restored in the state_dict, but not trained by the 
# optimizer, you should register them as buffers.
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np


class Utils: 

    def __init__(self):
        pass
    
    def soft_update(self, target, source, tau):

        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )


    def hard_update(self, target, source):

        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    


class PositionalEmbedding(nn.Module):
    
    def __init__(self, max_seq_len, embed_model_dim):
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_model_dim
        pe = torch.zeros(max_seq_len, self.embed_dim)
        
        for pos in range(max_seq_len):
            for i in range(0, self.embed_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2*i)/self.embed_dim)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2*(i+1))/self.embed_dim))) 
                
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # not calculated
        x = x * math.sqrt(self.embed_dim)
        # print(x.size())
        #add constant to embedding
        seq_len = x.size(1)
        # seq_len = x.shape[1]
        x = x + torch.autograd.Variable(self.pe[:,:seq_len], requires_grad=False)
        return x

class MultiHeadAttention(nn.Module):
    
    def __init__(self, embed_dim, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.single_head_dim = self.embed_dim // self.n_heads
        
        self.query_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias = False)
        self.key_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias = False)
        self.value_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias = False)
        self.out = nn.Linear(self.n_heads * self.single_head_dim, self.embed_dim)
        self.scores = None
        
    def forward(self, key, query, value):
        batch_size = key.size(0)
        seq_length = key.size(1)
        
        key = key.view(batch_size, seq_length, self.n_heads, self.single_head_dim)
        query = query.view(batch_size, seq_length, self.n_heads, self.single_head_dim)        
        value = value.view(batch_size, seq_length, self.n_heads, self.single_head_dim)                
        
        k = self.key_matrix(key)
        q = self.query_matrix(query)
        v = self.value_matrix(value)

        seq_length_query = query.size(1)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)        
        v = v.transpose(1, 2)
        
        k_adjusted = k.transpose(-1,-2)
        product = torch.matmul(q, k_adjusted)
        
        product = product / math.sqrt(self.single_head_dim)
        scores = F.softmax(product, dim = -1)

        scores = torch.matmul(scores, v)
        # self.scores = scores
        
        
        concat = scores.transpose(1,2).contiguous().view(batch_size, seq_length_query, self.single_head_dim*self.n_heads)
        self.scores = concat
        # print('concat inside', torch.sum(torch.mean(F.softmax(concat, dim = 1), dim = 1)))

        output = self.out(concat)
        
        return output

class OrnsteinUhlenbeckActionNoise:

    def __init__(self, action_dim, mu = 0, theta = 0.15, sigma = 0.2) -> None:
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx

        return self.X