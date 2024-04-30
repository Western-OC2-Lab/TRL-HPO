from Transformer.transformer_blocks import *

class TransformerActor(nn.Module):
    
    def __init__(self, seq_len, embed_dim, num_layers, expansion_factor, n_heads, action_space):
        super(TransformerActor, self).__init__()
        self.encoder = TransformerEncoder(seq_len, embed_dim, num_layers, expansion_factor, n_heads)
        self.actor_output = nn.Linear(embed_dim*seq_len, action_space)
        
    def forward(self, x):
        x = self.encoder(x)

        batch_size, seq_len, embed_dim = x.size()
        x = x.view(batch_size, seq_len*embed_dim)
        
        # actor_output = F.sigmoid(self.actor_output(x))
        # x = self.actor_output(x)
        actor_output = self.actor_output(x)

        actor_output = torch.sigmoid(actor_output)
        
        return actor_output


class TransformerCritic(nn.Module):
    
    def __init__(self, seq_len, embed_dim, num_layers, expansion_factor, n_heads, action_space):
        super(TransformerCritic, self).__init__()
        self.encoder = TransformerEncoder(seq_len, embed_dim, num_layers, expansion_factor, n_heads)
        self.critic_output = nn.Linear(embed_dim*seq_len, 1)
        
    def forward(self, x):
        x = self.encoder(x)
        batch_size, seq_len, embed_dim = x.size()
        x = x.view(batch_size, seq_len*embed_dim)
        
        critic_output = self.critic_output(x)
        critic_output = torch.tanh(critic_output)

        
        return critic_output