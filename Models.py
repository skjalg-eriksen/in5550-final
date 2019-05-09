import math
import torch
import torch.nn.functional as F

from torch import nn

class myInferenceModel(nn.Module):
    def __init__(self, token_vocab, tag_vocab):
        super().__init__()
        
        self.encoder = Encoder(token_vocab);
        
        input_size = 300 # output size of the encoder
        hidden_state_size = 512
        
        self.MLP = nn.Sequential(nn.Linear(input_size, hidden_state_size), # input layer
                                 nn.ReLU(),
                                 nn.Linear(hidden_state_size, hidden_state_size), # hidden layer
                                 nn.ReLU(),
                                 nn.Linear(hidden_state_size, len(tag_vocab))) # ouput layer
    def forward(self, batch):
        
        # u and v are encoded sentences
        u = self.encoder(batch.sentence1_tok)
        v = self.encoder(batch.sentence2_tok)
        
        # [u,v |u-v|, u*v]
        a = torch.cat([u, v])
        b = torch.abs(u-v)
        c = torch.cat(u*v)
        vec = torch.cat([a, b, c])
        print(vec.shape)
        return self.MLP( )
    
class Encoder(nn.Module):
    def __init__(self, token_vocab):
        super().__init__()
        self._embeds = nn.Embedding.from_pretrained(token_vocab.vectors)
        
        self._conv1 = nn.Conv1d(in_channels=token_vocab.vectors.shape[1],
                                out_channels=100,
                                kernel_size=2, 
                                padding=4)
        self._conv2 = nn.Conv1d(in_channels=token_vocab.vectors.shape[1],
                                out_channels=100,
                                kernel_size=3, 
                                padding=4)
        self._conv3 = nn.Conv1d(in_channels=token_vocab.vectors.shape[1],
                                out_channels=100,
                                kernel_size=4, 
                                padding=4)

        self._rnn = nn.RNN(input_size=300, 
                           hidden_size=150, 
                           num_layers=2,
                           bidirectional=True,
                           batch_first=True)
        self._out = nn.Linear(300, 150)

    def forward(self, sentence):
        tokens, lengths = sentence

        # embeddings go here
        embeds = self._embeds(tokens).transpose(1, 2)
       
        # conv + pooling
        c1 = self._conv1(embeds)
        c2 = self._conv2(embeds)
        c3 = self._conv3(embeds)
        
        p1 = F.relu(F.max_pool1d(c1, 2))
        p2 = F.relu(F.max_pool1d(c2, 2))
        p3 = F.relu(F.max_pool1d(c3, 2))
        max_dim = max(i.size(-1) for i in [p1, p2, p3])

        # funky padding stuff because of pooling  
        p1 = F.pad(p1, (0, max_dim-p1.size(-1)), 'constant', 0)
        p2 = F.pad(p2, (0, max_dim-p2.size(-1)), 'constant', 0)
        p3 = F.pad(p3, (0, max_dim-p3.size(-1)), 'constant', 0)

        rnn_input = torch.cat([p1, p2, p3], dim=1)
        rnn_input = rnn_input.transpose(1, 2)
        
        # recurrent
        rnn_out = self._rnn(rnn_input)[0]
        rnn_out = rnn_out[:,-1]
        
        return self._out(rnn_out)


