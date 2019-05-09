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
    
class convNetEncoder(nn.Module):
# hierarchical convnet architecture
    def __init__(self, token_vocab, tag_vocab):
        super().__init__()
        
        input_size = token_vocab.vectors.shape[1]*16 # output size of the encoder
        hidden_state_size = 512
        #self._embeds = nn.Embedding.from_pretrained(token_vocab.vectors)
        self._embeds = nn.Embedding(len(token_vocab), 300, padding_idx=token_vocab.stoi['<pad>']).from_pretrained(token_vocab.vectors)
        # define the rest of your components
        self._conv1 = nn.Conv1d(in_channels=token_vocab.vectors.shape[1],
                                out_channels=token_vocab.vectors.shape[1],
                                kernel_size=3, 
                                padding=1)
        self._conv2 = nn.Conv1d(in_channels=token_vocab.vectors.shape[1],
                                out_channels=token_vocab.vectors.shape[1],
                                kernel_size=3, 
                                padding=1)
        self._conv3 = nn.Conv1d(in_channels=token_vocab.vectors.shape[1],
                                out_channels=token_vocab.vectors.shape[1],
                                kernel_size=3, 
                                padding=1)
        self._conv4 = nn.Conv1d(in_channels=token_vocab.vectors.shape[1],
                                out_channels=token_vocab.vectors.shape[1],
                                kernel_size=3, 
                                padding=1)

        self.MLP = nn.Sequential(nn.Linear(input_size, hidden_state_size), # input layer
                                 nn.ReLU(), # hidden layer
                                 nn.Linear(hidden_state_size, len(tag_vocab))) # ouput layer
    def forward(self, batch):
        verbose = False
        #tokens, lengths = batch
        tokens = [batch.sentence1_tok, batch.sentence2_tok]
        
        for i, token in enumerate(tokens):
            token, lengths = token
            
            # embeddings go here
            embeds = self._embeds(token).transpose(1, 2)
            if verbose: print("embeds",embeds.shape)
            # conv + pooling
            c1 = self._conv1(embeds)
            if verbose: print("c1",c1.shape)
            c2 = self._conv2(c1)
            if verbose: print("c2",c2.shape)
            c3 = self._conv3(c2)
            if verbose: print("c3",c3.shape)
            c4 = self._conv4(c2)
            if verbose: print("c4",c4.shape)
            #tokens[i] = F.dropout(torch.cat([c1.max(dim=-1)[0], c2.max(dim=-1)[0], c3.max(dim=-1)[0], c4.max(dim=-1)[0]], dim=1), p=0.33, training=self.training)
            tokens[i] = torch.cat([c1.max(dim=-1)[0], c2.max(dim=-1)[0], c3.max(dim=-1)[0], c4.max(dim=-1)[0]], dim=1)
            if verbose: print("sentence {}".format(i), tokens[i].shape)
        
        u, v = tokens
        
        # [u,v |u-v|, u*v]
        a = torch.cat([u, v], dim=1)
        b = torch.abs(u-v)
        c = u*v
        vec = torch.cat([a, b, c], dim=1)
        if verbose: print('vec', vec.shape)
        return self.MLP(vec)

