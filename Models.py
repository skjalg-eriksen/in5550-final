import math
import torch
import torch.nn.functional as F

from torch import nn

class MLP(nn.Module):
    def __init__(self, token_vocab, tag_vocab, encoder, input_size=1024, hidden_size=512):
        super().__init__()
        attention_hops=4
        self.encoder = encoder #self_attention_Encoder3(token_vocab, attention_hops=attention_hops)
        
        self.MLP = nn.Sequential(nn.Linear(input_size*2*attention_hops*4, hidden_size), # input layer
                                 nn.ReLU(), # hidden layer
                                 nn.Linear(hidden_size, len(tag_vocab))) # ouput layer
    def forward(self, batch):
        u = self.encoder(batch.sentence1_tok)
        v = self.encoder(batch.sentence2_tok)
        
        # [u, v |u-v|, u*v]
        vec = torch.cat([torch.cat([u, v], dim=1), torch.abs(u-v), u*v], dim=1)
       
        return self.MLP(vec)
        
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
        
        #tokens, lengths = batch
        tokens = [batch.sentence1_tok, batch.sentence2_tok]
        
        for i, token in enumerate(tokens):
            token, lengths = token
            
            verbose = False
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



#https://arxiv.org/pdf/1703.03130.pdf
class self_attention_Encoder2_2(nn.Module):
    def __init__(self, token_vocab, batch_size, hidden_size=1024, dimension_a=256, attention_hops=4):
        super().__init__()
        self.hidden_size = hidden_size
        
        # word embeddings
        #self.embedding = nn.Embedding.from_pretrained(token_vocab.vectors)
        self.embedding = nn.Embedding(len(token_vocab), 300, padding_idx=token_vocab.stoi['<pad>']).from_pretrained(token_vocab.vectors)

        self.RNN = nn.LSTM(token_vocab.vectors.shape[1], hidden_size, bias=True, batch_first=True, bidirectional=True)
        #self.RNN_2 = nn.LSTM(token_vocab.vectors.shape[1], hidden_size, bias=True, batch_first=True, bidirectional=True)
        
        self.MLP = nn.Sequential(nn.Linear(hidden_size*2, dimension_a), # input layer
                                 nn.Tanh(), # hidden layer
                                 nn.Linear(dimension_a, attention_hops), # ouput layer
                                 nn.Softmax(dim=1)
                                )

        
    def forward(self, sentence):
        tokens, lengths = sentence;
        #print('len', lengths[0])
        lengths = list(torch.unbind(lengths))
        tokens = list(torch.unbind(tokens))
        A = [] 
        
        batch = []
        for i, (t, l) in enumerate(zip(tokens, lengths)):
            batch.append( [i, t, l, None] )
            
        batch = sorted(batch, reverse=True, key=lambda x: x[2]) # sort by lengths
        tokens = torch.stack([x[1] for x in batch])
        lengths = torch.stack([x[2] for x in batch])
        
        # get embedding vectors
        embedded = self.embedding(tokens)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)
        out, (hidden, _) = self.RNN(packed)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        A = self.MLP(out).transpose(2,1)
        u = A@out
        u = torch.sum(u, 1)
        # add the vectors to the batch list
        for i, v in enumerate(u):
            batch[i][3] = v
        batch = sorted(batch, key=lambda x: x[0]) # sort by original batch indexes
        u = torch.stack([x[3] for x in batch])
        u = u.transpose(1,0)
        
        u = torch.cat([v for v in u], dim=1)
        return u

class self_attention_Encoder2(nn.Module):
    def __init__(self, token_vocab, batch_size, hidden_size=1024, dimension_a=256, attention_hops=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.d_a = dimension_a
        self.batch_size = batch_size
        
        # word embeddings
        #self.embedding = nn.Embedding.from_pretrained(token_vocab.vectors)
        self.embedding = nn.Embedding(len(token_vocab), token_vocab.vectors.shape[1], padding_idx=token_vocab.stoi['<pad>']).from_pretrained(token_vocab.vectors)

        self.RNN = nn.LSTM(token_vocab.vectors.shape[1], hidden_size, bias=True, batch_first=True, bidirectional=True)
        
        self.MLP = nn.Sequential(nn.Linear(hidden_size*2, dimension_a), # input layer
                                 nn.Tanh(), # hidden layer
                                 nn.Linear(dimension_a, attention_hops), # ouput layer
                                 nn.Softmax(dim=1)
                                )
        self.init_state = (torch.autograd.Variable(torch.zeros(2,self.hidden_size)),torch.autograd.Variable(torch.zeros(2,self.hidden_size)))

    
    def forward(self, sentence):
        tokens, lengths = sentence;
        #print('len', lengths[0])
        lengths = list(torch.unbind(lengths))
        tokens = list(torch.unbind(tokens))
        A = [] 
        
        batch = []
        for i, (t, l) in enumerate(zip(tokens, lengths)):
            batch.append( [i, t, l, None] )
            
        batch = sorted(batch, reverse=True, key=lambda x: x[2]) # sort by lengths
        
        tokens = torch.stack([x[1] for x in batch])
        lengths = torch.stack([x[2] for x in batch])
        
        # get embedding vectors
        embedded = self.embedding(tokens)
        
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)
        
        init_hidden = [self.init_state[0] for l in lengths]
        init_cell = [self.init_state[1] for l in lengths]
        init_hidden = torch.stack(init_hidden).transpose(1,0)
        init_cell = torch.stack(init_cell).transpose(1,0)
        
        #print('init', init_hidden.shape)
        #print('Expected',2,len(lengths), self.hidden_size)
        
        out, (hidden, _) = self.RNN(packed, (init_hidden,init_cell))
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
 
        A = self.MLP(out).transpose(2,1)
        u = A@out
        
        # add the vectors to the batch list
        for i, v in enumerate(u):
            batch[i][3] = v
            
        batch = sorted(batch, key=lambda x: x[0]) # sort by original batch indexes
        u = torch.stack([x[3] for x in batch])
        u = u.transpose(1,0)
        
        u = torch.cat([v for v in u], dim=1)
        #print('rep u', u.shape)
        return u
     
