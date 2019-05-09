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
        #print(vec.shape)
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

class MLP(nn.Module):
    def __init__(self, token_vocab, tag_vocab, input_size=1024, hidden_size=512):
        super().__init__()
        attention_hops=3
        self.encoder = self_attention_Encoder3(token_vocab, attention_hops=attention_hops)
        
        self.MLP = nn.Sequential(nn.Linear(input_size*2*attention_hops*4, hidden_size), # input layer
                                 nn.ReLU(), # hidden layer
                                 nn.Linear(hidden_size, len(tag_vocab))) # ouput layer
    def forward(self, batch):
        u = self.encoder(batch.sentence1_tok)
        v = self.encoder(batch.sentence2_tok)
        
        # [u, v |u-v|, u*v]
        vec = torch.cat([torch.cat([u, v], dim=1), torch.abs(u-v), u*v], dim=1)
       
        return self.MLP(vec)

#https://arxiv.org/pdf/1703.03130.pdf
class self_attention_Encoder(nn.Module):
    def __init__(self, token_vocab, hidden_size=1024, dimension_a=100, attention_hops=4):
        super().__init__()
        self.hidden_size = hidden_size
        
        # word embeddings
        #self.embedding = nn.Embedding.from_pretrained(token_vocab.vectors)
        self.embedding = nn.Embedding(len(token_vocab), 300, padding_idx=token_vocab.stoi['<pad>']).from_pretrained(token_vocab.vectors)

        # to achive bi-direction i stack 2 LSTMCells, one for each direction; RNN_1 (->), RNN_2 (<-)
        self.RNN_1 = nn.LSTMCell(token_vocab.vectors.shape[1], hidden_size, bias=True)
        self.RNN_2 = nn.LSTMCell(token_vocab.vectors.shape[1], hidden_size, bias=True)
        
        #self.RNN_1 = nn.RNNCell(token_vocab.vectors.shape[1], hidden_size, bias=True)
        #self.RNN_2 = nn.RNNCell(token_vocab.vectors.shape[1], hidden_size, bias=True)
        
        
        # alpha weights
        Ws1 = torch.zeros(dimension_a, hidden_size*2)
        nn.init.normal_(Ws1, mean=0.0, std=1.0)
        self.Ws1 = nn.Parameter(Ws1)
        
        Ws2 = torch.zeros(attention_hops, dimension_a)
        nn.init.normal_(Ws2, mean=1.0, std=1.0)
        self.Ws2 = nn.Parameter(Ws2)
        
    def forward(self, sentence):
        tokens, _ = sentence;
        batch_size = len(_)
        H_1 = [] # layer 1, ->
        H_2 = [] # layer 2, <-
        A = [] # softmax(ws2*tanh(ws1*Ht))
        
        # get embedding vectors
        embedded = self.embedding(tokens)
        embedded = embedded.transpose(1,0)
     
        
        # initial word for layer 1
        hidden_state = self.RNN_1(embedded[0])
        H_1.append(hidden_state)
        
        # layer 1 RNN
        for token in embedded[1:]:
            hidden_state = self.RNN_1(token, hidden_state)
            H_1.append(hidden_state)
        
        # flip the words to do layer 2
        embedded = torch.flip(embedded, [0])
        
        # initial word for layer 2
        hidden_state = self.RNN_2(embedded[0])
        H_2.append(hidden_state)
       
        # layer 2 RNN
        for token in embedded[1:]:
            hidden_state = self.RNN_2(token, hidden_state)
            H_2.append(hidden_state)
       
        #flip H_2 so the hidden states line up with H_1
        H_2.reverse()
            
        # concat each hidden states for H_1, H_2
        H = []
        for h1, h2 in zip(H_1, H_2):
            H.append(torch.cat([h1[0], h2[0]], dim=1))
        H = torch.stack(H) # turn python list of tensors into 1 tensor for the hidden states
        H = H.transpose(1,0)
        
        # for every sentence in batch calculate attention
        for batched_H in H:
            A.append(F.softmax(self.Ws2 @ torch.tanh(self.Ws1 @ batched_H.transpose(0,1)), dim=1) )
        
        
        # attention distrubution
        A = torch.stack(A)
        u = A@H # weighted sums
        u = u.view((batch_size, -1, 1))
        u = torch.squeeze(u, -1)
        return u

class self_attention_Encoder2(nn.Module):
    def __init__(self, token_vocab, hidden_size=1024, dimension_a=100, attention_hops=4):
        super().__init__()
        self.hidden_size = hidden_size
        
        # word embeddings
        #self.embedding = nn.Embedding.from_pretrained(token_vocab.vectors)
        self.embedding = nn.Embedding(len(token_vocab), 300, padding_idx=token_vocab.stoi['<pad>']).from_pretrained(token_vocab.vectors)

        self.RNN = nn.LSTM(token_vocab.vectors.shape[1], hidden_size, bias=True, batch_first=True, bidirectional=True)
        #self.RNN_2 = nn.LSTM(token_vocab.vectors.shape[1], hidden_size, bias=True, batch_first=True, bidirectional=True)
        
        
        # alpha weights
        Ws1 = torch.zeros(dimension_a, hidden_size*2)
        nn.init.normal_(Ws1, mean=0.0, std=1.0)
        self.Ws1 = nn.Parameter(Ws1)
        
        Ws2 = torch.zeros(attention_hops, dimension_a)
        nn.init.normal_(Ws2, mean=1.0, std=1.0)
        self.Ws2 = nn.Parameter(Ws2)
        
    def forward(self, sentence):
        tokens, lengths = sentence;
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
        out, hidden = self.RNN(packed)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out)
        
        for h in out.transpose(0,1):
            A.append(F.softmax(self.Ws2 @ torch.tanh(self.Ws1 @ h.t()), dim=1))
        
        # attention distrubution
        A = torch.stack(A)
        #print('out',out.transpose(0,1).shape, 'A', A.shape)
        #u = []
        #for a in A.transpose(1,0): 
            #u.append(torch.mul(a, out.transpose(0,1)))
            #print(a.shape)
        #u = torch.mul(A, out.transpose(0,1)) # weighted sums
        u = A@out.transpose(0,1)
        #print('u',u.shape)
        
        
        
        # add the vectors to the batch list
        for i, v in enumerate(u):
            batch[i][3] = v
            
        batch = sorted(batch, key=lambda x: x[0]) # sort by original batch indexes
        u = torch.stack([x[3] for x in batch])
        u = u.transpose(1,0)
        
        u = torch.cat([v for v in u], dim=1)
        return u

class self_attention_Encoder3(nn.Module):
    def __init__(self, token_vocab, hidden_size=1024, dimension_a=100, attention_hops=3):
        super().__init__()
        self.hidden_size = hidden_size
        
        # word embeddings
        #self.embedding = nn.Embedding.from_pretrained(token_vocab.vectors)
        self.embedding = nn.Embedding(len(token_vocab), 300, padding_idx=token_vocab.stoi['<pad>']).from_pretrained(token_vocab.vectors)

        # to achive bi-direction i stack 2 LSTMCells, one for each direction; RNN_1 (->), RNN_2 (<-)
        self.RNN_1 = nn.LSTMCell(token_vocab.vectors.shape[1], hidden_size, bias=True)
        self.RNN_2 = nn.LSTMCell(token_vocab.vectors.shape[1], hidden_size, bias=True)
        
        #self.RNN_1 = nn.RNNCell(token_vocab.vectors.shape[1], hidden_size, bias=True)
        #self.RNN_2 = nn.RNNCell(token_vocab.vectors.shape[1], hidden_size, bias=True)
        
        
        # alpha weights
        Ws1 = torch.zeros(dimension_a, hidden_size*2)
        nn.init.normal_(Ws1, mean=0.0, std=1.0)
        self.Ws1 = nn.Parameter(Ws1)
        
        Ws2 = torch.zeros(attention_hops, dimension_a)
        nn.init.normal_(Ws2, mean=1.0, std=1.0)
        self.Ws2 = nn.Parameter(Ws2)
        
    def forward(self, sentence):
        tokens, _ = sentence;
        batch_size = len(_)
        H_1 = [] # layer 1, ->
        H_2 = [] # layer 2, <-
        A = [] # softmax(ws2*tanh(ws1*Ht))
        
        # get embedding vectors
        embedded = self.embedding(tokens)
        embedded = embedded.transpose(1,0)
     
        
        # initial word for layer 1
        hidden_state = self.RNN_1(embedded[0])
        H_1.append(hidden_state)
        
        # layer 1 RNN
        for token in embedded[1:]:
            hidden_state = self.RNN_1(token, hidden_state)
            H_1.append(hidden_state)
        
        # flip the words to do layer 2
        embedded = torch.flip(embedded, [0])
        
        # initial word for layer 2
        hidden_state = self.RNN_2(embedded[0])
        H_2.append(hidden_state)
       
        # layer 2 RNN
        for token in embedded[1:]:
            hidden_state = self.RNN_2(token, hidden_state)
            H_2.append(hidden_state)
       
        #flip H_2 so the hidden states line up with H_1
        H_2.reverse()
        
        # concat each hidden states for H_1, H_2
        H = []
        for h1, h2 in zip(H_1, H_2):
            H.append(torch.cat([h1[0], h2[0]], dim=1))

        H = torch.stack(H) # turn python list of tensors into 1 tensor for the hidden states
        H = H.transpose(1,0)
        
        a = [ F.softmax(self.Ws2 @ F.tanh(self.Ws1 @ h.t()), dim=1) for h in H ]
        
        # attention distrubution
        A = torch.stack(a)
        #go through all in batch and do operation
        v = []
        for a, h in zip(A,H):
            v.append(a@h)
        u = torch.stack(v)
        #print('u',u.shape)
        
        u_ = [ torch.cat([v for v in u_], dim=-1) for u_ in u ]
        u = torch.stack(u_)
       
        #print('U',u.shape)
        # batch_size x attention_hops x hidden_size*2
        return u