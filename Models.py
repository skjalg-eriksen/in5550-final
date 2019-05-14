import math
import torch
import torch.nn.functional as F
from collections import OrderedDict
from torch import nn

class MLP_classifier(nn.Module):
    def __init__(self, token_vocab, tag_vocab, encoder, hidden_size=512, num_layers=1, dropout=0):
        super().__init__()
        self.encoder = encoder # encoder to fixed-lenght-representation of sentences
        self.dropout = dropout # dropout preformed on resulting vector representation of the 2 encoded sentences
        
        # input size will be the encoder output size * 4, since we concatinate 2 encoded sentences u and v, [u, v |u-v|, u*v]
        layers = [('inputLayer', nn.Linear( self.encoder.output_size*4, hidden_size )), ('activation_0', nn.ReLU())]
        
        # add hidden layers
        [ 
            layers.extend([    
                            ('hiddenLayer_{}'.format(layer + 1), nn.Linear(hidden_size, hidden_size)),
                            ('activation_{}'.format(layer + 1), nn.ReLU())
                          ])
            for layer in range(num_layers-1)
        ]
        
        # add the output layer
        layers.append(('outputLayer', nn.Linear(hidden_size, len(tag_vocab))))
        
        # add all layers into a sequential model
        self.classifier = nn.Sequential(OrderedDict(layers))
        print(self)
        
    def forward(self, batch):
        p = torch.tensor(0) # penalization for similar attention dims, used only for sentenceEmbeddingEncoder
        # encode 2 sentences to fixed-lenght-representation u and v
        u = self.encoder(batch.sentence1_tok)
        v = self.encoder(batch.sentence2_tok)
        
        if isinstance(self.encoder, sentenceEmbeddingEncoder):
            # unpacking sentence representation and penalization
            u , p1 = u
            v, p2 = v
            p = (p1+p2)/2  # mean penalization between the 2 sentences
            
        # [u, v |u-v|, u*v]
        vec = torch.cat([torch.cat([u, v], dim=1), torch.abs(u-v), u*v], dim=1)
        
        # dropout regularization
        vec = F.dropout(vec, p=self.dropout, training=self.training)
        
        # return classification and penalty
        return self.classifier(vec), p
        
class convNetEncoder(nn.Module):
# hierarchical convnet architecture
    def __init__(self, token_vocab, filter_size=256, kernel_size=3, padding=1):
        super().__init__()
        
        # output = filter size * 4 layers of convolution
        self.output_size = filter_size*4
        
        self._embeds = nn.Embedding(  len(token_vocab), token_vocab.vectors.shape[1], # embedding dim
                                        padding_idx=token_vocab.stoi['<pad>']).from_pretrained(token_vocab.vectors)
        
        # convolution layers
        self._conv1 = nn.Conv1d(in_channels=token_vocab.vectors.shape[1],
                                out_channels=filter_size,
                                kernel_size=kernel_size, 
                                padding=padding)
        self._conv2 = nn.Conv1d(in_channels=filter_size,
                                out_channels=filter_size,
                                kernel_size=kernel_size, 
                                padding=padding)
        self._conv3 = nn.Conv1d(in_channels=filter_size,
                                out_channels=filter_size,
                                kernel_size=kernel_size, 
                                padding=padding)
        self._conv4 = nn.Conv1d(in_channels=filter_size,
                                out_channels=filter_size,
                                kernel_size=kernel_size, 
                                padding=padding)

    def forward(self, sentence):
        tokens, _ = sentence;
        
        # embeddings go here
        embeds = self._embeds(tokens).transpose(1, 2)
        
        # hierarchical convolution
        c1 = self._conv1(embeds)
        c2 = self._conv2(c1)
        c3 = self._conv3(c2)
        c4 = self._conv4(c2)
        
        # max pooling + concatinate
        u = torch.cat([ c1.max(dim=-1)[0],
                        c2.max(dim=-1)[0], 
                        c3.max(dim=-1)[0], 
                        c4.max(dim=-1)[0]], dim=1)
        return u

class LastStateEncoder(nn.Module):
# encodes a sentence to the last hidden state of a RNN
    def __init__(self, token_vocab, hidden_size=1024, RNN='LSTM'):
        super().__init__()
        # output size
        self.output_size = hidden_size
        self.type = RNN
        
        # Word embeddings
        self.embedding = nn.Embedding(  len(token_vocab), token_vocab.vectors.shape[1], # embedding dim
                                        padding_idx=token_vocab.stoi['<pad>']).from_pretrained(token_vocab.vectors)
        # RNN
        if RNN.lower() in 'lstm':
            self.RNN = nn.LSTM(token_vocab.vectors.shape[1], hidden_size, bias=True, batch_first=True)
        elif RNN.lower() in 'bilstm':
            self.RNN = nn.LSTM(token_vocab.vectors.shape[1], hidden_size, bias=True, batch_first=True, bidirectional=True)
        elif RNN.lower() in 'gru':
            self.RNN = nn.GRU(token_vocab.vectors.shape[1], hidden_size, bias=True, batch_first=True)
        elif RNN.lower() in 'bigru':
            self.RNN = nn.GRU(token_vocab.vectors.shape[1], hidden_size, bias=True, batch_first=True, bidirectional=True)
        elif RNN.lower() in 'rnn':
            self.RNN = nn.RNN(token_vocab.vectors.shape[1], hidden_size, bias=True, batch_first=True)
        elif RNN.lower() in 'birnn':
            self.RNN = nn.RNN(token_vocab.vectors.shape[1], hidden_size, bias=True, batch_first=True, bidirectional=True)
            
            
            
        if 'bi' in RNN.lower():  
            self.output_size *= 2
            self.init_state = ( # 2 directions
                            torch.autograd.Variable(torch.zeros(2, hidden_size)), # initial hidden state
                            torch.autograd.Variable(torch.zeros(2, hidden_size))) # initial cell state
        else:
            self.init_state = ( # 1 direction
                            torch.autograd.Variable(torch.zeros(1, hidden_size)), # initial hidden state
                            torch.autograd.Variable(torch.zeros(1, hidden_size))) # initial cell state
    def forward(self, sentence):
        tokens, lengths = sentence;
        
        # list(unbind()) turns tensors into lists. 
        lengths = list(torch.unbind(lengths))
        tokens = list(torch.unbind(tokens))
        
        # add batch to list object to keep track of original indexes
        batch = []
        for i, (t, l) in enumerate(zip(tokens, lengths)):
            # i batch index, t sentence tokens, l lenght, None will be sentence embedding u
            batch.append( [i, t, l, None] ) 
        
        # sort batch list by lenghts
        batch = sorted(batch, reverse=True, key=lambda x: x[2])
        
        # get sorted tokens and lengths
        tokens = torch.stack([x[1] for x in batch])
        lengths = torch.stack([x[2] for x in batch])
        
        # get embedding vectors
        embedded = self.embedding(tokens)
        
        # pack sentence
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)
        
        # for each sentence in batch use a initial hidden state init_hidden
        init_hidden = [self.init_state[0] for l in lengths]
        
        # turn lists into tensors
        init_hidden = torch.stack(init_hidden).transpose(1,0)
        
        init = init_hidden
        
        # if lstm add initial cell state to init
        if 'lstm' in self.type.lower():
            init_cell = [self.init_state[1] for l in lengths]
            init_cell = torch.stack(init_cell).transpose(1,0)
            init = (init_hidden,init_cell)
        
        # get rnn hidden states out
        _, u = self.RNN(packed, init)
        
        if 'lstm' in self.type.lower(): (u, _) = u
        #print('u, pre',u.shape)
        
        #u = torch.squeeze(u, 0)
        
        u = torch.cat( [x for x in u], 1 )
        
        #print('u, post',u.shape)
        
        # add the sentence embedding u_ in u to the batch list
        for i, u_ in enumerate(u):
            batch[i][3] = u_
                
        # sort by original batch indexes i
        batch = sorted(batch, key=lambda x: x[0])
        
        
        # return tensor u.
        u = torch.stack([x[3] for x in batch])
        return u

class BiLSTMEncoder(nn.Module):
# max/mean pooling BiLSTM encoder
    def __init__(self, token_vocab, hidden_size=1024, pooling='max'):
        super().__init__()
        self.pooling = pooling
        
        # output =
        self.output_size = hidden_size*2
        
        # Word embeddings
        self.embedding = nn.Embedding(  len(token_vocab), token_vocab.vectors.shape[1], # embedding dim
                                        padding_idx=token_vocab.stoi['<pad>']).from_pretrained(token_vocab.vectors)
        # BiLSTM
        self.RNN = nn.LSTM(token_vocab.vectors.shape[1], hidden_size, bias=True, batch_first=True, bidirectional=True)
        
        self.init_state = (
                        torch.autograd.Variable(torch.zeros(2, hidden_size)), # initial hidden state
                        torch.autograd.Variable(torch.zeros(2, hidden_size))) # initial cell state

    def forward(self, sentence):
        tokens, lengths = sentence;
        
        # list(unbind()) turns tensors into lists. 
        lengths = list(torch.unbind(lengths))
        tokens = list(torch.unbind(tokens))
        
        # add batch to list object to keep track of original indexes
        batch = []
        for i, (t, l) in enumerate(zip(tokens, lengths)):
            # i batch index, t sentence tokens, l lenght, None will be sentence embedding u
            batch.append( [i, t, l, None] ) 
        
        # sort batch list by lenghts
        batch = sorted(batch, reverse=True, key=lambda x: x[2])
        
        # get sorted tokens and lengths
        tokens = torch.stack([x[1] for x in batch])
        lengths = torch.stack([x[2] for x in batch])
        
        # get embedding vectors
        embedded = self.embedding(tokens)
        
        # pack sentence
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)
        
        # for each sentence in batch use a initial hidden state init_hidden, and a initial cell state init_cell
        init_hidden = [self.init_state[0] for l in lengths]
        init_cell = [self.init_state[1] for l in lengths]
        
        # turn lists into tensors
        init_hidden = torch.stack(init_hidden).transpose(1,0)
        init_cell = torch.stack(init_cell).transpose(1,0)
        
        # get rnn hidden states out
        out, _ = self.RNN(packed, (init_hidden,init_cell))
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        
        # pooling
        if self.pooling.lower() in 'max':
            u = torch.max(out, 1)[0]
        else:
            u = torch.mean(out, 1)[0]
        
        # add the sentence embedding u_ in u to the batch list
        for i, u_ in enumerate(u):
            batch[i][3] = u_
        
        # sort by original batch indexes i
        batch = sorted(batch, key=lambda x: x[0])
        
        # return tensor u.
        u = torch.stack([x[3] for x in batch])
        return u
        
class sentenceEmbeddingEncoder(nn.Module):
# structured self-attentive sentence embedding architecture, paper: https://arxiv.org/pdf/1703.03130.pdf
    def __init__(self, token_vocab, hidden_size=1024, attention_dim=256, attention_hops=4, penalty=0):
        super().__init__()
        self.penalty = penalty
        # output dimension, used by MLP_classifier to determin the dimensions of the input layer
        self.output_size = hidden_size*attention_hops*2
        
        # Word embeddings
        self.embedding = nn.Embedding(  len(token_vocab), token_vocab.vectors.shape[1], # embedding dim
                                        padding_idx=token_vocab.stoi['<pad>']).from_pretrained(token_vocab.vectors)
        
        # BiLSTM
        self.RNN = nn.LSTM(token_vocab.vectors.shape[1], hidden_size, bias=True, batch_first=True, bidirectional=True)
        
        # MLP described in the paper, first linear layer is Ws1, then tanh, then last linear layer is Ws2
        self.MLP = nn.Sequential(
                                 nn.Linear(hidden_size*2, attention_dim), # input layer
                                 nn.Tanh(), # hidden layer
                                 nn.Linear(attention_dim, attention_hops), # ouput layer
                                 nn.Softmax(dim=1) # softmax for attention distrubution
                                )
        
        # dimension (2, hidden_size) 
        # where hidden size is the hidden dim of the rnn, and 2 is the number of directions, BiLSTM has 2 directions
        # autograd means requires_grad is set to true, so it will be trained
        self.init_state = (
                            torch.autograd.Variable(torch.zeros(2, hidden_size)), # initial hidden state
                            torch.autograd.Variable(torch.zeros(2, hidden_size))) # initial cell state
    
    def forward(self, sentence):
        tokens, lengths = sentence;
        
        # list(unbind()) turns tensors into lists. 
        lengths = list(torch.unbind(lengths))
        tokens = list(torch.unbind(tokens))
        
        # add batch to list object to keep track of original indexes
        batch = []
        for i, (t, l) in enumerate(zip(tokens, lengths)):
            # i batch index, t sentence tokens, l lenght, None will be sentence embedding u
            batch.append( [i, t, l, None] ) 
        
        # sort batch list by lenghts
        batch = sorted(batch, reverse=True, key=lambda x: x[2])
        
        # get sorted tokens and lengths
        tokens = torch.stack([x[1] for x in batch])
        lengths = torch.stack([x[2] for x in batch])
        
        # get embedding vectors
        embedded = self.embedding(tokens)
        
        # pack sentence
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)
        
        # for each sentence in batch use a initial hidden state init_hidden, and a initial cell state init_cell
        init_hidden = [self.init_state[0] for l in lengths]
        init_cell = [self.init_state[1] for l in lengths]
        
        # turn lists into tensors
        init_hidden = torch.stack(init_hidden).transpose(1,0)
        init_cell = torch.stack(init_cell).transpose(1,0)
        
        # get rnn hidden states out
        out, _ = self.RNN(packed, (init_hidden,init_cell))
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        
        # make attention distrubution with MLP, where output will be a softmaxed distrubution
        A = self.MLP(out).transpose(2,1)
        
        # get weighted attention sentence matrix u
        u = A @ out
        
        #  penalization term for redundancy
        p =  A @ A.transpose(2,1)
        p = p - torch.matrix_power(p, 0) # matrix power 0 returns identity matrix
        p = torch.norm(p, p='fro')
        
        p = p * self.penalty
        
        # add the sentence embedding u_ in u to the batch list
        for i, u_ in enumerate(u):
            batch[i][3] = u_
        
        # sort by original batch indexes i
        batch = sorted(batch, key=lambda x: x[0])
        u = torch.stack([x[3] for x in batch])
        u = u.transpose(1,0)
        
        # concatinate the attention vectors
        u = torch.cat([v for v in u], dim=1)
        return u, p
    
    def AttentionExmaple(self, Example):
        tokens, lengths = Example
        
        # get embedding vectors
        embedded = self.embedding(tokens)
        
        # pack sentence
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)
        
        # for each sentence in batch use a initial hidden state init_hidden, and a initial cell state init_cell
        init_hidden = [self.init_state[0] for l in lengths]
        init_cell = [self.init_state[1] for l in lengths]
        
        # turn lists into tensors
        init_hidden = torch.stack(init_hidden).transpose(1,0)
        init_cell = torch.stack(init_cell).transpose(1,0)
        
        # get rnn hidden states out
        out, _ = self.RNN(packed, (init_hidden,init_cell))
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        
        # make attention distrubution with MLP, where output will be a softmaxed distrubution
        A = self.MLP(out).transpose(2,1)
        
        return A
    
