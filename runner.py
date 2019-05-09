import random
import argparse
import torch
import torch.nn.functional as F
from torch import nn
from torchtext import data, vocab
from tqdm import tqdm
from sklearn.metrics import f1_score
from helpers import *
from Models import *

def train(model, iterator, criterion, optimiser, epoch, writer=None):
    
    model.train()
    pbar = tqdm(total=len(iterator.data()))
    accuracy, total = 0, 0
    total_loss = 0
    iterator.init_epoch()
    for n, batch in enumerate(iterator):
        optimiser.zero_grad()
        
        predictions = model(batch)
        gold = batch.gold_label
        loss = criterion(predictions, gold)
        
        predmax = predictions.argmax(dim=-1)
        total_loss += loss.item()
        accuracy += ( predmax== gold).nonzero().size(0)
        total += predmax.size(0)
        
        loss.backward()
        optimiser.step()
        pbar.update(32) # batch_size
        pbar.set_postfix(loss=loss.item())
        if writer is not None: writer.add_scalar('Train/Loss', loss, n)
        
    if writer is not None: writer.add_scalar('Train/total_loss', total_loss, epoch)
    if writer is not None: writer.add_scalar('Train/mean_loss', total_loss/total, epoch)
    if writer is not None: writer.add_scalar('Train/accuracy', accuracy/total, epoch)
    
    pbar.close()

def evaluate(model, iterator, criterion, epoch, writer=None):
    model.eval()
    accuracy, total = 0, 0
    total_loss = 0
    for n, batch in enumerate(iterator):
        prediction = model(batch)
        predictions = prediction.argmax(dim=-1)
        gold = batch.gold_label
        loss = criterion(prediction, gold)
        total_loss += loss
        accuracy += (predictions == gold).nonzero().size(0)
        total += predictions.size(0)
        if writer is not None: writer.add_scalar('Dev/Loss', loss, n)
    
    if writer is not None: writer.add_scalar('Dev/total_loss', total_loss, epoch)
    if writer is not None: writer.add_scalar('Dev/mean_loss', total_loss/total, epoch)
    if writer is not None: writer.add_scalar('Dev/accuracy', accuracy/total, epoch)
    print("> dev accuracy: {}/{} = {}".format(accuracy, total, accuracy/total))

def main():
    torch.manual_seed(1337)
    random.seed(13370)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store', default="./nli5550/nli5550.test.jsonl")
    parser.add_argument('--dev', action='store', default="./nli5550/nli5550.dev.jsonl")
    parser.add_argument('--embeds', action='store', default=None)
    parser.add_argument('--batch_size', action='store', default=32)
    parser.add_argument('--lr', action='store', default=1e-3)
    parser.add_argument('--epochs', action='store', default=10)
    parser.add_argument('--name', action='store', default="MODEL_NAME")
    args = parser.parse_args()
    
    try:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter('./runs/{}'.format(args.name))
    except ImportError:
        writer = None
    
    token_field = data.Field(sequential=True, batch_first=True, include_lengths=True, tokenize=lambda x: x.split(), preprocessing=lambda x: x[1].split())
    label_field = data.Field(sequential=False, batch_first=True, preprocessing=lambda x: x[1])
    NaF = ('none', None)
    
    fields = [
            NaF, #('annotator_labels', label_field), 
            NaF, # captionID
            ('gold_label', label_field), 
            NaF, # pairID
            NaF, #('sentence1', token_field),
            NaF, #('sentence1_parse', token_field),
            NaF, #('sentence2', token_field),
            NaF, #('sentence2_parse', token_field),
            ('sentence1_tok', token_field),
            ('sentence2_tok', token_field)
        ]

    train_dataset = data.Dataset( load_jsonl_examples(args.train, fields) , fields=fields)
    dev_dataset = data.Dataset( load_jsonl_examples(args.dev, fields) , fields=fields)
    
    if args.embeds is not None:
        _vecs = vocab.Vectors(args.embeds)
    else:
        _vecs = "glove.6B.100d"
    
    token_field.build_vocab(train_dataset, vectors=_vecs)
    label_field.build_vocab(train_dataset)
    
    train_iter = data.Iterator(train_dataset, batch_size=args.batch_size, train=True, sort=True, repeat=False, 
                                sort_key=lambda x: (len(x.sentence1_tok)+len(x.sentence1_tok))/2)
    dev_iter = data.Iterator(dev_dataset, batch_size=1, train=False, sort=False)

    
    #net = convNetEncoder(token_field.vocab, label_field.vocab)
    encoder = self_attention_Encoder2(token_field.vocab)
    net = MLP(token_field.vocab, label_field.vocab, encoder)
    
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(net.parameters(), lr=args.lr)
    

    for epoch in range(args.epochs):
        train(net, train_iter, criterion, optimiser, epoch, writer)
        evaluate(net, dev_iter, criterion, epoch, writer)
        
    
if __name__ == '__main__':
    main()   