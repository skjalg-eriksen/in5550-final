import random
import argparse
import torch
import torch.nn.functional as F
from torch import nn
from torchtext import data, vocab
from tqdm import tqdm
from helpers import *
from Models import *
import os
import json
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def train(model, iterator, criterion, optimiser, epoch, args, writer=None):
    model.train()
    
    if args.tqdm: pbar = tqdm(total=len(iterator.data()))
    
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
        if args.tqdm: pbar.update( len(batch) ) # batch_size
        if args.tqdm: pbar.set_postfix(loss=loss.item())
        if writer is not None: writer.add_scalar('Train/Loss', loss, n)
        
    if writer is not None: writer.add_scalar('Train/total_loss', total_loss, epoch)
    if writer is not None: writer.add_scalar('Train/mean_loss', total_loss/total, epoch)
    if writer is not None: writer.add_scalar('Train/accuracy', accuracy/total, epoch)
    
    if args.tqdm: pbar.close()
    return accuracy/total, total_loss, total_loss/total

def evaluate(model, iterator, epoch, args, writer=None):
    model.eval()
    
    if args.tqdm: pbar = tqdm(total=len(iterator.data()))
    if args.tqdm: pbar.set_description('Evaluate')
    
    accuracy, total = 0, 0
    predicted = []
    golds = []
    loss = 0
    
    for n, batch in enumerate(iterator):
        out = model(batch)
        predictions = out.argmax(dim=-1)
        predicted.append(predictions)
        gold = batch.gold_label
        golds.append(gold)
        accuracy += (predictions == gold).nonzero().size(0)
        total += predictions.size(0)
        loss += F.cross_entropy(out, gold).item()
        
        if args.tqdm: pbar.update( len(batch) ) # batch_size
        if args.tqdm: pbar.set_postfix(accuracy=accuracy/total)
        
        
    if writer is not None: writer.add_scalar('Dev/accuracy', accuracy/total, epoch)
    if args.tqdm: pbar.close()
    
    print("> dev accuracy: {}/{} = {}".format(accuracy, total, accuracy/total))

    return accuracy / total, predicted, golds, loss, loss/total
    
def main():
    torch.manual_seed(1337)
    random.seed(13370)

    parser = argparse.ArgumentParser()
    # paths and files
    parser.add_argument('--train', action='store', default="./nli5550/nli5550.train.jsonl")
    parser.add_argument('--dev', action='store', default="./nli5550/nli5550.dev.jsonl")
    parser.add_argument('--embeds', action='store', default=None)
    parser.add_argument('--name', action='store', default=None)
    parser.add_argument('--dir', action='store', default="models")
    parser.add_argument('--testing', action='store', default="false")
    parser.add_argument('--tqdm', action='store', default="true")
    
    # optimizer and data hyperparamters
    parser.add_argument('--batch_size', action='store',type=int, default=32)
    parser.add_argument('--lr', action='store',type=float, default=1e-3)
    parser.add_argument('--epochs', action='store', type=int, default=3)
    parser.add_argument('--split', action='store', default=0.1, help="how much of train dataset to use")
    
    # MLP classifer hyperparameters
    parser.add_argument('--classifier_hidden_size',type=int, default=512)
    parser.add_argument('--classifier_num_layers',type=int, default=1)
    parser.add_argument('--classifier_dropout', type=float, default=0.0)
    
    # encoder hyperparameters
    parser.add_argument('--encoder', action='store', default="sentenceEmbedding", choices=['sentenceEmbedding', 'convNet', 'BiLSTM', 'LastStateEncoder'])
    parser.add_argument('--encoder_hidden_size',type=int, default=1024)  # sentenceEmbedding, convNet, BiLSTM, LastStateEncoder
    parser.add_argument('--encoder_attention_dim',type=int, default=512) # sentenceEmbedding
    parser.add_argument('--encoder_attention_hops',type=int, default=4)  # sentenceEmbedding
    parser.add_argument('--encoder_penalty', default='False')            # sentenceEmbedding
    parser.add_argument('--encoder_pooling', default='max',              # BiLSTM
                                            choices=['max', 'mean'])
    parser.add_argument('--encoder_RNN', default='LSTM')                 # LastStateEncoder
    parser.add_argument('--encoder_padding', type=int, default=1)        # convNet
    parser.add_argument('--encoder_kernel_size', type=int, default=3)    # convNet
    
    args = parser.parse_args()
    
    # get True boolean values for input arguments using them. (args gets strings, strings that is not None == true)
    args.encoder_penalty = correctBoolean(args.encoder_penalty, 'encoder_penalty')
    args.testing = correctBoolean(args.testing, 'testing')
    args.tqdm = correctBoolean(args.tqdm, 'tqdm')
    
    # if summary writer is available for logging learning curve
    try:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter('./runs/{}'.format(args.name))
    except ImportError:
        writer = None

    # save meta data about model
    if args.name is not None:
        # make dir.
        directory = "{}/{}".format(args.dir, args.name)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # save args
        with open('{}/args.json'.format(directory), 'w') as outfile:
            json.dump(args.__dict__, outfile)
    
    # field types
    token_field = data.Field(sequential=True, batch_first=True, include_lengths=True, tokenize=lambda x: x.split(), preprocessing=lambda x: x[1].split())
    label_field = data.Field(sequential=False, batch_first=True, preprocessing=lambda x: x[1])
    NaF = ('none', None)
    
    # data fields in dataset
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

    # load jsonl file with load_jsonl_examples to get a python list with torchtext.Example objects
    train_dataset = data.Dataset( load_jsonl_examples(args.train, fields) , fields=fields).split(args.split)[0]
    dev_dataset = data.Dataset( load_jsonl_examples(args.dev, fields) , fields=fields)
    
    if args.testing:
        dev_dataset = train_dataset
    
    # use glove.6b.100d word vectors
    if args.embeds is not None:
        _vecs = vocab.Vectors(args.embeds)
    else:
        _vecs = "glove.6B.100d"
    
    # build vocabulary 
    token_field.build_vocab(train_dataset, vectors=_vecs)
    label_field.build_vocab(train_dataset)
    
    # iterators 
    train_iter = data.Iterator(train_dataset, batch_size=args.batch_size, train=True, sort=True, repeat=False, 
                                sort_key=lambda x: (len(x.sentence1_tok)+len(x.sentence1_tok))/2)
    dev_iter = data.Iterator(dev_dataset, batch_size=1, train=False, sort=False)
    
    # encoder model, setenceEmbedding, convNet, BiLSTM, LastStateEncoder
    if args.encoder.lower() in 'sentenceembedding':
        encoder = sentenceEmbeddingEncoder(token_field.vocab, args.encoder_hidden_size, args.encoder_attention_dim, args.encoder_attention_hops)
    elif args.encoder.lower() in 'convnet':
        encoder = convNetEncoder(token_field.vocab, args.encoder_hidden_size, args.encoder_kernel_size, args.encoder_padding)
    elif args.encoder.lower() in 'bilstm':
        encoder = BiLSTMEncoder(token_field.vocab, args.encoder_hidden_size, args.encoder_pooling)
    elif args.encoder.lower() in 'laststateencoder':
        encoder = LastStateEncoder(token_field.vocab, args.encoder_hidden_size, args.encoder_RNN)
    else:
        raise Exception('not a valid encoder: {}'.format(args.encoder))
        
    # classifier model MLP_classifier
    net = MLP_classifier(token_field.vocab, label_field.vocab, encoder, args.classifier_hidden_size, args.classifier_num_layers, args.classifier_dropout)
    
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(net.parameters(), lr=args.lr)
    
    dev_acc = 0
    for epoch in range(args.epochs):
        train_accuracy, train_loss, train_mean_loss = train(net, train_iter, criterion, optimiser, epoch, args, writer)
        accuracy, predicted, gold_label, dev_loss, dev_mean_loss = evaluate(net, dev_iter, epoch, args, writer)
        
        gold_classes_human = [label_field.vocab.itos[x] for x in gold_label]
        predicted_dev_human = [label_field.vocab.itos[x] for x in predicted]
        
        if dev_acc <= accuracy:
            if args.name is not None: 
                torch.save(net, '{}/{}_classifier'.format(directory, args.name))
                torch.save(encoder, '{}/{}_encoder'.format(directory, args.name))
                save_data = {
                    'epoch': epoch, 
                    'train_accuracy': train_accuracy,
                    'train_loss': train_loss,
                    'train_mean': train_mean_loss,
                    'dev_accuracy': round(accuracy_score(gold_label, predicted), 3),
                    'dev_loss': dev_loss,
                    'dev_mean': dev_mean_loss,
                    'Report': classification_report(gold_classes_human, predicted_dev_human, output_dict=True)
                    }
                print(save_data)
                with open('{}/epochs.json'.format(directory), 'a') as outfile:
                    json.dump(save_data, outfile)
                    outfile.write('\n')
    
            dev_acc = accuracy;
            
if __name__ == '__main__':
    main()   
