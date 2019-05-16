#!/bin/env python3
# coding: utf-8

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

from argparse import ArgumentParser
import pandas as pd
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset
from helpers import *
from Models import *

#  This scripts load your pre-trained model and your vectorizer object (essentialy, vocabulary)
#  and evaluates it on a test set.

def evaluate(model, iterator):
    accuracy, total = 0, 0
    predicted = []
    golds = []
    
    if args.tqdm: pbar = tqdm(total=len(iterator.data()))
    if args.tqdm: pbar.set_description('Evaluate')
    
    for n, batch in enumerate(iterator):
        out = model(batch)
        if isinstance(out, tuple):
            out,  _ = out
        predictions = out.argmax(dim=-1)
        predicted.append(predictions)
        gold = batch.gold_label
        golds.append(gold)
        total += predictions.size(0)
        accuracy += (predictions == gold).nonzero().size(0)
        
        if args.tqdm: pbar.update( len(batch) )
        if args.tqdm: pbar.set_postfix(accuracy=accuracy/total)
        
    accuracy = accuracy / total
    return accuracy, predicted, golds

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--modelfile', help="PyTorch model saved with model.save()", action='store', default="./models/convNet_baseline1/convNet_baseline1_classifier")
    parser.add_argument('--testfile', help="Test dataset)", action='store', default="./nli5550/nli5550.test.jsonl")
    parser.add_argument('--embeds', action='store', default=None)
    parser.add_argument('--tqdm', action='store', default="false")
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    args.tqdm = correctBoolean(args.tqdm, 'tqdm')  
    
    logger.info('Loading the model...')
    model = torch.load(args.modelfile)  # Loading the model itself
    model.eval()
    print(model)
    
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

    logger.info('Loading the test set...')
    test_dataset = data.Dataset( load_jsonl_examples(args.testfile, fields) , fields=fields)
    logger.info('Finished loading the test set')
    
    logger.info('building vocab...')
    # use glove.6b.100d word vectors
    if args.embeds is not None:
        _vecs = vocab.Vectors(args.embeds)
    else:
        _vecs = "glove.6B.100d"
    
    # build vocabulary 
    token_field.build_vocab(test_dataset, vectors=_vecs)
    label_field.build_vocab(test_dataset)
    logger.info('Finished..')
    
    print( 'test dataset lenght:\t{}'.format(len(test_dataset)))

    
    classes = list(label_field.vocab.freqs)
    num_classes = len(classes)
    logger.info('%d classes' % num_classes)
    
    print('===========================')
    print('Class distribution in the test data:')
    print(label_field.vocab.freqs)

    
    dataset = data.Iterator(test_dataset, batch_size=1, train=False, sort=False)

    print('===========================')
    print('Evaluation:')
    print('===========================')

    test_accuracy, test_predictions, test_labels = evaluate(model, dataset)

    print("Accuracy on the test set:", round(test_accuracy, 3))
    print('Classification report for the test set:')
    
    gold_classes_human = [label_field.vocab.itos[x] for x in test_labels]
    predicted_test_human = [label_field.vocab.itos[x] for x in test_predictions]
    
    print(classification_report(gold_classes_human, predicted_test_human))