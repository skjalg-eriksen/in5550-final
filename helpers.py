#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import re
from gensim import utils, models
import logging
import zipfile
import json
from torchtext.data import Example
import torch

# function for loading jsonl files provided for track 2 dataset where there are json objects on every line
def load_jsonl_examples(path, fields):
    """
    :param path: path to jsonl file with json examples seperated by line-break
    :param fields: list with tuples for each field
    :return: list containing torchtext.data.example objects
    """
    examples = []
    # open file f from path
    with open(path) as f:
        # extract json dict from every line
        for line in f:
            examples.append(Example.fromlist(json.loads(line).items(), fields))
    # return list
    return examples

# since any argument passed to parser in command line to a bool type will be interpeted as true this function fixes that.
def correctBoolean (value, flag):
    """
    :param value: string with True/false written in it
    :param flag: the name of the variable you are trying to assing true/false
    :return: true or false boolean depending on the string value
    """
    if value.lower() in ('true'):
        return True
    elif value.lower() in ('false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Error: {} for {}'.format(value, flag))

def word2vocabindex(texts, embeddings, sequence_length=18):
    """
    :param texts: text string (words separated by space)
    :param embeddings: word embedding model in Gensim format
    :param sequence_length: length of the sequences you want to pad the sentences to, default is 18
    :return: indexes corresponding to words in the embeddings model
    """
    input_features = np.zeros( (len(texts), sequence_length) ).astype(np.float32)
    for n, sentence in enumerate(texts):
        words = sentence.split(' ')

        offset=0
        for i, word in enumerate(words):
            if i > sequence_length-1:
                  break
            if word in embeddings.vocab:
                  input_features[n][i - offset] = embeddings.vocab[word].index
            else:
                  offset += 1
    return input_features

def fingerprint(text, model=None):
    """
    :param text: text string (words separated by space)
    :param model: word embedding model in Gensim format
    :return: average vector of words in text
    """
    if not model:
        print('You need to provide a Gensim model to produce semantic fingerprints!')
        return None
    
    # make a list of all the words in string
    text = text.split()
    # remove all the words not present in the model
    text = list(set(text).intersection(model.vocab))
    # if there are no words
    if len(text) == 0:
      # return empty vec
      print("no words in model returning empty vec")
      return (np.zeros_like(model[ model.index2word[0] ]))
    
    # return the average of all the word vectors
    return (np.sum( model[text], 0 ) * 1/len(text)).astype(np.float32)

def tag_convert(token):
    """
    :param token: word with its PTB PoS after underscore ('say_VB')
    :return: the same word with PoS converted to Universal PoS tags ('say_VERB')
    """
    try:
        (word, pos) = token.split('_')
    except ValueError:
        word = 'UNKN'
        pos = 'X'
    if pos in ptb2upos:
        newpos = ptb2upos[pos]
        newtoken = word + '_' + newpos
    else:
        return word + '_' + pos
    return newtoken


ptb2upos = {"!": "PUNCT",
            "#": "PUNCT",
            "$": "PUNCT",
            "''": "PUNCT",
            "(": "PUNCT",
            ")": "PUNCT",
            ",": "PUNCT",
            "-LRB-": "PUNCT",
            "-RRB-": "PUNCT",
            ".": "PUNCT",
            ":": "PUNCT",
            "?": "PUNCT",
            "CC": "CCONJ",
            "CD": "NUM",
            "CD|RB": "X",
            "DT": "DET",
            "DT.": "DET",
            "EX": "DET",
            "FW": "X",
            "IN": "ADP",
            "IN|RP": "ADP",
            "JJ": "ADJ",
            "JJR": "ADJ",
            "JJRJR": "ADJ",
            "JJS": "ADJ",
            "JJ|RB": "ADJ",
            "JJ|VBG": "ADJ",
            "LS": "X",
            "MD": "AUX",
            "NN": "NOUN",
            "NNP": "PROPN",
            "NNPS": "PROPN",
            "NNS": "NOUN",
            "NN|NNS": "NOUN",
            "NN|SYM": "NOUN",
            "NN|VBG": "NOUN",
            "NP": "NOUN",
            "PDT": "DET",
            "POS": "PART",
            "PRP": "PRON",
            "PRP$": "PRON",
            "PRP|VBP": "PRON",
            "PRT": "PART",
            "RB": "ADV",
            "RBR": "ADV",
            "RBS": "ADV",
            "RB|RP": "ADV",
            "RB|VBG": "ADV",
            "RN": "X",
            "RP": "PART",
            "SYM": "SYM",
            "TO": "PART",
            "UH": "INTJ",
            "VB": "VERB",
            "VBD": "VERB",
            "VBD|VBN": "VERB",
            "VBG": "VERB",
            "VBG|NN": "VERB",
            "VBN": "VERB",
            "VBP": "VERB",
            "VBP|TO": "VERB",
            "VBZ": "VERB",
            "VP": "VERB",
            "V": "VERB",
            "WDT": "DET",
            "WH": "X",
            "WP": "PRON",
            "WP$": "PRON",
            "WRB": "ADV",
            "``": "PUNCT"}

    
def eval_func(batched_data, model2use):
    # This function uses a model to compute predictions on data coming in batches.
    # Then it calculates the accuracy of predictions with respect to the gold labels.
    correct = 0
    total = 0
    predicted = None
    gold_label = None

    # Iterating over all batches (can be 1 batch as well):
    for n, (input_data, gold_label) in enumerate(batched_data):
        out = model2use(input_data)
        predicted = out.argmax(1)
        
        correct += len((predicted == gold_label.type(torch.long)).nonzero())
        total += len(gold_label)
    accuracy = correct / total
    return accuracy, predicted, gold_label
    
def load_embeddings(embeddings_file, limit=None):
    # Detect the model format by its extension:
    # Binary word2vec format:
    if embeddings_file.endswith('.bin.gz') or embeddings_file.endswith('.bin'):
        emb_model = models.KeyedVectors.load_word2vec_format(embeddings_file, binary=True,
                                                             unicode_errors='replace',
                                                             limit=limit)
    # Text word2vec format:
    elif embeddings_file.endswith('.txt.gz') or embeddings_file.endswith('.txt') \
            or embeddings_file.endswith('.vec.gz') or embeddings_file.endswith('.vec'):
        emb_model = models.KeyedVectors.load_word2vec_format(
            embeddings_file, binary=False, unicode_errors='replace', limit=limit)
    # ZIP archive from the NLPL vector repository:
    elif embeddings_file.endswith('.zip'):
        with zipfile.ZipFile(embeddings_file, "r") as archive:
            # Loading and showing the metadata of the model:
            # metafile = archive.open('meta.json')
            # metadata = json.loads(metafile.read())
            # for key in metadata:
                # print(key, metadata[key])
            print('============')
            # Loading the model itself:
            stream = archive.open("model.bin")  # or model.txt, if you want to look at the model
            emb_model = models.KeyedVectors.load_word2vec_format(
                stream, binary=True, unicode_errors='replace', limit=limit)
    else:
        # Native Gensim format?
        emb_model = models.KeyedVectors.load(embeddings_file, limit=limit)
        # If you intend to train further: emb_model = models.Word2Vec.load(embeddings_file)

    emb_model.init_sims(replace=True)  # Unit-normalizing the vectors (if they aren't already)
    return emb_model