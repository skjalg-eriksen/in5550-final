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
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def main():
    torch.manual_seed(1337)
    random.seed(13370)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', action='store', default="./nli5550/nli5550.test.jsonl")
    parser.add_argument('--encoder', action='store', default="./encoder.pt")
    parser.add_argument('--example', action='store', default = None)
    parser.add_argument('--embeds', action='store', default = None)
    parser.add_argument('--save', action='store',type=str, default = None)
    args = parser.parse_args()

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

    dataset = data.Dataset( load_jsonl_examples(args.data, fields) , fields=fields)
    
    if args.embeds is not None:
        _vecs = vocab.Vectors(args.embeds)
    else:
        _vecs = "glove.6B.100d"
    
    token_field.build_vocab(dataset, vectors=_vecs)
    label_field.build_vocab(dataset)
    
    data_iter = data.Iterator(dataset, batch_size=1, train=False, sort=False)
    
    example_list = [ x for x in data_iter ]
    data_list = [x for x in data_iter.data()]
    
    encoder = torch.load(args.encoder)
    
    if args.example is None:
        args.example = np.random.randint(0, len(data_list))
        
    example = example_list[args.example]
    example_data = data_list[args.example]
    
    max_length = np.argmax([len(example_data.sentence1_tok), len(example_data.sentence1_tok)])
    print(max_length)
    fig = plt.figure(1, figsize=(15,10))
    fig.suptitle("entailment: {}".format(example_data.gold_label), x=0.5, y=0.98, fontsize=25)
    plt.rcParams.update({'font.size': 16})
    fig.patch.set_facecolor('white')
    
    sentence1 = plt.subplot(211)
    sentence1.yaxis.set_major_formatter(mtick.PercentFormatter())
    A = encoder.visualizeExmaple(example.sentence1_tok)
    A = torch.squeeze(A).mean(0)
    plt.bar(example_data.sentence1_tok, (A).detach().numpy()*100)
    
    sentence2 = plt.subplot(212)
    A = encoder.visualizeExmaple(example.sentence2_tok)
    A = torch.squeeze(A).mean(0)
    plt.bar(example_data.sentence2_tok, (A).detach().numpy()*100)
    sentence2.yaxis.set_major_formatter(mtick.PercentFormatter())
    
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0, hspace=0.1)
    
    if args.save is not None:
        plt.savefig(args.save)
    plt.show()
    
if __name__ == '__main__':
    main()   
