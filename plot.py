import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import json
import os
import pandas
from helpers import correctBoolean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', nargs='+', action='store')
    parser.add_argument('--labels', nargs='+', action='store', default = None)
    parser.add_argument('--save', action='store', type=str, default = None)
    parser.add_argument('--latex', action='store', default="false")
    args = parser.parse_args()
    
    args.latex = correctBoolean(args.latex, 'latex')
    
    
    fig = plt.figure(1, figsize=(5,5))
    ax = plt.subplot()
    ax.grid()
    plt.rcParams.update({'font.size': 12})
    fig.patch.set_facecolor('white')

    fig.suptitle('dev_accuracy')
    table = []
    for n, path in enumerate(args.paths):
        with open('{}/epochs.JSON'.format(path)) as epochs:
            # load json data
            data = [dict(json.loads(line)) for line in epochs]
            
            # get accuracy and f1-score
            accuracy = [ d['dev_accuracy'] for d in data]
            f1 = [ d['Report']['macro avg']['f1-score'] for d in data]
            
            
            # plot results
            if args.labels is not None and n < len(args.labels):
                plot, = plt.plot( accuracy, label=args.labels[n])
                # get best epoch
                best_iteration = np.argmax(accuracy)
                table_elem = {'model': args.labels[n], 
                              'dev_accuracy': accuracy[best_iteration],
                              'macro F1-score': f1[best_iteration]}
                table.append(table_elem)
            else:
                plot, = plt.plot( accuracy, label=path)
                # get best epoch
                best_iteration = np.argmax(accuracy)
                table_elem = {'model': path, 
                              'dev_accuracy': accuracy[best_iteration],
                              'macro F1-score': f1[best_iteration]}
                table.append(table_elem)

            
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    ax.legend()
    
    #table.sort(key=lambda x: x['macro F1-score'], reverse=True)
    df = pandas.DataFrame(table, columns=['model', 'dev_accuracy', 'macro F1-score'])
    
    if (args.latex):
        print(df.to_latex())
    else:
        print(df.to_string(index=False))
        
    if args.save is not None:
        plt.savefig(args.save)
    plt.show()
    
if __name__ == '__main__':
    main()   
