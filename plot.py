import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import json
import os
import pandas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', nargs='+', action='store')
    parser.add_argument('--save', action='store', type=str, default = None)
    args = parser.parse_args()
    
    fig = plt.figure(1, figsize=(5,5))
    ax = plt.subplot()
    ax.grid()
    plt.rcParams.update({'font.size': 11})
    fig.patch.set_facecolor('white')

    fig.suptitle('dev_accuracy')
    table = []
    for path in args.paths:
        print(path)
        try:
            with open('{}/epochs.JSON'.format(path)) as epochs:
                data = [dict(json.loads(line)) for line in epochs]
                accuracy = [ d['dev_accuracy'] for d in data]
                f1 = [ d['Report']['macro avg']['f1-score'] for d in data]
                best_iteration = np.argmax(accuracy)
                table_elem = {'model': path.split('/')[2], 
                              'dev_accuracy': accuracy[best_iteration],
                              'macro F1-score': f1[best_iteration]}
                table.append(table_elem)
                plot, = plt.plot( accuracy, label=path.split('/')[2])
        except: pass
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    ax.legend()
    
    #table.sort(key=lambda x: x['macro F1-score'], reverse=True)
    df = pandas.DataFrame(table, columns=['model', 'dev_accuracy', 'macro F1-score'])
    print(df.to_string(index=False))

    if args.save is not None:
        plt.savefig(args.save)
    plt.show()
    
if __name__ == '__main__':
    main()   
