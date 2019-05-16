# in5550-final

**abel directory** /cluster/home/skjale/in5550-final

## files, directories
 - [abel](https://github.uio.no/skjale/in5550-final/tree/master/abel) hold abel related slurm
 - [models](https://github.uio.no/skjale/in5550-final/tree/master/models) hold model tests, where each model has its own directory with epochs.json and args.json
 - [runs](https://github.uio.no/skjale/in5550-final/tree/master/runs) tensorboard log directory, these are unordered.
*********
in each model directory there is epochs.json and args.json
- args.json is a dump of the argument parser, holds a record of what spesifications the model had.
- epochs.json has a json object each line that holds information about each epoch

## model paths
todo insert some

## scripts

[**eval_on_test.py**](https://github.uio.no/skjale/in5550-final/blob/master/eval_on_test.py)

you can do a evaluation on your model
eval_on_test uses batches of 1 instead of one huge batch like we have done on previous assignments as my pc ran out of memory doing that.

  args | description | default
  --- | --- | ---
  --model | path to model | none
  --testfile | path to test data | relative path to NLI5550.test.jsonl
  --embeds | path to a txt model or torchtext glove embeds | None and it will use glove.6b.100d from .vector_cache
  --tqdm | true/false, if true uses a tqdm to show progress bar in commandline | false
  
  
  *****************************************************

[**plot.py** ](https://github.uio.no/skjale/in5550-final/blob/master/plot.py)

used to plot, will use matplotlib to plot a graph of dev_accuracy and print a table of model name, dev_accuracy, macro F1-score for given models

  args | description | default
  --- | --- | ---
  --paths | paths to models directory, can be multiple for multiple graphs togther | None
  --labels | labels for the models, must be given in same order as models, if left at None will simply use path name | None
  --save | a name to save the plot as. | default None, no plot will be saved
  --latex | true/false, if true will print latex table | false
*****************************************************

[**plotAttention.py** ](https://github.uio.no/skjale/in5550-final/blob/master/attentionPlot.py)

used to plot attention distrubution generated from an encoder

  args | description | default
  --- | --- | ---
  --data | path to jsonl file with examples | none
  --encoder | path to model | none
  --example | index for a spesific example in the data | None, a random example will be chosen
  --embeds | path to a txt model or torchtext glove embeds | None and it will use glove.6b.100d from .vector_cache
  --save | a name to save the plot as. | default None, no plot will be saved
  
*****************************************************
[**runner.py**](https://github.uio.no/skjale/in5550-final/blob/master/runner.py) and [**runner_abel**](https://github.uio.no/skjale/in5550-final/blob/master/runner_abel.py)

you can train a classifier and encoder, if name is specified will make a directory for the model and store relevant data each epoch in a file called epochs.json, very time dev_accuracy goes up it will save the encoder and whole model.

only difference between runner.py and runner_abel should be that runner_abel does not attempt to write with tensorboardX and tqdm is default to false.

  args | description | default
  --- | --- | ---
  --train | path to train data | relative path to NLI5550.train.jsonl
  --dev | path to dev data | relative path to NLI5550.dev.jsonl
  --embeds | path to a txt model or torchtext glove embeds | None and it will use glove.6b.100d from .vector_cache
  --name | name of the model, required to save data about it | None
  --dir | directory to save model | "models"
  --testing | true/false, if true train dataset will be set to be same as dev for fast testing | false
  --tqdm | true/false, if true displays a progressbar during training and evaluation | true
  --batch_size | training batch size | 32
  --lr | learning rate | 1e-3
  --epochs | how many epochs to train | 5
  --split | how much of the train dataset to use | 0.1
  --classifier_hidden_size | hidden dimension of the MLP classifier | 512
  --classifier_num_layers | number of hidden layers in the MLP classifier | 1
  --classifier_dropout | drop out to preform over the \[u, v |u-v|, u\*v] vector of the sentences | 0
  --encoder |what encoder to use 'sentenceEmbedding', 'convNet', 'BiLSTM', 'LastStateEncoder', 'gruSentenceEmbeddingEncoder' | 'sentenceEmbedding'
  --encoder_hidden_size | hidden size of RNN encoders hidden state,  or filter size in the case of convNet encoder | 1024
  --encoder_attention_dim | dimension of the hidden layers in the attention MLP that makes the self-attentive sentence embeddings (sentenceEmbedding)
  --encoder_attention_hops | how many layers of attention in sentenceEmbedding encoder | 4
  --encoder_penalty | penalty coefficient for sentenceEmbedding | 0
  --encoder_pooling | 'max' or 'mean' pooling for BiLSTM encoder | 'max'
  --encoder_RNN | what type of rnn to use for LastStateEncoder, supports(LSTM, BiLSTM, GRU, BiGRU, vanilla RNN and BiRNN) | LSTM
  --encoder_padding | how much padding in the conovolutional layers for ConvNet encoder | 1
  --encoder_kernel_size | kernel size for ConvNet encoder | 3
  
  

