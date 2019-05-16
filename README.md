# in5550-final

**abel directory** /cluster/home/skjale/in5550-final

## scripts

**eval_on_test.py** you can do a evaluation on your model

eval_on_test uses batches of 1 instead of one huge batch like we have done on previous assignments as my pc ran out of memory doing that.

  args | description | default
  --- | --- | ---
  --model | path to model | none
  --testfile | path to test data | relative path to NLI5550.test.jsonl
  --embeds | path to a txt model or torchtext glove embeds | None and it will use glove.6b.100d from .vector_cache
  --tqdm | true/false, if true uses a tqdm to show progress bar in commandline | false
*****************************************************
**runner.py** and **runner_abel** you can train a classifier and encoder

only difference between runner.py and runner_abel should be that runner_abel does not attempt to write with tensorboardX and tqdm is default to false.
  args | description | default
  --- | --- | ---
  --model | path to model | none
  --testfile | path to test data | relative path to NLI5550.test.jsonl
  --embeds | path to a txt model or torchtext glove embeds | None and it will use glove.6b.100d from .vector_cache
*****************************************************

**plotAttention.py** you can do a evaluation on your model

  args | description | default
  --- | --- | ---
  --data | path to jsonl file with examples | none
  --encoder | path to model | none
  --example | index for a spesific example in the data | None, a random example will be chosen
  --embeds | path to a txt model or torchtext glove embeds | None and it will use glove.6b.100d from .vector_cache
  --save | a name to save the plot as. | default None, no plot will be saved
*****************************************************

**plot.py** used to plot models dev_accuracy
used to plot models, will use matplotlib to plot a graph of dev_accuracy, and print a table of model name, dev_accuracy, macro F1-score'

  args | description | default
  --- | --- | ---
  --paths | paths to models directory, can be multiple for multiple graphs togther | none
  --save | a name to save the plot as. | default None, no plot will be saved
## files and tests
for each model i store a args.json with arguments, and epochs.json with data on each epoch.

## model paths
### path for best model
