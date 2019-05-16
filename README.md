# in5550-final

**abel directory** /cluster/home/skjale/in5550-final

## scripts

**runner.py** you can train a classifier and encoder


  args | description | default
  --- | --- | ---
  --model | path to model | none
  --testfile | path to test data | relative path to NLI5550.test.jsonl
  --embeds | path to a txt model or torchtext glove embeds | None and it will use glove.6b.100d from .vector_cache
*****************************************************
**eval_on_test.py** you can do a evaluation on your model, eval_on_test uses batches of 1 instead of one huge batch like we have done on previous batches as my pc ran out of memory doing that.

  args | description | default
  --- | --- | ---
  --model | path to model | none
  --testfile | path to test data | relative path to NLI5550.test.jsonl
  --embeds | path to a txt model or torchtext glove embeds | None and it will use glove.6b.100d from .vector_cache
  --tqdm | true/false, if true uses a tqdm to show progress bar in commandline | false
*****************************************************
**plotAttention.py** you can do a evaluation on your model

  args | description | default
  --- | --- | ---
  --data | path to jsonl file with examples | none
  --encoder | path to model | none
  --example | index for a spesific example in the data | None, a random example will be chosen
  --embeds | path to a txt model or torchtext glove embeds | None and it will use glove.6b.100d from .vector_cache
  --save | a name to save the plot as. | default None, no plot will be saved
  
## files and tests
### ...

## model paths
### path for best model
