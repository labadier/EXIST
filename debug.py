#%%
from matplotlib.pyplot import axis
from more_itertools import take
import pandas as pd, csv
from utils.params import params
from utils.utils import evaluate, MajorityVote, mergePredsByLanguage, ProbabilitiesAnalysis

mode = 'stl'
tasks = [1]
# Evaluate Individual Language

print('*'*5 + " Individual Languages " + '*'*5)

for task in tasks:
  for lang in ['en', 'es', 'de', 'fr', 'pt', 'it']:
    print(f"Lang {lang} on task {task}:")
    evaluate(input=f'logs/{mode}/task{task}_LPtower_1_p={lang}_{lang}.csv', task=task)

# Majority Vote

def evaluate_aggregation(mode, tasks, agregation='major'):
  print('*'*5 + " Backtranslation Prediction Augmentation " + '*'*5)

  for task in tasks:
    print(f"Augmented Backtranslation on task {task}:")
    evaluate(input=f'logs/{mode}/task{task}_LPtower_1_{agregation}.csv', task=task)


  print('*'*5 + " Major Voting by Language Models " + '*'*5)

  for task in tasks:
    print(f"Augmented Backtranslation on task {task}:")
    evaluate(input=f'logs/{mode}/task{task}_LPtower_1_ensemble_{agregation}.csv', task=task)


mergePredsByLanguage(mode, tasks, submit=1)
MajorityVote(mode, tasks, submit=1)


evaluate_aggregation(mode, tasks, agregation='major')

#SVM Probabilities Analysis

ProbabilitiesAnalysis(mode, tasks, submit=1)

for task in tasks:
  print(f"Prob Analysis on task {task}:")
  evaluate(input=f'logs/{mode}/task{task}_LPtower_1_svm.csv', task=task)


# %%
