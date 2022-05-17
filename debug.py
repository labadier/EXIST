#%%
import pandas as pd, csv
from utils.params import params
from utils.utils import evaluate, MajorityVote, mergePredsByLanguage, ProbabilitiesAnalysis


mode = 'mtl'
tasks = [1, 2]
lang = ['de', 'fr', 'pt', 'it']
plang_major = ['en', 'es', 'fr', 'pt', 'it']
plang_prob = ['en', 'es', 'it']


# mergePredsByLanguage('test', tasks, submit=1)
MajorityVote('test', tasks, plang_major, submit=1)
# ProbabilitiesAnalysis('mtl', tasks, plang_prob, submit=1)

for task in tasks:
    print(f"{task}:")
    evaluate(input=f'logs/{mode}/task{task}_LPtower_1_ensemble_major.csv', task=task, lang = ['es'])


#%%
# # Evaluate Individual Language

print('*'*5 + " Individual Languages " + '*'*5)

for task in tasks:
  for lang in ['en', 'es', 'de', 'fr', 'pt', 'it']:
    print(f"Lang {lang} on task {task}:")
    evaluate(input=f'logs/{mode}/task{task}_LPtower_1_p={lang}_{lang}.csv', task=task)

#%% Majority Vote
def evaluate_aggregation(mode, tasks, languages, agregation='major'):

  print('*'*5 + f" Major Voting by Language {languages} Models " + '*'*5)

  for task in tasks:
    print(f"{task}:")
    evaluate(input=f'logs/{mode}/task{task}_LPtower_1_ensemble_{agregation}.csv', task=task)


mergePredsByLanguage(mode, tasks, submit=1)
print('*'*5 + " Backtranslation Prediction Augmentation " + '*'*5)

# for task in tasks:
#   print(f"{task}:")
#   evaluate(input=f'logs/{mode}/task{task}_LPtower_1_major.csv', task=task)


for i in range(1 << 4):

  lp = ['en', 'es'] + [lang[j] for j in range(4) if i&(1 << j)]
  MajorityVote(mode, tasks, lp, submit=1)
  evaluate_aggregation(mode, tasks, lp, agregation='major')

#%% SVM Probabilities Analysis
for i in range(1 << 4):

  lp = ['en', 'es'] + [lang[j] for j in range(4) if i&(1 << j)]
  ProbabilitiesAnalysis(mode, tasks, lp, submit=1)
  print(lp)
  for task in tasks:
    print(f"Prob Analysis on task {task}:")
    evaluate(input=f'logs/{mode}/task{task}_LPtower_1_svm.csv', task=task)


# %%
for task in tasks:
  print(f"{task}:")
  evaluate(input=f'logs/{mode}/task{task}_LPtower_1_major.csv', task=task)
# %%
