import pandas as pd, numpy as np, os, csv
from sklearn.metrics import f1_score, accuracy_score
from utils.params import params
from matplotlib import pyplot as plt

def load_data(filename):

  data = pd.read_csv(filename, dtype=str)
  text = data['text'].to_numpy()
    
  if len(data.keys()) < 4:
    return text
  labels = data[params.columns[3:]].astype(int).to_numpy()
  return text, labels

def evalData(filename, lang, pivotlang):

  data = pd.read_csv(filename, dtype=str)
  if pivotlang == 'all':
    data = data[data['language'] == lang]

  testcase = data['test_case'].to_numpy()
  ids = data['id'].to_numpy()
  text = data['text'].to_numpy()
  if 'task1' in data.keys():
    return testcase, ids, text, {'task1':data['task1'].to_numpy(), 'task2':data['task2'].to_numpy()}
  else: return testcase, ids, text
    

def plot_training(history, model, output, measure='loss'):
    
    plotdev = 'dev_' + measure

    plt.plot(history[measure])
    plt.plot(history['dev_' + measure])
    plt.legend(['train', 'dev'], loc='upper left')
    plt.ylabel(measure)
    plt.xlabel('Epoch')
    if measure == 'loss':
        x = np.argmin(history['dev_loss'])
    else: x = np.argmax(history['dev_acc'])

    plt.plot(x,history['dev_' + measure][x], marker="o", color="red")

    if os.path.exists('./logs') == False:
        os.system('mkdir logs')

    plt.savefig(os.path.join(output, f'train_history_{model}.png'))

def mergePredsByLanguage(submit = 1) -> None:

  ''' 
    
    Merge the predition of english and spanish models for back transalted trategie of prediction
  
  '''
  for task in range(1, 3, 1):
    en = pd.read_csv(f'logs/task{task}_LPtower_{submit}_p=all_en.csv', sep='\t',  dtype=str, header=None)
    es = pd.read_csv(f'logs/task{task}_LPtower_{submit}_p=all_es.csv', sep='\t',  dtype=str, header=None)
    pd.concat([en, es]).to_csv(f'logs/task{task}_LPtower_{submit}.csv', sep='\t', index=False, header=False)


def MajorityVote(submit = 1) -> None:


  ''' 
    
   Make Majority voting with ensemble of languages
  
  '''

  for task in range(1, 3, 1):
    df = []
    for i in ['en', 'es', 'de', 'fr', 'pt']:
      df += [pd.read_csv(f'logs/task{task}_LPtower_{submit}_p={i}_{i}.csv', sep='\t',  dtype=str, header=None)]
      df[-1] = df[-1].sort_values(by=[1])

    with open(f'logs/task{task}_LPtower_{submit}_ensemble_major.csv', 'wt', newline='', encoding="utf-8") as csvfile:
      spamwriter = csv.writer(csvfile, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)

      for i in range(len(df[0])):
        ans = [j.iloc[i][2] for j in df]
        if len(set([j.iloc[i][1] for j in df])) > 1:
          print('Wrong arrangemet')
        spamwriter.writerow([df[0].iloc[i][0], df[0].iloc[i][1], max(set(ans), key=ans.count)])

    df_agumentedP = pd.read_csv(f'logs/task{task}_LPtower_{submit}.csv', sep='\t',  dtype=str, header=None)

    logs = {}
    testcase = {}

    for i in range(len(df_agumentedP[0])):
      if df_agumentedP[1][i] not in logs.keys():
        logs[df_agumentedP[1][i]] = [df_agumentedP[2][i]]
        testcase[df_agumentedP[1][i]] = [df_agumentedP[0][i]]
      else: 
        logs[df_agumentedP[1][i]] += [df_agumentedP[2][i]]
        testcase[df_agumentedP[1][i]] += [df_agumentedP[0][i]]
      
    with open(f'logs/task{task}_LPtower_{submit}_major.csv', 'wt', newline='', encoding="utf-8") as csvfile:
      spamwriter = csv.writer(csvfile, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)

      for i in logs.keys():

        outlog = max(set(logs[i]), key=logs[i].count)
        if len(set(testcase[i])) > 1:
          print('Wrong arrangemet')
        outtestcase = testcase[i][0]
        spamwriter.writerow([outtestcase, i, outlog])

def ProbabilitiesAnalysis(trainingFile =  'data/augmented.csv', submit = 1) -> None:


  ''' 
    
   Make ML voting with ensemble of languages Probabilities
  
  '''

  trainingData = pd.read_csv(trainingFile,dtype=str, header=None)
  trainingData = trainingData[trainingData['campain'] != 'HAHACKATHON']

  fetures = []

  for task in range(1, 3, 1):
    df = []
    for i in ['en', 'es', 'de', 'fr', 'pt']:
      df += [pd.read_csv(f'logs/task{task}_LPtower_{submit}_p={i}_{i}.csv', sep='\t',  dtype=str, header=None)]
      df[-1] = df[-1].sort_values(by=[1])

    with open(f'logs/task{task}_LPtower_{submit}_ensemble_major.csv', 'wt', newline='', encoding="utf-8") as csvfile:
      spamwriter = csv.writer(csvfile, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)

      for i in range(len(df[0])):
        ans = [j.iloc[i][2] for j in df]
        if len(set([j.iloc[i][1] for j in df])) > 1:
          print('Wrong arrangemet')
        spamwriter.writerow([df[0].iloc[i][0], df[0].iloc[i][1], max(set(ans), key=ans.count)])

    df_agumentedP = pd.read_csv(f'logs/task{task}_LPtower_{submit}.csv', sep='\t',  dtype=str, header=None)

    logs = {}
    testcase = {}

    for i in range(len(df_agumentedP[0])):
      if df_agumentedP[1][i] not in logs.keys():
        logs[df_agumentedP[1][i]] = [df_agumentedP[2][i]]
        testcase[df_agumentedP[1][i]] = [df_agumentedP[0][i]]
      else: 
        logs[df_agumentedP[1][i]] += [df_agumentedP[2][i]]
        testcase[df_agumentedP[1][i]] += [df_agumentedP[0][i]]
      
    with open(f'logs/task{task}_LPtower_{submit}_major.csv', 'wt', newline='', encoding="utf-8") as csvfile:
      spamwriter = csv.writer(csvfile, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)

      for i in logs.keys():

        outlog = max(set(logs[i]), key=logs[i].count)
        if len(set(testcase[i])) > 1:
          print('Wrong arrangemet')
        outtestcase = testcase[i][0]
        spamwriter.writerow([outtestcase, i, outlog])

def evaluate(input, task):


  labels = ['non-sexist', 'sexist'] if task == 1 else ['non-sexist'] + params.columns_exist
  file = pd.read_csv(input, sep='\t',  dtype=str, header=None).sort_values(by=[1])
  gold = pd.read_csv(f'data/EXIST/training/EXIST2021_test.tsv', sep='\t',  dtype=str, usecols=['id', f'task{task}']).sort_values(by=['id'])

  y = []
  y_hat = []
  for i in range(len(gold)):
    if file.iloc[i][1] != gold.iloc[i]['id']:
      print('Wrong arrangemet')
    y_hat += [labels.index(file.iloc[i][2])]
    y += [labels.index(gold.iloc[i][f'task{task}'])]

  
  print(f"acc: {accuracy_score(y, y_hat)}\nf1: {f1_score(y, y_hat, average='macro')}")
