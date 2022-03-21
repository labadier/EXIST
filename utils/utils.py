import pandas as pd, numpy as np, os
from utils.params import params
from matplotlib import pyplot as plt

def load_data(filename):

  data = pd.read_csv(filename, dtype=str)
  text = data['text'].to_numpy()
    
  if len(data.keys()) < 4:
    return text
  labels = data[params.columns[3:]].astype(int).to_numpy()
  return text, labels

def evalData(filename, lang):

  data = pd.read_csv(filename, dtype=str, sep='\t')
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