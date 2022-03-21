#%%
from googletrans import Translator 
import pandas as pd
import os, csv, time, random, argparse, sys
from params import params

def TranslatePivotLang(sourceFile = 'data/augmented.csv', outputFile = 'training'):
  for target_lang in ['en', 'es', 'fr', 'de']:

    print(f'Pivot Language {target_lang}: 0%', end="")

    perc = 0
    data_frame = pd.read_csv(os.path.join(params.root, sourceFile),dtype=str)
    data_frame['text'] = data_frame['text'].replace('\n', ' ')

    with open(os.path.join(params.root, f'data/{outputFile}_{target_lang}.csv'), 'at', newline='', encoding="utf-8") as csvfile:
      spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
      spamwriter.writerow(list(data_frame.columns))
      
      for i in range(0, len(data_frame), 20):
        
        if (i*100.0)/len(data_frame) - perc > 1:
          perc = (i*100.0)/len(data_frame)
          print(f'\rPivot Language {target_lang}: {perc:.2f}%', end = "")

        data = data_frame[i:i + 20].copy() 

        if len(set(data['language'].to_list())) == 1 and data.iloc[0]['language'] != target_lang:
          ts = Translator()
          time.sleep(random.random()*3)
          try:
            data['text'] = (ts.translate(text='\n'.join(data['text'].to_list()), src=data.iloc[0]['language'], dest=target_lang).text).split('\n')
          except:
            print(f'An exception occurred on index {i}')
        elif len(set(data['language'].to_list())) > 1:
          ts = Translator()
          for j in range(20):
            if data.iloc[j]['language'] != target_lang:
              data.iloc[j]['text'] = ts.translate(text=data.iloc[j]['text'], dest=target_lang, src = data.iloc[j]['language']).text
              time.sleep(random.random()*3)

        for j in data.iterrows():
          spamwriter.writerow(j[1].to_list())
        
    print(f'\rPivot Language {target_lang}: 100%')

def backTranslation(sourceFile = 'training'):
  
  for back_target in ['en', 'es']:
    data_frame = pd.read_csv(os.path.join(params.root, f'data/{sourceFile}_en.csv'), dtype=str)
    with open(os.path.join(params.root, f'data/{sourceFile}_backTo_{back_target}.csv'), 'wt', newline='', encoding="utf-8") as csvfile:
      spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
      spamwriter.writerow(list(data_frame.columns))

      for pivot in ['en', 'es', 'fr', 'de']:
        data_frame = pd.read_csv(os.path.join(params.root, f'data/{sourceFile}_{pivot}.csv'), dtype=str)
        data_frame['text'] = data_frame['text'].replace('\n', ' ')
        print(f'{pivot} -> {back_target}: 0%', end = "")
        perc = 0

        for i in range(0, len(data_frame), 15):
            
          if (i*100.0)/len(data_frame) - perc > 1:
            perc = (i*100.0)/len(data_frame)
            print(f'\r{pivot} -> {back_target}: {perc:.2f}%', end = "")

          data = data_frame[i:i + 15].copy() 

          if pivot != back_target:
            ts = Translator()
            time.sleep(random.random()*3)
            try:
              data['text'] = (ts.translate(text='\n'.join(data['text'].to_list()), src=pivot, dest=back_target).text).split('\n')
            except:
              print(f'An exception occurred on index {i}')

          for j in data.iterrows():
            spamwriter.writerow(j[1].to_list())
          
        print(f'\r{pivot} -> {back_target}: 100%')


def check_params(args=None):
  
  parser = argparse.ArgumentParser(description='Language Model Encoder')

  parser.add_argument('-task', metavar='task', help='Phase')
  parser.add_argument('-sf', metavar='source', help='Source File for pivot language')
  parser.add_argument('-of', metavar='output', help='Output file for pivot language')
  return parser.parse_args(args)


if __name__ == '__main__':

  parameters = check_params(sys.argv[1:])

  task = parameters.task
  sourceFile = parameters.sf
  outputFile = parameters.of

  if task == 'pivot':
    TranslatePivotLang(sourceFile, outputFile)
  if task == 'back':
    backTranslation(sourceFile)
# %%
