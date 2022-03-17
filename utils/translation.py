#%%
from googletrans import Translator 
import pandas as pd
import os, csv, time, random
from params import params

def TranslatePivotLang():
  for target_lang in ['en', 'es', 'fr', 'de']:

    print(f'Pivot Language {target_lang}: 0%', end="")

    perc = 0
    data_frame = pd.read_csv(os.path.join(params.root, 'data/augmented.csv'),dtype=str)

    with open(os.path.join(params.root, f'data/{target_lang}.csv'), 'at', newline='', encoding="utf-8") as csvfile:
      spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
      spamwriter.writerow(params.columns)
      
      for i in range(0, len(data_frame), 20):
        
        if (i*100.0)/len(data_frame) - perc > 1:
          perc = (i*100.0)/len(data_frame)
          print(f'\rPivot Language {target_lang}: {perc:.2f}%', end = "")

        data = data_frame[i:i + 20].copy()
        
        if len(set(data['lang'].to_list())) == 1 and data.iloc[0]['lang'] != target_lang:
          ts = Translator()
          time.sleep(random.random()*3)
          try:
            data['tweet'] = (ts.translate(text='\n'.join(data['tweet'].to_list()), src=data.iloc[0]['lang'], dest=target_lang).text).split('\n')
          except:
            print(f'An exception occurred on index {i}')
        elif len(set(data['lang'].to_list())) > 1:
          ts = Translator()
          for j in range(20):
            if data.iloc[j]['lang'] != target_lang:
              data.iloc[j]['tweet'] = ts.translate(text=data.iloc[j]['tweet'], dest=target_lang, src = data.iloc[j]['lang']).text
              time.sleep(random.random()*3)

        for j in data.iterrows():
          spamwriter.writerow(j[1].to_list())
        
    print(f'\rPivot Language {target_lang}: 100%')

for back_target in ['en', 'es']:

  with open(os.path.join(params.root, f'data/back_to_{back_target}.csv'), 'wt', newline='', encoding="utf-8") as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(params.columns)

    for pivot in ['en', 'es', 'fr', 'de']:
      data_frame = pd.read_csv(os.path.join(params.root, f'data/{pivot}.csv'), dtype=str)
     
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
            data['tweet'] = (ts.translate(text='\n'.join(data['tweet'].to_list()), src=pivot, dest=back_target).text).split('\n')
          except:
            print(f'An exception occurred on index {i}')

        for j in data.iterrows():
          spamwriter.writerow(j[1].to_list())
        
      print(f'\r{pivot} -> {back_target}: 100%')

   
# %%
